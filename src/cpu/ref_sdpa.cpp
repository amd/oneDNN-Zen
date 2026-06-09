/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "cpu/ref_sdpa.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// ---------------------------------------------------------------------------
// Reference implementation for Scaled Dot-Product Attention.
//
//   O = softmax( (Q * K^T) * scale , dim=Skv ) * V
//
// Tensor layout. The SDPA contract gives K the logical shape (N, H, D, Skv)
// (head-size axis before Skv), so scores = Q * K is a plain matmul:
//   Q : (N, H, Sq,  D)    queries
//   K : (N, H, D,   Skv)  keys (logical)
//   V : (N, H, Skv, D)    values
//   O : (N, H, Sq,  D)    output
//
// Layouts are fully stride-driven. Every axis of every tensor is addressed
// through its memory-descriptor stride, so dense, transposed (BHSD / BHDS
// views from K.transpose(-2,-1)), and packed/strided (e.g. fused-QKV BSHD)
// inputs are all read correctly -- nothing assumes a dense row-major order.
//
// Scope: f32 and bf16, no quantization, single-threaded. The math core is
// always f32; bf16 inputs are up-converted to f32 element-wise as they are
// read (via their strides) and the result is converted back to bf16 on write,
// so there is a single source of truth for the computation.
//
// Masking. Two forms are supported, applied to the raw (already scaled)
// scores before softmax:
//   * explicit additive buffer mask : scores[s] += mask[n, h, sq, s]
//   * top-left causal mask           : keys after the query position -> -inf
// The additive mask may be f32 or bf16 and 2D (Sq, Skv) or 4D (N, H, Sq, Skv);
// any size-1 axis broadcasts.
// ---------------------------------------------------------------------------

// Resolved view of the attention mask for the reference kernel. s_n/s_h/s_sq
// are the strides (in elements) of the outer axes; a 0 stride means that axis
// is broadcast (or absent, e.g. N/H for a 2D mask). The inner (Skv) axis is
// contiguous, so the mask for one (n, h, sq) row is just a plain array.
struct ref_mask_t {
    const void *base = nullptr; // additive mask buffer, or null
    data_type_t dt = data_type::undef;
    dim_t s_n = 0, s_h = 0, s_sq = 0;
    bool has_buffer = false;
    bool causal = false;
};

// Applies the additive buffer mask and/or causal masking to one score row.
// Causal "future" positions are set to -inf so softmax zeroes them out.
static void apply_mask(
        float *scores, const ref_mask_t &mk, int n, int h, int sq, int Skv) {
    if (mk.has_buffer) {
        // The mask row for this (n, h, sq); inner (Skv) axis is contiguous.
        const dim_t off = n * mk.s_n + h * mk.s_h + sq * mk.s_sq;
        if (mk.dt == data_type::f32) {
            const float *row = reinterpret_cast<const float *>(mk.base) + off;
            for (int s = 0; s < Skv; ++s)
                scores[s] += row[s];
        } else {
            const bfloat16_t *row
                    = reinterpret_cast<const bfloat16_t *>(mk.base) + off;
            for (int s = 0; s < Skv; ++s)
                scores[s] += static_cast<float>(row[s]);
        }
    }
    // Top-left causal: query sq attends to keys [0, sq]; mask out the future.
    if (mk.causal) {
        for (int s = sq + 1; s < Skv; ++s)
            scores[s] = -std::numeric_limits<float>::infinity();
    }
}

// Per-tensor element strides for Q/K/V/O, pulled from each memory descriptor.
// Logical axes: Q, O use (N, H, Sq, D); V uses (N, H, Skv, D); K is the
// transposed (N, H, D, Skv) view. Addressing every axis through its stride
// (rather than assuming a dense order) is what makes dense, transposed, and
// packed/strided inputs all work.
struct qkvo_strides_t {
    dim_t q_n, q_h, q_s, q_d;
    dim_t k_n, k_h, k_d, k_s; // K: (N, H, D, Skv)
    dim_t v_n, v_h, v_s, v_d; // V: (N, H, Skv, D)
    dim_t o_n, o_h, o_s, o_d;
};

// In-place numerically-stable softmax: x <- exp(x - max(x)) / sum(...).
// Subtracting the max keeps exp() from overflowing on large scores.
static void softmax_inplace(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i)
        mx = std::max(mx, x[i]);

    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    for (int i = 0; i < n; ++i)
        x[i] /= sum;
}

// scores[s] = (q · K[:, s]) * scale,  for s in [0, Skv). q points at one query
// row; element d is at q[d * st.q_d]. K element (d, s) is at
// k[d * st.k_d + s * st.k_s]. Reads promote to f32 (so bf16 up-converts).
template <typename T>
static void compute_attention_scores(const T *q, const T *k, float *scores,
        int Skv, int D, float scale, const qkvo_strides_t &st) {
    for (int s = 0; s < Skv; ++s) {
        float acc = 0.f;
        for (int d = 0; d < D; ++d)
            acc += static_cast<float>(q[d * st.q_d])
                    * static_cast<float>(k[d * st.k_d + s * st.k_s]);
        scores[s] = acc * scale;
    }
}

// out[d] = sum_s probs[s] * v[s, d],  for d in [0, D). Output element d is at
// o[d * st.o_d]; V element (s, d) is at v[s * st.v_s + d * st.v_d]. Arithmetic
// stays in f32 (probs kept f32, V widened on read) -- a high-precision
// reference, not bit-identical to a bf16 P*V GEMM.
template <typename T>
static void accumulate_weighted_values(T *o, const T *v, const float *probs,
        int Skv, int D, const qkvo_strides_t &st) {
    for (int d = 0; d < D; ++d) {
        float acc = 0.f;
        for (int s = 0; s < Skv; ++s)
            acc += probs[s] * static_cast<float>(v[s * st.v_s + d * st.v_d]);
        o[d * st.o_d] = static_cast<T>(acc);
    }
}

// Full SDPA for a single (batch, head, query) row: scores -> mask -> softmax
// -> V. q/k/v/o already point at this row's/head's base; strides index within.
// n/h/sq locate this row for mask addressing.
template <typename T>
static void attend_single_query(const T *q, const T *k, const T *v, T *o,
        int Skv, int D, float scale, const qkvo_strides_t &st,
        const ref_mask_t &mk, int n, int h, int sq) {
    std::vector<float> probs(Skv);
    compute_attention_scores(q, k, probs.data(), Skv, D, scale, st);
    apply_mask(probs.data(), mk, n, h, sq, Skv);
    softmax_inplace(probs.data(), Skv);
    accumulate_weighted_values(o, v, probs.data(), Skv, D, st);
}

// Top-level driver: walks every (n, h, sq), locating each tensor's row/head
// base via its strides, so dense, transposed, and packed layouts all work.
template <typename T>
static void sdpa_forward_ref(const T *Q, const T *K, const T *V, T *O, int N,
        int H, int Sq, int Skv, int D, float scale, const qkvo_strides_t &st,
        const ref_mask_t &mk) {

    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int sq = 0; sq < Sq; ++sq) {
                const T *q = Q + n * st.q_n + h * st.q_h + sq * st.q_s;
                const T *k = K + n * st.k_n + h * st.k_h;
                const T *v = V + n * st.v_n + h * st.v_h;
                T *o = O + n * st.o_n + h * st.o_h + sq * st.o_s;

                attend_single_query(
                        q, k, v, o, Skv, D, scale, st, mk, n, h, sq);
            }
}

// ---------------------------------------------------------------------------
// oneDNN glue: dispatch checks + extract pointers/dims and call sdpa_forward_ref.
// Each VDISPATCH_SDPA narrows the set of problems this kernel accepts; if a
// check fails the dispatcher tries the next impl on the list.
// ---------------------------------------------------------------------------
status_t ref_sdpa_fwd_t::pd_t::init(engine_t *engine) {
    using namespace data_type;
    const auto *d = desc();

    // Shape: only the canonical 4D (N, H, S, D) form is handled.
    VDISPATCH_SDPA(d->q_desc.ndims == 4 && d->k_desc.ndims == 4
                    && d->v_desc.ndims == 4 && d->dst_desc.ndims == 4,
            "ref_sdpa: expected 4D Q/K/V/dst");

    // Datatype: f32 or bf16, but uniform across all tensors. bf16 is handled by
    // up-converting to f32 for the math core (see execute()).
    const auto qdt = d->q_desc.data_type;
    VDISPATCH_SDPA(utils::one_of(qdt, f32, bf16)
                    && utils::everyone_is(qdt, d->k_desc.data_type,
                            d->v_desc.data_type, d->dst_desc.data_type),
            "ref_sdpa: only uniform f32 or bf16 supported");

    // Masks: explicit additive buffer masks (f32/bf16, 2D or 4D) and top-left
    // causal masks are handled in execute().
    if (with_attn_mask()) {
        const auto &mmd = d->attn_mask_desc;
        VDISPATCH_SDPA(utils::one_of(mmd.data_type, f32, bf16),
                "ref_sdpa: attn mask must be f32 or bf16");
        VDISPATCH_SDPA(utils::one_of(mmd.ndims, 2, 4),
                "ref_sdpa: attn mask must be 2D or 4D");
        VDISPATCH_SDPA(
                memory_desc_wrapper(&mmd).blocking_desc().strides[mmd.ndims - 1]
                        == 1,
                "ref_sdpa: attn mask inner (Skv) axis must be contiguous");
    }
    VDISPATCH_SDPA(d->mask_type != attn_mask_type::bottom_right,
            "ref_sdpa: bottom-right causal mask not supported");

    // Unsupported features — keep the reference small; reject loudly so the
    // dispatcher can pick a richer impl when one exists.
    VDISPATCH_SDPA(!with_key_scales() && !with_value_scales(),
            "ref_sdpa: KQ/VS quantization not supported");
    VDISPATCH_SDPA(d->softmax_alg == alg_kind::softmax_accurate,
            "ref_sdpa: only softmax_accurate supported");

    // Require concrete (non-any) layouts. execute() addresses every tensor
    // through its per-axis strides, so dense, transposed (BHSD / BHDS views),
    // and packed/strided inputs are all supported.
    VDISPATCH_SDPA(set_default_formats(), "ref_sdpa: bad default formats");
    return status::success;
}

status_t ref_sdpa_fwd_t::execute(const exec_ctx_t &ctx) const {
    const auto *d = pd()->desc();

    // Shapes. K is logically (N, H, D, Skv), so Skv is its LAST dim; D is Q's.
    const int N = static_cast<int>(d->q_desc.dims[0]);
    const int H = static_cast<int>(d->q_desc.dims[1]);
    const int Sq = static_cast<int>(d->q_desc.dims[2]);
    const int D = static_cast<int>(d->q_desc.dims[3]);
    const int Skv = static_cast<int>(d->k_desc.dims[3]);

    // Per-axis strides for every tensor, so any layout (dense, transposed, or
    // packed/strided) is read correctly. K's logical axes are (N, H, D, Skv).
    const memory_desc_wrapper q_mdw(&d->q_desc), k_mdw(&d->k_desc),
            v_mdw(&d->v_desc), o_mdw(&d->dst_desc);
    const auto *qs = q_mdw.blocking_desc().strides;
    const auto *ks = k_mdw.blocking_desc().strides;
    const auto *vs = v_mdw.blocking_desc().strides;
    const auto *os = o_mdw.blocking_desc().strides;
    qkvo_strides_t st {};
    st.q_n = qs[0];
    st.q_h = qs[1];
    st.q_s = qs[2];
    st.q_d = qs[3];
    st.k_n = ks[0];
    st.k_h = ks[1];
    st.k_d = ks[2];
    st.k_s = ks[3];
    st.v_n = vs[0];
    st.v_h = vs[1];
    st.v_s = vs[2];
    st.v_d = vs[3];
    st.o_n = os[0];
    st.o_h = os[1];
    st.o_s = os[2];
    st.o_d = os[3];

    // Standard SDPA scale; explicit scale tensors aren't accepted here.
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Resolve the mask view (additive buffer and/or causal) once up front; it
    // applies identically to the f32 and bf16 code paths below.
    ref_mask_t mk;
    mk.causal = pd()->with_causal_mask();
    if (pd()->with_attn_mask()) {
        const memory_desc_wrapper m_mdw(&d->attn_mask_desc);
        const auto *str = m_mdw.blocking_desc().strides;
        const auto *dim = d->attn_mask_desc.dims;
        const int nd
                = d->attn_mask_desc.ndims; // 2 (Sq, Skv) or 4 (N, H, Sq, Skv)
        mk.has_buffer = true;
        mk.dt = d->attn_mask_desc.data_type;
        mk.base = CTX_IN_MEM(const void *, DNNL_ARG_ATTN_MASK);
        // Outer-axis strides; a size-1 or absent (2D) axis broadcasts -> 0.
        // The inner (Skv) axis is contiguous (enforced at init).
        mk.s_sq = (dim[nd - 2] > 1) ? str[nd - 2] : 0;
        mk.s_h = (nd == 4 && dim[1] > 1) ? str[1] : 0;
        mk.s_n = (nd == 4 && dim[0] > 1) ? str[0] : 0;
    }

    // Run the f32 or bf16 kernel directly on the user buffers; bf16 elements
    // are up-/down-converted on the fly as they are read/written via strides.
    if (d->q_desc.data_type == data_type::f32) {
        sdpa_forward_ref<float>(CTX_IN_MEM(const float *, DNNL_ARG_QUERIES),
                CTX_IN_MEM(const float *, DNNL_ARG_KEYS),
                CTX_IN_MEM(const float *, DNNL_ARG_VALUES),
                CTX_OUT_MEM(float *, DNNL_ARG_DST), N, H, Sq, Skv, D, scale, st,
                mk);
    } else { // bf16
        sdpa_forward_ref<bfloat16_t>(
                CTX_IN_MEM(const bfloat16_t *, DNNL_ARG_QUERIES),
                CTX_IN_MEM(const bfloat16_t *, DNNL_ARG_KEYS),
                CTX_IN_MEM(const bfloat16_t *, DNNL_ARG_VALUES),
                CTX_OUT_MEM(bfloat16_t *, DNNL_ARG_DST), N, H, Sq, Skv, D,
                scale, st, mk);
    }
    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
