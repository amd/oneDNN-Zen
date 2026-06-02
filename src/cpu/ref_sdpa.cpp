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
#include <cstdio>
#include <limits>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/sdpa_types.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// ---------------------------------------------------------------------------
// Static reference SDPA implementation.
//
// Layout (matches sdpa_desc_t accessors):
//   Q : (N, H, Sq,  D )    queries  -> dims[ndims-2]=Sq, dims[ndims-1]=D
//   K : (N, H, D,   Skv)   keys     -> dims[ndims-1]=Skv  (K is "pre-transposed")
//   V : (N, H, Skv, D )    values   -> dims[ndims-1]=D
//   O : (N, H, Sq,  D )    output
//
// Math:
//   S(b, sq, skv) = sum_d  Q(b, sq, d) * K(b, d, skv)
//   S            *= scale                       (default 1/sqrt(D))
//   S(b, sq, skv) += -inf where causal mask hides keys
//   P            = softmax(S, axis=skv)
//   O(b, sq, d)  = sum_skv P(b, sq, skv) * V(b, skv, d)
//
// Reference scope (intentionally narrow):
//   * f32 only
//   * 4D Q/K/V/dst
//   * Default scale 1/sqrt(D) when no explicit scale tensor is provided.
//     If with_attn_scale() is true, the first element of scale_md is used as
//     a host scalar (host_scalar format or single-element buffer).
//   * Causal mask: top_left and bottom_right supported.
//   * Explicit attention-mask tensor: NOT supported (rejected at init).
//   * Quant (kq/vs scales / zps) and softmax_alg != softmax_accurate: NOT
//     supported (rejected at init).
// ---------------------------------------------------------------------------

status_t ref_sdpa_fwd_t::pd_t::init(engine_t *engine) {
    using namespace data_type;
    const auto *d = desc();

    VDISPATCH_SDPA(d->q_desc.ndims == 4 && d->k_desc.ndims == 4
                    && d->v_desc.ndims == 4 && d->dst_desc.ndims == 4,
            "ref_sdpa: expected 4D Q/K/V/dst");

    VDISPATCH_SDPA(utils::everyone_is(f32, d->q_desc.data_type,
                           d->k_desc.data_type, d->v_desc.data_type,
                           d->dst_desc.data_type),
            "ref_sdpa: only f32 supported in reference impl");

    // Reject features we don't compute in the reference yet.
    VDISPATCH_SDPA(!with_attn_mask(),
            "ref_sdpa: explicit attention-mask tensor not supported");
    VDISPATCH_SDPA(!with_key_scales() && !with_value_scales(),
            "ref_sdpa: KQ/VS quantization not supported");
    VDISPATCH_SDPA(d->softmax_alg == alg_kind::softmax_accurate,
            "ref_sdpa: only softmax_accurate is supported");

    // Q/K/V/dst dim consistency.
    const dim_t Sq  = d->q_desc.dims[2];
    const dim_t Dq  = d->q_desc.dims[3];
    const dim_t Dk  = d->k_desc.dims[2];
    const dim_t Skv = d->k_desc.dims[3];
    const dim_t Sv  = d->v_desc.dims[2];
    const dim_t Dv  = d->v_desc.dims[3];
    VDISPATCH_SDPA(Dq == Dk, "ref_sdpa: Q.D must equal K.D");
    VDISPATCH_SDPA(Skv == Sv, "ref_sdpa: K.Skv must equal V.Skv");
    VDISPATCH_SDPA(Dq == Dv, "ref_sdpa: Q.D must equal V.D");
    VDISPATCH_SDPA(d->dst_desc.dims[2] == Sq && d->dst_desc.dims[3] == Dq,
            "ref_sdpa: dst dims must match (Sq, D)");

    VDISPATCH_SDPA(set_default_formats(), "ref_sdpa: bad default formats");

    return status::success;
}

status_t ref_sdpa_fwd_t::execute(const exec_ctx_t &ctx) const {
    const auto *d = pd()->desc();

    const auto *Q   = CTX_IN_MEM(const float *, DNNL_ARG_QUERIES);
    const auto *K   = CTX_IN_MEM(const float *, DNNL_ARG_KEYS);
    const auto *V   = CTX_IN_MEM(const float *, DNNL_ARG_VALUES);
    auto       *DST = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    const memory_desc_wrapper q_w(&d->q_desc);
    const memory_desc_wrapper k_w(&d->k_desc);
    const memory_desc_wrapper v_w(&d->v_desc);
    const memory_desc_wrapper o_w(&d->dst_desc);

    const dim_t N   = d->q_desc.dims[0];
    const dim_t H   = d->q_desc.dims[1];
    const dim_t Sq  = d->q_desc.dims[2];
    const dim_t D   = d->q_desc.dims[3];
    const dim_t Skv = d->k_desc.dims[3]; // K is (N, H, D, Skv)

    // Resolve attention scale.
    // Default: 1/sqrt(D). With explicit scale tensor (host-scalar today), use it.
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    if (pd()->with_attn_scale()) {
        const auto *attn_scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
        if (attn_scale) {
            const float s = attn_scale[0];
            scale = d->invert_scale ? (1.0f / s) : s;
        }
    }

    const bool causal = pd()->with_causal_mask();
    const bool causal_bottom_right
            = (d->mask_type == attn_mask_type::bottom_right);
    // top_left: position(skv) > position(sq) is masked.
    // bottom_right: shift so the last query attends to all keys.
    //   masked iff skv > sq + (Skv - Sq).
    const dim_t shift = causal_bottom_right ? (Skv - Sq) : 0;

    const float NEG_INF = -std::numeric_limits<float>::infinity();

    parallel_nd(N, H, Sq, [&](dim_t n, dim_t h, dim_t sq) {
        std::vector<float> S(static_cast<size_t>(Skv), 0.0f);

        // S[skv] = sum_d Q[n,h,sq,d] * K[n,h,d,skv]
        float row_max = NEG_INF;
        for (dim_t skv = 0; skv < Skv; ++skv) {
            const bool masked
                    = causal && (skv > sq + shift);
            if (masked) {
                S[skv] = NEG_INF;
                continue;
            }
            float acc = 0.0f;
            for (dim_t dd = 0; dd < D; ++dd) {
                acc += Q[q_w.off(n, h, sq, dd)]
                        * K[k_w.off(n, h, dd, skv)];
            }
            acc *= scale;
            S[skv] = acc;
            if (acc > row_max) row_max = acc;
        }

        // Numerically-stable softmax along skv.
        if (row_max == NEG_INF) {
            // Fully masked row: output zeros.
            std::fill(S.begin(), S.end(), 0.0f);
        } else {
            float sum = 0.0f;
            for (dim_t skv = 0; skv < Skv; ++skv) {
                const float e
                        = (S[skv] == NEG_INF) ? 0.0f : std::exp(S[skv] - row_max);
                S[skv] = e;
                sum += e;
            }
            const float inv_sum
                    = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (dim_t skv = 0; skv < Skv; ++skv) S[skv] *= inv_sum;
        }

        // O[n,h,sq,d] = sum_skv P[skv] * V[n,h,skv,d]
        for (dim_t dd = 0; dd < D; ++dd) {
            float acc = 0.0f;
            for (dim_t skv = 0; skv < Skv; ++skv) {
                acc += S[skv] * V[v_w.off(n, h, skv, dd)];
            }
            DST[o_w.off(n, h, sq, dd)] = acc;
        }
    });

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
