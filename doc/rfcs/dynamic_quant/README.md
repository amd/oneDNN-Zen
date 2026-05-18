# RFC: Dynamic Quantization Primitive in oneDNN

## Status
**Draft** &nbsp;|&nbsp; AMD-Zenai/oneDNN-ZenDNN &nbsp;|&nbsp; Branch: `rfc/zendnn-integration`

## Authors
- AMD ZenDNN team

## Summary

This RFC proposes adding a new `dynamic_quantize` primitive to oneDNN. The primitive computes the quantization parameters (scale and, for asymmetric mode, zero-point) **at runtime from the source data** at the granularity requested by the user (per-tensor, per-row, per-column, or per-group along either axis), then quantizes the source into an INT8 / UINT8 destination &mdash; in a single fused pass.

oneDNN today only supports *static* quantization: the user provides scales and zero-points (via `attr.scales` / `attr.zero_points`) and the library applies them. There is no primitive that can derive scales / zero-points from input data at inference time. This is the core operation behind dynamic-quantised LLM inference (activation quantisation per token, per row, or per group of rows / cols), and it is the missing primitive that frameworks have to implement on top of oneDNN today &mdash; via two memory passes and a separate reduction kernel &mdash; instead of asking oneDNN for it.

The kernels for this contribution come from AMD ZenDNN's existing reorder-API surface, where dynamic quantisation already ships with all the required granularities and is in production use through `zentorch`. As with the companion `embedding_bag` RFC, this RFC proposes upstreaming the kernels **natively** into oneDNN's `src/cpu/x64/` tree as **AVX-512 intrinsics** &mdash; no Xbyak, no JIT, no link-time or runtime dependency on ZenDNN. A portable reference implementation under `src/cpu/` is always available.

A standalone primitive is the right shape for this operation in oneDNN because it has multiple outputs (quantised dst + computed scale + optional zero-point) and a compute-only mode (compute scale / zp without writing the quantised dst), neither of which fits cleanly under the existing `reorder` primitive's single-output, parametric-scales model.

---

## 1. Background and Motivation

### 1.1 What dynamic quantisation is, and why it matters

In a standard quantised inference path, scales and zero-points are *known ahead of time* &mdash; either baked into the model from a calibration pass (static quantisation), or fixed by design (e.g. weight-only quantisation with frozen scales). oneDNN's existing `attr.scales` / `attr.zero_points` machinery covers exactly that: the caller provides the parameters and oneDNN applies them.

Dynamic quantisation is the opposite shape. The activation tensor changes every inference step, so the scale and zero-point need to be recomputed *per call* from the actual data, at the right granularity (per-tensor / per-row / per-group), before the matmul or convolution that consumes them. This is the dominant pattern in modern LLM inference at INT8 / FP8 / INT4 / UINT8 precisions:

- **Per-token activation quantisation** (per row of `[seq_len, hidden]`) used by AWQ, SmoothQuant, GPTQ-style decode paths, and most "W4A8" / "W8A8" production deployments.
- **Per-group quantisation** (groups of K columns or M rows) used by AWQ, GPTQ, and modern MoE inference where each expert's activation distribution is different.
- **Per-tensor quantisation** for low-overhead cases on smaller activations.

Today, every framework that wants this on CPU has to implement it itself &mdash; typically as a min/max reduction kernel followed by a separate quantisation kernel &mdash; or rely on a vendor library that does. The two-pass approach over the same activation tensor is wasteful (memory bandwidth dominated) and prevents fusion with the downstream operator.

### 1.2 What ZenDNN already provides

ZenDNN ships dynamic quantisation today as an extension of its reorder API: setting `dynamic_quant = true` in `reorder_params_t` switches the path from "apply caller-supplied scales/zps" to "compute scales/zps from input data, write them to caller-supplied output buffers, and produce the quantised dst" &mdash; all in a single fused AVX-512 pass with multiple threading strategies tuned for AMD CPUs. All five granularities are supported (per-tensor, per-row, per-col, per-group-row, per-group-col), in both symmetric (S8 output) and asymmetric (U8 output) modes, with a compute-only mode that skips the quantised dst write.

### 1.3 What this RFC proposes

Upstream that capability into oneDNN as a first-class primitive `dynamic_quantize`, re-implemented natively in oneDNN's `src/cpu/x64/` tree using AVX-512 intrinsics. After this contribution lands, oneDNN consumers (PyTorch, TensorFlow, ONNX Runtime, vLLM, llama.cpp via custom paths) get a single oneDNN call that:

1. Scans the source tensor at the requested granularity to find min/max.
2. Computes scale (and optionally zero-point) into caller-supplied output buffers.
3. Quantises the source into an INT8 / UINT8 destination using the just-computed parameters.

All in one pass over the source data, with cache-resident intermediates and per-CCD-aware threading.

### 1.4 Library functionality criteria (CONTRIBUTING.md)

| Criterion | How |
|---|---|
| **Performance** | Single-pass dynamic quantisation is the bottleneck of the activation path in dynamic-quantised LLM inference. Replacing two memory passes (min/max + quantise) with one fused AVX-512 pass measurably reduces both latency and memory bandwidth pressure. Validated at model level via vLLM W8A8 / W4A8 inference before each production PR. |
| **Generality** | The primitive's semantics (granularity expressed via output-tensor shape; symmetric vs asymmetric via presence of a zero-point output) match every modern framework's dynamic-quant abstraction. PyTorch (`torch.quantize_per_tensor` / `_per_channel` / dynamic), TensorFlow (`tf.quantization.quantize`), ONNX Runtime's `DynamicQuantizeLinear`, and vLLM's per-token quantisation all map to this primitive. |
| **Complexity** | High-quality dynamic-quant CPU kernels combine: SIMD min/max reductions over arbitrary-shape groups, fused scale/zp computation in floating point, and saturating cast to INT8/UINT8 &mdash; with multiple threading strategies depending on granularity. Frameworks repeatedly re-implement this; centralising it in oneDNN saves duplication. |

## 2. Goals and Non-Goals

### Goals
- One new primitive kind: `dnnl_dynamic_quantize`.
- Forward pass on CPU, single-pass: scan + compute params + quantise.
- Two algorithm modes: **symmetric** (`scale` only, zero-point implicitly 0, INT8 output) and **asymmetric** (`scale` + `zero_point`, UINT8 output).
- All five granularities in scope: **per-tensor**, **per-row**, **per-col**, **per-group-row**, **per-group-col**, expressed via the shape of the scale/zp memory descriptors.
- Compute-only mode (skip dst, only emit scale/zp) selectable via a flag.
- Source dtypes: `f32`, `bf16` (Phase 1). Destination dtypes: `s8` (symmetric), `u8` (asymmetric). Scale dtype: `f32` (Phase 1; `bf16` Phase 2).
- AVX-512 / AVX-512-FP16 intrinsic kernel native to oneDNN; no Xbyak / no JIT (consistent with the `embedding_bag` companion RFC).
- benchdnn driver, gtest coverage, example, user-guide doc.

### Non-Goals (initial release)
- **Backward / training.** No gradient computation; dynamic quantisation is an inference-side primitive.
- **GPU paths.** GPU impl-list `CASE` returns `empty_list` initially.
- **Fusion with matmul.** A user can call `dynamic_quantize` then call `matmul` separately with the produced scale / zp passed as `attr.scales` / `attr.zero_points`. A future RFC may propose a fused `matmul_with_dynamic_quant` primitive or a primitive-attribute extension; this RFC does not block that path but does not deliver it.
- **INT4 / FP8 / FP4 destination dtypes.** Tracked as Phase 3; the AVX-512 kernel structure generalises naturally but the saturating-cast paths and storage-packing are non-trivial and worth their own design pass.
- **Stochastic rounding modes.** Round-to-nearest-even only in this RFC.
- **Static / WoQ quantisation.** Already covered by oneDNN's existing `attr.scales` / `attr.zero_points` and not changed by this RFC.

## 3. Operation Definition

Let `S` be a 2D source tensor of shape `[M, N]` (higher-rank src is supported by treating the trailing two dims as `[M, N]` and broadcasting; see §3.4). Let `G_md` denote the granularity, expressed by the dimensions of the scale memory descriptor (and, for asymmetric mode, the zero-point memory descriptor, which must match).

### 3.1 Granularity → group shape

| Granularity | `scale_md.dims` (2D) | Output count | Group shape over `S` |
| --- | --- | --- | --- |
| Per-tensor | `[1, 1]` | `1` | one group covering the whole `M × N` |
| Per-row (per-token) | `[M, 1]` | `M` | one group per row of `S`: `S[i, :]` |
| Per-col | `[1, N]` | `N` | one group per col of `S`: `S[:, j]` |
| Per-group-row | `[Gr, N]` with `Gr` dividing `M` | `Gr × N` | rows of `S` partitioned into `Gr` contiguous row-groups; one group per `(row-group, col)` |
| Per-group-col | `[M, Gc]` with `Gc` dividing `N` | `M × Gc` | cols of `S` partitioned into `Gc` contiguous col-groups; one group per `(row, col-group)` |

The scale and zero-point descriptors must have identical `dims`. The destination memory descriptor `dst_md` must match `src_md.dims`.

For higher-rank sources, the same rules apply with the leading dims treated as a batch axis: `scale_md.dims = [..., Gr, Gc]` produces one set of scale / zp values per batch element.

### 3.2 Symmetric mode (S8 output, zero-point implicitly 0)

For each group `g` defined by the granularity:

```
abs_max(g) = max(|S[i]| : i ∈ g)
scale(g)   = abs_max(g) / 127.0          ; in f32 (or bf16) per scale_md.data_type
                                         ; eps-floored to avoid divide-by-zero
zp(g)      = 0                           ; not stored; zero_point output is unused
dst[i]     = saturate_cast<int8>(round(S[i] / scale(g))),  for i ∈ g
```

The destination dtype is `s8` and the value range is `[-127, +127]` (`-128` is intentionally not produced &mdash; matches PyTorch's symmetric int8 convention).

### 3.3 Asymmetric mode (U8 output, zero-point computed)

For each group `g`:

```
min(g)   = min(S[i] : i ∈ g)
max(g)   = max(S[i] : i ∈ g)
scale(g) = (max(g) - min(g)) / 255.0     ; eps-floored
zp(g)    = round(-min(g) / scale(g))     ; clamped to [0, 255]; stored as s32
dst[i]   = saturate_cast<uint8>(round(S[i] / scale(g)) + zp(g)),  for i ∈ g
```

The destination dtype is `u8` and the value range is `[0, 255]`. The zero-point output dtype is `s32` (oneDNN's standard zero-point type) to match the `attr.zero_points` convention used elsewhere.

### 3.4 Compute-only mode

When the `dnnl_dynamic_quantize_compute_only` flag is set in the op descriptor's flags bitmask, the primitive performs only the min/max scan and writes the scale (and, in asymmetric mode, the zero-point) outputs. The `dst_md` may be `zero_md`, and `DNNL_ARG_DST` is omitted from the args map at execute time. This mode is the one frameworks need when they want to compute dynamic-quant parameters once and reuse them across multiple consumer kernels.

In compute-only mode the kernel does the same min/max reduction but skips the per-element quantise loop, halving the memory-traffic-per-element and fitting cleanly inside the existing parallel scheme.

### 3.5 Edge cases and validation

- `abs_max(g) == 0` (all-zero group): `scale(g)` is set to a small epsilon (`1e-30f`) to avoid divide-by-zero; the resulting dst is all-zero, which round-trips dequantisation correctly.
- `min(g) > 0` or `max(g) < 0`: still valid; the formulas reduce to a uniform shift. The kernel does not special-case these.
- `M` not divisible by `Gr` (per-group-row) or `N` not divisible by `Gc` (per-group-col): rejected at PD `init()` time with `status::invalid_arguments`. No padding is applied.
- `padding_idx`-like skip semantics: not defined for dynamic quant. Every element of every group contributes to the min/max.

## 4. Public API Design

### 4.1 Primitive kind

```c
typedef enum {
    /* ... existing kinds ... */
    dnnl_dynamic_quantize,            /* new */
    /* ... */
} dnnl_primitive_kind_t;
```

C++ mirror in `dnnl::primitive::kind`:

```cpp
enum class kind {
    /* ... */
    dynamic_quantize = dnnl_dynamic_quantize,
};
```

### 4.2 Algorithm enum

Two new entries in `dnnl_alg_kind_t`, in a fresh numeric band to avoid collisions:

```c
typedef enum {
    /* ... */
    dnnl_dynamic_quantize_symmetric  = 0x50000,  /* S8 output, zp implicit 0 */
    dnnl_dynamic_quantize_asymmetric,            /* U8 output, zp computed     */
    /* ... */
} dnnl_alg_kind_t;
```

C++ aliases under `dnnl::algorithm::dynamic_quantize_symmetric` and `dnnl::algorithm::dynamic_quantize_asymmetric`.

### 4.3 Flags

Mirrors `dnnl_normalization_flags_t` shape:

```c
typedef enum {
    dnnl_dynamic_quantize_flags_none    = 0x0,
    dnnl_dynamic_quantize_compute_only  = 0x1,   /* skip dst write; emit scale/zp only */
} dnnl_dynamic_quantize_flags_t;
```

### 4.4 Argument map

| Argument | Memory | Required for |
| --- | --- | --- |
| `DNNL_ARG_SRC` | source tensor `[..., M, N]`, dtype `f32` or `bf16` | always |
| `DNNL_ARG_DST` | quantised output tensor, same shape as `SRC`, dtype `s8` (symmetric) or `u8` (asymmetric) | full quantise mode (i.e. when `compute_only` flag is **not** set) |
| `DNNL_ARG_DYNAMIC_SCALE` | computed scales, shape per granularity (§3.1), dtype `f32` (Phase 1) or `bf16` (Phase 2) | always |
| `DNNL_ARG_DYNAMIC_ZERO_POINT` | computed zero-points, same shape as scale, dtype `s32` | asymmetric mode only |
| `DNNL_ARG_SCRATCHPAD` | scratchpad | when `attr.scratchpad_mode = user` |

Two new `DNNL_ARG_*` constants are added as aliases in `include/oneapi/dnnl/dnnl_types.h`:

```c
#define DNNL_ARG_DYNAMIC_SCALE        DNNL_ARG_DST_1
#define DNNL_ARG_DYNAMIC_ZERO_POINT   DNNL_ARG_DST_2
```

Reusing the existing `DST_1` / `DST_2` slots keeps the argument-map layout consistent with how oneDNN already handles auxiliary outputs (e.g., RNN's `DST_ITER`).

### 4.5 Internal op descriptor

Lives in `src/common/opdesc.hpp` (consistent with every other primitive &mdash; never in the public C header):

```cpp
struct dynamic_quantize_desc_t : public op_desc_t {
    dynamic_quantize_desc_t() : op_desc_t(primitive_kind::dynamic_quantize) {}

    alg_kind_t      alg_kind {};
    memory_desc_t   src_desc;       // [..., M, N], f32 / bf16
    memory_desc_t   dst_desc;       // same shape, s8 (sym) or u8 (asym); zero_md if compute_only
    memory_desc_t   scale_desc;     // shape encodes granularity
    memory_desc_t   zero_point_desc;// matches scale_desc; zero_md if symmetric
    unsigned        flags { 0 };    // dnnl_dynamic_quantize_flags_t
};
```

### 4.6 C API

```c
/// Creates a primitive descriptor for a dynamic-quantize primitive.
dnnl_status_t DNNL_API dnnl_dynamic_quantize_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc,
        dnnl_engine_t engine,
        dnnl_alg_kind_t alg_kind,
        const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t dst_desc,        // NULL or zero-md when compute_only
        const_dnnl_memory_desc_t scale_desc,
        const_dnnl_memory_desc_t zero_point_desc, // NULL or zero-md when symmetric
        unsigned flags,
        const_dnnl_primitive_attr_t attr);
```

The implementation in `src/common/dynamic_quantize.cpp` follows the same pattern as `dnnl_softmax_forward_primitive_desc_create`: build a `dynamic_quantize_desc_t`, then call `primitive_desc_create(...)`.

### 4.7 C++ wrapper

```cpp
struct dynamic_quantize : public primitive {
    enum class flags : unsigned {
        none           = dnnl_dynamic_quantize_flags_none,
        compute_only   = dnnl_dynamic_quantize_compute_only,
    };

    struct primitive_desc : public dnnl::primitive_desc {
        primitive_desc() = default;

        primitive_desc(const engine &aengine,
                algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &dst_desc,
                const memory::desc &scale_desc,
                const memory::desc &zero_point_desc,
                flags  aflags = flags::none,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false);

        memory::desc src_desc()        const;
        memory::desc dst_desc()        const;
        memory::desc scale_desc()      const;
        memory::desc zero_point_desc() const;
    };

    dynamic_quantize() = default;
    explicit dynamic_quantize(const primitive_desc &pd) : primitive(pd.get()) {}
};
```

### 4.8 Validation rules

Performed in `dynamic_quantize_desc_init` and `dynamic_quantize_pd_t::init()`:

1. `src_desc` rank `>=` 2; trailing two dims `[M, N]` blocked or plain.
2. `src_desc.data_type` &isin; `{f32, bf16}` (Phase 1).
3. **Symmetric mode** (`alg_kind == dynamic_quantize_symmetric`):
   - `dst_desc.data_type == s8`, same shape as `src_desc`.
   - `zero_point_desc` must be `zero_md` (no zero-point output).
4. **Asymmetric mode**:
   - `dst_desc.data_type == u8`, same shape as `src_desc`.
   - `zero_point_desc.data_type == s32`, same dims as `scale_desc`.
5. `scale_desc.data_type` &isin; `{f32}` (Phase 1; `bf16` Phase 2).
6. `scale_desc.dims` matches one of the five granularity templates in §3.1; in particular for per-group modes, `Gr` must divide `M` and `Gc` must divide `N`.
7. **Compute-only flag**:
   - When set, `dst_desc` must be `zero_md`.
   - When unset, `dst_desc` must be a fully-described non-empty memory descriptor.
8. `attr` may include `scratchpad_mode` only (Phase 1); other attributes (post-ops, scales, zero-points) are rejected &mdash; they don't make sense for a quantising primitive whose outputs are themselves the parameters other primitives consume.

### 4.9 Attributes

| Attribute | Phase 1 | Phase 2 |
| --- | --- | --- |
| `scratchpad_mode` | yes | yes |
| `post_ops` | no | no (fundamentally doesn't apply) |
| `scales` / `zero_points` | no | no (the primitive *produces* these) |

## 5. Implementation Architecture

### 5.1 Source layout (additions only)

```
include/oneapi/dnnl/
  dnnl_types.h                              [+] primitive kind, alg kinds, flags, named ARG aliases
  dnnl.h                                    [+] dnnl_dynamic_quantize_primitive_desc_create
  dnnl.hpp                                  [+] dnnl::dynamic_quantize

src/common/
  opdesc.hpp                                [+] dynamic_quantize_desc_t
  dynamic_quantize.cpp                      [+] desc_init + C API entry point
  dynamic_quantize_pd.hpp                   [+] primitive-desc base class
  c_types_map.hpp                           [+] enum mirrors
  primitive_desc_iface.cpp                  [+] add to known_primitive_kind whitelist

src/cpu/
  cpu_engine.hpp                            [+] DECLARE_IMPL_LIST + CASE
  cpu_dynamic_quantize_pd.hpp               [+] thin CPU PD typedef
  cpu_dynamic_quantize_list.cpp             [+] impl_list registration
  ref_dynamic_quantize.hpp                  [+] reference implementation
  ref_dynamic_quantize.cpp                  [+]
  x64/dynamic_quantize.hpp                  [+] optimised x64 primitive (intrinsic-based, no JIT)
  x64/dynamic_quantize.cpp                  [+] AVX-512 / AVX-512-FP16 intrinsic kernels +
                                                ISA dispatch + execute()

src/gpu/
  gpu_impl_list.cpp                         [+] CASE returns empty_list (no GPU impl in Phase 1)

tests/
  benchdnn/dynamic_quantize/                [+] driver
  benchdnn/inputs/dynamic_quantize/         [+] test harnesses
  benchdnn/doc/driver_dynamic_quantize.md   [+] driver doc
  benchdnn.cpp                              [+] dispatch for --dynamic-quantize
  gtests/test_dynamic_quantize.cpp          [+] gtest coverage

examples/primitives/dynamic_quantize.cpp    [+] tutorial example
doc/primitives/dynamic_quantize.md          [+] user-guide markdown
```

The optimised file is `src/cpu/x64/dynamic_quantize.{hpp,cpp}` &mdash; **no `jit_` prefix**, consistent with the `embedding_bag` companion RFC's choice of intrinsics over JIT.

### 5.2 Primitive descriptor base

`src/common/dynamic_quantize_pd.hpp` defines `dynamic_quantize_pd_t : public primitive_desc_t` with:

- MD accessors: `src_md(0)`, `dst_md(0)`, `dst_md(1)` (= `scale_md`), `dst_md(2)` (= `zero_point_md`).
- `arg_usage(int arg)` returning `input` for `SRC`, `output` for `DST` / `DYNAMIC_SCALE` / `DYNAMIC_ZERO_POINT`. `DST` is `output_optional` when `compute_only` flag is set so the dispatcher does not require `DNNL_ARG_DST` in the args map.
- `arg_md(int arg)` mapping each arg to the right MD.
- Helpers: `is_symmetric()`, `is_asymmetric()`, `is_compute_only()`, `granularity()` (returns one of `per_tensor / per_row / per_col / per_group_row / per_group_col`), `M()`, `N()`, `Gr()`, `Gc()`.

`src/cpu/cpu_dynamic_quantize_pd.hpp` is a thin CPU-side typedef matching `src/cpu/cpu_softmax_pd.hpp`. CPU impls inherit `pd_t : public cpu_dynamic_quantize_pd_t` and use `DECLARE_COMMON_PD_T(...)` from `src/common/primitive_desc.hpp`.

### 5.3 Reference implementation

`src/cpu/ref_dynamic_quantize.{hpp,cpp}` &mdash; `ref_dynamic_quantize_t : public primitive_t`. Forward only; algorithm and granularity are read from `pd()->desc()` at execute time so a single class covers all combinations. The body uses `parallel_nd(num_groups, ...)` from `src/common/dnnl_thread.hpp` to parallelise across groups (per-group-row / per-group-col modes are the most parallel-friendly; per-tensor is single-group and parallelises over the single reduction internally with `parallel_reduce`-style splitting).

`pd_t::init()` skeleton:

```cpp
status_t init(engine_t *engine) {
    using namespace data_type;
    VDISPATCH_DYNAMIC_QUANTIZE(utils::one_of(src_md(0)->data_type, f32, bf16),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_DYNAMIC_QUANTIZE(check_granularity_dims(),
            VERBOSE_BAD_PARAM);
    VDISPATCH_DYNAMIC_QUANTIZE(set_default_formats() == status::success,
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_DYNAMIC_QUANTIZE(attr()->has_default_values(skip_mask_t::none),
            VERBOSE_UNSUPPORTED_ATTR);
    return status::success;
}
```

`VDISPATCH_DYNAMIC_QUANTIZE` is a new macro mirroring `VDISPATCH_SOFTMAX` in `src/common/softmax_pd.hpp`.

### 5.4 Optimised x64 implementation (AVX-512 intrinsics)

This is intentionally not a JIT primitive. Same rationale as the `embedding_bag` companion RFC: the kernel logic (min/max reduction over a group, then a scale/zp computation, then a saturating-cast quantise loop) is structurally simple, has no shape-dependent code-gen opportunity that benefits from Xbyak's runtime emission, and ZenDNN's existing intrinsic implementation is well-validated and ships today. Porting as intrinsics preserves that validation surface and keeps the diff readable for upstream reviewers. A future JIT impl can be slotted above the intrinsic one in the impl list if profiling ever justifies it (no API change).

**Class declaration** &mdash; `src/cpu/x64/dynamic_quantize.hpp`:

```cpp
struct dynamic_quantize_t : public primitive_t {
    struct pd_t : public cpu_dynamic_quantize_pd_t {
        using cpu_dynamic_quantize_pd_t::cpu_dynamic_quantize_pd_t;
        DECLARE_COMMON_PD_T(impl_name(), dynamic_quantize_t);

        status_t init(engine_t *engine);
        const char *impl_name() const;     // "intrin:avx512_core_*" / "intrin:avx2"

        cpu_isa_t isa_ = isa_undef;
        int nthr_ = 1;
    };

    explicit dynamic_quantize_t(const pd_t *apd) : primitive_t(apd) {}
    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    using kernel_fn_t = void (*)(const call_params_t *);
    kernel_fn_t kernel_fn_ = nullptr;     // resolved in primitive_t::init(), called in execute()
};
```

There is no `jit_generator_t`, no `Xbyak` headers, no `create_kernel()`, no `generate()`.

**ISA-templated kernel functions.** The intrinsic kernels are ordinary free functions templated on `cpu_isa_t isa`, source dtype, algorithm (symmetric / asymmetric), and granularity (so the inner loops are branch-free):

```cpp
template <cpu_isa_t isa, data_type_t src_dt, alg_kind_t alg,
          granularity_t gran, bool compute_only>
void dynamic_quantize_kernel(const call_params_t *p);   // pure C++ + AVX-512 intrinsics
```

Every supported `(isa, src_dt, alg, gran, compute_only)` instantiation is compiled once into the library at build time. The ISA-targeted compile flags (`-mavx512f -mavx512bw -mavx512vbmi -mavx512fp16` etc.) are applied per translation unit so that AVX-512 instructions never appear in code paths reached on hosts that don't have the ISA.

**Kernel pseudocode** (symmetric, per-row mode; the most common LLM activation case):

```cpp
template <cpu_isa_t isa, data_type_t src_dt>
void dynamic_quantize_kernel_symmetric_per_row(const call_params_t *p) {
    // For each row r in [r0, r1):
    //   1. SIMD min/max reduction over S[r, :] -> abs_max(r)
    //   2. scale(r) = abs_max(r) / 127.0   (with eps floor)
    //   3. For each col c: dst[r, c] = saturate_int8(round(S[r, c] / scale(r)))
    //   4. Write scale(r) to scale[r]
    // All three passes share one outer row-tile loop; the inner SIMD loop
    // streams once over the row and accumulates the abs_max in a vector
    // accumulator that is reduced to a scalar at row end.
}
```

**PD `init()`** picks `isa_` via a `mayiuse(avx512_core_fp16) -> avx512_core_bf16 -> avx512_core -> avx2` ladder. **`primitive_t::init()`** resolves `kernel_fn_` to the right specialisation through a small `static constexpr` lookup. **No JIT compilation, no allocation, no Xbyak.**

**`execute()`** retrieves args via `CTX_IN_MEM` / `CTX_OUT_MEM`, fans out work with `parallel_nd_ext(nthr, num_groups, ...)`, and invokes the kernel function pointer per work item. No allocations on the hot path.

### 5.5 Dispatch

`src/cpu/cpu_dynamic_quantize_list.cpp`:

```cpp
namespace dnnl::impl::cpu {
const impl_list_item_t *get_dynamic_quantize_impl_list(
        const dynamic_quantize_desc_t *desc) {
    UNUSED(desc);
    static constexpr impl_list_item_t impl_list[] = REG_DYNAMIC_QUANTIZE_P({
        CPU_INSTANCE_X64(dynamic_quantize_t)         // intrinsic-based
        CPU_INSTANCE(ref_dynamic_quantize_t)         // portable C++ reference
        nullptr,
    });
    return impl_list;
}
} // namespace dnnl::impl::cpu
```

`src/cpu/cpu_engine.hpp` adds `DECLARE_IMPL_LIST(dynamic_quantize);` and a `CASE(dynamic_quantize);` arm in the dispatch switch. `REG_DYNAMIC_QUANTIZE_P` is a new macro added to `src/common/impl_registration.hpp` mirroring `REG_SOFTMAX_P`.

### 5.6 Threading

The threading strategy varies with granularity and is the part of ZenDNN's parallel primitive that most directly translates here:

- **Per-tensor.** Single group covering `M × N` elements. Parallelise the min/max reduction across threads using a per-thread partial accumulator + a tree reduction, then a parallel quantise pass. Implemented with `parallel_nd_ext(nthr, M, ...)` over rows with shared scale.
- **Per-row (per-token).** Independent groups. `parallel_nd(M, [&](dim_t m) { ... })`. Optimal for LLM activations.
- **Per-col.** Independent groups across columns. `parallel_nd(N, [&](dim_t n) { ... })`. Memory access is column-major over a row-major tensor &mdash; the kernel uses tiled traversal to keep cache reuse high.
- **Per-group-row.** `parallel_nd(Gr * N, [&](dim_t i) { ... })` where each work item is one `(row-group, col)` pair.
- **Per-group-col.** `parallel_nd(M * Gc, [&](dim_t i) { ... })` where each work item is one `(row, col-group)` pair.

All threading uses oneDNN's standard `parallel_nd*` from `src/common/dnnl_thread.hpp`. There are no per-primitive threading knobs in the public API.

### 5.7 Performance considerations

The design deliberately avoids unnecessary layers in the hot path:

- **Validation only at PD `init()`.** No re-validation in `execute()`. ISA selection (`pd()->isa_`), scratchpad sizing, format defaults all resolved at PD time.
- **No JIT in the API flow.** `primitive_t::init()` resolves a function pointer; no Xbyak emit, no executable mapping at runtime.
- **Single-pass kernel.** The min/max reduction and the quantise loop share one outer tile loop; the source tensor is streamed exactly once, no extra memory pass over the data. This is the headline difference vs the framework-side two-pass approach.
- **Branch-light inner loop.** Algorithm (symmetric / asymmetric), granularity, and `compute_only` are template parameters on the kernel function so the compiler emits a separate specialisation per combination &mdash; no conditional branches inside the SIMD loop.
- **No scratchpad in the common case.** For per-row / per-col / per-group-row / per-group-col modes, the only intermediate is one scalar accumulator per work item, held in registers / spill slots. Per-tensor mode books per-thread partial accumulators (one f32 each) in the scratchpad.
- **`format_kind::any` for dst / scale / zp.** PDs let the impl pick layouts; the intrinsic impl prefers the same layout as `src_md` for `dst_md` (so the streaming pattern is identical) and plain layouts for scale / zp.

## 6. Data Type Support Matrix

`src` is the input; `dst` is the quantised output; `scale` and `zp` are the parameter outputs. Supported combinations per phase:

| Phase | `src` | `dst` (sym) | `dst` (asym) | `scale` | `zp` (asym) | Algorithms | Engine | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `f32` | `s8` | `u8` | `f32` | `s32` | sym, asym | CPU (ref + intrinsic x64) | Baseline. Intrinsic impl on `avx2` / `avx512_core`. |
| 2 | `bf16` | `s8` | `u8` | `f32` or `bf16` | `s32` | sym, asym | CPU (ref + intrinsic x64) | Requires `avx512_core_bf16` for the intrinsic path; ref always works. |
| 3 | `f32`, `bf16` | `s4`, `nf4`, `fp8` | `u4` | `f32` or `bf16` | `s32` | sym, asym | CPU (intrinsic x64) | Sub-byte / FP8 dtypes; storage packing handled inside the intrinsic kernel. Beyond Phase 1 scope. |
| Future | f32 / bf16 | s8 / u8 / sub-byte | as above | f32 / bf16 | s32 | sym, asym | GPU (Intel SYCL) | Not in this RFC. |

Granularity coverage is identical across phases &mdash; all five granularities listed in §3.1 are supported in every phase.

## 7. Testing Strategy

### 7.1 benchdnn driver

`tests/benchdnn/dynamic_quantize/` follows the pattern of `tests/benchdnn/reduction/`:

- `dynamic_quantize.hpp` &mdash; `prb_t`, `settings_t`, dim parsing, granularity parsing.
- `dynamic_quantize.cpp` &mdash; driver entry (`bench()`), problem creation, primitive lifecycle.
- `dynamic_quantize_aux.cpp` &mdash; algorithm / dtype / granularity parsers and pretty-printers.
- `ref_dynamic_quantize.cpp` &mdash; CPU-only reference for correctness checks.
- `bench_dynamic_quantize.cpp` &mdash; option parser entry called from `tests/benchdnn/benchdnn.cpp`.

`tests/benchdnn/benchdnn.cpp` adds `#include "dynamic_quantize/dynamic_quantize.hpp"` and a new `else if (!strcmp("--dynamic-quantize", argv[0])) { ... }` branch.

`tests/benchdnn/inputs/dynamic_quantize/`:

- `shapes_basic` &mdash; small cases for correctness across all granularities.
- `shapes_llm` &mdash; representative LLM activation shapes (`[seq, hidden]` for various seq / hidden).
- `test_dynamic_quantize_ci` &mdash; the CI harness fed by CTest.

CLI shape (proposed):

```
--dynamic-quantize --algo=sym --src-dt=bf16 --dst-dt=s8 \
                   --granularity=per-row --compute-only=false \
                   M:N
--dynamic-quantize --algo=asym --src-dt=f32 --dst-dt=u8 \
                   --granularity=per-group-row --gr=8 \
                   M:N
```

### 7.2 Unit tests

`tests/gtests/test_dynamic_quantize.cpp` covers:

- API surface: ctor / accessors / arg map / compute-only flag.
- Validation errors: wrong rank, wrong dtypes, granularity mismatch (`Gr` not dividing `M`, etc.), zero-point on symmetric mode, non-zero `dst_md` with `compute_only`, etc.
- Correctness on small shapes vs hand-computed values for each granularity / algorithm combo.

### 7.3 Examples

`examples/primitives/dynamic_quantize.cpp` shows two flows:

1. **Per-token activation quant.** Input `[seq=8, hidden=128]` BF16; `algorithm::dynamic_quantize_symmetric`; scale shape `[8, 1]`; produce `s8` activations + per-row `f32` scales.
2. **Per-group quant for AWQ-style.** Input `[M=64, N=128]` FP32; `algorithm::dynamic_quantize_asymmetric`; scale shape `[1, 4]` (per-group-col with `Gc=4`); produce `u8` quantised + scales + zero-points.

Both examples include a final dequantise loop that round-trips the input within numerical tolerance.

## 8. Documentation

`doc/primitives/dynamic_quantize.md` follows the structure of `doc/primitives/reduction.md`:

1. Title with Doxygen anchor: `Dynamic Quantize {#dev_guide_dynamic_quantize}`.
2. **General** &mdash; one-paragraph description with link to API ref.
3. **Operation Definition** &mdash; the formulas from §3 of this RFC.
4. **Execution Arguments** table: `DNNL_ARG_SRC`, `DNNL_ARG_DST`, `DNNL_ARG_DYNAMIC_SCALE`, `DNNL_ARG_DYNAMIC_ZERO_POINT`, `DNNL_ARG_SCRATCHPAD`.
5. **Granularity** &mdash; how the scale-tensor shape encodes the five granularities.
6. **Implementation Details** &mdash; supported dtypes per algorithm, compute-only mode, edge cases.
7. **Performance Tips** &mdash; choose the granularity that matches downstream consumer expectations; prefer per-row for LLM activations.

A short sentence in the developer-guide TOC links the new page.

## 9. Phasing and Delivery

Each gate has a concrete done-when criterion and the next gate cannot open until the previous one is closed.

| Gate | Scope | Done-when |
|---|---|---|
| **G1 &mdash; API plumbing + ref impl** | Public API (enums, op-desc, C / C++ entry points, whitelist, `c_types_map`); ref impl skeleton handling all five granularities and both algorithms for `f32 -> s8 / u8` | `dnnl::dynamic_quantize` PD can be created and executed against the ref impl for every granularity / algorithm combination. |
| **G2 &mdash; ref impl complete + tests + docs** | Ref impl correctness for all granularities, both algorithms, compute-only flag, edge cases; benchdnn driver passes `shapes_basic`; gtests; example; user-guide markdown | benchdnn `--dynamic-quantize` shape sets pass against ref. |
| **G3 &mdash; Optimised x64 intrinsic impl (`f32`)** | `src/cpu/x64/dynamic_quantize.{hpp,cpp}` for `f32 -> s8 / u8`, all granularities, both algorithms, `avx2` and `avx512_core` paths | Intrinsic impl wins dispatch over ref on AVX-512 hosts; benchdnn `shapes_llm` accuracy matches ref. |
| **G4 &mdash; `bf16` source path** | `bf16 -> s8 / u8` in the intrinsic impl; `avx512_core_bf16` ladder entry | Coverage on AVX-512 BF16 hosts; ref fallback verified on hosts without the ISA. |
| **G5 &mdash; Performance validation** | benchdnn perf on representative LLM shapes, vLLM W8A8 / W4A8 model-level | Measured speed-up vs framework-side two-pass dynamic quantisation reported in PR cover letter, satisfying the `CONTRIBUTING.md` performance gate. |
| **G6 &mdash; Sub-byte / FP8 destinations (Phase 3)** | `s4`, `nf4`, `fp8`, `u4` destinations; storage packing handled inside the intrinsic kernel | Out of scope for the first PR series; tracked here for completeness. |

## 10. Risks and Open Questions

1. **Multi-output primitive ergonomics.** oneDNN primitives traditionally have one DST. Expressing scale and zero-point as `DNNL_ARG_DST_1` / `DST_2` works mechanically but is novel enough that the named aliases (`DNNL_ARG_DYNAMIC_SCALE`, `DNNL_ARG_DYNAMIC_ZERO_POINT`) are important for readability. Mitigation: documented prominently in the user-guide doc and the example.
2. **Granularity expression via MD shape.** The granularity is implicit in `scale_md.dims`; reviewers may prefer an explicit enum (`per_tensor / per_row / ...`) on the op-desc. Option held open: an explicit `granularity_t` field can be added to the op-desc with an `auto` value that infers from MD, without breaking the proposed API. Decision deferred to first review.
3. **Scale dtype `bf16`.** Ship `f32` scales in Phase 1 and add `bf16` in Phase 2. Risk: frameworks already producing `bf16` scales will need a dtype conversion for one release. Acceptable.
4. **Compute-only via flag vs separate primitive.** The flag-vs-primitive choice is similar to the `embedding_lookup` algorithm choice in the `embedding_bag` companion RFC: both algorithms / modes share the same kernel structure, so a single primitive with a flag is cleaner than splitting.
5. **Per-tensor parallelism.** Per-tensor mode has a single output; the kernel parallelises the min/max reduction internally. Risk: small inputs may not benefit from threading. Mitigation: the kernel falls back to a single-thread path under a size threshold (resolved at PD time, not in `execute()`).
6. **Performance gate (CONTRIBUTING.md).** Every new primitive must show "material workload-level impact". Mitigation: ship benchdnn perf and vLLM W8A8 / W4A8 numbers with each production PR.

## 11. Alternatives Considered

### A. Extend `reorder` instead of adding a new primitive (rejected)

ZenDNN's reference implementation lives inside its reorder API (`reorder_params_t::dynamic_quant = true`). One option is to mirror that and extend oneDNN's `reorder` primitive with a `dynamic_quant` mode.

- **Pros:** No new primitive; smaller API surface.
- **Cons:** `reorder` has one DST; dynamic quant has up to three (DST + scale + zp). Bolting auxiliary outputs onto a single-output primitive breaks the abstraction. `reorder`'s scales / zps are *parametric inputs* today; flipping the same fields to be *outputs* in a special mode is confusing. Compute-only mode (no DST) is even harder to express on top of reorder.

**Decision:** Reject. Add a dedicated `dynamic_quantize` primitive with multiple DST args and an explicit `compute_only` flag.

### B. Express dynamic quantisation as an attribute on consumer primitives (rejected for now)

oneDNN's `attr.scales` / `attr.zero_points` could in principle carry a "compute-from-input" tag, so e.g. `matmul` would dynamically quantise its activation argument before computing.

- **Pros:** Tightest fusion with the consumer; no separate primitive call.
- **Cons:** Locks the dynamic-quant computation to a specific consumer primitive's pipeline; the same activation can't be reused across multiple consumers without re-quantising; the API change is invasive (attribute mode that produces side-effect outputs?). Complete redesign of the attribute system.

**Decision:** Reject for this RFC. A future RFC may revisit fused dynamic quant + matmul once the standalone primitive has a year of usage.

### C. AVX-512 intrinsics vs Xbyak JIT port (intrinsics chosen)

Same trade-off as the `embedding_bag` companion RFC. The kernel logic (min/max + scale/zp + saturating cast) is structurally simple, has no shape-dependent code-gen opportunity, and ZenDNN's existing intrinsic implementation is well-validated. Porting as intrinsics preserves that validation surface and lowers porting risk; a future Xbyak JIT impl can slot above the intrinsic one in the impl list if profiling ever justifies it &mdash; users see no API change.

## 12. End-to-End Lifecycle

```
user code (PyTorch / TF / vLLM / app)
            │
            ▼   dnnl::dynamic_quantize::primitive_desc(eng, alg, src_md,
                                                        dst_md, scale_md,
                                                        zero_point_md,
                                                        flags, attr)
            │
            ▼   C++ wrapper -> C API
   dnnl_dynamic_quantize_primitive_desc_create(...)
            │
            ▼   build internal dynamic_quantize_desc_t, validate
            │
            ▼   primitive-kind whitelist check
   known_primitive_kind contains dynamic_quantize ?
            │
            ▼   cpu_engine -> CASE(dynamic_quantize) -> impl_list
   ┌────────────────────────────────────────────────────────────┐
   │ dynamic_quantize_t::pd_t::init()  (intrinsic x64)          │
   │   - mayiuse(avx512_core_*) ladder -> isa_                  │
   │   - dtype + granularity + format + attr checks             │
   │   - status::success → selected                             │
   │   (or status::unimplemented → iterator advances)           │
   └────────────────────────────────────────────────────────────┘
            │ unimpl
            ▼
   ref_dynamic_quantize_t  (terminal fallback)
            │ success
            ▼
   primitive_desc ready  ▶  prim(pd);
            │
            ▼   primitive_t::create_primitive_common -> impl::init(engine)
   dynamic_quantize_t::init(engine):
       kernel_fn_ = pick_kernel(isa, src_dt, alg, granularity, compute_only);
       // NO Xbyak, NO create_kernel(), NO allocation
            │
            ▼   prim.execute(stream, args);
   dynamic_quantize_t::execute(ctx):
       src        = CTX_IN_MEM (... DNNL_ARG_SRC)
       dst        = CTX_OUT_MEM(... DNNL_ARG_DST)            // null in compute-only
       scale      = CTX_OUT_MEM(... DNNL_ARG_DYNAMIC_SCALE)
       zero_point = CTX_OUT_MEM(... DNNL_ARG_DYNAMIC_ZERO_POINT)  // null if symmetric

       parallel_nd_ext(nthr, num_groups, [&](ithr, nthr_, g) {
           call_params_t p { src_group_g, dst_group_g, scale + g_idx,
                             zero_point + g_idx, group_size, ... };
           kernel_fn_(&p);   // single-pass min/max + compute + quantise
       });
            │
            ▼
   scale (and zp) memory written;
   dst memory written (unless compute-only)
```

## Appendix A. ZenDNN Reference &rarr; Native oneDNN Mapping

ZenDNN's reorder-based dynamic-quant implementation is the porting reference. After the implementation lands in oneDNN, none of these ZenDNN names appear in oneDNN code &mdash; everything is replaced by an idiomatic oneDNN equivalent.

| ZenDNN reference | Translates to in oneDNN | Notes |
| --- | --- | --- |
| `reorder_params_t::dynamic_quant = true` | `primitive_kind::dynamic_quantize` | Promoted to a first-class primitive instead of a reorder mode |
| `reorder_params_t` source / dst types | `src_md.data_type` / `dst_md.data_type` | Standard oneDNN memory descriptors |
| Symmetric mode (zp.buff = nullptr) | `algorithm::dynamic_quantize_symmetric` | Explicit alg-kind, not implicit by buffer presence |
| Asymmetric mode (zp.buff != nullptr) | `algorithm::dynamic_quantize_asymmetric` | Explicit alg-kind |
| `scale.buff` (f32 or bf16) | `DNNL_ARG_DYNAMIC_SCALE` output (= `DNNL_ARG_DST_1`) | Output via standard arg mechanism |
| `zero_point.buff` (s32) | `DNNL_ARG_DYNAMIC_ZERO_POINT` output (= `DNNL_ARG_DST_2`) | Output via standard arg mechanism |
| `scale.dims` (= granularity) | `scale_md.dims` | Same shape, same five granularity templates |
| `zero_point.dims` matches `scale.dims` | `zero_point_md.dims == scale_md.dims` | Validated in PD `init()` |
| Compute-only (`dst = nullptr`) | `flags & dnnl_dynamic_quantize_compute_only` | Flag-driven instead of inferred from arg presence |
| Per-tensor / per-row / per-col / per-group-row / per-group-col | identical, expressed via `scale_md.dims` shape | One-to-one |
| ZenDNN's internal threading dispatch | `parallel_nd*` from `src/common/dnnl_thread.hpp` per granularity (§5.6) | One canonical strategy per granularity baked in; no public knob |
| ZenDNN AVX-512 intrinsic kernels | `src/cpu/x64/dynamic_quantize.{hpp,cpp}` intrinsic kernels | Direct intrinsic-to-intrinsic port (no JIT rewrite) |

## Appendix B. File Touchpoints Checklist

For Phase 1, each PR maps to a sub-bullet. Crossed off when merged.

**Public headers**
- [ ] `include/oneapi/dnnl/dnnl_types.h`: add `dnnl_dynamic_quantize` to `dnnl_primitive_kind_t`; add `dnnl_dynamic_quantize_symmetric` / `_asymmetric` to `dnnl_alg_kind_t`; add `dnnl_dynamic_quantize_flags_t`; add `DNNL_ARG_DYNAMIC_SCALE` / `DNNL_ARG_DYNAMIC_ZERO_POINT` aliases.
- [ ] `include/oneapi/dnnl/dnnl.h`: declare `dnnl_dynamic_quantize_primitive_desc_create`.
- [ ] `include/oneapi/dnnl/dnnl.hpp`: add `dynamic_quantize` wrapper struct.

**Common**
- [ ] `src/common/opdesc.hpp`: `dynamic_quantize_desc_t`.
- [ ] `src/common/dynamic_quantize.cpp`: `dynamic_quantize_desc_init` + C entry impl.
- [ ] `src/common/dynamic_quantize_pd.hpp`: `dynamic_quantize_pd_t` base class.
- [ ] `src/common/c_types_map.hpp`: enum mirrors.
- [ ] `src/common/primitive_desc_iface.cpp`: add `dynamic_quantize` to `known_primitive_kind` whitelist.

**CPU**
- [ ] `src/cpu/cpu_engine.hpp`: `DECLARE_IMPL_LIST(dynamic_quantize)`, `CASE(dynamic_quantize)`.
- [ ] `src/cpu/cpu_dynamic_quantize_pd.hpp`: thin CPU PD typedef.
- [ ] `src/cpu/cpu_dynamic_quantize_list.cpp`: impl list registration.
- [ ] `src/cpu/ref_dynamic_quantize.{hpp,cpp}`: reference implementation.

**Optimised x64 intrinsic impl**
- [ ] `src/cpu/x64/dynamic_quantize.hpp`: `dynamic_quantize_t` + `pd_t` + `DECLARE_COMMON_PD_T`. **No `jit_` prefix.**
- [ ] `src/cpu/x64/dynamic_quantize.cpp`: AVX-512 / AVX-512-FP16 intrinsic kernel template instantiations, `pick_kernel(...)` lookup, `init()`, `execute()`.
- [ ] No new CMake flag; gating is via the existing `DNNL_X64` macro through `CPU_INSTANCE_X64`.

**GPU (Phase 1, stub only)**
- [ ] `src/gpu/gpu_impl_list.cpp`: `case primitive_kind::dynamic_quantize: return empty_list;`.

**Tests / docs / examples**
- [ ] `tests/benchdnn/benchdnn.cpp`: dispatch `--dynamic-quantize`.
- [ ] `tests/benchdnn/dynamic_quantize/`: full driver.
- [ ] `tests/benchdnn/inputs/dynamic_quantize/`: harnesses (`shapes_basic`, `shapes_llm`, CI harness).
- [ ] `tests/benchdnn/doc/driver_dynamic_quantize.md`: driver doc.
- [ ] `tests/gtests/test_dynamic_quantize.cpp`.
- [ ] `examples/primitives/dynamic_quantize.cpp`.
- [ ] `doc/primitives/dynamic_quantize.md` and TOC link.

## Appendix C. References

**Project guidelines.**
- [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) &mdash; library functionality criteria, RFC process, commit-message format, unit-test policy.
- [`CODING_STANDARDS.md`](../../../CODING_STANDARDS.md) &mdash; clang-tidy, clang-format, naming conventions.

**Existing oneDNN patterns referenced by this design.**
- `src/common/reduction.cpp`, `src/common/reduction_pd.hpp`, `src/cpu/cpu_reduction_list.cpp` &mdash; closest existing primitive for layout / argument conventions.
- `src/common/softmax.cpp`, `src/cpu/x64/jit_uni_softmax.{hpp,cpp}` &mdash; PD `init()` flow with ISA-aware dispatch and scratchpad usage. (We mirror the dispatch flow but **not** the Xbyak JIT path; see §11.C.)
- `src/cpu/ref_softmax.{hpp,cpp}` &mdash; canonical layout for a non-JIT `primitive_t` subclass.
- `src/cpu/x64/cpu_isa_traits.hpp` &mdash; `cpu_isa_traits_t<isa>` used to template intrinsic kernels.
- `src/common/dnnl_thread.hpp` &mdash; `parallel_nd`, `parallel_nd_ext`, `balance211`.

**Source of the kernels being upstreamed.**
- AMD ZenDNN's reorder API with `dynamic_quant = true` &mdash; the reference implementation used during porting; not a runtime dependency of the upstreamed primitive.

**Consumer semantics targets.**
- PyTorch `torch.quantize_per_tensor` / `quantize_per_channel` / dynamic-quant ops &mdash; per-tensor and per-row coverage.
- ONNX Runtime `DynamicQuantizeLinear` &mdash; standard ONNX op for dynamic per-tensor / per-row asymmetric quantisation.
- TensorFlow `tf.quantization.quantize` &mdash; per-tensor symmetric / asymmetric.
- vLLM W8A8 / W4A8 inference paths &mdash; per-token activation quantisation, the dominant LLM use case.
- AWQ, SmoothQuant, GPTQ &mdash; per-group quantisation for weight + activation paths.

