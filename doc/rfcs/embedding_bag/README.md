# RFC: Embedding and Embedding-Bag Primitive in oneDNN

## Status
**Draft** &nbsp;|&nbsp; AMD-Zenai/oneDNN-ZenDNN &nbsp;|&nbsp; Branch: `rfc/zendnn-integration` &nbsp;|&nbsp; Umbrella RFC: [`doc/rfcs/zendnn_integration/`](../zendnn_integration/README.md)

## Authors
- AMD ZenDNN team

## Summary
This RFC proposes adding a new `embedding_bag` primitive to oneDNN, with `embedding` (lookup-only) exposed as a thin C++ wrapper over the same primitive. The design follows existing oneDNN conventions for primitives such as `softmax` and `reduction`, complies with the project [Coding Standards](../../../CODING_STANDARDS.md) and [Contributing guidelines](../../../CONTRIBUTING.md), and is compatible with PyTorch `nn.EmbeddingBag`, TensorFlow `tf.nn.embedding_lookup_sparse`, and ONNX Runtime `EmbedLayerNormalization` consumer semantics.

Optimized x64 kernels live **natively** under `src/cpu/x64/` and are written with **AVX-512 / AVX-512-FP16 intrinsics** &mdash; **not** Xbyak / JIT. The kernels are ported directly from AMD ZenDNN's existing `lowoha::embag` intrinsic implementations, so the porting work preserves code structure already validated by ZenDNN's testing rather than re-deriving it as Xbyak (see §6.4 for the rationale and §12.E for the JIT-vs-intrinsics alternative analysis). ZenDNN is the *source* of those kernels for porting, **not** a runtime dependency. After this contribution lands, oneDNN has no link-time, runtime, or build-time dependency on ZenDNN, and there is no JIT involvement anywhere in the primitive's API flow. A portable reference implementation under `src/cpu/` is always available as the correctness fallback and lights up for any unsupported (algorithm, dtype, ISA) combination.

---

## 1. Background and Motivation

Embedding lookups and embedding-bag reductions are core building blocks for recommendation models (DLRM, DCNv2, MMoE, DIN, etc.) and NLP/transformer models that consume token-id sequences. Today, frameworks (PyTorch `nn.EmbeddingBag`, TensorFlow `embedding_lookup_sparse`, ONNX Runtime `EmbedLayerNormalization`) implement these on CPU either via per-framework hand-rolled kernels, FBGEMM, or vendor libraries. oneDNN currently has **no equivalent primitive** &mdash; the closest dense neighbours (`reduction`, graph `gather`) do not cover *indexed lookup + variable-length bag reduction* semantics.

The kernels for this contribution come from AMD's ZenDNN, which has shipped highly optimized AVX-512 / AVX-512-FP16 implementations through its internal `lowoha::embag` API. ZenDNN's implementation is written with **compiler intrinsics** (`<immintrin.h>`), not JIT. This RFC proposes upstreaming those intrinsic kernels into oneDNN as a first-class primitive, lifted into `src/cpu/x64/` with their structure preserved &mdash; ISA-targeted compile-time specializations, runtime dispatch via `mayiuse(...)` ladder in PD `init()`, and oneDNN's standard `parallel_nd*` / scratchpad / `primitive_attr` machinery wrapped around them. After integration, ZenDNN itself is no longer in the runtime path. A discussion of the JIT-port-versus-intrinsic-port choice and why intrinsics is the right call here lives in §6.4 and §12.E.

The contribution must satisfy the three [Library Functionality Guidelines](../../../CONTRIBUTING.md#library-functionality-guidelines) that gate every new oneDNN primitive:

| Criterion | Justification |
|---|---|
| **Performance** &mdash; material workload-level impact | DLRM-class recommendation models spend a substantial fraction of CPU time in the embedding stage; ZenDNN's existing AVX-512 intrinsic implementation has demonstrated meaningful speed-ups over framework-native paths on AMD CPUs. The optimized intrinsic impl proposed here ports those gains into oneDNN, validated end-to-end with benchdnn shape sets representative of DLRM. |
| **Generality** &mdash; usable across multiple frameworks | The primitive's semantics (indices + offsets + per-sample weights + reduction mode + `padding_idx` + `include_last_offset`) match PyTorch `nn.EmbeddingBag` exactly; the same algebraic shape covers TensorFlow's `embedding_lookup_sparse` (dense weights mode) and ONNX Runtime's embedding fast paths. The lookup-only (no-reduction) mode covers PyTorch `nn.Embedding` and ONNX `Gather` (axis 0) for most practical cases. |
| **Complexity** &mdash; non-trivial to implement directly in apps | High-quality CPU embedding implementations require careful work splitting (table-threaded vs bag-threaded vs CCD-aware), int32/int64 indices, optional per-sample weights without branchy inner loops, ISA-specific load/scatter strategies (AVX-512 BF16, AVX-512-FP16), and quantized-table support. Frameworks repeatedly re-implement this; centralizing it in oneDNN saves duplication and keeps the kernel-level optimizations in one place. |

## 2. Goals and Non-Goals

### Goals
- One new primitive kind: `dnnl_embedding_bag`.
- Forward pass on CPU.
- Algorithms: `sum`, `mean`, `max`, and `lookup` (no reduction).
- Data types (Phase 1): `f32` table and output, `s32`/`s64` indices and offsets, `f32` per-sample weights.
- Data types (Phase 2): `bf16`, `f16` table/output (lights up on the ISAs that support them; ref impl always works).
- Optional per-sample weights, optional `padding_idx`, optional `include_last_offset`.
- Dispatch through oneDNN's standard impl-list mechanism &mdash; no public API for kernel selection or threading strategy.
- benchdnn driver, gtest coverage, example, user-guide doc.

### Non-Goals (initial release)
- Backward / training. (Tracked as future work in Phase 4; see §10.)
- GPU implementation. (Impl-list `CASE` returns `empty_list` initially, matching the precedent of `sdpa` / `gated_mlp` in `src/cpu/cpu_engine.hpp`.)
- Quantized (`s8` / `s4` / `u4`) embedding tables &mdash; valuable for DLRM-style inference, tracked in Phase 3 (§10).
- A separate `group_embedding_bag` primitive &mdash; see §12.B for the alternative considered and why we defer it.

## 3. oneDNN Conventions and Coding Standards

### 3.1 Per-primitive convention map

Every primitive in oneDNN follows the same skeleton. The new primitive must respect it. Files cited below are the patterns we mirror.

| Layer | Existing examples | What we add |
| --- | --- | --- |
| Primitive kind enum | `include/oneapi/dnnl/dnnl_types.h` (`dnnl_primitive_kind_t`) | `dnnl_embedding_bag` |
| Algorithm enum | `include/oneapi/dnnl/dnnl_types.h` (`dnnl_alg_kind_t`) | `dnnl_embedding_bag_sum/_mean/_max`, `dnnl_embedding_lookup` |
| Argument map | `DNNL_ARG_*` in `dnnl_types.h` | reuse `SRC`, `SRC_1`, `SRC_2`, `WEIGHTS`, `DST` (named aliases optional) |
| Internal op-desc | `src/common/opdesc.hpp` (`softmax_desc_t`, `reduction_desc_t`) | `embedding_bag_desc_t` |
| C API | `include/oneapi/dnnl/dnnl.h` (e.g. `dnnl_softmax_forward_primitive_desc_create`) | `dnnl_embedding_bag_primitive_desc_create` |
| C++ wrapper | `include/oneapi/dnnl/dnnl.hpp` (`dnnl::softmax_forward`, `dnnl::reduction`) | `dnnl::embedding_bag` (+ `dnnl::embedding` thin wrapper) |
| Desc init + C entry | `src/common/<prim>.cpp` (`src/common/softmax.cpp`) | `src/common/embedding_bag.cpp` |
| PD base class | `src/common/<prim>_pd.hpp` (`src/common/reduction_pd.hpp`) | `src/common/embedding_bag_pd.hpp` |
| CPU PD typedef | `src/cpu/cpu_<prim>_pd.hpp` (`src/cpu/cpu_softmax_pd.hpp`) | `src/cpu/cpu_embedding_bag_pd.hpp` |
| Whitelist | `src/common/primitive_desc_iface.cpp::known_primitive_kind` | add `embedding_bag` |
| C↔C++ enum mirror | `src/common/c_types_map.hpp` | mirror new enums |
| CPU impl list | `src/cpu/cpu_<prim>_list.cpp` (`cpu_reduction_list.cpp`) + `DECLARE_IMPL_LIST` & `CASE` in `src/cpu/cpu_engine.hpp` | `src/cpu/cpu_embedding_bag_list.cpp` + engine plumbing |
| Reference impl | `src/cpu/ref_<prim>.{hpp,cpp}` | `src/cpu/ref_embedding_bag.{hpp,cpp}` |
| Optimized x64 impl | Most existing optimized x64 primitives use Xbyak JIT (`src/cpu/x64/jit_uni_*`); a small number of higher-level orchestration files in `src/cpu/x64/` are non-JIT (e.g. `gemm_bf16_convolution.{hpp,cpp}`, `ip_convolution.{hpp,cpp}`). | `src/cpu/x64/embedding_bag.{hpp,cpp}` &mdash; AVX-512 / AVX-512-FP16 **intrinsics**, no Xbyak; ISA-templated kernel functions compiled into the library and selected by function pointer at PD time. See §6.4 for design and §12.E for the JIT-vs-intrinsics decision. |
| GPU impl list | `src/gpu/gpu_<prim>_list.cpp` + `CASE` in `gpu_impl_list.cpp` | `CASE` returns `empty_list` for now (same shape as `sdpa`, `gated_mlp` precedent) |
| benchdnn driver | `tests/benchdnn/<prim>/` (5 files) + dispatch in `benchdnn.cpp` + harnesses in `tests/benchdnn/inputs/<prim>/` + doc `tests/benchdnn/doc/driver_<prim>.md` | full driver |
| Example | `examples/primitives/<prim>.cpp` | `examples/primitives/embedding_bag.cpp` |
| User-guide doc | `doc/primitives/<prim>.md` | `doc/primitives/embedding_bag.md` |

oneDNN exposes only memory descriptors and scalar parameters in the public C API; it does not expose op-descriptor structs publicly. Threading and kernel selection are internal concerns; users never pick a kernel.

### 3.2 Coding-standards adherence

Every file added by this contribution must comply with the project [`CODING_STANDARDS.md`](../../../CODING_STANDARDS.md), automatically enforced where possible:

| Rule | How we comply |
| --- | --- |
| `clang-format -style=file -i` on every changed `.cpp/.hpp` | Run as a pre-commit step; CI rejects unformatted patches |
| `clang-tidy` checks per `.clang-tidy` (top-level) | All new code passes locally before PR; pre-existing exceptions inherited only when surrounding code already has them |
| No `using namespace ...` in headers | Per CODING_STANDARDS §General; we keep `using` directives inside anonymous namespaces in `.cpp` only |
| Naming: `lower_case_t` types, `lower_case` functions, `_t` suffix on type aliases | Matches `softmax_pd_t`, `jit_uni_softmax_fwd_t`, `embag_params_t` style |
| Use `src` / `dst` (not `input` / `output`) | Already adopted in our argument naming and op-desc fields |
| Prefer `IMPLICATION`, `one_of`, `everyone_is`, `utils::one_of` | Used throughout PD `init()` validation |
| Innermost-scope variable declarations | Standard C++; enforced by review |
| Properly named constants instead of magic numbers | E.g., `static constexpr int simd_w = ...;` rather than literal `16` |
| `Xbyak::Label` for labels (no `char[]`) | N/A &mdash; this primitive uses AVX-512 intrinsics, not Xbyak. The rule is preserved for any future JIT addition (Phase 4+). |
| Commit messages: `<scope>[: <subscope>...]: <imperative summary>`, body wrapped at 72 | Each PR commit follows this; example: `cpu: x64: embedding_bag: add avx512 intrinsic kernel for f32 sum mode` |
| Linear history before merge | Rebase on `main` before each PR |

### 3.3 Library-functionality criteria checklist (CONTRIBUTING.md)

This RFC explicitly addresses each gate that CONTRIBUTING.md sets for a new primitive:

- **Performance** &mdash; demonstrated via benchdnn perf runs vs. reference and vs. framework baselines on representative DLRM shapes; reported in the implementation PR cover letter.
- **Generality** &mdash; consumer-side compatibility verified by mapping to PyTorch `nn.EmbeddingBag`, TensorFlow `embedding_lookup_sparse`, ONNX Runtime embedding paths (see §1).
- **Complexity** &mdash; non-trivial to implement in user code: variable-bag reduction, ISA-aware vectorization, int32/int64 index handling, padding-idx skipping, optional per-sample weights, BF16/FP16 paths.
- **Tested** &mdash; gtests for API + correctness; benchdnn `embedding_bag` driver; CI harness `test_embedding_bag_ci`; `ONEDNN_TEST_SET=NIGHTLY` validated before submit.
- **Documented** &mdash; Doxygen comments in public headers; markdown user guide at `doc/primitives/embedding_bag.md`.
- **Portable** &mdash; builds on every supported OS/compiler; ref impl is portable C++; the intrinsic impl is gated on `DNNL_X64` (via `CPU_INSTANCE_X64`).

## 4. Operation Definition

Let T be a 2D embedding table of shape `[V, D]` (V = num_embeddings, D = embedding_dim), I a 1D indices vector of length N, O a 1D offsets vector defining B bags, and `algo` one of {sum, mean, max, lookup}.

Lookup mode (no offsets, no reduction):

    Y[n, :] = T[I[n], :]                  for n in [0, N), shape Y = [N, D]

Bag modes (sum / mean / max):

    bag_b   = { I[k] : O[b] <= k < O[b+1] }
    Y[b, :] = REDUCE_algo over bag_b of T[I[k], :] * w[k]
    shape Y = [B, D]

with:
- REDUCE_sum  = elementwise sum
- REDUCE_mean = elementwise mean (after weights)
- REDUCE_max  = elementwise max
- w[k]        = per_sample_weights[k] if provided, else 1.0
- indices equal to padding_idx are skipped

If `include_last_offset == true` the offsets vector has length B+1 and O[B] is read from the data (matches PyTorch's `include_last_offset=True`). Otherwise the implicit terminator O[B] = N is used.

`max` mode does not combine with per-sample weights (matches PyTorch behavior).

## 5. Public API Design

### 5.1 Primitive kind

```c
typedef enum {
    /* ... existing kinds ... */
    dnnl_softmax,
    dnnl_embedding_bag,            /* new */
    /* ... */
} dnnl_primitive_kind_t;
```

C++ mirror in `dnnl::primitive::kind`:

```cpp
enum class kind {
    /* ... */
    embedding_bag = dnnl_embedding_bag,
};
```

### 5.2 Algorithm enum

Three new reduction algorithms plus `embedding_lookup` (no reduction). New numeric band to avoid collisions with existing softmax/reduction values:

```c
typedef enum {
    /* ... */
    dnnl_embedding_bag_sum   = 0x40000,
    dnnl_embedding_bag_mean,
    dnnl_embedding_bag_max,
    dnnl_embedding_lookup,
    /* ... */
} dnnl_alg_kind_t;
```

C++ aliases under `dnnl::algorithm::embedding_bag_{sum,mean,max}` and `dnnl::algorithm::embedding_lookup`.

### 5.3 Flags

A small bitmask mirroring `dnnl_normalization_flags_t`:

```c
typedef enum {
    dnnl_embedding_bag_flags_none          = 0x0,
    dnnl_embedding_bag_include_last_offset = 0x1,
} dnnl_embedding_bag_flags_t;
```

### 5.4 Argument map

| Argument | Memory | When required |
| --- | --- | --- |
| `DNNL_ARG_SRC` | embedding table `[V, D]` | always |
| `DNNL_ARG_SRC_1` | indices `[N]`, `s32` or `s64` | always |
| `DNNL_ARG_SRC_2` | offsets `[B]` or `[B+1]`, same dtype as indices | bag modes only |
| `DNNL_ARG_WEIGHTS` | per-sample weights `[N]`, `f32` | optional; not allowed with `max` |
| `DNNL_ARG_DST` | output `[B, D]` (bag) or `[N, D]` (lookup) | always |
| `DNNL_ARG_SCRATCHPAD` | scratchpad | when `attr.scratchpad_mode = user` |

Optional named aliases `DNNL_ARG_INDICES` (= `DNNL_ARG_SRC_1`) and `DNNL_ARG_OFFSETS` (= `DNNL_ARG_SRC_2`) can be added as `#define`s for ergonomics; they don't change the layout.

### 5.5 Internal op descriptor

Lives in `src/common/opdesc.hpp` (consistent with every existing primitive — never in the public C header):

```cpp
struct embedding_bag_desc_t : public op_desc_t {
    embedding_bag_desc_t() : op_desc_t(primitive_kind::embedding_bag) {}

    alg_kind_t      alg_kind {};
    memory_desc_t   src_desc;       // table [V, D]
    memory_desc_t   indices_desc;   // [N]
    memory_desc_t   offsets_desc;   // [B] or [B+1]; zero_md() in lookup mode
    memory_desc_t   weights_desc;   // [N];           zero_md() if no weights
    memory_desc_t   dst_desc;       // [B, D] or [N, D]
    int64_t         padding_idx { -1 };
    unsigned        flags { 0 };
};
```

### 5.6 C API

```c
/// Creates a primitive descriptor for an embedding-bag primitive.
dnnl_status_t DNNL_API dnnl_embedding_bag_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc,
        dnnl_engine_t engine,
        dnnl_alg_kind_t alg_kind,
        const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t indices_desc,
        const_dnnl_memory_desc_t offsets_desc,    // NULL or zero-md for lookup mode
        const_dnnl_memory_desc_t weights_desc,    // NULL or zero-md when not used
        const_dnnl_memory_desc_t dst_desc,
        int64_t  padding_idx,
        unsigned flags,
        const_dnnl_primitive_attr_t attr);
```

The implementation in `src/common/embedding_bag.cpp` follows the same pattern as `dnnl_softmax_forward_primitive_desc_create`: build an `embedding_bag_desc_t`, then call `primitive_desc_create(...)`.

### 5.7 C++ wrapper

```cpp
struct embedding_bag : public primitive {
    enum class flags : unsigned {
        none                = dnnl_embedding_bag_flags_none,
        include_last_offset = dnnl_embedding_bag_include_last_offset,
    };

    struct primitive_desc : public dnnl::primitive_desc {
        primitive_desc() = default;

        primitive_desc(const engine &aengine,
                algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &indices_desc,
                const memory::desc &offsets_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                int64_t  padding_idx = -1,
                flags    aflags = flags::none,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false);

        memory::desc src_desc()     const;
        memory::desc indices_desc() const;
        memory::desc offsets_desc() const;
        memory::desc weights_desc() const;
        memory::desc dst_desc()     const;
    };

    embedding_bag() = default;
    explicit embedding_bag(const primitive_desc &pd) : primitive(pd.get()) {}
};

// Ergonomic wrapper: lookup-only mode.
struct embedding : public primitive {
    struct primitive_desc : public dnnl::primitive_desc {
        primitive_desc(const engine &aengine,
                const memory::desc &src_desc,
                const memory::desc &indices_desc,
                const memory::desc &dst_desc,
                int64_t padding_idx = -1,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false);
    };
    explicit embedding(const primitive_desc &pd) : primitive(pd.get()) {}
};
```

### 5.8 Validation rules

Performed in `embedding_bag_desc_init` and `embedding_bag_pd_t::init_*`:

1. `src_desc` is 2D, blocked, dtype in the supported set (Phase 1: `f32`).
2. `indices_desc` is 1D; dtype in {`s32`, `s64`}.
3. Bag modes: `offsets_desc` is 1D, same dtype as indices, length B (or B+1 with `include_last_offset`); `dst_desc` is 2D `[B, D]`.
4. Lookup mode: `offsets_desc` and `weights_desc` must be empty (`zero_md`); `dst_desc` is `[N, D]`.
5. If `weights_desc` is non-empty: `f32`, length N, and `alg_kind != embedding_bag_max`.
6. `padding_idx` in `[-1, V)`; `-1` means "no padding".
7. `dst_desc` may use `format_kind::any`; `src_desc` and index/offset MDs must be fully described.
8. `dst_desc.dims[1] == src_desc.dims[1]` (D matches).
9. Phase 1: `dst_desc.data_type == src_desc.data_type` (mixed dtypes deferred to Phase 2 via attributes).

### 5.9 Attributes

| Attribute | Phase 1 | Phase 2 | Phase 3 |
| --- | --- | --- | --- |
| `scratchpad_mode` | yes | yes | yes |
| `post_ops` (eltwise / binary on output) | no | optional | yes |
| `scales` (per-tensor / per-row on table) | no | no | yes (quantized tables) |
| `zero_points` | no | no | yes (quantized tables) |

## 6. Implementation Architecture

### 6.1 Source layout (additions only)

All additions live in the same directories as their analogous `softmax` / `reduction` counterparts &mdash; the optimized kernel is a regular `src/cpu/x64/` JIT primitive, not a wrapper around an external library.

```
include/oneapi/dnnl/
  dnnl_types.h                              [+] primitive kind, alg kinds, flags
  dnnl.h                                    [+] dnnl_embedding_bag_primitive_desc_create
  dnnl.hpp                                  [+] dnnl::embedding_bag, dnnl::embedding

src/common/
  opdesc.hpp                                [+] embedding_bag_desc_t
  embedding_bag.cpp                         [+] desc_init + C API entry point
  embedding_bag_pd.hpp                      [+] primitive-desc base class
  c_types_map.hpp                           [+] enum mirrors
  primitive_desc_iface.cpp                  [+] add to known_primitive_kind whitelist

src/cpu/
  cpu_engine.hpp                            [+] DECLARE_IMPL_LIST + CASE
  cpu_embedding_bag_pd.hpp                  [+] thin CPU PD typedef (mirrors cpu_softmax_pd.hpp)
  cpu_embedding_bag_list.cpp                [+] impl_list registration
  ref_embedding_bag.hpp                     [+] reference implementation
  ref_embedding_bag.cpp                     [+]
  x64/embedding_bag.hpp                     [+] optimized x64 primitive (no `jit_` prefix; intrinsics-based)
  x64/embedding_bag.cpp                     [+] AVX-512/AVX-512-FP16 intrinsic kernels + ISA dispatch + execute()
  # If the .cpp grows large we may split into x64/embedding_bag_kernels_avx512.{hpp,cpp}
  # and x64/embedding_bag_kernels_avx2.{hpp,cpp} (per-ISA translation units), but the
  # primitive itself stays in a single x64/embedding_bag.{hpp,cpp} pair.

src/gpu/
  gpu_impl_list.cpp                         [+] CASE returns empty_list (no GPU impl in Phase 1)

tests/
  benchdnn/embedding_bag/                   [+] driver (5 files: bench_*, *.{hpp,cpp}, *_aux, ref_*)
  benchdnn/inputs/embedding_bag/            [+] test harness files
  benchdnn/doc/driver_embedding_bag.md      [+] driver doc
  benchdnn.cpp                              [+] dispatch for --embedding-bag
  gtests/test_embedding_bag.cpp             [+] gtest coverage

examples/primitives/embedding_bag.cpp       [+] tutorial example
doc/primitives/embedding_bag.md             [+] user-guide markdown
```

**No new CMake options.** No `ONEDNN_ENABLE_*` flag is introduced; the optimized kernel is part of the regular x64 build (gated by the existing `DNNL_X64` macro via `CPU_INSTANCE_X64`). Per-translation-unit ISA flags (e.g. `-mavx512f -mavx512bw`) are applied to `src/cpu/x64/embedding_bag.cpp` via the existing CMake helper that other ISA-targeted source files use; this is the normal way oneDNN compiles ISA-specific code without forcing the whole library to that ISA.

### 6.2 Primitive descriptor base

`src/common/embedding_bag_pd.hpp` defines `embedding_bag_pd_t : public primitive_desc_t` with:

- MD accessors: `src_md(0)` (table), `src_md(1)` (indices), `src_md(2)` (offsets), `weights_md(0)`, `dst_md(0)`.
- `arg_usage(int arg)` &rarr; input / output per `DNNL_ARG_*` (mirrors `reduction_pd_t::arg_usage`).
- `arg_md(int arg)` &rarr; the right MD (mirrors `reduction_pd_t::arg_md`).
- Helpers: `is_lookup()`, `has_offsets()`, `has_weights()`, `padding_idx()`, `flags()`, `D()` (= embedding dim).

`src/cpu/cpu_embedding_bag_pd.hpp` is a thin CPU-side typedef that mirrors `src/cpu/cpu_softmax_pd.hpp`. CPU impls inherit `pd_t : public cpu_embedding_bag_pd_t` and use `DECLARE_COMMON_PD_T(...)` from `src/common/primitive_desc.hpp` for the standard `clone()` / `create_primitive()` / `name()` boilerplate.

### 6.3 Reference implementation

`src/cpu/ref_embedding_bag.{hpp,cpp}` &mdash; `ref_embedding_bag_t : public primitive_t`. Forward only; algorithm is read from `pd()->desc()->alg_kind` at execute time so a single class covers all four modes. Parallelism is via `parallel_nd(B, ...)` (or `parallel_nd(N, ...)` for lookup) from `src/common/dnnl_thread.hpp`. Always present; benchdnn uses it as the correctness oracle.

`pd_t::init()` skeleton:

```cpp
status_t init(engine_t *engine) {
    using namespace data_type;
    VDISPATCH_EMBEDDING_BAG(utils::one_of(src_md(0)->data_type, f32, bf16, f16),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_EMBEDDING_BAG(utils::one_of(src_md(1)->data_type, s32, s64),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_EMBEDDING_BAG(set_default_formats() == status::success,
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_EMBEDDING_BAG(attr()->has_default_values(),
            VERBOSE_UNSUPPORTED_ATTR);
    return status::success;
}
```

The `VDISPATCH_EMBEDDING_BAG` macro is a new entry mirroring `VDISPATCH_SOFTMAX` in `src/common/softmax_pd.hpp` &mdash; it returns `status::unimplemented` with a verbose-mode message on mismatch.

### 6.4 Optimized x64 implementation (AVX-512 intrinsics &mdash; not JIT)

This is the part of the design that intentionally departs from the typical oneDNN optimized-CPU pattern. **No JIT (no Xbyak) is used anywhere in this primitive.** The optimized kernel is plain C++ with `<immintrin.h>` AVX-512 / AVX-512-FP16 intrinsics, compiled into the library by the regular toolchain.

**Why intrinsics, not JIT.** Detailed analysis lives in §12.E. In short: the kernel logic (gather a row, multiply by an optional scalar weight, reduce element-wise) is simple enough that there is no shape-dependent code-gen opportunity that would benefit from Xbyak's runtime emission; ZenDNN's existing intrinsic implementation is well-validated and ships today; porting it as intrinsics preserves that validation surface, lowers porting risk, and keeps the diff readable for upstream reviewers. The impl-list mechanism leaves room to add a `jit_uni_embedding_bag_t` impl above the intrinsic one in a follow-up if profiling later shows JIT could meaningfully outperform &mdash; users see no API change.

**Class declaration** &mdash; `src/cpu/x64/embedding_bag.hpp`:

```cpp
struct embedding_bag_t : public primitive_t {
    struct pd_t : public cpu_embedding_bag_pd_t {
        using cpu_embedding_bag_pd_t::cpu_embedding_bag_pd_t;

        // impl name reported via primitive descriptor:
        //   "intrin:avx512_core_fp16" / "intrin:avx512_core_bf16" /
        //   "intrin:avx512_core"     / "intrin:avx2"
        DECLARE_COMMON_PD_T(impl_name(), embedding_bag_t);

        status_t init(engine_t *engine);
        const char *impl_name() const;

        cpu_isa_t isa_ = isa_undef;   // chosen by mayiuse() ladder in init()
        int nthr_ = 1;
    };

    explicit embedding_bag_t(const pd_t *apd) : primitive_t(apd) {}
    status_t init(engine_t *engine) override;             // selects kernel_fn_
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    using kernel_fn_t = void (*)(const call_params_t *);  // intrinsic kernel signature
    kernel_fn_t kernel_fn_ = nullptr;                     // function pointer set in init()
};
```

There is **no `jit_generator_t`, no `Xbyak` headers, no `create_kernel()`, no `generate()`**. `embedding_bag_t` inherits the standard `primitive_t` directly and overrides `init()` / `execute()`, like any non-JIT primitive does (`ref_softmax_fwd_t` is a similar shape).

**ISA-templated kernel functions.** The intrinsic kernels are ordinary free functions templated on `cpu_isa_t isa` (and on dtype / algorithm / feature flags so the inner loops are branch-free):

```cpp
template <cpu_isa_t isa, data_type_t dst_dt,
          alg_kind_t alg, bool has_weights, bool has_padding>
void embedding_bag_kernel(const call_params_t *p);   // pure C++ + AVX-512 intrinsics
```

Every `(isa, dst_dt, alg, has_weights, has_padding)` instantiation we support is compiled once into the library at build time. This is regular template instantiation; the compiler emits ordinary text-section code, no runtime emission. ISA-targeted compile flags (e.g. `-mavx512f -mavx512bw -mavx512fp16`) are applied per-translation-unit so that, for example, AVX-512-FP16 instructions never appear in code paths reached on hosts that don't have the ISA &mdash; the same technique oneDNN already uses to isolate other ISA-specific code.

**PD `init()`.** Walks an ordered list of supported ISAs (`avx512_core_fp16`, `avx512_core_bf16`, `avx512_core`, `avx2`) and picks the first satisfying `mayiuse(isa_) && layout_check(...)`. Sets `isa_` once at PD time so `execute()` is dispatch-free. Validates dtypes per algorithm, format kinds, attributes; calls `set_default_formats()`. Books any required scratchpad (see §6.7). Returns `status::success` on match or `status::unimplemented` (via `VDISPATCH_EMBEDDING_BAG`) otherwise &mdash; the impl iterator advances to `ref_embedding_bag_t`.

**`embedding_bag_t::init(engine_t *)`** sets `kernel_fn_` to point at the right specialization, by resolving `(pd()->isa_, dst dt, alg_kind, has_weights, has_padding)` through a small `static constexpr` lookup (or switch). **No JIT compilation, no allocation, no Xbyak.** It is a function-pointer assignment:

```cpp
status_t embedding_bag_t::init(engine_t *engine) {
    kernel_fn_ = pick_kernel(pd()->isa_,
                             pd()->dst_md(0)->data_type,
                             pd()->desc()->alg_kind,
                             pd()->has_weights(),
                             pd()->padding_idx() != -1);
    if (!kernel_fn_) return status::runtime_error;  // unreachable if PD init was thorough
    return status::success;
}
```

**`execute(const exec_ctx_t &ctx)`** retrieves arguments via `CTX_IN_MEM` / `CTX_OUT_MEM`, fetches the scratchpad grantor, fans out work with `parallel_nd_ext(nthr, B, ...)` (or `parallel_nd(N, ...)` for lookup mode), and invokes the kernel function pointer per work item:

```cpp
status_t embedding_bag_t::execute(const exec_ctx_t &ctx) const {
    const auto *table = CTX_IN_MEM (const char *,  DNNL_ARG_SRC);
    const auto *idx   = CTX_IN_MEM (const char *,  DNNL_ARG_SRC_1);
    const auto *off   = CTX_IN_MEM (const char *,  DNNL_ARG_SRC_2);    // null in lookup mode
    const auto *w     = CTX_IN_MEM (const float *, DNNL_ARG_WEIGHTS);  // null if absent
    auto       *dst   = CTX_OUT_MEM(char *,        DNNL_ARG_DST);

    const dim_t B = pd()->desc()->dst_desc.dims[0];
    parallel_nd_ext(pd()->nthr_, B,
            [&](int ithr, int /*nthr*/, dim_t b) {
        call_params_t p { table, idx, off, w, dst, b /*, ...sizes, flags... */ };
        kernel_fn_(&p);     // direct call into the compiled-in intrinsic kernel
    });
    return status::success;
}
```

**No external dependencies.** No header, link-time dep, or runtime call into ZenDNN; **no JIT runtime anywhere in the API flow**; no allocations on the hot path.

### 6.5 Dispatch

`src/cpu/cpu_embedding_bag_list.cpp`:

```cpp
namespace dnnl::impl::cpu {
const impl_list_item_t *get_embedding_bag_impl_list(
        const embedding_bag_desc_t *desc) {
    UNUSED(desc);
    static constexpr impl_list_item_t impl_list[] = REG_EMBEDDING_BAG_P({
        CPU_INSTANCE_X64(embedding_bag_t)        // intrinsic-based, x64 only
        CPU_INSTANCE(ref_embedding_bag_t)        // portable C++ reference
        nullptr,
    });
    return impl_list;
}
} // namespace dnnl::impl::cpu
```

`src/cpu/cpu_engine.hpp` adds `DECLARE_IMPL_LIST(embedding_bag);` and a `CASE(embedding_bag);` arm in the dispatch switch. `REG_EMBEDDING_BAG_P` is a new macro added to `src/common/impl_registration.hpp` mirroring `REG_SOFTMAX_P` / `REG_REDUCTION_P`.

`primitive_desc_iterator_t::operator++` (`src/common/primitive_desc_iterator.hpp` lines 83&ndash;91) walks the list and skips entries whose `init()` returns `status::unimplemented` &mdash; the intrinsic impl rejects unsupported ISA / dtype combos in PD `init()`, and the iterator falls through to `ref_embedding_bag_t` automatically. (A future JIT impl, if added, would slot in *above* `embedding_bag_t` in this list and follow the same fallthrough behavior.)

### 6.6 Threading

All threading uses oneDNN's standard helpers in `src/common/dnnl_thread.hpp`:

- **Bag modes** (`sum`/`mean`/`max`): `parallel_nd_ext(nthr, B, [&](int ithr, int nthr_, dim_t b) { ... })`. Each thread handles a contiguous slice of bags; per-thread scratch (if any) is indexed by `ithr`. (Same parallel-region shape used by `jit_uni_softmax_fwd_t::execute`; the body just calls our intrinsic kernel function pointer instead of a JIT'd kernel.)
- **Lookup mode** (`embedding_lookup`): `parallel_nd(N, [&](dim_t n) { ... })`. One row per work item.
- **Optional inner split** (future): for very long bags relative to thread count, the kernel can split a single bag across threads using `balance211(bag_len, nthr, tid, n_start, n_end)`. Not required for Phase 1.

There are no per-primitive threading knobs in the public API. Internal threading respects `dnnl_get_max_threads()` and the active runtime (OpenMP / TBB / threadpool), following the same model as every other oneDNN primitive.

### 6.7 Performance considerations

The design deliberately avoids unnecessary layers in the hot path. Specifically:

- **Validation is done once, in PD `init()`.** No re-validation in `execute()`. ISA selection (`pd()->isa_`), scratchpad sizing, format defaults are all resolved at PD time.
- **No JIT in the API flow.** Because the optimized impl is intrinsics, not Xbyak, there is no `create_kernel()` step, no Xbyak emit, no executable mapping at primitive-creation time. `embedding_bag_t::init()` just resolves a function pointer (`kernel_fn_ = pick_kernel(...)`) into one of the AVX-512 specializations the compiler already emitted at build time. The cost of that resolution is a few comparisons; primitive cache amortizes even that.
- **`execute()` does not allocate.** Per-thread scratch is pre-booked with `scratchpad_registry().registrar().book(...)` in PD `init()` and consumed via `ctx.get_scratchpad_grantor().get<T>(key)`. No `std::vector` growth, no `new` / `make_unique` on the hot path.
- **No vendor / external indirection.** The kernel is a direct function-pointer call into compiled-in code; there is no adapter, shim, or library boundary between `execute()` and the AVX-512 instructions. ZenDNN's tunables (`embag_kernel_t`, `eb_thread_algo_t`, `ZENDNNL_EMBAG_*` env vars) collapse into a single, fixed dispatch path chosen at PD time.
- **Branch-light inner loop.** Per-sample-weight handling and `padding_idx` skipping are encoded as template parameters on the kernel function (see §6.4) so the compiler emits a separate specialization per `(has_weights, has_padding)` combination &mdash; no conditional branch inside the SIMD reduction loop when those features are disabled.
- **Scratchpad opt-in.** The kernel needs scratch only when (a) the output dtype is bf16/f16 and we want f32 accumulation, or (b) post-ops require an intermediate buffer. For pure f32 sum/mean/max with no post-ops, no scratchpad is registered.
- **format_kind::any for dst.** The PD lets the impl pick the dst layout; the intrinsic impl accepts plain (`ab`) layouts and the ref impl accepts both plain and blocked. Indices/offsets are forced to plain 1D &mdash; there is no benefit to letting the user request other layouts for them.

## 7. Data Type Support Matrix

`I/O` columns refer to the table (input) and dst (output) data types. Indices and offsets are always integer (`s32` or `s64`). Per-sample weights are always `f32`.

| Phase | Table dtype | Dst dtype | Algorithms | Engine | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `f32` | `f32` | sum, mean, max, lookup | CPU (ref + intrinsic x64) | Baseline. Intrinsic impl lights up on `avx2` / `avx512_core`; ref covers everything else. |
| 2 | `bf16` | `bf16` | sum, mean, max, lookup | CPU (ref + intrinsic x64) | Intrinsic impl requires `avx512_core_bf16`; ref always works. |
| 2 | `f16` | `f16` | sum, mean, max, lookup | CPU (ref + intrinsic x64) | Intrinsic impl requires `avx512_core_fp16`; ref always works. |
| 3 | `s8`, `s4`, `u4` | `f32` | sum, mean, lookup | CPU (intrinsic x64) | Quantized table; uses `attr.scales` / `attr.zero_points`. |
| Future | f32/bf16/f16 | f32 (acc) | sum, mean, max | GPU (Intel SYCL) | Not in this RFC. |

## 8. Testing Strategy

### 8.1 benchdnn driver

`tests/benchdnn/embedding_bag/` follows the pattern of `tests/benchdnn/reduction/`:

- `embedding_bag.hpp` &mdash; `prb_t`, `settings_t`, dim parsing.
- `embedding_bag.cpp` &mdash; the driver entry (`bench()`), problem creation, primitive lifecycle.
- `embedding_bag_aux.cpp` &mdash; algo / dt parsers and pretty-printers.
- `ref_embedding_bag.cpp` &mdash; CPU-only reference for correctness checks.
- `bench_embedding_bag.cpp` &mdash; option parser entry called from `tests/benchdnn/benchdnn.cpp`.

`tests/benchdnn/benchdnn.cpp` adds `#include "embedding_bag/embedding_bag.hpp"` and a new `else if (!strcmp("--embedding-bag", argv[0])) { embedding_bag::bench(...); }` branch.

`tests/benchdnn/inputs/embedding_bag/` contains:
- `shapes_basic` &mdash; small cases for correctness.
- `shapes_dlrm` &mdash; representative DLRM-style shapes (large V, small D, many bags).
- `test_embedding_bag_ci` &mdash; the CI harness fed by CTest.
- `harness_embedding_bag_*` &mdash; nightly / regression sets.

CLI shape (proposed):
```
--embedding-bag --algo=sum --dt=f32 --idt=s32 --pad=-1 \
  --include-last-offset=false --weights=false \
  V:D:N:B
```

### 8.2 Unit tests

`tests/gtests/test_embedding_bag.cpp` covers:
- API surface: ctor / accessors / arg map.
- Validation errors: wrong rank, wrong dtypes, missing offsets in bag mode, weights with `max`, etc.
- Correctness on small shapes vs hand-computed values.

### 8.3 Examples

`examples/primitives/embedding_bag.cpp` shows:
1. Build a small `[V=10, D=4]` table.
2. Build indices `[0,1,2,3,4]` and offsets `[0,2,5]` (two bags of sizes 2 and 3).
3. Construct the primitive with `algorithm::embedding_bag_sum`.
4. Execute and print the output.

`examples/CMakeLists.txt` already globs `*.cpp`; no per-file registration needed beyond optional exclusions if a config can't run it.

## 9. Documentation

`doc/primitives/embedding_bag.md` follows the structure of `doc/primitives/reduction.md`:

1. Title with Doxygen anchor: `Embedding Bag {#dev_guide_embedding_bag}`.
2. **General** &mdash; one-paragraph description, with link to API ref.
3. **Operation Definition** &mdash; the formulas from §4 of this RFC.
4. **Execution Arguments** table: `DNNL_ARG_SRC`, `DNNL_ARG_SRC_1`, `DNNL_ARG_SRC_2`, `DNNL_ARG_WEIGHTS`, `DNNL_ARG_DST`, `DNNL_ARG_SCRATCHPAD`.
5. **Implementation Details** &mdash; supported data types per algorithm; padding; include_last_offset.
6. **Performance Tips** &mdash; layout recommendations (contiguous table rows), int32 vs int64 indices.

A short sentence in the developer guide TOC links the new page.

## 10. Phasing and Delivery

Each phase ends with a green CI run and is reviewable as a self-contained PR series. Per CONTRIBUTING.md, each commit follows the `<scope>: <subscope>: <imperative summary>` format with body wrapped at 72 chars.

| Phase | Scope | Commit-scope examples | Done-when |
| --- | --- | --- | --- |
| 1a | Public API plumbing: enums, op-desc, C / C++ entry points, whitelist, `c_types_map`, ref-impl skeleton | `api: embedding_bag: ...`, `common: embedding_bag: ...` | `dnnl::embedding_bag` PD can be created and executed for `f32` `algorithm::embedding_bag_sum` against the ref impl. |
| 1b | Reference impl complete for all 4 algorithms; `padding_idx`; `include_last_offset`; per-sample weights; scratchpad attribute | `cpu: embedding_bag: ref: ...` | benchdnn driver passes `shapes_basic`. |
| 1c | benchdnn driver; gtests; example; user-guide doc | `tests: benchdnn: embedding_bag: ...`, `doc: embedding_bag: ...` | All new tests run in CI under `ONEDNN_TEST_SET=NIGHTLY`. |
| 2a | Optimized x64 intrinsic impl `embedding_bag_t` (AVX-512 + AVX2) for `f32` | `cpu: x64: embedding_bag: ...` | Intrinsic impl wins dispatch over ref on AVX-512 hosts; benchdnn `shapes_dlrm` accuracy matches ref. |
| 2b | `bf16` and `f16` paths in the intrinsic impl | `cpu: x64: embedding_bag: ...` | Coverage on AVX-512 BF16 and AVX-512-FP16 hosts; ref fallback verified on hosts without those ISAs. |
| 2c | Performance validation: benchdnn perf runs vs framework baselines on representative DLRM shapes | `tests: benchdnn: embedding_bag: ...` | Measured workload-level speed-up reported in PR cover letter, satisfying CONTRIBUTING.md performance gate. |
| 3 | `s8` / `s4` / `u4` quantized embedding tables via `attr.scales` / `attr.zero_points`; post-ops support | `cpu: x64: embedding_bag: ...`, `api: embedding_bag: ...` | Accuracy matches a reference dequantize+sum path; perf comparable to or better than equivalent paths in frameworks. |
| 4 (future) | GPU implementation (Intel SYCL); backward pass for training | &mdash; | Out of scope for this RFC. |

## 11. Risks and Open Questions

1. **Lookup mode argument shape.** When `algorithm::embedding_lookup` is used, the dst MD is `[N, D]` rather than `[B, D]`, and offsets are absent. Two ways to express this in the C API:
   - Pass `NULL` for `offsets_desc`. *Preferred &mdash; consistent with how `dnnl_primitive_attr_t` is allowed to be `NULL`.*
   - Require a `zero_md`. Less ergonomic.
   The C++ wrapper hides this by exposing `dnnl::embedding` which never asks for offsets.

2. **Argument naming (`SRC_1` / `SRC_2` vs named aliases).** Other primitives that take auxiliary inputs (`binary` uses `SRC_0`/`SRC_1`; `rnn` uses named ones like `SRC_LAYER`) have inconsistent conventions. Recommendation: stick with `SRC` / `SRC_1` / `SRC_2` to avoid bikeshedding the alias names; revisit if framework integrators ask for clarity.

3. **`max` mode + per-sample weights.** PyTorch forbids this; we mirror that. Some users may expect `max(T[i] * w[i])`; the RFC defaults to the PyTorch behavior to avoid surprises.

4. **Indices dtype.** `s32` is enough for `V <= ~2.1B` but DLRM-scale recommendation models routinely exceed that and use `s64`. Both must be supported &mdash; the impl fan-outs by indices dtype.

5. **`include_last_offset` semantic clash.** Some frameworks always include the last offset; oneDNN must support both. Enforce via flag rather than inferring from dimensions.

6. **Should `embedding_bag_pd_t` allow `format_kind::any` for the table?** For now, no &mdash; we keep the table fully described to avoid the impl having to recompute strides per call. The dst can be `any`.

7. **What happens when all indices in a bag are `padding_idx`?** PyTorch returns zeros for sum/mean and sentinel `-inf`-like for max (actually defaults to 0). We should match PyTorch &mdash; document and test.

8. **Performance gate (CONTRIBUTING.md).** Every new oneDNN primitive must show "material workload-level impact." We need a documented benchdnn perf comparison vs. framework-native paths and ref-impl, on representative DLRM shapes, to land the PR series. Risk: if the intrinsic port underperforms ZenDNN's stand-alone implementation in some shape regions, we either tune further or narrow the impl's accept criteria so ref handles those cases.

9. **Keeping ref and intrinsic impls in sync.** The reference and intrinsic implementations must produce bit-equal results (within fp tolerance) for all shapes / dtypes / algorithms. We mitigate by making benchdnn run both impls (`--skip-impl=ref` and not) on every change, with explicit cross-check in CI.

10. **Template-instantiation fan-out.** Each combination of `(isa, dst_dt, alg, has_weights, has_padding, indices_dt)` is a distinct kernel specialization compiled into the library. With the Phase 1+2 matrix this is a manageable two-digit count, but it grows quickly once Phase 3 adds quantization. Mitigation: keep the template axes orthogonal, generate the dispatch table once with a script-friendly macro, and revisit the matrix at the start of Phase 3 to prune impossible combinations (e.g. `max` + weights, lookup + offsets, etc.).

11. **Intrinsics-based optimized CPU primitive is unusual in oneDNN.** Almost every existing optimized x64 primitive uses Xbyak JIT (`src/cpu/x64/jit_uni_*`). Reviewers will reasonably ask why we deviate. Mitigation: §6.4 and §12.E carry the rationale (kernel logic is simple, validated intrinsic source already exists, runtime code-gen is not the win here), and the impl-list mechanism reserves room for a future JIT impl above this one if profiling later justifies it &mdash; users see no API change.

## 12. Alternatives Considered

### A. Two separate primitives `embedding` and `embedding_bag`

PyTorch exposes `nn.Embedding` and `nn.EmbeddingBag` as separate modules. We considered mirroring this in oneDNN.

- **Pros:** Slightly clearer per-operation API; matches consumer mental model.
- **Cons:** Doubles the surface area (two `dnnl_*_primitive_desc_create` functions, two `_pd_t` classes, two impl lists, two benchdnn drivers, two doc pages). Internally, ZenDNN already implements them as one operation; oneDNN's existing `softmax`/`logsoftmax` and `eltwise` precedent shows that a single primitive with multiple algorithms is the idiomatic choice.

**Decision:** One primitive, multiple algorithms. `dnnl::embedding` is a thin C++-only wrapper for ergonomics; it does not introduce a new C primitive kind.

### B. Add a `group_embedding_bag` primitive

ZenDNN has a `group_embedding_bag_direct` API that batches multiple lookups in one call with smart threading.

- **Pros:** Lower per-call overhead, useful for DLRM-style models with many parallel embedding tables.
- **Cons:** No precedent in oneDNN for "vector of primitives" APIs. Most consumers want one primitive per call so it composes with the rest of oneDNN. Threading benefits can be recovered at the framework integration layer (the framework calls `dnnl_primitive_execute` in parallel) or by extending the same primitive with a "group" dimension on inputs.

**Decision:** Defer. If profiling shows real overhead from per-bag dispatch, revisit by either (a) extending `embedding_bag` to accept a "group" dimension, or (b) adding a separate `group_embedding_bag` primitive in a follow-up RFC.

### C. Wrap ZenDNN as a runtime backend (rejected)

An earlier draft of this RFC proposed compiling against ZenDNN and dispatching `embedding_bag` calls to ZenDNN's `lowoha::embag::embedding_bag_direct(...)` behind an `ONEDNN_ENABLE_ZENDNN` CMake flag.

- **Pros:** Lower porting cost; reuses ZenDNN's tuning history (`embag_kernel_t`, `eb_thread_algo_t`, env-var knobs) directly.
- **Cons:** Adds a third-party runtime dep to oneDNN; bleeds vendor-specific taxonomy into the build / packaging story; couples oneDNN release cadence to ZenDNN; obscures the implementation from oneDNN reviewers, complicating future maintenance.

**Decision:** Reject. The kernels are upstreamed natively into `src/cpu/x64/` and re-implemented to oneDNN's coding standards. ZenDNN remains a *reference implementation* during porting, not a runtime dependency. After Phase 2, ZenDNN does not appear anywhere in the oneDNN build, link, or runtime path.

### D. One big PR vs phased PR series (chosen)

**Decision:** Phased PR series per §10. CONTRIBUTING.md requires linear history, encourages splitting unrelated changes, and asks reviewers to evaluate one logical concern at a time. The phasing keeps each PR focused on a single layer (API plumbing &rarr; ref impl &rarr; tests &rarr; intrinsic impl &rarr; quantization), each commit follows the project commit-message format, and each PR runs `ONEDNN_TEST_SET=NIGHTLY` before merge.

### E. AVX-512 intrinsics port vs Xbyak JIT rewrite (intrinsics chosen)

oneDNN's optimized x64 primitives are overwhelmingly Xbyak JIT-based (`jit_uni_softmax`, `jit_uni_reduction`, etc.). For embedding_bag we considered two ports:

| | Intrinsic port (chosen) | JIT (Xbyak) rewrite |
| --- | --- | --- |
| Source code | Lifted from ZenDNN's existing `lowoha::embag` AVX-512 / AVX-512-FP16 intrinsic kernels, validated for years | New Xbyak generator written from scratch, modelled on `jit_uni_softmax_*_kernel_t` |
| Porting risk | Low: preserve the intrinsic structure, swap surrounding scaffolding for oneDNN's `primitive_t` / `parallel_nd*` / scratchpad | Higher: rewrite kernels in Xbyak, validate from scratch |
| Runtime code-gen benefit | None: the inner loop (gather + multiply + reduce) has no shape-dependent code-gen opportunity that intrinsics can't already express | None obvious for this op &mdash; JIT typically wins on convolution / GEMM where blocking shapes vary widely; embedding_bag's hot loop doesn't have those degrees of freedom |
| Compile / link overhead | None: ordinary template instantiations, compiled once into the library | Lower per-instance overhead than intrinsic dispatch is theoretical; JIT compile *adds* per-PD overhead (small but non-zero) |
| Reviewer familiarity in upstream oneDNN | Lower: deviates from convention | Higher: matches every other `src/cpu/x64/` primitive |
| Maintainability for the AMD team | Higher: team knows the intrinsic code | Lower: requires Xbyak expertise the team would need to grow |
| Future flexibility | Adding a JIT impl later is non-breaking: it goes *above* `embedding_bag_t` in the impl list and replaces it on success | Going from JIT back to intrinsics is similarly non-breaking but unusual |

**Decision:** Port as intrinsics for Phase 2. The intrinsic kernels are well-validated, the kernel does not benefit from JIT's runtime code-gen, and the porting cost is materially lower. The impl list keeps the door open for a future Xbyak JIT impl if benchmarking ever shows it would win on a meaningful set of shapes &mdash; that addition would be a separate, scope-limited contribution and would not require any user-visible change.

**Caveat for upstream review.** Because this would be one of the first non-JIT optimized CPU primitives in oneDNN, the RFC explicitly calls out the choice (Summary, §1, §6.4, this section, §11.11) so reviewers can engage with the trade-off directly rather than discovering it in the implementation PR.

## 13. End-to-End Lifecycle (illustrative)

The flow follows the standard oneDNN primitive shape (`softmax`, `reduction`), with **one important difference**: there is **no JIT compilation step anywhere**. The optimized kernel is plain compiled-in code (AVX-512 / AVX-512-FP16 intrinsics) selected by function pointer at PD `init()` time. No external libraries are involved &mdash; all kernel code is part of the oneDNN x64 build.

### Phase A &mdash; primitive-descriptor creation (one-time, may be cached)

```
  user code (PyTorch / TF / ONNX RT / application)
        |
        v   dnnl::embedding_bag::primitive_desc(eng, alg, table_md, idx_md,
                                                off_md, w_md, dst_md, pad, flags, attr)
                                                    [include/oneapi/dnnl/dnnl.hpp]
        |
        v   C++ wrapper -> C API
  dnnl_embedding_bag_primitive_desc_create(...)
                                                    [include/oneapi/dnnl/dnnl.h]
        |
        v   build internal embedding_bag_desc_t, validate scalar params
  embedding_bag_desc_init(...) --> primitive_desc_create(...)
                                                    [src/common/embedding_bag.cpp]
        |
        v   primitive-kind whitelist check
  known_primitive_kind contains embedding_bag ?
                                                    [src/common/primitive_desc_iface.cpp]
        |
        v   engine-side dispatch
  cpu_engine::get_*_impl_list -> CASE(embedding_bag)
                                                    [src/cpu/cpu_engine.hpp]
        |
        v   ordered impl_list walked by primitive_desc_iterator_t
  +-> embedding_bag_t::pd_t::init()         <-- intrinsic impl (x64)
  |     - mayiuse(avx512_core_*) ladder -> sets pd()->isa_
  |     - dtype + format + attr checks via VDISPATCH_EMBEDDING_BAG
  |     - set_default_formats(); init_scratchpad()
  |     - status::success      ---> selected
  |   (or status::unimplemented ---> iterator advances)
  |                                                 [src/cpu/x64/embedding_bag.cpp]
  |
  +-> ref_embedding_bag_t::pd_t::init()     <-- terminal fallback
                                                    [src/cpu/ref_embedding_bag.cpp]
        |
        v
  primitive_desc ready (with selected impl's pd_t)
```

### Phase B &mdash; primitive creation (one-time, no JIT)

```
  user: dnnl::embedding_bag prim(pd);
        |
        v   primitive_t::create_primitive_common -> impl::init(engine)
  embedding_bag_t::init(engine):
      // NO JIT, NO Xbyak, NO create_kernel(). Just function-pointer resolution:
      kernel_fn_ = pick_kernel(pd()->isa_,
                               pd()->dst_md(0)->data_type,
                               pd()->desc()->alg_kind,
                               pd()->has_weights(),
                               pd()->padding_idx() != -1);
                                                    [src/cpu/x64/embedding_bag.cpp]
```

The function pointer points into AVX-512 / AVX-512-FP16 code that the C++ compiler emitted into the library at oneDNN build time. There is no executable mapping created at runtime; nothing is JIT'd; nothing is allocated.

### Phase C &mdash; execute (hot path, no allocations, no JIT, no ZenDNN)

```
  user: prim.execute(stream, args);
        |
        v
  embedding_bag_t::execute(ctx):
      table  = CTX_IN_MEM (const char *,  DNNL_ARG_SRC);
      idx    = CTX_IN_MEM (const char *,  DNNL_ARG_SRC_1);
      off    = CTX_IN_MEM (const char *,  DNNL_ARG_SRC_2);     // null in lookup mode
      w      = CTX_IN_MEM (const float *, DNNL_ARG_WEIGHTS);   // null if absent
      dst    = CTX_OUT_MEM(char *,         DNNL_ARG_DST);
      auto sp = ctx.get_scratchpad_grantor();                  // pre-booked, no alloc

      parallel_nd_ext(nthr, B, [&](int ithr, int nthr_, dim_t b) {
          call_params_t p { table, idx, off, w, dst, b /*, sizes, flags... */ };
          kernel_fn_(&p);   // direct call into compiled-in AVX-512 intrinsic kernel
      });
        |
        v
  dst memory written
```

The reference fallback follows the same shape but inlines a portable C++ row-reduction loop instead of calling the intrinsic kernel.

For `algorithm::embedding_lookup` the fan-out is over `N` (number of indices) instead of `B`, the offset arithmetic disappears (each work item handles exactly one row), and `DNNL_ARG_SRC_2` is unused.

## Appendix A. ZenDNN Reference &rarr; Native oneDNN Mapping

ZenDNN's `lowoha::embag` source tree is the *reference implementation* used during porting. After Phase 2 lands, none of these ZenDNN names appear in oneDNN code &mdash; everything is replaced by an idiomatic oneDNN equivalent. The table below documents that translation so reviewers can cross-check the port.

| ZenDNN reference | Translates to in oneDNN | Notes |
| --- | --- | --- |
| `embag_algo_t::sum / mean / max` | `algorithm::embedding_bag_sum / _mean / _max` | New entries in `dnnl_alg_kind_t` enum |
| `embag_algo_t::none` (lookup) | `algorithm::embedding_lookup` | New entry in `dnnl_alg_kind_t` enum |
| `embag_data_types_t::table` / `output` | `src_desc.data_type` / `dst_desc.data_type` | Standard oneDNN memory descriptors |
| `embag_data_types_t::indices` / `offsets` | `indices_desc.data_type` / `offsets_desc.data_type` | Standard oneDNN memory descriptors |
| `embag_data_types_t::scale` / `bias` | `attr.scales` / `attr.zero_points` (Phase 3) | Use existing `primitive_attr` machinery |
| `embag_params_t::num_embeddings` / `embedding_dim` | `src_desc.dims[0]` / `src_desc.dims[1]` | Inferred from MD, no separate field |
| `embag_params_t::num_indices` / `num_bags` | `indices_desc.dims[0]` / `dst_desc.dims[0]` | Inferred from MDs |
| `embag_params_t::is_weights` | `!weights_desc.is_zero()` | Inferred from MD presence |
| `embag_params_t::include_last_offset` | `flags & dnnl_embedding_bag_include_last_offset` | New flags bitmask in op-desc |
| `embag_params_t::padding_idx` | `padding_idx` field in `embedding_bag_desc_t` | New op-desc field |
| `embag_params_t::dst_stride` | derived from `dst_desc.format_desc` | Memory layout, not an explicit param |
| `embag_params_t::num_threads` | `dnnl_get_max_threads()` | Owned by oneDNN's threading runtime |
| `embag_params_t::kernel` (`fbgemm` / `native` / `reference` / ...) | impl-list ordering: `embedding_bag_t` (intrinsic) then `ref_embedding_bag_t` | No public knob; PD `init()` picks the right impl |
| `eb_thread_algo_t` (`table_threaded` / `ccd_threaded` / `hybrid_threaded` / `batch_threaded`) | `parallel_nd_ext(nthr, B, ...)` work split | One canonical strategy chosen at port time; tunable internally if profiling justifies it later |
| `lowoha::embag::embedding_bag_direct(...)` | `embedding_bag_t::execute(ctx)` + the AVX-512 intrinsic kernel function it dispatches to | Direct intrinsic-to-intrinsic port (no JIT rewrite); no call through to ZenDNN at runtime |
| `lowoha::embag::embedding_direct(...)` | same `execute(ctx)` with `alg_kind == embedding_lookup` | Single primitive class, not a separate function |
| `lowoha::embag::group_embedding_bag_direct(...)` | not ported as a single primitive | See §12.B; deferred to a follow-up RFC if perf data justifies it |
| ZenDNN env vars `ZENDNNL_EMBAG_ALGO`, `ZENDNNL_EMBAG_THREAD_ALGO` | none | Tuning baked into the intrinsic impl at port time; no env-var surface |

## Appendix B. File Touchpoints Checklist

For Phase 1, each PR maps to a sub-bullet here. Crossed-off when merged.

**Public headers**
- [ ] `include/oneapi/dnnl/dnnl_types.h`: add `dnnl_embedding_bag` to `dnnl_primitive_kind_t`; add `dnnl_embedding_bag_sum/_mean/_max/_lookup` to `dnnl_alg_kind_t`; add `dnnl_embedding_bag_flags_t`.
- [ ] `include/oneapi/dnnl/dnnl.h`: declare `dnnl_embedding_bag_primitive_desc_create`.
- [ ] `include/oneapi/dnnl/dnnl.hpp`: add `embedding_bag` and `embedding` wrapper structs.

**Common**
- [ ] `src/common/opdesc.hpp`: `embedding_bag_desc_t`.
- [ ] `src/common/embedding_bag.cpp`: `embedding_bag_desc_init` + C entry impl.
- [ ] `src/common/embedding_bag_pd.hpp`: `embedding_bag_pd_t` base class.
- [ ] `src/common/c_types_map.hpp`: enum mirrors.
- [ ] `src/common/primitive_desc_iface.cpp`: add `embedding_bag` to `known_primitive_kind` whitelist.

**CPU**
- [ ] `src/cpu/cpu_engine.hpp`: `DECLARE_IMPL_LIST(embedding_bag)`, `CASE(embedding_bag)`.
- [ ] `src/cpu/cpu_embedding_bag_list.cpp`: impl list registration.
- [ ] `src/cpu/ref_embedding_bag.{hpp,cpp}`: reference implementation.

**Optimized x64 intrinsic impl (Phase 2)**
- [ ] `src/cpu/x64/embedding_bag.hpp`: `embedding_bag_t` + `pd_t` + `DECLARE_COMMON_PD_T`. **No `jit_` prefix**; this is intrinsics, not JIT.
- [ ] `src/cpu/x64/embedding_bag.cpp`: AVX-512 / AVX-512-FP16 intrinsic kernel template instantiations (`<immintrin.h>`), `pick_kernel(...)` lookup, `init()`, `execute()`. Optionally split into `src/cpu/x64/embedding_bag_kernels_avx512.{hpp,cpp}` and `src/cpu/x64/embedding_bag_kernels_avx2.{hpp,cpp}` if the single `.cpp` grows large; the per-translation-unit ISA flags isolate AVX-512 / AVX-512-FP16 code.
- [ ] No new CMake flag; gating is via the existing `DNNL_X64` macro through `CPU_INSTANCE_X64`.
- [ ] Per-source-file ISA compile flags applied via the existing oneDNN CMake helper (same mechanism used for other ISA-targeted code).

**GPU (Phase 1, stub only)**
- [ ] `src/gpu/gpu_impl_list.cpp`: `case primitive_kind::embedding_bag: return empty_list;` (mirrors `sdpa` / `gated_mlp`).

**Tests / docs / examples**
- [ ] `tests/benchdnn/benchdnn.cpp`: dispatch `--embedding-bag`.
- [ ] `tests/benchdnn/embedding_bag/`: full driver.
- [ ] `tests/benchdnn/inputs/embedding_bag/`: harnesses.
- [ ] `tests/benchdnn/doc/driver_embedding_bag.md`: driver doc.
- [ ] `tests/gtests/test_embedding_bag.cpp`.
- [ ] `examples/primitives/embedding_bag.cpp`.
- [ ] `doc/primitives/embedding_bag.md` and TOC link.

## Appendix C. References

**Project guidelines (must read before reviewing).**
- [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) &mdash; library functionality criteria, RFC process, commit-message format, unit-test policy.
- [`CODING_STANDARDS.md`](../../../CODING_STANDARDS.md) &mdash; clang-tidy, clang-format, naming, Xbyak conventions.
- [`.clang-tidy`](../../../.clang-tidy) (top-level) &mdash; the enforced check list.
- [`.clang-format`](../../../.clang-format) (top-level) &mdash; the enforced style file.

**Most relevant existing primitives (patterns to mirror).**
- `src/common/reduction.cpp`, `src/common/reduction_pd.hpp`, `src/cpu/cpu_reduction_list.cpp` &mdash; closest existing primitive for layout / argument conventions.
- `src/common/softmax.cpp`, `src/common/softmax_pd.hpp`, `src/cpu/cpu_softmax_list.cpp` &mdash; forward-only PD with algorithm enum, scratchpad usage, ISA-aware dispatch from PD `init()`.
- `src/cpu/ref_softmax.{hpp,cpp}` &mdash; canonical layout for a non-JIT `primitive_t` subclass: `init()` resolves any per-instance state, `execute()` does the work via `parallel_nd*`. Our intrinsic impl follows this overall shape (state-only `init()`, no `create_kernel()`), with the body of `execute()` calling AVX-512 intrinsic kernels.
- `src/cpu/x64/jit_uni_softmax.{hpp,cpp}` and `src/cpu/x64/jit_uni_reduction.{hpp,cpp}` (+ `*_kernel.{hpp,cpp}`) &mdash; the *JIT* pattern, referenced for PD `init()` / dispatch flow / scratchpad booking. We deliberately do **not** mirror their Xbyak `jit_generator_t` parts (see §6.4 / §12.E for the rationale).
- `src/cpu/x64/cpu_isa_traits.{hpp,cpp}`, `src/cpu/x64/cpu_isa_traits_t<isa>` &mdash; the canonical way to template C++ code on `cpu_isa_t`, including for non-JIT translation units; we use this for our intrinsic kernel templates.

**Internal infrastructure used by every JIT primitive.**
- `src/common/dnnl_thread.hpp` &mdash; `parallel_nd`, `parallel_nd_ext`, `balance211`, `nd_iterator_init`/`step`.
- `src/common/primitive.hpp` &mdash; `CTX_IN_MEM` / `CTX_OUT_MEM` macros.
- `src/common/primitive_desc.hpp` &mdash; `DECLARE_COMMON_PD_T` macro.
- `src/common/primitive_desc_iterator.hpp` &mdash; impl-list walking semantics (`unimplemented` &rarr; advance).
- `src/cpu/cpu_engine.hpp` &mdash; `CPU_INSTANCE`, `CPU_INSTANCE_X64`, `DECLARE_IMPL_LIST`, `CASE`.
- `src/common/impl_registration.hpp` &mdash; `REG_<PRIM>_P` macros.

**Source of the kernels being upstreamed.**
- AMD ZenDNN `lowoha::embag` &mdash; `embedding_bag_direct`, `embedding_direct`, `group_embedding_bag_direct`. Used as reference during Phase 2 porting; not a runtime dependency after the contribution lands.

**Consumer semantics targets.**
- PyTorch `nn.EmbeddingBag` &mdash; primary consumer semantics target.
- TensorFlow `tf.nn.embedding_lookup_sparse` &mdash; sum / mean / max with combiner.
- ONNX Runtime `EmbedLayerNormalization` &mdash; inference-time embedding fast path.

