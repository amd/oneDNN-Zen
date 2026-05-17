# RFC: Embedding and Embedding-Bag Primitive in oneDNN

## Status
**Draft** &nbsp;|&nbsp; AMD-Zenai/oneDNN-ZenDNN &nbsp;|&nbsp; Branch: `rfc/embedding-bag`

## Authors
- AMD-Zenai team

## Summary
This RFC proposes adding a new `embedding_bag` primitive to oneDNN, with `embedding` (lookup-only) exposed as a thin C++ wrapper over the same primitive. The design follows existing oneDNN conventions for primitives such as `softmax` and `reduction`, and is compatible with PyTorch's `nn.EmbeddingBag` semantics. AMD-specific acceleration is provided by an internal implementation that delegates to ZenDNN's existing `embedding_bag_direct` / `group_embedding_bag_direct` kernels; a portable reference implementation is always available.

---

## 1. Background and Motivation

Embedding lookups and embedding-bag reductions are core building blocks for recommendation models (DLRM, DCNv2, etc.) and NLP models that consume token-id sequences. Today, frameworks (PyTorch's `nn.EmbeddingBag`, TF, ONNX Runtime) implement these on CPU either via per-framework kernels, FBGEMM, or vendor libraries. oneDNN currently has **no equivalent primitive**: the closest dense neighbours (`reduction`, `gather` in graph) do not cover the *indexed lookup + variable-length bag reduction* semantics.

ZenDNN already has highly optimized AVX-512 / AVX-512-FP16 implementations for these operations exposed through its internal `lowoha::embag` API (see appendix A). The goal of this RFC is to expose this functionality through a **first-class oneDNN primitive** that:

1. Looks and feels like every other oneDNN primitive (op-desc → primitive-desc → primitive → execute).
2. Plugs into oneDNN's CPU dispatch system so that the ZenDNN-backed implementation is selected automatically on AMD CPUs, with a portable reference implementation as the fallback on every other platform.
3. Maps cleanly onto PyTorch / ONNX consumer semantics so frameworks can adopt it with minimal glue.

## 2. Goals and Non-Goals

### Goals
- One new primitive kind: `dnnl_embedding_bag`.
- Forward pass on CPU.
- Algorithms: `sum`, `mean`, `max`, and `lookup` (no reduction).
- Data types (Phase 1): `f32` table, `f32` output, `s32`/`s64` indices and offsets.
- Data types (Phase 2): `bf16`, `f16` table/output (subject to ISA support).
- Optional per-sample weights, optional `padding_idx`, optional `include_last_offset`.
- ZenDNN-backed CPU implementation behind oneDNN's standard dispatch — no API surface for kernel/threading choice.
- benchdnn driver, gtest coverage, example, user-guide doc.

### Non-Goals (initial release)
- Backward / training. (Tracked as future work in Phase 4; see §10.)
- GPU implementation. (Stub list returns empty until an Intel/SYCL impl exists.)
- Quantized (`s8` / `s4` / `u4`) embedding tables — these are valuable for DLRM-style inference and are tracked in Phase 3 (§10).
- A separate `group_embedding_bag` primitive — see §12.B for the alternative considered and why we defer it.

## 3. oneDNN Convention Recap

Every primitive in oneDNN follows the same skeleton. The new primitive must respect it. Files cited below are the patterns we copy.

| Layer | Files (existing examples) | What we add |
| --- | --- | --- |
| Primitive kind enum | `include/oneapi/dnnl/dnnl_types.h` (`dnnl_primitive_kind_t`) | `dnnl_embedding_bag` |
| Algorithm enum | `include/oneapi/dnnl/dnnl_types.h` (`dnnl_alg_kind_t`) | `dnnl_embedding_bag_sum/_mean/_max`, `dnnl_embedding_lookup` |
| Argument map | `DNNL_ARG_*` in `dnnl_types.h` | reuse `SRC`, `SRC_1`, `SRC_2`, `WEIGHTS`, `DST` (named aliases optional) |
| Internal op-desc struct | `src/common/opdesc.hpp` (e.g. `softmax_desc_t`, `reduction_desc_t`) | `embedding_bag_desc_t` |
| C API | `include/oneapi/dnnl/dnnl.h` (e.g. `dnnl_softmax_forward_primitive_desc_create`) | `dnnl_embedding_bag_primitive_desc_create` |
| C++ wrapper | `include/oneapi/dnnl/dnnl.hpp` (e.g. `dnnl::softmax_forward`, `dnnl::reduction`) | `dnnl::embedding_bag` (+ `dnnl::embedding` thin wrapper) |
| Desc init + entry impl | `src/common/<prim>.cpp` (e.g. `src/common/softmax.cpp`) | `src/common/embedding_bag.cpp` |
| Primitive-desc base | `src/common/<prim>_pd.hpp` (e.g. `src/common/reduction_pd.hpp`) | `src/common/embedding_bag_pd.hpp` |
| Whitelist | `src/common/primitive_desc_iface.cpp::known_primitive_kind` | add `embedding_bag` |
| C↔C++ kind/alg map | `src/common/c_types_map.hpp` | mirror new enums |
| CPU impl list | `src/cpu/cpu_<prim>_list.cpp` (e.g. `cpu_reduction_list.cpp`) + `DECLARE_IMPL_LIST` and `CASE` in `src/cpu/cpu_engine.hpp` | `src/cpu/cpu_embedding_bag_list.cpp`, list reg in engine |
| Reference impl | `src/cpu/ref_<prim>.{hpp,cpp}` | `src/cpu/ref_embedding_bag.{hpp,cpp}` |
| Optimized impl | `src/cpu/x64/jit_*` (e.g. `jit_uni_reduction.cpp`) | `src/cpu/x64/zendnn/zendnn_embedding_bag.{hpp,cpp}` |
| GPU impl list | `src/gpu/gpu_<prim>_list.cpp` + `CASE` in `gpu_impl_list.cpp` | empty list initially (returns `empty_list`, like `sdpa`/`gated_mlp` do today) |
| benchdnn driver | `tests/benchdnn/<prim>/`, dispatch in `benchdnn.cpp`, harnesses in `tests/benchdnn/inputs/<prim>/`, doc `tests/benchdnn/doc/driver_<prim>.md` | full driver |
| Example | `examples/primitives/<prim>.cpp` | `examples/primitives/embedding_bag.cpp` |
| User-guide doc | `doc/primitives/<prim>.md` | `doc/primitives/embedding_bag.md` |

oneDNN exposes only memory descriptors and scalar parameters in the public C API; it does not expose op-descriptor structs publicly. Threading and kernel selection are internal concerns; users never pick a kernel. ZenDNN's `embag_kernel_t` and `eb_thread_algo_t` therefore have no counterpart in the public oneDNN API.

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

The new files mirror the existing layout exactly. Modifications are minimal touches in already-existing dispatch / whitelist files; everything else is greenfield additions inside per-primitive directories.

```
include/oneapi/dnnl/
  dnnl_types.h                              [+] primitive kind, alg kinds, flags
  dnnl.h                                    [+] dnnl_embedding_bag_primitive_desc_create
  dnnl.hpp                                  [+] dnnl::embedding_bag, dnnl::embedding

src/common/
  opdesc.hpp                                [+] embedding_bag_desc_t
  embedding_bag.cpp                         [+] desc_init + C API entry point
  embedding_bag_pd.hpp                      [+] primitive_desc base class
  c_types_map.hpp                           [+] enum mirrors
  primitive_desc_iface.cpp                  [+] add to known_primitive_kind whitelist

src/cpu/
  cpu_engine.hpp                            [+] DECLARE_IMPL_LIST + CASE
  cpu_embedding_bag_list.cpp                [+] impl_list registration
  ref_embedding_bag.hpp                     [+] reference implementation
  ref_embedding_bag.cpp                     [+]
  x64/zendnn/                               [+] new dir for ZenDNN-backed impls
    zendnn_embedding_bag.hpp                [+]
    zendnn_embedding_bag.cpp                [+]

src/gpu/
  gpu_impl_list.cpp                         [+] CASE returns empty_list

tests/
  benchdnn/embedding_bag/                   [+] driver
  benchdnn/inputs/embedding_bag/            [+] test harness files
  benchdnn/doc/driver_embedding_bag.md      [+] driver doc
  benchdnn.cpp                              [+] dispatch for --embedding-bag
  gtests/test_embedding_bag.cpp             [+] gtest coverage

examples/primitives/embedding_bag.cpp       [+] tutorial example
doc/primitives/embedding_bag.md             [+] user-guide markdown
```

### 6.2 Primitive descriptor base

`src/common/embedding_bag_pd.hpp` defines `embedding_bag_pd_t : public primitive_desc_t` with:

- Field accessors `src_md(0)`, `src_md(1)` (indices), `src_md(2)` (offsets), `weights_md(0)`, `dst_md(0)`.
- `arg_usage(int arg)` returning input/output for each `DNNL_ARG_*`.
- `arg_md(int arg)` mapping to the right MD (see `reduction_pd_t::arg_md` for the pattern).
- Helpers: `desc()->alg_kind`, `desc()->padding_idx`, `desc()->flags`, `is_lookup()`, `has_offsets()`, `has_weights()`.

### 6.3 Reference implementation

`src/cpu/ref_embedding_bag.{hpp,cpp}` — `ref_embedding_bag_t : public primitive_t`, single fwd template parameterized over alg_kind. Body uses `parallel_nd(B, ...)` from `src/common/dnnl_thread.hpp` to parallelize across bags. Always present, used as ground truth in benchdnn `--engine=cpu --skip-impl=ref` exclusion logic.

### 6.4 ZenDNN-backed implementation

`src/cpu/x64/zendnn/zendnn_embedding_bag_t` is a thin adapter:

- `init()` checks ISA (AVX-512 for `f32`/`bf16`, AVX-512-FP16 for `f16`) and rejects unsupported configurations.
- `execute()`:
  - Reads memory arguments via `CTX_IN_MEM` / `CTX_OUT_MEM`.
  - Builds a ZenDNN `embag_params_t` from the PD (see Appendix A for field-by-field mapping).
  - Calls `zendnnl::lowoha::embag::embedding_bag_direct(...)` (or `embedding_direct(...)` for lookup mode).
  - Returns `status::success` / `status::unimplemented` based on the ZenDNN return code.

The `group_embedding_bag_direct` ZenDNN entry point is **not** wrapped at the primitive level. oneDNN users invoke a single primitive per call; framework integration code that wants to amortize launch overhead can build a small adapter on top (see §12.B).

Build-time gating: the new `src/cpu/x64/zendnn/` directory and its registration in `cpu_embedding_bag_list.cpp` are guarded by an existing or new CMake option (e.g. `ONEDNN_ENABLE_ZENDNN`). When disabled, only `ref_embedding_bag_t` is in the list.

### 6.5 Dispatch

`src/cpu/cpu_embedding_bag_list.cpp`:

```cpp
namespace dnnl::impl::cpu {
const impl_list_item_t *get_embedding_bag_impl_list(
        const embedding_bag_desc_t *desc) {
    UNUSED(desc);
    constexpr impl_list_item_t impl_list[] = REG_EMBEDDING_BAG_P({
        CPU_INSTANCE_X64(zendnn_embedding_bag_t)   // gated by ZENDNN
        CPU_INSTANCE(ref_embedding_bag_t)
        nullptr,
    });
    return impl_list;
}
} // namespace
```

`src/cpu/cpu_engine.hpp`:

```cpp
DECLARE_IMPL_LIST(embedding_bag);

// inside switch in get_implementation_list():
CASE(embedding_bag);
```

### 6.6 Threading

All threading happens inside `execute()` via oneDNN's standard `parallel_nd` / `parallel_nd_ext` (OMP / TBB / threadpool selected at build time). The ZenDNN-backed implementation uses ZenDNN's internal threading inside `embedding_bag_direct`; it must respect the calling thread limit via `dnnl_get_max_threads()` and ZenDNN's `thread_guard` (already done inside ZenDNN's API).

oneDNN does **not** expose `eb_thread_algo_t` or `embag_kernel_t` to users. Internally, the ZenDNN adapter may consult ZenDNN environment variables (`ZENDNNL_EMBAG_ALGO`, `ZENDNNL_EMBAG_THREAD_ALGO`) for tuning, but this is invisible to the oneDNN API.

## 7. Data Type Support Matrix

`I/O` columns refer to the table (input) and dst (output) data types. Indices and offsets are always integer (`s32` or `s64`). Per-sample weights are always `f32`.

| Phase | Table dtype | Dst dtype | Algorithms | Engine | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `f32` | `f32` | sum, mean, max, lookup | CPU (ref + ZenDNN) | Baseline. Must work on every supported ISA. |
| 2 | `bf16` | `bf16` | sum, mean, max, lookup | CPU (ref + ZenDNN) | ZenDNN dispatches AVX-512 BF16 path. |
| 2 | `f16` | `f16` | sum, mean, max, lookup | CPU (ref + ZenDNN) | Requires AVX-512-FP16 in ZenDNN; ref always works. |
| 3 | `s8`, `s4`, `u4` | `f32` | sum, mean, lookup | CPU (ZenDNN) | Quantized table; uses `attr.scales`. |
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

Each phase ends with a green CI on the AMD-Zenai/oneDNN-ZenDNN main branch and is reviewable as a single PR series.

| Phase | Scope | Done-when |
| --- | --- | --- |
| 1a | Public API plumbing: enums, op-desc, C and C++ entry points, whitelist, c_types_map, ref impl skeleton | A `dnnl::embedding_bag` PD can be created and executed for `f32` `algorithm::embedding_bag_sum` against the ref impl. |
| 1b | Reference impl complete for all 4 algorithms; padding_idx; include_last_offset; per-sample weights; primitive_attr scratchpad | benchdnn driver passes `shapes_basic`. |
| 1c | benchdnn driver wired up; gtests; example; doc/primitives/embedding_bag.md | All new tests run in `make test` and CI. |
| 2 | ZenDNN-backed CPU impl behind `ONEDNN_ENABLE_ZENDNN`; bf16 / f16 dtypes | benchdnn `shapes_dlrm` matches ref within tolerance; ZenDNN dispatch verified on AMD CPU. |
| 3 | int8 / int4 quantized embedding tables via `attr.scales`; post-ops | Performance and accuracy match or beat the standalone ZenDNN `lowoha::embag` on representative DLRM shapes. |
| 4 (future) | GPU implementation; backward pass for training | Out of scope for this RFC. |

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

8. **ZenDNN compile-time vs runtime gating.** `ONEDNN_ENABLE_ZENDNN=ON` adds a build dep on ZenDNN; runtime detection is done with ISA checks already inside ZenDNN. Need to decide whether the ZenDNN dependency is statically or dynamically linked &mdash; affects packaging.

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

### C. Expose ZenDNN's kernel/thread tuning knobs in `primitive_attr`

ZenDNN allows the caller to pick `embag_kernel_t` (fbgemm, native, reference) and `eb_thread_algo_t` (table_threaded, ccd_threaded, hybrid_threaded, batch_threaded).

- **Pros:** Power users can tune.
- **Cons:** Bleeds vendor-specific concerns into oneDNN's public API. Locks oneDNN to ZenDNN's internal taxonomy.

**Decision:** Do not expose. Tuning happens via ZenDNN env vars (`ZENDNNL_EMBAG_*`) inside the ZenDNN-backed implementation, opaque to oneDNN users.

## 13. End-to-End Lifecycle (illustrative)

```
            user code (PyTorch / TF / app)
                         |
                         v
   dnnl::embedding_bag::primitive_desc(eng, sum, table, idx, off, w, dst, ...)
                         |
                         v   (C++ wrapper -> C entry)
   dnnl_embedding_bag_primitive_desc_create(...)
                         |
                         v   (src/common/embedding_bag.cpp)
   embedding_bag_desc_init(...) --> primitive_desc_create(...)
                         |
                         v   (src/common/primitive_desc_iface.cpp)
   known_primitive_kind whitelist check
                         |
                         v   (src/cpu/cpu_engine.hpp - CASE(embedding_bag))
   get_embedding_bag_impl_list(desc)
                         |
                         v   (src/cpu/cpu_embedding_bag_list.cpp)
   try zendnn_embedding_bag_t::pd_t::create() ... if rejected, fall through
   try ref_embedding_bag_t::pd_t::create()
                         |
                         v
                  primitive_desc ready
                         |
   user: dnnl::embedding_bag prim(pd); prim.execute(stream, args);
                         |
                         v
   primitive_t::execute() -> ZenDNN's lowoha::embag::embedding_bag_direct(...)
                         OR ref_embedding_bag_t::execute_forward(...)
                         |
                         v
                    dst memory written
```

## Appendix A. ZenDNN-to-oneDNN Concept Mapping

How each ZenDNN concept lands in oneDNN:

| ZenDNN concept | oneDNN equivalent | Where it lives |
| --- | --- | --- |
| `embag_algo_t::sum / mean / max` | `algorithm::embedding_bag_sum / _mean / _max` | `dnnl_alg_kind_t` enum |
| `embag_algo_t::none` (lookup) | `algorithm::embedding_lookup` | `dnnl_alg_kind_t` enum |
| `embag_data_types_t::table` | `src_desc.data_type` | memory descriptor |
| `embag_data_types_t::output` | `dst_desc.data_type` | memory descriptor |
| `embag_data_types_t::indices / offsets` | `indices_desc.data_type / offsets_desc.data_type` | memory descriptor |
| `embag_data_types_t::scale / bias` | `attr.scales` / `attr.zero_points` (Phase 3) | `primitive_attr` |
| `embag_params_t::num_embeddings / embedding_dim` | `src_desc.dims[0]`, `src_desc.dims[1]` | memory descriptor |
| `embag_params_t::num_indices` | `indices_desc.dims[0]` | memory descriptor |
| `embag_params_t::num_bags` | `dst_desc.dims[0]` (bag mode) | memory descriptor |
| `embag_params_t::is_weights` | `weights_desc.is_zero() == false` | inferred |
| `embag_params_t::include_last_offset` | `flags & include_last_offset` | flags bitmask |
| `embag_params_t::padding_idx` | `padding_idx` op-desc field | op descriptor |
| `embag_params_t::dst_stride` | derived from `dst_desc.format_desc` | memory descriptor |
| `embag_params_t::num_threads` | inherited from oneDNN's threading runtime | not exposed |
| `embag_params_t::kernel` | hidden inside the impl list | not exposed |
| `eb_thread_algo_t` | hidden inside the impl list | not exposed |
| `embedding_bag_direct(...)` | called from `zendnn_embedding_bag_t::execute()` | impl |
| `embedding_direct(...)` | called when alg_kind == lookup | impl |
| `group_embedding_bag_direct(...)` | not wrapped at primitive level | see §12.B |

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

**ZenDNN backend (Phase 2)**
- [ ] `src/cpu/x64/zendnn/zendnn_embedding_bag.{hpp,cpp}`.
- [ ] CMake option `ONEDNN_ENABLE_ZENDNN`.

**GPU (Phase 1, stub only)**
- [ ] `src/gpu/gpu_impl_list.cpp`: `case primitive_kind::embedding_bag: return empty_list;`

**Tests / docs / examples**
- [ ] `tests/benchdnn/benchdnn.cpp`: dispatch `--embedding-bag`.
- [ ] `tests/benchdnn/embedding_bag/`: full driver.
- [ ] `tests/benchdnn/inputs/embedding_bag/`: harnesses.
- [ ] `tests/benchdnn/doc/driver_embedding_bag.md`: driver doc.
- [ ] `tests/gtests/test_embedding_bag.cpp`.
- [ ] `examples/primitives/embedding_bag.cpp`.
- [ ] `doc/primitives/embedding_bag.md` and TOC link.

## Appendix C. References

- ZenDNN `lowoha::embag::embedding_bag_direct` &mdash; reference implementation we wrap. (See ZenDNN tree `operators/embag/` and `lowoha_operators/embag/`.)
- oneDNN `reduction` primitive &mdash; closest existing primitive for layout / argument conventions: `src/common/reduction.cpp`, `src/common/reduction_pd.hpp`, `src/cpu/cpu_reduction_list.cpp`.
- oneDNN `softmax` primitive &mdash; reference for forward-only PD with algorithm enum: `src/common/softmax.cpp`, `src/common/softmax_pd.hpp`, `src/cpu/cpu_softmax_list.cpp`.
- PyTorch `nn.EmbeddingBag` &mdash; consumer semantics target.
- FBGEMM EmbeddingBag kernels &mdash; ZenDNN's underlying optimized backend on AVX-512.

