# RFC: Constant Memory Property and Library-Managed Weight Cache (Primitive API)

## Status
**Draft** &nbsp;|&nbsp; AMD-Zenai/oneDNN-ZenDNN &nbsp;|&nbsp; Branch: `rfc/zendnn-integration`

## Authors
- AMD ZenDNN team

## Summary

This RFC proposes adding a **constant memory property** to oneDNN's Primitive API, plus a small **library-managed weight cache** that uses the property to prepack and reuse user-supplied weight tensors across `execute()` calls. The user marks a weight (or any other invariant input) as constant on its `dnnl::memory::desc`; the library is then free to convert it to whichever blocked layout the chosen impl prefers, perform the reorder lazily on the first execute, and reuse the prepacked buffer on subsequent executes &mdash; all without the framework having to know the impl's preferred layout, query it, allocate a second buffer, or manage the reorder lifecycle itself.

This pattern already exists and ships in production inside oneDNN's **Graph API** (`dnnl_graph_tensor_property_constant` plus `dnnl_graph_set_constant_tensor_cache_capacity`). It does **not** exist in the Primitive API. Today, primitive-API consumers (PyTorch's `at::native::onednn`, llama.cpp via custom paths, vLLM CPU paths, the many in-house framework integrations) implement their own weight-reorder caching outside the library, each one differently and each one carrying the same boilerplate. This RFC closes the gap by lifting the Graph-API behaviour into the Primitive API in a clean, opt-in way.

The design is opt-in (default behaviour unchanged), capacity-bound (the cache size is user-controlled, default `0` = disabled), and per-engine (CPU and GPU caches are independent). The first implementation targets `matmul`; the same shape generalises to `convolution`, `inner_product`, and any other primitive that produces a noticeably better impl when its weight is in a non-plain layout.

---

## 1. Motivation

### 1.1 oneDNN's Primitive API is stateless today

The contract of every Primitive-API call today is:

1. The user constructs a `primitive_desc` with `format_tag::any` for inputs / outputs they don't care about.
2. The user **queries** the primitive's chosen layout (`pd.weights_desc()` etc.).
3. If the chosen layout differs from what the user is holding, the user constructs a separate `dnnl::memory` in the chosen layout, runs an explicit `reorder` primitive once, and **caches the reordered `dnnl::memory` in framework state**.
4. On every subsequent inference call, the user passes the cached reordered memory.

This works, and it has been the official guidance in `doc/usage_models/inference.md`. But it has well-known drawbacks:

- **The framework re-implements the same caching, differently, every time.** PyTorch, TensorFlow, ONNX Runtime, vLLM CPU paths, llama.cpp, and the dozens of in-house integrations each carry their own "queried-layout to cached-memory" plumbing. The bookkeeping is non-trivial (the cache is keyed by the chosen impl's preferred layout, which can change across impl-list updates and oneDNN versions).
- **The framework has to decide upfront, before dispatch resolves, what layout it will commit to.** If the impl that wins dispatch wants a different layout than the one the framework prepacked for, the framework either runs an extra reorder per call (defeating the prepack) or settles for a sub-optimal impl.
- **Multi-backend dispatch breaks the static prepack assumption.** When more than one impl could win for a given problem (e.g. an AMD-tuned backend competing with the existing JIT, an ISA-specific impl competing with the generic one), the framework cannot pre-commit to a single layout without losing access to the impls that prefer a different one.

### 1.2 The pattern already exists in Graph API

oneDNN's Graph API has solved exactly this problem inside its own surface. A logical tensor can be marked **constant** (`dnnl_graph_tensor_property_constant`) to declare it stays unchanged across iterations; the backend then converts the weight memory desc to `format_kind::any`, lets the primitive pick a blocked layout, performs the reorder lazily on the first execute, and stores the prepacked buffer in a per-engine, MB-bounded cache. On a cache hit the prepack is skipped and the cached buffer is reused. The cache capacity is controlled by `dnnl_graph_set_constant_tensor_cache_capacity()`.

This pattern is well-validated: it has been in oneDNN Graph for several releases and is the recommended path for inference workloads when consumers go through Graph.

### 1.3 The gap

Many oneDNN consumers do not go through Graph. They use the Primitive API directly &mdash; either because they predate Graph adoption, or because their dispatch logic doesn't fit Graph's partition model, or because they want fine-grained control over each kernel call. Those consumers cannot use Graph's constant-tensor cache; they reimplement it externally, with each implementation carrying its own subtle bugs (missed cache invalidation when the user swaps weight tensors, wrong key when the impl-list changes, no shared cache across primitives).

This RFC closes that gap by **lifting the Graph-API constant-tensor mechanism into the Primitive API** &mdash; same semantic, same cache shape, same opt-in capacity-bound contract, but accessible via `dnnl::memory::desc` and `dnnl::primitive` instead of `dnnl::graph::logical_tensor` and `dnnl::graph::partition`.

### 1.4 Library-functionality criteria (CONTRIBUTING.md)

| Criterion | How |
|---|---|
| **Performance** | The library-side cache eliminates per-call reorder cost on inference workloads with constant weights. Validated at model level via vLLM / PyTorch eager / ONNX Runtime CPU on representative LLM and recommendation workloads before each production PR. |
| **Generality** | Every Primitive-API consumer benefits with no framework-side change beyond the one-line property setter. PyTorch (`at::native::onednn`), TensorFlow, ONNX Runtime, vLLM CPU paths, llama.cpp custom integrations, plus any in-house consumer all map onto the same property + cache surface. |
| **Complexity** | Correct, race-free, capacity-bound weight caching with cache-key invalidation on data-pointer change is non-trivial. Today every framework does this differently; centralising it in oneDNN saves duplication and keeps the invariants in one place. |

## 2. Goals and Non-Goals

### Goals
- One new memory-descriptor property: `dnnl_memory_property_constant`, settable via a new public API.
- One new global, per-engine, capacity-bound cache for prepacked constant weights.
- **Opt-in.** Default capacity is `0` (disabled). When capacity is `0`, the property is recorded but the library does not allocate any prepack buffer; behaviour is identical to today's stateless flow.
- **Idiomatic for primitives that already use `format_tag::any`** &mdash; the cache plugs into the existing `set_default_formats()` flow without changes to op-descriptor structs.
- **Phase 1 implementation: `matmul`.** Other primitives (`convolution`, `inner_product`) follow in subsequent PRs using the same hooks.
- **No public API change to existing primitives.** `dnnl::matmul`, `dnnl::convolution_forward`, etc. continue to work unchanged for users who don't set the property.
- benchdnn coverage, gtest correctness, user-guide doc.

### Non-Goals (initial release)
- **GPU backend.** The first PR is CPU-only; GPU follows naturally because the cache is engine-keyed, but the GPU memory-management path needs its own scoping.
- **Cache inspection / introspection API.** Users do not query the cache contents or the prepacked layout. The cache is internal.
- **Sharing cached weights across processes.** Cache is per-process. (The existing `cache_blob` mechanism, which serialises primitive *creation*, is orthogonal and not changed by this RFC.)
- **Automatic constant-detection.** The library does not analyse data to infer a tensor is constant; the user marks it explicitly. This matches Graph API semantics and avoids guessing.
- **Replacing the existing `format_tag::any` + explicit-reorder pattern.** The existing pattern continues to work; this RFC only adds an alternative that is more ergonomic for invariant inputs.
- **Cross-primitive cache sharing.** A weight prepacked for `matmul` is not automatically reused for a different primitive (e.g. `convolution`) on the same data, because the chosen layouts may differ. Each primitive consumes its own cache entry keyed on its own preferred layout.

## 3. Existing Precedent: oneDNN Graph API

The Graph API ships the substantive feature today. This RFC proposes lifting the same shape into the Primitive API, with the smallest possible API delta. The relevant Graph-API surface is summarised here so the lift is concrete.

### 3.1 Tensor property

```c
typedef enum {
    dnnl_graph_tensor_property_undef       = 0,
    dnnl_graph_tensor_property_variable    = 1,
    dnnl_graph_tensor_property_constant    = 2,
    dnnl_graph_tensor_property_host_scalar = 3,
} dnnl_graph_tensor_property_t;
```

The Graph documentation for `constant` says: *"The tensor will keep unchanged during computation and between different iterations. It's useful for the library to apply optimizations for constant tensors or cache constant tensors inside the library. For example, constant weight tensors in inference scenarios."*

### 3.2 Cache control

```c
dnnl_status_t dnnl_graph_set_constant_tensor_cache_capacity(
        dnnl_engine_kind_t engine_kind, size_t size_mb);
dnnl_status_t dnnl_graph_get_constant_tensor_cache_capacity(
        dnnl_engine_kind_t engine_kind, size_t *size_mb);
```

Default `size_mb = 0` (disabled). Per-engine, capacity-bound, with LRU-style eviction when full.

### 3.3 Backend behaviour

When the cache is enabled and a tensor is marked `constant`, the Graph backend (e.g. `src/graph/backend/dnnl/kernels/matmul.cpp`):

1. Converts the weight memory descriptor to `format_kind::any` so the primitive can pick a blocked layout.
2. On the first `execute()`, allocates a persistent buffer, runs the reorder into it, and stores the buffer in `constant_tensor_cache_t` keyed on `(partition_hash, weight_data_ptr)`.
3. On subsequent `execute()` calls, looks up the buffer by the same key and reuses it; non-constant subgraph nodes still execute normally.
4. The buffer lives until either the partition is destroyed or the cache evicts it under capacity pressure.

This RFC mirrors that flow inside the Primitive API.

## 4. The Gap in the Primitive API

A direct comparison of what's available today:

| Concept | Graph API | Primitive API |
|---|---|---|
| Mark a tensor as "constant / won't change across iterations" | yes (`tensor_property::constant` on `logical_tensor`) | **no** &mdash; no public flag on `dnnl_memory_desc_t` |
| Library swaps the marked tensor's MD to `format_kind::any` | yes (in backend `matmul`, `convolution`, etc.) | n/a |
| Library reorders into the chosen layout on the first execute | yes (lazy on cache miss) | **no** |
| Library caches the reordered buffer keyed on data pointer | yes (`constant_tensor_cache_t`) | **no** |
| Subsequent executes hit the cache and skip the reorder | yes | **no** |
| Cache size is capacity-bound and per-engine | yes (MB, per `engine::kind`) | **no** |
| User can query / set cache capacity at runtime | yes (`dnnl_graph_set_constant_tensor_cache_capacity`) | **no** |
| Env var to set capacity at start-up | yes (`ONEDNN_GRAPH_CONSTANT_TENSOR_CACHE_CAPACITY`) | **no** |

The current oneDNN guidance for primitive-API consumers (`doc/usage_models/inference.md`) is: *"use `format_tag::any`, query the preferred layout, the framework reorders once and reuses the reordered `dnnl::memory` across calls."* That works mechanically, but pushes every consumer to re-implement the cache externally and to commit to a single layout before dispatch is known.

This RFC adds the missing capability without disturbing the existing flow &mdash; users who don't set the property keep the current behaviour exactly.

## 5. Proposed API

### 5.1 New memory-descriptor property

A new opaque enum, settable via a new pair of functions on `dnnl_memory_desc_t`:

```c
/* include/oneapi/dnnl/dnnl_types.h */

typedef enum {
    dnnl_memory_property_undef    = 0,
    /// The tensor will not change between calls to `dnnl_primitive_execute`
    /// that consume it. The library is free to reorder it to its preferred
    /// internal layout once and cache the reordered buffer for reuse on
    /// subsequent executes. The user must guarantee data invariance for as
    /// long as primitives created with this descriptor are in use.
    dnnl_memory_property_constant = 1,
} dnnl_memory_property_t;
```

```c
/* include/oneapi/dnnl/dnnl.h */

dnnl_status_t DNNL_API dnnl_memory_desc_set_property(
        dnnl_memory_desc_t memory_desc,
        dnnl_memory_property_t property);

dnnl_status_t DNNL_API dnnl_memory_desc_get_property(
        const_dnnl_memory_desc_t memory_desc,
        dnnl_memory_property_t *property);
```

The property is stored in the existing `extra` field of the internal `memory_desc_t` so the public `dnnl_memory_desc_t` opaque handle stays unchanged in size and layout. Implementations of `set_property` operate on the internal struct via the standard `memory_desc_wrapper`.

### 5.2 C++ wrapper

```cpp
/* include/oneapi/dnnl/dnnl.hpp */

namespace dnnl {

enum class memory_property : unsigned {
    undef    = dnnl_memory_property_undef,
    constant = dnnl_memory_property_constant,
};

struct memory {
    struct desc {
        // ... existing API unchanged ...
        void           set_property(memory_property p);
        memory_property get_property() const;
    };
};

} // namespace dnnl
```

Both methods are thin wrappers over the C entry points. No existing constructors, accessors, or helpers change behaviour; the property defaults to `undef` for every existing code path.

### 5.3 Cache capacity control

A new pair of global, per-engine functions controls how much memory the cache may hold. Default capacity is `0` (cache disabled) so behaviour is unchanged for users who don't opt in.

```c
/* include/oneapi/dnnl/dnnl.h */

dnnl_status_t DNNL_API dnnl_set_const_weight_cache_capacity(
        dnnl_engine_kind_t engine_kind, size_t size_mb);

dnnl_status_t DNNL_API dnnl_get_const_weight_cache_capacity(
        dnnl_engine_kind_t engine_kind, size_t *size_mb);
```

C++:

```cpp
namespace dnnl {
void   set_const_weight_cache_capacity(engine::kind ekind, size_t size_mb);
size_t get_const_weight_cache_capacity(engine::kind ekind);
}
```

### 5.4 Environment variable

Setting `ONEDNN_CONST_WEIGHT_CACHE_CAPACITY=<MB>` at process start has the same effect as calling `dnnl_set_const_weight_cache_capacity(<each engine>, <MB>)` after engine creation. This matches the precedent of `ONEDNN_PRIMITIVE_CACHE_CAPACITY` and `ONEDNN_GRAPH_CONSTANT_TENSOR_CACHE_CAPACITY`. The runtime API takes precedence over the env var if both are set.

### 5.5 User contract

The user's promise when setting `dnnl_memory_property_constant`:

1. **Data invariance.** The bytes pointed to by the `dnnl::memory` constructed against this descriptor will not change for as long as any primitive consuming it is in use.
2. **Pointer stability is the cache key.** If the user destroys the underlying buffer and allocates a new one with the same descriptor, they should expect a cache miss on the next execute that uses the new pointer; the library will reorder again. This matches Graph API semantics (cache key includes the data pointer).
3. **Lifetime alignment with primitives.** The user must keep the underlying memory alive at least until all primitives that have observed it have been destroyed. If the user frees the buffer while a cache entry still references it, behaviour is undefined.
4. **No edits via re-quantisation, in-place updates, etc.** If the user needs to update the weight (e.g. continued fine-tuning, reloading a model), they should either avoid the property altogether or call a new explicit `dnnl_invalidate_const_weight_cache_entry()` (proposed as a follow-up if real workloads ask for it; not in this RFC's scope).

The user may set the property on **any** input memory descriptor &mdash; not only weights. The name "constant" is the semantic; "weight" appears in the cache function names because that is the dominant use case, but the contract is symmetric for any invariant input (e.g. learned positional encodings, per-token bias tables, and so on).

### 5.6 Validation

`dnnl_memory_desc_set_property` rejects:

- Setting a property on a `zero_md` (no destination to attach the flag to).
- Setting a property on a memory descriptor that already has incompatible `extra` flags (e.g. compensation flags from quantised conv) &mdash; the two are not combined in Phase 1.

Setting `dnnl_memory_property_constant` on an MD that the consumer primitive does not use as a constant (e.g. a destination MD on a forward primitive) is **allowed and silently ignored** by the consumer. This keeps the property attribute-of-data semantics independent of where the data is consumed.

## 6. Implementation Architecture

### 6.1 Source layout (additions only)

```
include/oneapi/dnnl/
  dnnl_types.h                              [+] dnnl_memory_property_t enum
  dnnl.h                                    [+] dnnl_memory_desc_set_property /
                                                _get_property,
                                                dnnl_set_const_weight_cache_capacity /
                                                _get_capacity
  dnnl.hpp                                  [+] memory::desc::set_property /
                                                _get_property,
                                                set_const_weight_cache_capacity,
                                                _get_capacity

src/common/
  memory_desc.hpp                           [M] add property field to memory_extra_desc_t
                                                (or new sibling)
  memory_desc.cpp                           [M] property getter/setter implementation
  memory_desc_wrapper.hpp                   [M] is_constant() helper
  const_weight_cache.hpp                    [+] new module: cache class + global instance
                                                lookup
  const_weight_cache.cpp                    [+]
  primitive.cpp / primitive_iface.cpp       [M] thread cached buffer through execute()

src/cpu/                                    (Phase 1: matmul integration)
  matmul/matmul_pd.cpp                      [M] honour constant property in
                                                set_default_formats() (swap to any) and
                                                book a slot in the cache plan

src/cpu/x64/matmul/                         (depending on which existing matmul impl
                                             gates the cache lookup)
  brgemm_matmul.cpp                         [M] minor: use the helper that returns either
                                                the user MD or a cached prepacked buffer

doc/programming_model/
  const_weight_cache.md                     [+] user-facing doc

tests/
  benchdnn/matmul/                          [M] add `--const-weights=true|false` knob;
                                                verify warm-cache skips the reorder
  gtests/test_const_weight_cache.cpp        [+] gtest correctness + invalidation cases
```

### 6.2 The cache module: `src/common/const_weight_cache.{hpp,cpp}`

Modelled after `src/graph/interface/constant_tensor_cache.*`, with the same shape but in `dnnl::impl::common::` rather than `dnnl::graph::impl::interface::`:

- `class const_weight_cache_t` &mdash; per-engine instance, capacity-bound (MB), thread-safe (read-mostly: shared mutex).
- **Cache key:** `cache_key_t { primitive_kind, hash_of_pd_args, src_ptr, src_md_hash }`. Including the data pointer matches Graph-API behaviour and makes invalidation automatic when the user swaps tensors.
- **Cache value:** `std::shared_ptr<dnnl::impl::memory_storage_t>` &mdash; the prepacked buffer; held under `std::shared_ptr` so lookups can return a stable pointer even if the cache evicts the entry concurrently.
- **Eviction:** LRU on capacity overflow. When evicted, the underlying buffer is released the moment its last `shared_ptr` goes out of scope.
- **API:**
  ```cpp
  std::shared_ptr<memory_storage_t>
  get_or_compute(const cache_key_t &key,
                 size_t size_bytes,
                 std::function<status_t(memory_storage_t &)> fill);
  ```
  `fill` is the impl-supplied lambda that runs the reorder on a cache miss; on a hit, the cached buffer is returned without invoking `fill`.

A global `const_weight_cache_t &get_const_weight_cache(engine_kind_t)` returns the per-engine instance. The cache instances are constructed lazily on first use after the user sets a non-zero capacity.

### 6.3 Integration with `format_kind::any` flow

When a primitive's `pd_t::set_default_formats()` runs, it currently picks the impl's preferred layout for any input MD with `format_kind::any`. The new flow:

1. The PD checks each input MD for `dnnl_memory_property_constant`.
2. If the property is set **and** the cache is enabled (`get_const_weight_cache_capacity(engine) > 0`):
   - The PD treats the user-supplied MD as a logical declaration and additionally records a *cached layout* MD that carries the impl's preferred blocked layout.
   - The PD books a cache slot of the corresponding size with a stable key derived from the PD's internal hash plus the user-supplied MD hash.
3. If the property is **not** set or the cache capacity is `0`:
   - The PD behaves exactly as today &mdash; the primitive consumes the user-supplied MD as-is, and the user is responsible for any external reorder.

This integration is local to the per-primitive `*_pd.cpp` and does not require changes to op-descriptor structs.

### 6.4 Primitive-side execute path

`primitive_t::execute(const exec_ctx_t &ctx)` for primitives that opt in:

```cpp
status_t matmul_t::execute(const exec_ctx_t &ctx) const {
    auto wei_arg = ctx.input(DNNL_ARG_WEIGHTS);
    bool wei_is_const = wei_arg.desc().is_constant();         // helper

    const memory_storage_t *effective_wei = wei_arg.storage();
    if (wei_is_const && pd()->const_cache_enabled_) {
        // Look up or compute the prepacked buffer.
        auto cached = const_weight_cache(ctx.engine())
                .get_or_compute(pd()->const_cache_key(wei_arg),
                                pd()->cached_wei_md().size(),
                                [&](memory_storage_t &dst) {
                                    return run_internal_reorder(
                                            wei_arg.storage(),
                                            wei_arg.desc(),
                                            dst,
                                            pd()->cached_wei_md());
                                });
        effective_wei = cached.get();
    }

    return run_kernel_with_inputs(/*src=*/..., /*wei=*/effective_wei, ...);
}
```

`run_internal_reorder()` is implemented as a one-shot internal reorder primitive constructed at the consumer PD's `init()` time and held inside the consumer primitive (compiled once, reused across cache misses). The cache value is a `memory_storage_t`, which means the impl gets a raw pointer back the same way as for any other primitive input &mdash; no extra abstraction layer on the hot path.

### 6.5 Eviction and lifetime

- Each cache entry holds a `shared_ptr<memory_storage_t>`. When the cache evicts the entry under capacity pressure, it drops the entry's `shared_ptr`; the buffer is freed when the last user of it (typically the running `execute()` if any) finishes.
- Primitive destruction does **not** evict the entry; the cache outlives the primitive. This is desirable because the same weight tensor is often consumed by multiple primitives over a workload's lifetime.
- The cache exposes a `clear()` method (not in the public API; available to oneDNN internals) to allow `dnnl_engine_destroy` to drop entries belonging to that engine.

### 6.6 Thread safety

- The cache uses a `std::shared_mutex`: the read path (lookups) takes the shared lock; the write path (insertion / eviction) takes the unique lock.
- The cache miss path runs the reorder under a *per-key* fine-grained mutex so two threads racing to fill the same entry don't run the reorder twice.
- All buffer reads on cache hits are lock-free after the initial lookup.

### 6.7 Performance considerations

- **No allocation on cache hit.** The hot path is one cache lookup (read-locked) and one pointer fetch.
- **Reorder runs once per `(PD, weight pointer)` tuple.** The reorder cost is amortised across the lifetime of the cached buffer.
- **No extra indirection on the kernel hot path.** The cached buffer is returned as a `memory_storage_t *` which the impl uses identically to a user-supplied storage.
- **No per-call validation of the property.** PD `init()` resolves whether the cache path is taken; `execute()` does not re-check the property bit.

### 6.8 Interaction with existing oneDNN mechanisms

- **Primitive cache (`dnnl_set_primitive_cache_capacity`)** &mdash; orthogonal. The primitive cache stores compiled primitives and their state; the const-weight cache stores prepacked weight buffers. Both can be enabled simultaneously.
- **`cache_blob` / persistent cache** &mdash; orthogonal. Persistent cache serialises primitive *creation*; this cache lives only in process memory.
- **`attr.scratchpad_mode`** &mdash; orthogonal. Scratchpad is per-execute transient; this cache is across-execute persistent.
- **Graph API constant tensor cache** &mdash; independent. Graph keeps using its own cache for partition-level execution; the new primitive-level cache is for the Primitive-API surface. The two caches do not share state. Future work could unify them once both have stabilised.

## 7. Use Cases

### 7.1 Standard inference with constant weights

The dominant use case. The framework constructs the weight `dnnl::memory` once at model-load time, marks it constant, and runs many `execute()` calls with varying inputs. The library prepacks on the first execute and reuses on every subsequent call. The framework does not need to query the impl's preferred layout, allocate a second buffer, or run an explicit reorder.

```cpp
auto wei_md = memory::desc({K, N}, dt::f32, tag::ab);
wei_md.set_property(memory_property::constant);
auto wei = memory(wei_md, eng, weight_data_ptr);

auto pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md);
auto prim = matmul(pd);

dnnl::set_const_weight_cache_capacity(engine::kind::cpu, /*MB=*/256);

for (...) {
    prim.execute(stream, { {DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei}, {DNNL_ARG_DST, dst} });
    // first execute does the reorder + caches; subsequent executes hit the cache.
}
```

### 7.2 Multi-backend dispatch

When more than one impl could win for a given problem &mdash; for example, an arch-tuned backend competing with the generic JIT &mdash; the chosen impl's preferred layout is decided at `pd_t::init()` time and is generally *not* knowable to the framework in advance. Today this forces the framework to either pre-commit to a single impl's layout (and lose access to better impls that prefer a different one) or skip the prepack altogether (and pay the reorder cost on every call).

With the constant property:

- The framework declares the weight constant and supplies it in its canonical user-visible layout.
- The dispatcher selects the best impl for the problem; the impl's PD `init()` declares its preferred weight layout.
- The first `execute()` runs the reorder into that layout; the result is cached for the lifetime of the workload.

The framework does not have to know which impl won, what layout it wanted, or when to reorder. The same code works across stock JIT, ISA-specific tuned backends, vendor backends, and arch-specific optimised paths.

### 7.3 Low-latency inference with shape variability

For workloads where activation shape varies across calls (e.g. variable-length LLM decode) but weights are fixed, the framework today either:

- prepacks the weight in one layout and uses it across all shapes, accepting that the prepack may not be optimal for every shape; or
- maintains a per-shape prepacked-weight cache itself.

With the library-side cache, the second option becomes the default behaviour without framework code &mdash; if two PDs (one per shape bucket) want different layouts, the cache holds two entries keyed on the PD hash and the same data pointer; both prepacks reuse the same source data and the cache enforces capacity globally.

### 7.4 Group MatMul and other multi-weight workloads

When a workload uses many weight tensors (e.g. MoE inference with many experts, transformer-block stacks where each layer has its own weights), the cache concentrates the per-tensor reorder cost into the first call and reuses across all subsequent calls. The MB-bound capacity gives the framework a single global knob to limit oneDNN's working-set growth.

## 8. End-to-End Lifecycle

### 8.1 First `execute()` (cache miss)

```
user code:
   wei_md.set_property(memory_property::constant);
   set_const_weight_cache_capacity(cpu, 256);

   auto pd   = matmul::primitive_desc(eng, src_md, wei_md, dst_md);
   auto prim = matmul(pd);
   prim.execute(stream, args);
                                 │
                                 ▼   pd_t::init() (during primitive_desc creation):
   ┌────────────────────────────────────────────────────────────────────┐
   │  set_default_formats():                                            │
   │    - if wei_md.is_constant() AND cache capacity > 0:               │
   │        wei_md_logical    = user-supplied (canonical layout)        │
   │        wei_md_cached     = impl-preferred blocked layout           │
   │        const_cache_key   = hash(pd, wei_md_logical)                │
   │        const_cache_size  = wei_md_cached.size_bytes                │
   │    - else:                                                         │
   │        behave exactly as today (no cache path)                     │
   └────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼   first execute():
   matmul_t::execute(ctx):
        wei_storage = ctx.input(WEIGHTS).storage()
        if wei_is_const && cache_enabled:
            cached = const_weight_cache(eng).get_or_compute(
                key:  const_cache_key + ptr(wei_storage),
                size: const_cache_size,
                fill: [&](dst_storage){
                    run_internal_reorder(wei_storage, wei_md_logical,
                                         dst_storage,  wei_md_cached);
                });
            effective_wei = cached.get();   // freshly populated
        else:
            effective_wei = wei_storage;

        run_kernel(src, effective_wei, dst);
                                 │
                                 ▼
   dst written;
   reorder cost paid once;
   prepacked buffer now resident in const_weight_cache.
```

### 8.2 Subsequent `execute()` calls (cache hit)

```
prim.execute(stream, args);
                                 │
                                 ▼
   matmul_t::execute(ctx):
        cached = const_weight_cache(eng).get_or_compute(
            key:  const_cache_key + ptr(wei_storage),     // same key as before
            size: const_cache_size,
            fill: <not invoked>);                          // cache hit
        effective_wei = cached.get();                      // cached buffer
        run_kernel(src, effective_wei, dst);
                                 │
                                 ▼
   dst written;
   no reorder; one shared-lock cache lookup.
```

### 8.3 User swaps the weight tensor

If the user constructs a new `dnnl::memory` with a different `data_ptr` and re-runs:

```
prim.execute(stream, { ..., {DNNL_ARG_WEIGHTS, new_wei}, ... });
                                 │
                                 ▼
   key changes (data pointer differs)
   --> cache miss
   --> new entry computed and cached
   --> previous entry remains until evicted under capacity pressure or
       until the engine that owns the cache is destroyed.
```

This matches Graph-API behaviour and keeps the user contract simple: pointer equality is the cache key.

## 9. Risks and Open Questions

1. **Adding a property to an opaque public type.** `dnnl_memory_desc_t` is documented as opaque; the property must be stored in the existing `extra` field to keep the public handle stable. The setter / getter functions are the user-facing surface. No ABI break.
2. **Cache key choice (data pointer vs content hash).** Pointer-equality matches Graph API and is cheap; content-hash is safer (catches the case where the user mutates the buffer in place despite the contract) but costs a hash on every execute. The proposal uses pointer-equality (matches Graph). Consideration: add a debug-build assertion that hashes the first/last cache lines and compares on cache hit, to surface contract violations early without paying the hash on release builds. Decision deferred to first review.
3. **Lifetime / dangling pointers.** If the user frees the underlying buffer while a cache entry still references it, the cache holds a dangling reference. Mitigation: the user-contract documentation makes this an explicit responsibility (same as Graph API); a follow-up RFC could propose an explicit weak-ref / invalidation API if real workloads ask for it.
4. **Multi-thread cache coherency.** A common pattern is multiple inference threads sharing the same model. The cache uses a `shared_mutex` for read-mostly access; the per-key fine-grained lock prevents two threads from running the reorder concurrently on a cold key.
5. **Memory accounting opacity.** The user sees only the cache capacity; they cannot easily tell how much of it is in use. A diagnostic API (`dnnl_get_const_weight_cache_usage()`) is a candidate follow-up. Phase 1 keeps the API minimal.
6. **Interaction with `attr.scratchpad_mode = user`.** The cache buffers are owned by the library, not by user-supplied scratchpad. The two coexist cleanly because they serve different purposes (transient vs persistent). Document this in the user-guide doc.
7. **Cache scope across engines.** The cache is per-engine to mirror Graph API. If a user constructs multiple CPU engines (rare but legal), each engine has its own cache. This is consistent with Graph API and keeps the lifetime story simple.
8. **Performance gate (CONTRIBUTING.md).** Every new mechanism must show "material workload-level impact." Mitigation: ship benchdnn perf evidence (warm-cache vs cold-cache for representative LLM and recommendation MatMul shapes) and model-level numbers (vLLM warm-up vs steady-state inference latency) with the production PR.

## 10. Alternatives Considered

### A. Use a `primitive_attr` flag instead of a memory-desc property (rejected)

Add `attr.set_const_weight(arg_index, true)` to `primitive_attr`, identifying which input argument is constant.

- **Pros:** Sidesteps any debate over adding a new property to the opaque `memory_desc`. The flag travels with the PD.
- **Cons:** The thing that is constant is the *tensor*, not the *operation*. Putting it on `primitive_attr` makes it harder to reuse the same logical tensor across multiple primitives (each consumer would need to set the attribute again with the right `arg_index`). It also splits constancy across two places once we add it for non-weight tensors (e.g. learned positional encodings). Memory-property is the right level of abstraction.

**Decision:** Reject. Use the memory-desc property.

### B. Document the gap and recommend framework-side caching (rejected)

Don't add the API; keep documenting the existing `format_tag::any` + framework-side reorder cache pattern as the canonical inference path.

- **Pros:** No API change.
- **Cons:** Multi-backend dispatch (where the framework cannot pre-commit to a layout) breaks under this approach. Every framework reimplements the cache differently. Diagnostics and bug-fixes have to land in N different places.

**Decision:** Reject.

### C. Wait for Graph API adoption (rejected)

Direct primitive-API consumers to migrate to oneDNN Graph, where the constant tensor cache already exists.

- **Pros:** Reuses existing code; no Primitive-API change.
- **Cons:** Many production CPU consumers are tied to the Primitive API for legitimate reasons (control over per-op dispatch, predictability, simpler integration model than partitions). Forcing Graph adoption is not a tractable migration ask, and it would not happen on the timescale on which CPU inference workloads need the feature.

**Decision:** Reject.

### D. Content-hashed cache key instead of pointer-equality (deferred)

Use a SHA / xxHash of the weight buffer contents as the cache key, so the cache is correct even if the user violates the data-invariance contract.

- **Pros:** Correctness under buggy user code.
- **Cons:** Hashing the whole buffer on every execute defeats the cache's purpose. Sample-hashing (first / last / middle lines) is a partial mitigation but adds complexity and has its own corner cases.

**Decision:** Defer. Match Graph API's pointer-equality default. Revisit if real workloads demonstrate contract violations slipping through.

### E. Automatic constant detection by the library (rejected)

Have the library infer that a tensor is constant by observing its data pointer and contents across executes.

- **Pros:** No user opt-in needed.
- **Cons:** False positives are catastrophic (caching something that does change silently produces wrong results); false negatives are common (the library has no way to be sure across calls). Worse than the explicit user contract.

**Decision:** Reject. Keep the contract explicit, as Graph API does.

## Appendix A. Graph API → Primitive API Concept Mapping

| Graph API (today) | Primitive API (this RFC) | Notes |
|---|---|---|
| `dnnl_graph_logical_tensor_t.property` | `dnnl_memory_desc_t` property (via `set_property`) | Same semantic; different host type |
| `dnnl_graph_tensor_property_constant` | `dnnl_memory_property_constant` | Identical meaning |
| `dnnl_graph_set_constant_tensor_cache_capacity(engine_kind, MB)` | `dnnl_set_const_weight_cache_capacity(engine_kind, MB)` | One-to-one |
| `ONEDNN_GRAPH_CONSTANT_TENSOR_CACHE_CAPACITY` env var | `ONEDNN_CONST_WEIGHT_CACHE_CAPACITY` env var | Same shape |
| `src/graph/interface/constant_tensor_cache_t` | `src/common/const_weight_cache_t` | Same data structure shape |
| Backend swaps weight MD to `format_kind::any` for marked tensors | Per-primitive `set_default_formats()` does the same | Same flow, different host code |
| Cache key = `(partition_hash, data_ptr)` | Cache key = `(pd_hash, src_md_hash, data_ptr)` | Slightly broader for primitives |
| Capacity-bound LRU eviction | Capacity-bound LRU eviction | Identical |
| Per-engine independent caches | Per-engine independent caches | Identical |

## Appendix B. File Touchpoints Checklist (Phase 1: matmul)

**Public headers**
- [ ] `include/oneapi/dnnl/dnnl_types.h`: add `dnnl_memory_property_t`.
- [ ] `include/oneapi/dnnl/dnnl.h`: declare `dnnl_memory_desc_set_property` / `_get_property`, `dnnl_set_const_weight_cache_capacity` / `_get_capacity`.
- [ ] `include/oneapi/dnnl/dnnl.hpp`: add `memory_property` enum, `memory::desc::set_property` / `_get_property`, `set_const_weight_cache_capacity`, `get_const_weight_cache_capacity`.

**Common**
- [ ] `src/common/memory_desc.{hpp,cpp}`: store the property in `extra` (or in a dedicated property field on the internal struct).
- [ ] `src/common/memory_desc_wrapper.hpp`: `is_constant()` helper.
- [ ] `src/common/const_weight_cache.{hpp,cpp}`: cache class, per-engine instance lookup, capacity control, env-var integration.
- [ ] `src/common/primitive_iface.cpp`: thread the cached buffer through `execute()` for primitives that opt in.

**Matmul integration (Phase 1)**
- [ ] `src/common/matmul_pd.hpp` / `src/common/matmul.cpp`: honour the constant property in `set_default_formats()`; record the cached layout MD and the cache slot key.
- [ ] `src/cpu/matmul/cpu_matmul_pd.hpp`: thin CPU PD typedef changes only if needed.
- [ ] One x64 matmul impl as the test bed (e.g. `brgemm_matmul_t`): consume the cached buffer via the helper.

**Tests / docs**
- [ ] `tests/benchdnn/matmul/`: add `--const-weights=true|false` knob; assert warm-cache executes show the reorder skipped.
- [ ] `tests/gtests/test_const_weight_cache.cpp`: correctness and invalidation cases (pointer change → cache miss; capacity = 0 → cache disabled; capacity overflow → eviction).
- [ ] `examples/primitives/matmul_const_weight.cpp`: tutorial example.
- [ ] `doc/programming_model/const_weight_cache.md`: user-guide page covering the property, the cache, the user contract, and the env var.

**No CMake changes.** No new build flag. No external dependencies introduced.

## Appendix C. References

**Project guidelines.**
- [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) &mdash; library functionality criteria, RFC process, commit-message format, unit-test policy.
- [`CODING_STANDARDS.md`](../../../CODING_STANDARDS.md) &mdash; clang-tidy, clang-format, naming conventions.

**Existing oneDNN mechanisms referenced by this design.**
- `include/oneapi/dnnl/dnnl_graph_types.h` &mdash; `dnnl_graph_tensor_property_t` enum (the precedent we lift into Primitive API).
- `include/oneapi/dnnl/dnnl_graph.h` &mdash; `dnnl_graph_set_constant_tensor_cache_capacity` (capacity API shape we mirror).
- `doc/graph/constant_tensor_cache.md` &mdash; user-facing description of Graph's cache; the Primitive-API equivalent doc will follow the same structure.
- `src/graph/interface/constant_tensor_cache.{hpp,cpp}` &mdash; reference implementation of the cache class.
- `src/graph/backend/dnnl/kernels/matmul.cpp` (lines 212&ndash;254) &mdash; reference flow for cache-keyed prepack + reuse.
- `src/graph/backend/dnnl/kernels/kernel_base.cpp` &mdash; cache key encoding including data-pointer.
- `src/common/memory_desc.hpp` &mdash; the internal struct where the property is stored.
- `src/common/primitive_cache.cpp` &mdash; the existing primitive cache (orthogonal mechanism); referenced for plumbing inspiration.
- `doc/usage_models/inference.md` &mdash; existing oneDNN guidance on stateless weight-reorder management; this RFC adds an alternative path without changing the existing one.
- `doc/advanced/persistent_cache.md` &mdash; the existing `cache_blob` mechanism (orthogonal: serialises primitive *creation*, not weight buffers).

