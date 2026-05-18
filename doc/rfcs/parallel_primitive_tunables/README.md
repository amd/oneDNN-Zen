# RFC: Performance Tunables on the Existing Primitive API (BRGEMM matmul, Phase 1)

## Status
**Draft**

## Authors
- AMD ZenDNN team

## Summary

This RFC proposes a small, opt-in extension to `dnnl::primitive_attr` that lets advanced users **override the auto-derived performance choices** that oneDNN's BRGEMM-based matmul makes today &mdash; cache block sizes (M / N / K), outer chunk sizes, thread partitioning (per-call max-threads, K-parallel reduction depth, chunk-axis thread split), and copy / pack policy &mdash; without any change to how primitives are constructed or executed.

oneDNN today exposes **zero** performance tunables on the primitive API. Every `dnnl_primitive_attr_set_*` is behavioural / numerical (`dropout`, `fpmath_mode`, `deterministic`, `accumulation_mode`, `scratchpad_mode`, `scales`, `zero_points`, `precomputed_reductions`, `rounding`, `post_ops`, `rnn_qparams`). The blocking heuristics that determine matmul's hot-path performance live in `src/cpu/x64/matmul/brgemm_matmul_utils.cpp` and `src/cpu/x64/matmul/amx_blocking_heuristics.cpp`, are entirely auto-derived from shape, dtype, and ISA, and have **no `getenv` hook anywhere in the matmul / brgemm tree** &mdash; so a user who knows their workload and wants to compare a candidate blocking against the auto choice has no path other than rebuilding the library with edited heuristics.

The proposal is conservative: every tunable defaults to **auto** (current behaviour); users only set the one(s) they care about; if the chosen impl cannot honour a user-supplied tunable, `pd_t::init()` returns `status::unimplemented` via the standard `VDISPATCH_*` flow and dispatch falls through to the next impl. No new primitive kind, no new dispatch path, no new public types beyond a few enums. Phase 1 is **matmul only**; convolution / inner_product / others can adopt the same hooks in subsequent RFCs without changing the API surface.

This RFC is fully independent of any other RFC on this branch and has no dependency on or reference to any external library.

---

## 1. Motivation

oneDNN's BRGEMM matmul has *good* auto-heuristics. They are tuned, ISA-aware, shape-aware, and right *most* of the time. But they are still heuristics, and there are three concrete situations where an advanced user has knowledge the heuristic doesn't:

1. **Workload-specific tuning.** A team deploying a fixed model on a fixed CPU SKU profiles the workload and finds, for example, that `K_block = 512` outperforms the auto-derived `K_block = 1024` by 8% on their decode-time M=1 BMM shapes. There is no oneDNN API surface to apply that finding short of recompiling the library.
2. **Cross-CPU benchmarking sweeps.** A performance engineer needs to benchmark candidate blockings against the auto choice for ten different shapes on a CPU model the heuristic was not specifically tuned for. Today this requires modifying `compute_blocking_heuristic_avx512` in `src/cpu/x64/matmul/brgemm_matmul_utils.cpp` and rebuilding &mdash; a workflow no production user can adopt.
3. **Thread provisioning that doesn't match `dnnl_get_max_threads()`.** A latency-sensitive caller wants matmul to use 4 threads even though the global max is 32, because the rest of its 32-thread pool is doing other work concurrently. There is no per-call thread limit on a primitive today.

The auto-heuristic defaults stay unchanged. This RFC adds an **opt-in override path** &mdash; if the user knows better, they can say so; if they don't, nothing changes.

The contribution must satisfy the three [Library Functionality Guidelines](../../../CONTRIBUTING.md#library-functionality-guidelines):

| Criterion | How |
|---|---|
| **Performance** | The tunables expose the exact knobs that are already proven (by `compute_blocking_heuristic_avx512`'s own search loop and by the AMX macro / micro blocking classes) to materially affect matmul performance. Validated workload-by-workload with benchdnn perf comparisons before each production PR. |
| **Generality** | Every Primitive-API consumer benefits with no framework-side change beyond the one-line `attr.set_*` call. PyTorch (`at::native::onednn`), TensorFlow, ONNX Runtime, vLLM CPU paths, llama.cpp, and any in-house consumer all map onto the same tunables surface. |
| **Complexity** | Choosing a good blocking is non-trivial; today every advanced user has to either accept the heuristic or fork the library. Centralising the override path in oneDNN saves duplication and keeps the validation logic in one place. |

## 2. Goals and Non-Goals

### Goals

- Add a small, opt-in set of performance tunables to `dnnl::primitive_attr`, all defaulting to `auto`.
- Phase 1 scope: **BRGEMM matmul on x64**. Same hooks generalise mechanically to other primitives (convolution, inner_product) but their adoption is out of scope for this RFC.
- The override mechanism must be **safe**: if the user sets a tunable that the chosen impl can't honour, dispatch returns `unimplemented` cleanly and the next impl in the list runs.
- Tunables are described in **workload-shaped terms** (block sizes on M / N / K, thread counts along axes, buffer policy on A / B), not in terms of internal kernel taxonomy that may shift across ISAs.
- One env-var counterpart per tunable for benchmarking workflows that don't want to touch source.

### Non-Goals

- **No new primitive kind, no new dispatch infra, no new memory descriptor flag.** Just additions to `primitive_attr`.
- **No public exposure of vendor- or ISA-specific kernel taxonomy.** Tunables stay descriptive (e.g. "block size on M") rather than prescriptive (e.g. "use AMX tile shape 16&times;16&times;64").
- **No changes to default behaviour.** Existing programs see exactly today's performance with no recompile and no API change.
- **No cross-primitive coordination.** A tunable set on one matmul instance does not influence any other primitive.
- **No promotion of brgemm ukernel out of experimental.** Orthogonal change, not part of this RFC.
- **No GPU coverage.** All proposed tunables are CPU-only; GPU follows naturally if the same model fits, in a future RFC.

## 3. What Auto-Derivation Does Today (BRGEMM matmul on x64)

The configuration that drives the hot path is held in `brgemm_matmul_conf_t`, declared at `src/cpu/x64/matmul/brgemm_matmul_utils.hpp` lines 117&ndash;255. Every field there is filled by the auto-heuristic, then frozen at PD-init time and consumed by the kernel at execute time.

The heuristic itself lives in three places:

- `init_brgemm_matmul_conf` (entry point, `src/cpu/x64/matmul/brgemm_matmul_utils.cpp`) &mdash; figures out dtype / ISA / layout / buffer-policy bits, then dispatches into the blocking strategy.
- `compute_blocking_heuristic_avx512` (`src/cpu/x64/matmul/brgemm_matmul_utils.cpp` lines 977&ndash;1077) &mdash; AVX-512 / AVX2 path. Searches a small grid of `(M_blk, N_blk, K_blk, batch_size, nthr_k)` candidates and picks the one that minimises a parallel-imbalance score.
- `matmul_amx_blocking_params_macro_t` and `matmul_amx_blocking_params_micro_t` (`src/cpu/x64/matmul/amx_blocking_heuristics.{hpp,cpp}`) &mdash; AMX path. Tries a macro (L2-oriented) blocking first; falls back to micro (smaller-tile) blocking if the macro choice fails or for shapes the macro path doesn't fit.

The kernel-side blocking (register tiling inside one BRGEMM call) is then derived from `M_blk` / `N_blk` / `K_blk` by `brgemm_blocking_*` in `src/cpu/x64/brgemm/brgemm_utils.cpp` lines 910&ndash;948 (`brgemm_blocking_tmm` for AMX, `brgemm_blocking_vmm` for AVX-512 / AVX2). That step produces `bd_block` / `ld_block` / `rd_block` &mdash; the *microkernel* register tiles, distinct from the matmul-driver tiles even though both are sometimes called "M / N / K block" in conversation.

There is **no `getenv` hook anywhere** in `src/cpu/x64/matmul/` or `src/cpu/x64/brgemm/`, so today none of these can be influenced from outside the library short of recompiling.

## 4. Tunables in Scope

Each tunable below is a field (or small group of fields) in `brgemm_matmul_conf_t` that the auto-heuristic chooses today. The ones we propose to expose were picked using three criteria: **(a)** typical-case impact on performance, **(b)** stability of the meaning across ISAs (we want the same enum to mean the same thing on AVX-512 and AMX), and **(c)** ease of validating user input.

### 4.1 Cache blocking on M / N / K (recommended for most users)

`brgemm_matmul_conf_t::M_blk`, `N_blk`, `K_blk` &mdash; the matmul-driver tile sizes, fed into each BRGEMM call.

- **What they do.** Drive cache footprint, parallel work granularity, and the kernel-count split between body and tail.
- **Auto-derive.** AVX-512 search loop in `compute_blocking_heuristic_avx512` picks `M_blk` from a candidate range (lines 957&ndash;963 set bounds, lines 1063&ndash;1083 do the search). `N_blk` starts from the layout-driven `wei_n_blk` (line 1258) and may shrink under low parallelism (lines 998&ndash;1000). `K_blk` is `512` or `1024` rounded up to the dtype's K granularity (lines 974&ndash;977). AMX paths are in `matmul_amx_blocking_params_macro_t::find_best_blocking` and the micro counterpart in `amx_blocking_heuristics.cpp`.
- **Why expose.** Highest-impact knob in the matmul. Any benchmarking sweep starts here.
- **Validation.** User-supplied `M_blk` / `N_blk` / `K_blk` must (a) be a positive multiple of the dtype's vector / VNNI / AMX granularity, (b) satisfy `block ≤ dim`, (c) be honourable by the chosen impl. If any condition fails, `pd_t::init()` returns `unimplemented` with a verbose-mode reason; dispatch advances to the next impl.

### 4.2 Outer chunking on M / N / K (recommended for L2-reuse tuning)

`M_chunk_size`, `N_chunk_size`, `K_chunk_size` &mdash; how many tiles of size `*_blk` are grouped before the outer parallel loop hands a chunk to a thread.

- **What they do.** Drive L2 reuse for copied A / B, plus the static work-chunk granularity for `parallel_nd*`. AMX macro path uses `N_chunk_size` up to 16 when copying A (lines 969&ndash;971, 1130&ndash;1132).
- **Auto-derive.** Initialised to 1 (line 1261); then AMX macro / micro paths can grow them based on a parallel-balance heuristic. AVX-512 keeps `M_chunk_size = 1` in its search (line 1087).
- **Why expose.** Second highest-impact knob. Crucial for shapes where L2 reuse over multiple tiles matters.
- **Validation.** Must be a positive integer; `chunk_size > 1` requires the corresponding buffer (`use_buffer_a` for `M_chunk_size > 1`, `use_buffer_b` for `N_chunk_size > 1`) to be active &mdash; the impl can either fold the request in or reject.

### 4.3 Thread partitioning

| Tunable | Underlying field | What it does | Auto-derive |
|---|---|---|---|
| `max_threads` (per-call) | clamps `nthr_` in `brg_matmul_exec_ctx_t` ctor at lines 1717&ndash;1718 | Per-primitive override of `dnnl_get_max_threads()` for **this** primitive's `execute()`; lets a caller cap matmul to 4 threads even if the global pool is larger | inherits from `dnnl_get_max_threads()` |
| `threads_along(axis::k)` | `nthr_k` in conf, derived in `compute_blocking_heuristic_avx512` lines 1011&ndash;1058 (AVX) and `matmul_amx_blocking_params_micro_t` lines 1124&ndash;1128 (AMX) | K-parallel reduction depth; splits K into `nthr_k` partial accumulators that are tree-reduced at the end | search-derived |
| `threads_along(axis::m)`, `threads_along(axis::n)`, `threads_along(axis::batch)` | `nthr_m`, `nthr_n`, `nthr_b`, currently set only when AMX macro path applies (`update_configuration` lines 35&ndash;38) | Static thread split across M chunks / N chunks / batch dim | usually all 1 outside AMX macro |

- **Why expose.** `nthr_k` is searched today &mdash; meaning the heuristic *already cares* about the right value but may pick wrong on shapes outside its tuned set. `max_threads` lets callers integrate matmul into multi-tenant inference loops without contending with the global thread limit.
- **Validation.** `max_threads ≥ 1`; `nthr_k * nthr_m * nthr_n * nthr_b ≤ max_threads`; honouring `nthr_k > 1` requires the impl to use the K-parallel reduction path (not all impls support this; the rest reject cleanly).

### 4.4 Copy / pack policy on A and B (recommended for known-pathological strides)

| Tunable | Underlying field | What it does | Auto-derive |
|---|---|---|---|
| `buffer_policy(input::a, ...)` | `use_buffer_a` (auto-set lines 1788&ndash;1804 of `brgemm_matmul_utils.cpp`) | When to copy A into a packed buffer before BRGEMM consumes it (forced for AMX bf32/tf32, for transposed A, for weight-zp without decompression; otherwise heuristic via `prefer_copy_a` lines 1263&ndash;1292) | data-driven + heuristic |
| `buffer_policy(input::b, ...)` | `use_buffer_b` (set in `bm_conf_utils.use_buffer_b()`, definitions at `brgemm_matmul_utils.hpp` lines 326&ndash;357) | When to copy B / compensate for s8s8 / decompress weights | data-driven + heuristic |

Three values in scope: `auto` (default; library decides), `force` (always copy), `never` (never copy &mdash; reject if the impl needs to copy).

- **Why expose.** The `prefer_copy_a` heuristic (`brgemm_matmul_utils.cpp` lines 1263&ndash;1292) is annotated with TODO comments today; users with pathological strides see noticeable wins from forcing or suppressing the copy. Forcing the copy is also useful for benchmarking the cost of the prepack itself.
- **Validation.** `force` requires the impl to be able to issue the copy; `never` requires the data to be already in a layout the kernel can consume directly &mdash; otherwise reject.

### 4.5 Microkernel hints (advanced; lower priority)

`brgemm_attr_t::hint_bd_block`, `hint_ld_block`, `hint_bd_block2`, `hint_ld_block2` (`src/cpu/x64/brgemm/brgemm_types.hpp` lines 165&ndash;251). These are register-blocking hints inside one BRGEMM kernel call &mdash; the *microkernel* tiles, not the matmul-driver tiles.

- **Why expose only as advanced.** The semantics of `bd_block` / `ld_block` differ across AVX-512 (vector-width-tied) and AMX (tile-shape-tied). Validating user input here is hard. Surface them under a separate `attr.advanced_*` namespace so casual users don't reach for them by accident.
- **Recommendation for Phase 1:** ship the four advanced setters but document them clearly as "subject to ISA-specific constraints; prefer §4.1 unless you've confirmed the matmul-driver tile is right and the bottleneck is microkernel register utilisation".

### 4.6 Explicitly out of scope (with reasons)

- **`brgemm_batch_size`** (K batch inside one BRGEMM call). ISA-specific meaning, hard to validate, low typical impact unless paired with `K_blk`. Revisit after Phase 1 lands and there's evidence the joint search of `(K_blk, brgemm_batch_size)` matters in production.
- **`brg_type`** (`brgemm_addr` / `brgemm_offs` / `brgemm_strd`). Matmul today uses only `brgemm_addr` (`brgemm_matmul_utils.cpp` lines 1426&ndash;1427); exposing the choice would be premature.
- **Prefetch hints** (`hint_prfA/B/C`, `set_nt`, `mem_advice`). CPU-model-specific; surface area too large for the value. Revisit if profiling shows they matter on a particular target.
- **`postops_inst_count` / `is_macro_heuristics` / other internal scoring fields.** These are intermediate values inside the heuristic, not knobs the user could meaningfully set.

## 5. Proposed API

### 5.1 Axis enums

```c
/* include/oneapi/dnnl/dnnl_types.h */

typedef enum {
    dnnl_tunable_axis_undef = 0,
    dnnl_tunable_axis_m,
    dnnl_tunable_axis_n,
    dnnl_tunable_axis_k,
    dnnl_tunable_axis_batch,
} dnnl_tunable_axis_t;

typedef enum {
    dnnl_tunable_input_undef = 0,
    dnnl_tunable_input_a,        /* matmul SRC */
    dnnl_tunable_input_b,        /* matmul WEIGHTS */
} dnnl_tunable_input_t;

typedef enum {
    dnnl_buffer_policy_auto      = 0,   /* default; library decides */
    dnnl_buffer_policy_force     = 1,   /* always copy / pack */
    dnnl_buffer_policy_never     = 2,   /* never copy; reject if impl needs to */
} dnnl_buffer_policy_t;
```

### 5.2 New `primitive_attr` setters

All values default to **0** which is treated as "auto" (i.e. current heuristic behaviour). The user only sets the tunables they want to override.

```c
/* include/oneapi/dnnl/dnnl.h */

/* §4.1 cache blocking on M / N / K */
dnnl_status_t DNNL_API dnnl_primitive_attr_set_block_size(
        dnnl_primitive_attr_t attr,
        dnnl_tunable_axis_t axis,
        dnnl_dim_t block_size);              /* 0 = auto */

/* §4.2 outer chunking on M / N / K */
dnnl_status_t DNNL_API dnnl_primitive_attr_set_chunk_size(
        dnnl_primitive_attr_t attr,
        dnnl_tunable_axis_t axis,
        dnnl_dim_t chunk_size);              /* 0 = auto */

/* §4.3 thread provisioning */
dnnl_status_t DNNL_API dnnl_primitive_attr_set_max_threads(
        dnnl_primitive_attr_t attr,
        int max_threads);                    /* 0 = inherit dnnl_get_max_threads() */

dnnl_status_t DNNL_API dnnl_primitive_attr_set_threads_along(
        dnnl_primitive_attr_t attr,
        dnnl_tunable_axis_t axis,
        int n_threads);                      /* 0 = auto */

/* §4.4 copy / pack policy on A and B */
dnnl_status_t DNNL_API dnnl_primitive_attr_set_buffer_policy(
        dnnl_primitive_attr_t attr,
        dnnl_tunable_input_t input,
        dnnl_buffer_policy_t policy);

/* §4.5 advanced microkernel hints (clearly marked advanced) */
dnnl_status_t DNNL_API dnnl_primitive_attr_set_advanced_microkernel_block(
        dnnl_primitive_attr_t attr,
        dnnl_tunable_axis_t axis,            /* m -> bd_block, n -> ld_block, k -> rd_block */
        dnnl_dim_t block_size);              /* 0 = auto */

/* and matching getters for each setter */
```

### 5.3 C++ wrapper

```cpp
/* include/oneapi/dnnl/dnnl.hpp */

namespace dnnl {

enum class tunable_axis  { m, n, k, batch };
enum class tunable_input { a, b };
enum class buffer_policy { auto_, force, never_ };

struct primitive_attr {
    /* ... existing API unchanged ... */

    /* §4.1 */
    void set_block_size(tunable_axis axis, memory::dim block);
    memory::dim get_block_size(tunable_axis axis) const;

    /* §4.2 */
    void set_chunk_size(tunable_axis axis, memory::dim chunk);
    memory::dim get_chunk_size(tunable_axis axis) const;

    /* §4.3 */
    void set_max_threads(int n);
    int  get_max_threads() const;
    void set_threads_along(tunable_axis axis, int n);
    int  get_threads_along(tunable_axis axis) const;

    /* §4.4 */
    void set_buffer_policy(tunable_input input, buffer_policy p);
    buffer_policy get_buffer_policy(tunable_input input) const;

    /* §4.5 advanced */
    void set_advanced_microkernel_block(tunable_axis axis, memory::dim block);
    memory::dim get_advanced_microkernel_block(tunable_axis axis) const;
};

} // namespace dnnl
```

### 5.4 Environment-variable counterparts

For benchmarking workflows that don't want to touch source. Each env var corresponds 1:1 to a setter; the runtime API takes precedence if both are set.

| Env var | Equivalent setter |
|---|---|
| `ONEDNN_MATMUL_M_BLOCK=<n>` | `set_block_size(tunable_axis::m, n)` |
| `ONEDNN_MATMUL_N_BLOCK=<n>` | `set_block_size(tunable_axis::n, n)` |
| `ONEDNN_MATMUL_K_BLOCK=<n>` | `set_block_size(tunable_axis::k, n)` |
| `ONEDNN_MATMUL_M_CHUNK=<n>` | `set_chunk_size(tunable_axis::m, n)` |
| `ONEDNN_MATMUL_N_CHUNK=<n>` | `set_chunk_size(tunable_axis::n, n)` |
| `ONEDNN_MATMUL_K_CHUNK=<n>` | `set_chunk_size(tunable_axis::k, n)` |
| `ONEDNN_MATMUL_MAX_THREADS=<n>` | `set_max_threads(n)` |
| `ONEDNN_MATMUL_K_THREADS=<n>` | `set_threads_along(tunable_axis::k, n)` |
| `ONEDNN_MATMUL_M_THREADS=<n>` | `set_threads_along(tunable_axis::m, n)` |
| `ONEDNN_MATMUL_N_THREADS=<n>` | `set_threads_along(tunable_axis::n, n)` |
| `ONEDNN_MATMUL_BUFFER_A=auto\|force\|never` | `set_buffer_policy(tunable_input::a, ...)` |
| `ONEDNN_MATMUL_BUFFER_B=auto\|force\|never` | `set_buffer_policy(tunable_input::b, ...)` |

Env vars apply globally across every matmul primitive in the process &mdash; useful for sweeps, dangerous in long-running services. The runtime API is the supported path for production.

### 5.5 Example: tuning a candidate blocking against the auto choice

```cpp
auto try_blocking = [&](int Mb, int Nb, int Kb) {
    dnnl::primitive_attr attr;
    attr.set_block_size(tunable_axis::m, Mb);
    attr.set_block_size(tunable_axis::n, Nb);
    attr.set_block_size(tunable_axis::k, Kb);

    auto pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
    auto p  = matmul(pd);
    return time_executes(p, /*iters=*/200);
};

double t_auto = try_blocking(0, 0, 0);   // current heuristic
double t_cand = try_blocking(64, 64, 256);
```

If `(64, 64, 256)` is incompatible with the chosen impl on the running ISA, the second call's `pd_t::init()` returns `unimplemented` for the BRGEMM matmul impl and dispatch falls through to the next impl in `cpu_matmul_list.cpp` &mdash; `try_blocking` still returns a number, but the user can detect the fall-through via `pd.impl_info_str()`.

## 6. Implementation

### 6.1 Storage in `primitive_attr_t`

A new struct in `src/common/primitive_attr.hpp`:

```cpp
struct tunables_t {
    int64_t block_size[axis_count]   = {0, 0, 0, 0};   // 0 = auto
    int64_t chunk_size[axis_count]   = {0, 0, 0, 0};
    int     max_threads              = 0;              // 0 = inherit
    int     threads_along[axis_count] = {0, 0, 0, 0};
    buffer_policy_t buffer_policy[input_count] = {auto_, auto_};
    int64_t advanced_microkernel_block[axis_count] = {0, 0, 0, 0};

    bool has_any_set() const;   // fast check: skip the override path entirely
                                // if the user touched nothing
};
```

Held as one new field on `primitive_attr_t`. Default-constructed; serialised into the primitive cache key only if `has_any_set()` so primitives without any tunable override hash identically to today.

### 6.2 Consumption in BRGEMM matmul `pd_t::init()`

`init_brgemm_matmul_conf` (`src/cpu/x64/matmul/brgemm_matmul_utils.cpp`) gains a new step: after the auto-heuristic has populated `brgemm_matmul_conf_t`, **override** any field whose corresponding tunable is non-default. Then validate:

```cpp
status_t apply_tunables(brgemm_matmul_conf_t &bgmmc,
                        const primitive_attr_t *attr) {
    const auto &t = attr->tunables_;
    if (!t.has_any_set()) return status::success;

    if (t.block_size[m] != 0) bgmmc.M_blk = t.block_size[m];
    if (t.block_size[n] != 0) bgmmc.N_blk = t.block_size[n];
    if (t.block_size[k] != 0) bgmmc.K_blk = t.block_size[k];
    if (t.chunk_size[m] != 0) bgmmc.M_chunk_size = t.chunk_size[m];
    /* ... */

    return validate_tunables(bgmmc, t);   // returns unimplemented on conflict
}
```

`validate_tunables` enforces the rules listed under each tunable in §4 (granularity, divisibility, layout compatibility, K-parallel-reduction support). Any failure returns `status::unimplemented` via `VDISPATCH_MATMUL` &mdash; the impl-list iterator advances and the next impl runs.

### 6.3 Per-call `max_threads`

Today, `brg_matmul_exec_ctx_t` ctor at `brgemm_matmul.cpp` lines 1717&ndash;1718 clamps `nthr_ = min(dnnl_get_current_num_threads(), bgmmc.nthr)`. We add one extra clamp:

```cpp
nthr_ = std::min({dnnl_get_current_num_threads(),
                  bgmmc.nthr,
                  attr->tunables_.max_threads ? attr->tunables_.max_threads
                                              : INT_MAX});
```

This is the only `execute()`-side change; everything else is a PD-time decision.

### 6.4 Verbose output

`onednn_verbose,...` lines for matmul gain a `tunables:...` section that lists any non-default tunables, so `--mode=p` perf comparisons can attribute differences to the override:

```
onednn_verbose,...,matmul,brgemm_matmul:avx512_core,...,tunables:M_blk=64;N_blk=64;K_blk=256,...
```

Empty if no tunable was set &mdash; existing log lines unchanged for users who don't opt in.

### 6.5 Source layout (additions only)

```
include/oneapi/dnnl/
  dnnl_types.h                     [+] dnnl_tunable_axis_t, dnnl_tunable_input_t,
                                       dnnl_buffer_policy_t enums
  dnnl.h                           [+] dnnl_primitive_attr_set_block_size,
                                       _set_chunk_size, _set_max_threads,
                                       _set_threads_along, _set_buffer_policy,
                                       _set_advanced_microkernel_block + getters
  dnnl.hpp                         [+] tunable_axis, tunable_input, buffer_policy,
                                       primitive_attr::set_*/get_* methods

src/common/
  primitive_attr.hpp               [M] add tunables_t field
  primitive_attr.cpp               [M] setters / getters + serialisation
  primitive_hashing.hpp            [M] hash tunables when has_any_set()

src/cpu/x64/matmul/
  brgemm_matmul_utils.cpp          [M] apply_tunables() + validate_tunables()
                                       called from init_brgemm_matmul_conf
  brgemm_matmul.cpp                [M] honour max_threads in exec_ctx ctor

doc/programming_model/
  primitive_tunables.md            [+] user-facing doc + examples + when-not-to-use

tests/
  benchdnn/matmul/                 [M] add knobs for each tunable; perf sweeps
  gtests/test_primitive_tunables.cpp [+] correctness + validation tests
```

No new build flag, no external dependencies, no new CMake option.

## 7. Use Cases

### 7.1 Workload-specific tuning of a deployed model

A team profiles a transformer-decoder model on a fixed CPU SKU and discovers that the default `K_blk = 1024` is slightly worse than `K_blk = 512` on the M=1 BMM in the attention block. Today they have no way to apply this finding short of forking oneDNN. With the proposed API:

```cpp
attr.set_block_size(tunable_axis::k, 512);    // one-line override
```

The change is local to that primitive; every other matmul in the rest of the model keeps using the auto choice.

### 7.2 Benchmarking sweeps

Benchmarking the auto choice against candidate blockings is the canonical workflow for finding tuning wins. With the env-var counterparts:

```bash
for K_blk in 256 384 512 768 1024 1536 2048; do
    ONEDNN_MATMUL_K_BLOCK=$K_blk ./benchdnn --matmul --mode=p ...
done
```

No source change, no rebuild, no fork.

### 7.3 Per-call thread budgeting in multi-tenant inference

A latency-sensitive service runs matmul concurrently with other oneDNN primitives in different threads of its own pool. It wants matmul to use 4 threads even though the global thread limit is 32. Today the only knob is `dnnl_set_max_threads()` which is global. With the proposed API:

```cpp
attr.set_max_threads(4);              // this matmul only
```

Other primitives in the same process keep their full thread budget.

### 7.4 K-parallel reduction on long-K shapes

Some shapes (e.g. very tall-skinny matmul with K &gt;&gt; M, N) benefit from splitting K across threads. The auto-heuristic tries this on AVX-512 but its search is limited; users with knowledge of their specific shape can force a `nthr_k` value:

```cpp
attr.set_threads_along(tunable_axis::k, 4);
```

### 7.5 Forcing or suppressing the A-copy

The `prefer_copy_a` heuristic (`brgemm_matmul_utils.cpp` lines 1263&ndash;1292, with multiple TODO comments) is conservative. Users with pathological strides can override:

```cpp
attr.set_buffer_policy(tunable_input::a, buffer_policy::force);    // always copy
// or
attr.set_buffer_policy(tunable_input::a, buffer_policy::never_);   // never copy;
                                                                   // reject if impl needs to
```

## 8. End-to-End Lifecycle

```
user code:
   primitive_attr attr;
   attr.set_block_size(tunable_axis::k, 512);       // single override
                                 │
                                 ▼
   matmul::primitive_desc(eng, src, wei, dst, attr)
                                 │
                                 ▼   cpu_engine -> CASE(matmul) -> impl_list
   ┌─────────────────────────────────────────────────────────────────┐
   │ brgemm_matmul_t::pd_t::init():                                  │
   │   1. init_brgemm_matmul_conf(...)        // auto-heuristic runs │
   │   2. apply_tunables(bgmmc, attr)         // user overrides land │
   │   3. validate_tunables(bgmmc, attr)                             │
   │      ▶ all conditions met → status::success                     │
   │      ▶ conflict → VDISPATCH_MATMUL → status::unimplemented      │
   └─────────────────────────────────────────────────────────────────┘
                                 │  unimpl
                                 ▼
   next impl in list (e.g. ref_matmul) runs unchanged
                                 │  success
                                 ▼
   primitive_desc ready  ▶  prim.execute(stream, args)
                                 │
                                 ▼
   brg_matmul_exec_ctx_t:
        nthr_ = min(current_num_threads, bgmmc.nthr,
                    attr.max_threads ? attr.max_threads : ∞);
                                 │
                                 ▼
   parallel work distribution honours nthr_, nthr_k, nthr_m, nthr_n, nthr_b;
   kernel runs with the (possibly overridden) M_blk / N_blk / K_blk / chunks.
```

## 9. Alternatives Considered

### A. Environment variables only (rejected)

Skip the `primitive_attr` setters; expose only `ONEDNN_MATMUL_*` env vars.

- **Pros:** Zero changes to the public C / C++ API; benchmarking sweeps work with no code change.
- **Cons:** No per-call control. Two matmuls in the same process can't have different tunables. Long-running services can't change tunables based on the request shape. Env vars are useful for sweeps but insufficient as the only path.

**Decision:** Reject as the *only* mechanism. Include them as a *complement* to the programmatic API (§5.4).

### B. Use the BRGEMM ukernel API (rejected)

Tell users who want fine-grained control to drop down to `dnnl::ukernel::brgemm` (currently experimental).

- **Pros:** Already exists in the tree, more direct control over tile sizes.
- **Cons:** The ukernel API requires the user to write threading, work distribution, post-op application, and packing themselves. The whole point of this RFC is to keep users on the existing primitive API where the library handles all of that, and to surface only the small set of knobs that actually affect performance. Asking advanced users to rewrite their entire matmul invocation in ukernel terms just to change a block size is the wrong cost-benefit.

**Decision:** Reject. Ukernel remains its own API for users who genuinely want kernel-level control; this RFC is for users who want auto-management plus a small set of overrides.

### C. Compile-time configuration (rejected)

Add CMake flags that the user can set when building oneDNN.

- **Pros:** Zero runtime cost.
- **Cons:** Doesn't survive deployment; requires every workload to ship its own libdnnl build. Defeats the "production tuning without rebuild" use case (§7.1). Useless for benchmarking sweeps.

**Decision:** Reject.

### D. Add a separate "low-overhead" or "direct kernel" API (out of scope)

A different RFC could propose a function-style entry point that bypasses primitive construction. That is a different problem (API overhead, not blocking choice) and is **not** addressed here. This RFC is strictly about exposing the existing primitive's internal tunables.

## Appendix A. Tunable &rarr; `brgemm_matmul_conf_t` Field Mapping

| User-facing tunable | Internal field (`brgemm_matmul_conf_t`) | Source location |
|---|---|---|
| `block_size(m)` | `M_blk` | `brgemm_matmul_utils.hpp:117&ndash;255` |
| `block_size(n)` | `N_blk` | same |
| `block_size(k)` | `K_blk` | same |
| `chunk_size(m)` | `M_chunk_size` | same |
| `chunk_size(n)` | `N_chunk_size` | same |
| `chunk_size(k)` | `K_chunk_size` | same |
| `max_threads` | clamps `nthr_` in exec ctx | `brgemm_matmul.cpp:1717&ndash;1718` |
| `threads_along(k)` | `nthr_k` | `brgemm_matmul_utils.cpp:977&ndash;1077` (AVX), `amx_blocking_heuristics.cpp` (AMX) |
| `threads_along(m)` | `nthr_m` | `amx_blocking_heuristics::update_configuration:35&ndash;38` |
| `threads_along(n)` | `nthr_n` | same |
| `threads_along(batch)` | `nthr_b` | same |
| `buffer_policy(a)` | `use_buffer_a` | `brgemm_matmul_utils.cpp:1788&ndash;1804` |
| `buffer_policy(b)` | `use_buffer_b` | `brgemm_matmul_utils.hpp:326&ndash;357` |
| `advanced_microkernel_block(m)` | `brgemm_attr_t::hint_bd_block` | `brgemm_types.hpp:165&ndash;251` |
| `advanced_microkernel_block(n)` | `brgemm_attr_t::hint_ld_block` | same |
| `advanced_microkernel_block(k)` | `brgemm_attr_t::hint_rd_block` | same |

## Appendix B. File Touchpoints Checklist (Phase 1, matmul x64)

**Public headers**
- [ ] `include/oneapi/dnnl/dnnl_types.h`: `dnnl_tunable_axis_t`, `dnnl_tunable_input_t`, `dnnl_buffer_policy_t` enums.
- [ ] `include/oneapi/dnnl/dnnl.h`: 6 setter / 6 getter declarations under §5.2.
- [ ] `include/oneapi/dnnl/dnnl.hpp`: `tunable_axis`, `tunable_input`, `buffer_policy` C++ enums; `primitive_attr` member methods.

**Common**
- [ ] `src/common/primitive_attr.hpp`: `tunables_t` struct + member.
- [ ] `src/common/primitive_attr.cpp`: setters / getters / `has_any_set()`.
- [ ] `src/common/primitive_hashing.hpp`: include `tunables_t` in the cache key when `has_any_set()`.

**CPU x64 matmul (Phase 1)**
- [ ] `src/cpu/x64/matmul/brgemm_matmul_utils.cpp`: `apply_tunables` + `validate_tunables`, called from `init_brgemm_matmul_conf` after auto-heuristic.
- [ ] `src/cpu/x64/matmul/brgemm_matmul.cpp`: honour `max_threads` in `brg_matmul_exec_ctx_t` ctor.

**Verbose output**
- [ ] `src/common/verbose.cpp` (and matmul-side hooks): emit `tunables:...` section when any tunable is set.

**Tests**
- [ ] `tests/benchdnn/matmul/`: knobs for each tunable; benchdnn perf sweeps consume the env-var counterparts.
- [ ] `tests/gtests/test_primitive_tunables.cpp`: correctness, validation rejection, fall-through-to-next-impl checks.
- [ ] `doc/programming_model/primitive_tunables.md`: user-facing doc with the §7 use-case examples.

**Env-var plumbing**
- [ ] `src/common/utils/env.cpp` (or wherever oneDNN reads env): bind the 12 `ONEDNN_MATMUL_*` env vars to `tunables_t` defaults.

**No CMake changes. No public ABI break. No external dependencies.**

## Appendix C. References

- [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) &mdash; library functionality criteria, RFC process, commit-message format.
- [`CODING_STANDARDS.md`](../../../CODING_STANDARDS.md) &mdash; clang-tidy, clang-format, naming conventions.
- `src/cpu/x64/matmul/brgemm_matmul_utils.hpp` (`brgemm_matmul_conf_t` declaration, lines 117&ndash;255).
- `src/cpu/x64/matmul/brgemm_matmul_utils.cpp` (`init_brgemm_matmul_conf`, `compute_blocking_heuristic_avx512`, `prefer_copy_a` heuristic with TODO comments at lines 1263&ndash;1292).
- `src/cpu/x64/matmul/amx_blocking_heuristics.{hpp,cpp}` (`matmul_amx_blocking_params_t` macro / micro classes; `update_configuration`).
- `src/cpu/x64/matmul/brgemm_matmul.cpp` (execute-time threading clamp at lines 1717&ndash;1718; work distribution via `parallel_nd*` and `balance211`).
- `src/cpu/x64/brgemm/brgemm_types.hpp` (`brgemm_attr_t` and `brgemm_desc_t`, lines 165&ndash;385).
- `src/cpu/x64/brgemm/brgemm_utils.cpp` (`brgemm_blocking_*`, lines 910&ndash;948).
- `src/common/primitive_attr.{hpp,cpp}` (existing attribute storage and serialisation patterns).
- `src/common/dnnl_thread.hpp` (`parallel_nd`, `parallel_nd_ext`, `balance211`).

