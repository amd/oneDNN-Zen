# RFC: ZenDNN Integration in oneDNN — `zen64` Module + Native Upstream Contributions

## Status
**Draft** &nbsp;|&nbsp; AMD-Zenai/oneDNN-ZenDNN &nbsp;|&nbsp; Branch: `rfc/embedding-bag` &nbsp;|&nbsp; Companion RFC: [`doc/rfcs/embedding_bag/`](../embedding_bag/README.md) &nbsp;|&nbsp; Target: mid-June 2026

## Authors
- AMD-Zenai team

## Summary

This RFC describes the AMD plan to bring ZenDNN-class CPU performance into oneDNN through **two complementary verticals**, both delivered into upstream `uxlfoundation/oneDNN` over Q2&ndash;Q4&nbsp;2026. The two verticals share strategic intent (production-grade AMD-CPU performance for every framework that already consumes oneDNN) but use different integration mechanisms by design:

1. **Vertical 1 &mdash; `zen64` module.** A new ISA-flavoured CPU sub-target under `src/cpu/x64/zen64/` that registers ZenDNN as one entry in oneDNN's existing per-primitive `impl_list`, gated at build time by `DNNL_ENABLE_ZENDNN=ON` and at runtime by `DNNL_ENABLE_ZENDNN` plus per-primitive env vars. Ahead-of-time selected on AMD CPUs with the right uArch and ISA, falls through to oneDNN's native impls otherwise. Initial scope: MatMul (+ fused) and Reorder for fp32 / bf16. Follow-up scope (Q3): BMM, GroupGEMM (Grouped Memory Format for variable-size batching), SDPA, Quantized MatMul (WoQ + Static + Dynamic), and a ZenDNN LOWOHA hook into oneDNN's BRGEMM microkernel API.

2. **Vertical 2 &mdash; native upstream contributions.** New oneDNN primitives and primitive optimisations implemented natively in oneDNN's tree (no `zen64` dependency, no `DNNL_ENABLE_ZENDNN` gating). Initial scope: a new `embedding_bag` primitive (covered in detail by the companion RFC at [`doc/rfcs/embedding_bag/`](../embedding_bag/README.md)), Dynamic Quantization support (its own RFC at end of June), MatMul / BMM tiling and threading heuristics specifically tuned for AMD CPUs, an internal AutoTuner pass over runtime kernel/backend selection, and Low-Overhead API hooks. Each item has its own RFC and PR series; this document is the umbrella that places them in context.

The two verticals are **not** alternatives. Vertical&nbsp;1 ships ZenDNN's existing kernels behind a gated backend so AMD users get production performance immediately on already-existing oneDNN primitives. Vertical&nbsp;2 invests in new oneDNN-native code so the operations that ZenDNN provides today but oneDNN does not yet have (embedding bag, dynamic quant, etc.) become permanently part of oneDNN's standard surface, available to every framework on every CPU. §4 explains the rationale for the split, §12 describes how the embedding_bag RFC fits in.

A working PoC of Vertical&nbsp;1 (MatMul fp32/bf16 with Reorder) is already running locally; verbose logs are reproduced in §6. vLLM-level framework validation is in progress. After review of this umbrella RFC and its companion(s), the production PR series begins end of June.

---

## 1. Motivation

oneDNN is the de-facto CPU primitive library that frameworks (PyTorch, TensorFlow, ONNX Runtime, llama.cpp via custom paths, vLLM) depend on. On Intel CPUs, oneDNN has years of dedicated optimisation. On AMD CPUs (Zen3 / Zen4 / Zen5 / Zen6), the same primitives leave performance on the table when kernel selection, memory layout, or threading choices don't match Zen-microarchitecture preferences. AMD's ZenDNN library has been the production path for closing that gap, today shipping inside zentorch and consumed by frameworks via a separate AMD-specific code path.

**Problem.** Every framework that wants competitive AMD-CPU performance has to consume ZenDNN directly, bypassing oneDNN. That fragments the integration story:
- PyTorch users go through `zentorch`, TensorFlow users go through a separate path, vLLM users have to wire `zentorch` themselves, etc.
- Improvements in the underlying kernels reach each framework on a different cadence.
- Dynamic Quant, Embedding-Bag, Group MatMul, and other AMD-tuned operations exist nowhere in oneDNN's standard surface, so frameworks that consume only oneDNN cannot see them at all.

**Proposal.** Bring the value into oneDNN itself, through two integration mechanisms chosen per operator:
- For **operators oneDNN already has** (MatMul, BMM, SDPA, Reorder, GroupGEMM): land ZenDNN as a runtime-selectable backend (`zen64` module, Vertical&nbsp;1). This is the lowest-friction path to put production AMD-tuned kernels under existing oneDNN primitive APIs &mdash; no framework-side change required.
- For **operators oneDNN does not have** (Embedding+Bag, Dynamic Quant, etc.): add the primitive natively to oneDNN, with AVX-512 intrinsic kernels written in oneDNN's own code base (Vertical&nbsp;2). These become permanent oneDNN primitives benefiting every CPU, with ZenDNN's existing implementation used only as the porting reference (no link-time, runtime, or build-time dependency).

This dual approach mirrors how oneDNN itself already structures its impl space: optimised paths register ahead of reference paths in `cpu_<prim>_list.cpp`, with `status::unimplemented` fall-through driving graceful degradation. We are adding one more optimised path (zen64) for primitives already covered, and adding entirely new primitives for the gaps.

The contribution must satisfy the three [Library Functionality Guidelines](../../../CONTRIBUTING.md#library-functionality-guidelines) that gate every oneDNN change:

| Criterion | How this RFC addresses it |
|---|---|
| **Performance** &mdash; material workload-level impact | Vertical&nbsp;1 brings ZenDNN's existing AVX-512 / AVX-512-FP16 / AOCL-DLP / FBGEMM dispatch onto AMD CPUs end-to-end through oneDNN; Vertical&nbsp;2 adds new primitives that materially change framework-level numbers (DLRM-class embedding throughput, dynamic-quantised LLM decode latency). Performance is validated at model level via vLLM and benchdnn before each production PR. |
| **Generality** &mdash; usable across multiple frameworks | Both verticals work through oneDNN's standard primitive API. PyTorch (via `at::native::onednn`), TensorFlow, ONNX Runtime, vLLM &mdash; every consumer that already calls oneDNN benefits with no integration work. |
| **Complexity** &mdash; not trivial to implement in apps | High-quality CPU MatMul / BMM / SDPA / Embedding-Bag / Dynamic-Quant kernels are several engineer-years of work. Centralising them in oneDNN avoids each framework re-implementing them. |

## 2. Goals and Non-Goals

### Goals

- **No public API change.** Both verticals are invisible to users of `dnnl::matmul`, `dnnl::reduction`, `dnnl::embedding_bag`, etc.
- **No mandatory build-time dependency.** Vertical&nbsp;1 is gated by `DNNL_ENABLE_ZENDNN` (default `OFF`); Vertical&nbsp;2 has no ZenDNN dependency at all. A standard `cmake` build with no extra flags produces a oneDNN that is byte-for-byte unchanged from today.
- **Graceful fallback.** Any time the `zen64` impl rejects a configuration (wrong CPU vendor / uArch / ISA, unsupported shape / dtype / layout / post-op, runtime kill-switch via env var), the existing `impl_list` walks to the next candidate and oneDNN's stock impl runs.
- **One PR series per logical concern.** Each Vertical&nbsp;1 follow-up (BMM, GroupGEMM, SDPA, Quantization, BRGEMM-microkernel hook) and each Vertical&nbsp;2 contribution (Embedding+Bag, Dynamic Quant, AMD-CPU MatMul/BMM tuning) is its own RFC + PR series, reviewed independently.
- **Validation evidence in every PR.** benchdnn correctness, benchdnn perf, and a representative model-level run (vLLM for LLM workloads) before each production PR merges.

### Non-Goals (initial release)

- **GPU paths.** Vertical&nbsp;1's `zen64` is a CPU-only sub-target; Vertical&nbsp;2's primitives leave GPU `case` arms returning `empty_list` initially (precedent: `sdpa`, `gated_mlp`).
- **Replacing oneDNN's existing CPU kernels.** Vertical&nbsp;1 only *registers ahead* of the existing kernels in the impl list on AMD systems; it does not delete or modify any existing kernel. On Intel CPUs the `zen64` impl rejects in `pd_t::init()` and the existing kernel runs unchanged.
- **Exposing ZenDNN tuning knobs in the public API.** ZenDNN's `embag_kernel_t`, `eb_thread_algo_t`, `ZENDNNL_*` env vars stay internal to the `zen64` module and are not visible to oneDNN users.
- **Mandatory ZenDNN versioning coupling.** Vertical&nbsp;1 pins a ZenDNN version range at build time; runtime ABI compatibility is the build system's problem, not the user's.
- **A Vertical&nbsp;1 path for ops oneDNN does not yet have.** Embedding-bag, dynamic quant, etc. land via Vertical&nbsp;2 (native primitives). This avoids leaking ZenDNN-only APIs into oneDNN's public surface.

## 3. ZenDNN Value Add (Background)

ZenDNN ships several capabilities that we want to bring through into oneDNN. Each is what differentiates the AMD-tuned path from a stock JIT-only impl on Zen-class CPUs.

### 3.1 Multi-backend Auto Tuner (TBP + Decision Tree)

ZenDNN's MatMul / BMM / GEMV path runs an internal Auto Tuner that picks between four candidate backends per problem size: native ZenDNN kernels, AOCL-DLP, libxsmm, and FBGEMM. Selection is driven by:
- **Time-Based Profiling (TBP)** &mdash; the first call for a given shape probes a small set of candidates and caches the winner.
- **Decision Tree** &mdash; static heuristics (problem dimensions, dtype, layout) that resolve without profiling for shapes the tree already covers.

Vertical&nbsp;1 makes this Auto Tuner the dispatch core inside the `zen64::matmul_t::execute()` body; oneDNN sees a single primitive that picks the best backend per call.

### 3.2 Parallel primitive

Fine-grained control over CPU parallelism for batched / small-shape operators that defeat coarse `parallel_nd` strategies:
- **Cache blocking** with **dynamic tile selection** per problem size.
- **Threading strategies** &mdash; Inner-dimension, Outer-dimension, Batch-threaded, Table-threaded, Hybrid (CCD-aware) for embedding-bag-class operators.
- **Group support** &mdash; Group MatMul and Group Embedding Bag using a Grouped Memory Format that handles variable-size batches without padding.

Vertical&nbsp;1 brings these to MatMul/BMM via `zen64`; Vertical&nbsp;2 ports the embedding-bag flavor of the parallel primitive into oneDNN's native intrinsic kernels (see [`embedding_bag` RFC §6.6](../embedding_bag/README.md)).

### 3.3 Low-Overhead API (LOWOHA)

A kernel-direct surface that minimises per-call book-keeping:
- No descriptor re-validation per call &mdash; everything that can be resolved at "init" time is.
- Direct kernel-pointer dispatch from caller stack into the pre-compiled kernel.
- Caller passes raw pointers + sizes; ZenDNN's normal high-level API is bypassed for tight inner loops.

Critical for small-shape GEMM-heavy workloads &mdash; SDPA, BMM, low-batch decode &mdash; where API overhead is a measurable fraction of kernel time. Vertical&nbsp;1 surfaces LOWOHA at the `zen64::execute()` boundary; the broader oneDNN-side hook into BRGEMM microkernel APIs is a Q3 deliverable.

### 3.4 Custom ops and kernels

Operations that do not map onto a single dense primitive:
- **Group MatMul** and **Group Embedding Bag** with a Grouped Memory Format that batches variable-size problems without padding.
- **Custom GEMV path** specialised for transformer decode (M=1 or M=batch).
- **Fused MoE** for mixture-of-experts inference.

Vertical&nbsp;1 covers these to the extent they fit existing oneDNN primitive shapes; Vertical&nbsp;2 picks up the rest as new primitives or as oneDNN-native fused-op support.

### 3.5 Operator coverage today

ZenDNN already ships, today, in production: MatMul (+ fused), BMM, GroupMatMul, Embedding Bag, Group Embedding Bag, Reorder (Quant / Dequant / Dynamic Quant), Normalization, plus other key DL primitives. All are candidates for upstream contribution, split across the two verticals per the rule in §4.

## 4. Two-Vertical Strategy

The split between Vertical&nbsp;1 (`zen64` backend) and Vertical&nbsp;2 (native upstream) is driven by a single question: **does oneDNN already have a primitive with the right semantics?**

| Operator | oneDNN today | Vertical | Why |
|---|---|---|---|
| MatMul (+ fused) | Yes | 1 | Existing primitive, ZenDNN registers as one more impl in `cpu_matmul_list.cpp` |
| Reorder (incl. format conversion) | Yes | 1 | Existing primitive; ZenDNN handles ahead-of-time prepack and post-pack |
| BMM | Yes (via `matmul` with batch dim) | 1 | Existing primitive |
| SDPA | Yes (`dnnl_sdpa`) | 1 | Existing primitive; ZenDNN provides AMD-tuned dispatch |
| GroupGEMM | Partial (graph) | 1 + RFC | Existing graph-level support; native primitive may follow |
| Quantized MatMul (WoQ / static / dynamic) | Partial via `attr.scales` / `attr.zero_points` | 1 + 2 | Backend in V1; new oneDNN API for *Dynamic* Quant in V2 (RFC end June) |
| BRGEMM microkernel | Yes | 1 (LOWOHA hook) | Existing microkernel API; ZenDNN plugs in a low-overhead path |
| **Embedding (+ Bag)** | **No** | **2** | New primitive entirely &mdash; native AVX-512 intrinsic kernel; companion RFC |
| **Dynamic Quant (compute scale + quantize)** | **No standalone primitive** | **2** | New primitive + companion API extension; companion RFC end June |
| AMD-CPU MatMul/BMM tiling/threading heuristics | n/a | 2 | Native tweaks inside oneDNN's existing kernel paths; no ZenDNN dep |
| AutoTuner (runtime kernel/backend selection) | No global mechanism today | 2 | Native infra contribution to oneDNN itself |
| Low-Overhead APIs (oneDNN-wide) | Partial | 2 | Native infra contribution |

The rule is simple: **wrap if the API exists, upstream natively if it doesn't.** Vertical&nbsp;1 is dependency-free for users (default-OFF flag); Vertical&nbsp;2 leaves no ZenDNN dependency in oneDNN's tree.

## 5. Vertical 1: `zen64` Module (ZenDNN as a oneDNN CPU Backend)

### 5.1 Architecture overview

```
            user code (PyTorch / TF / ONNX RT / vLLM / app)
                            │
                            ▼
                       Primitive APIs            (include/oneapi/dnnl/dnnl.{h,hpp})
                            │
                            ▼
                  CPU  GPU  XPU  Graph
                            │
                            ▼
              x64  aarch64  riscv64  ppc64  s390x
                  │
                  ▼
   ┌─────────────────────────────────────────────────────────┐
   │ x64 impl space:                                         │
   │   BRGEMM · GEMM · jit_uni_* · ref · ...                 │
   │   ┌──────────────────────────────┐                      │
   │   │ zen64 (NEW, opt-in)          │   ← this RFC, V1    │
   │   │   • registers in cpu_*_list  │                      │
   │   │   • PD::init() AMD vendor /  │                      │
   │   │     uArch / ISA / shape gate │                      │
   │   │   • execute() calls ZenDNN   │                      │
   │   └──────────┬───────────────────┘                      │
   └──────────────┼──────────────────────────────────────────┘
                  │
                  ▼  (only when DNNL_ENABLE_ZENDNN=ON and runtime checks pass)
              ZenDNN library
              (linked, default OFF)
```

The new `src/cpu/x64/zen64/` directory is a regular `src/cpu/x64/` sub-tree, gated by `DNNL_ENABLE_ZENDNN=ON`. It contributes one impl class per primitive (e.g. `zen64::matmul_t`, `zen64::reorder_t`) registered in the existing `src/cpu/<prim>/cpu_<prim>_list.cpp` files. From oneDNN's dispatcher's point of view it is one more `CPU_INSTANCE_X64(...)` entry; from a user's point of view, opting in or out changes nothing about the public API.

### 5.2 Build-time gating: `DNNL_ENABLE_ZENDNN` CMake option

A new top-level CMake option:

```cmake
option(DNNL_ENABLE_ZENDNN
       "Build with ZenDNN as an opt-in AMD-CPU backend (Linux x86_64 only)."
       OFF)
```

When `OFF` (default): zero source files from `src/cpu/x64/zen64/` are compiled, no ZenDNN headers are referenced, and no ZenDNN library is linked. Behaviour is byte-identical to today's oneDNN.

When `ON`: the build:
1. Locates ZenDNN via `find_package(ZenDNN REQUIRED)` (or a vendored thin-CMake module if the user has a non-system ZenDNN install).
2. Compiles `src/cpu/x64/zen64/*.cpp`.
3. Links `libzendnn` to the produced `libdnnl`.
4. Adds the `zen64::*_t` impl entries inside `cpu_<prim>_list.cpp` via existing `REG_<PRIM>_P(...)` macros, conditionally compiled with `#if DNNL_X64 && DNNL_ENABLE_ZENDNN` guards.

A second option, `DNNL_ENABLE_ZENDNN_STATIC=ON|OFF`, controls whether the link is static or dynamic; default `OFF` (dynamic) for distribution-friendliness. Linkage choice is internal to the build; no impact on the consumer.

The option is documented in `doc/build/build_options.md` alongside the existing build options.

### 5.3 Runtime gating

Even with `DNNL_ENABLE_ZENDNN=ON` at build time, the impl dispatch is gated at runtime by environment variables, so users can disable the backend on a per-process or per-primitive basis:

| Env var | Default | Semantics |
|---|---|---|
| `DNNL_ENABLE_ZENDNN` | `1` (when build flag is ON) | Master switch &mdash; `0` makes every `zen64::*_t::pd_t::init()` return `unimplemented`. |
| `DNNL_ZENDNN_MATMUL` | `1` | Per-primitive override for MatMul. |
| `DNNL_ZENDNN_REORDER` | `1` | Per-primitive override for Reorder. |
| `DNNL_ZENDNN_BMM` | `1` (Q3) | Per-primitive override for BMM. |
| `DNNL_ZENDNN_SDPA` | `1` (Q3) | Per-primitive override for SDPA. |

The env vars are read once per process during oneDNN initialisation, cached, and queried inside each `pd_t::init()` before any other check. Setting any to `0` is equivalent to a non-AMD CPU from the dispatcher's point of view &mdash; the next entry in the impl list runs.

### 5.4 Per-primitive backend registration

Existing `src/cpu/<prim>/cpu_<prim>_list.cpp` files gain one new entry, conditionally compiled. For MatMul:

```cpp
// src/cpu/matmul/cpu_matmul_list.cpp  (sketch)
#include "cpu/matmul/ref_matmul.hpp"
#include "cpu/x64/jit_brgemm_matmul.hpp"
#if DNNL_X64 && DNNL_ENABLE_ZENDNN
#include "cpu/x64/zen64/zen64_matmul.hpp"
#endif

const impl_list_item_t *get_matmul_impl_list(const matmul_desc_t *desc) {
    UNUSED(desc);
    static constexpr impl_list_item_t impl_list[] = REG_MATMUL_P({
        DNNL_X64_ONLY_IF(DNNL_ENABLE_ZENDNN, CPU_INSTANCE(zen64::matmul_t))  // ahead of stock impls
        CPU_INSTANCE_X64(brgemm_matmul_t)
        // ...existing entries...
        CPU_INSTANCE(ref_matmul_t)
        nullptr,
    });
    return impl_list;
}
```

`DNNL_X64_ONLY_IF(DNNL_ENABLE_ZENDNN, ...)` is a small new macro in `src/cpu/cpu_engine.hpp` mirroring the existing `DNNL_X64_ONLY(...)` &mdash; it expands to the entry only when both `DNNL_X64` and `DNNL_ENABLE_ZENDNN` are defined, otherwise to nothing.

The `zen64::matmul_t` entry is registered **before** the existing `brgemm_matmul_t`. On AMD CPUs that pass the runtime gates, the dispatcher selects ZenDNN; on Intel CPUs, ZenDNN's PD `init()` returns `unimplemented` (vendor mismatch) and the standard dispatcher continues to the existing entries.

### 5.5 PD `init()` validation flow

Each `zen64` impl's `pd_t::init(engine_t *engine)` runs the same staged check:

```cpp
status_t zen64::matmul_t::pd_t::init(engine_t *engine) {
    // Stage 1 - runtime kill-switch
    VDISPATCH_MATMUL(zen64::is_runtime_enabled("matmul"),
            "ZenDNN backend disabled at runtime (DNNL_ENABLE_ZENDNN / DNNL_ZENDNN_MATMUL)");

    // Stage 2 - CPU vendor / uArch / ISA
    VDISPATCH_MATMUL(zen64::cpu_supported(),  // AuthenticAMD AND uArch in {Zen4, Zen5, Zen6}
            "CPU is not a supported AMD micro-architecture");
    VDISPATCH_MATMUL(zen64::isa_supported(desc()),  // BF16 / FP16 / VNNI as needed
            "Required ISA features not present");

    // Stage 3 - shapes / dtypes / layouts / attributes / post-ops
    VDISPATCH_MATMUL(zen64::shapes_supported(desc()), VERBOSE_BAD_PARAM);
    VDISPATCH_MATMUL(zen64::dtypes_supported(desc()), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_MATMUL(zen64::layouts_supported(desc()), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(zen64::attrs_supported(attr()), VERBOSE_UNSUPPORTED_ATTR);

    // Stage 4 - oneDNN-side bookkeeping (formats, scratchpad)
    CHECK(set_default_formats());
    CHECK(zen64::book_scratchpad(this));

    return status::success;
}
```

Every `VDISPATCH_*` is a thin wrapper over oneDNN's existing dispatch macros that returns `status::unimplemented` and emits a verbose-mode reason string. Dispatch failure is *normal* &mdash; the impl-list iterator simply advances to the next candidate.

### 5.6 `execute()` flow

`zen64::matmul_t::execute(ctx)` is small. It pulls memory descriptors out of the context, prepacks weights if needed (via either oneDNN's reorder or ZenDNN's internal cache &mdash; see §5.7), and calls into the appropriate ZenDNN entry point:

```cpp
status_t zen64::matmul_t::execute(const exec_ctx_t &ctx) const {
    const auto *src   = CTX_IN_MEM (const char *, DNNL_ARG_SRC);
    const auto *wei   = CTX_IN_MEM (const char *, DNNL_ARG_WEIGHTS);
    const auto *bias  = CTX_IN_MEM (const char *, DNNL_ARG_BIAS);   // optional
    auto       *dst   = CTX_OUT_MEM(char *,        DNNL_ARG_DST);
    auto        sp    = ctx.get_scratchpad_grantor();

    zen64::matmul_params p = pd()->kernel_params();   // resolved at PD time
    p.src = src; p.wei = wei; p.bias = bias; p.dst = dst;

    // ZenDNN's LOWOHA entry; AutoTuner picks ZenDNN-native vs AOCL-DLP vs FBGEMM internally.
    return zen64::translate_status(
            zendnnl::lowoha::matmul_direct(p));
}
```

Inside `zendnnl::lowoha::matmul_direct(...)`, ZenDNN's Auto Tuner selects the actual kernel (ZenDNN-native, AOCL-DLP, libxsmm, FBGEMM), the parallel primitive picks the threading strategy and tile size, and the LOWOHA path avoids re-validation on hot calls. From oneDNN's dispatcher and from the user's view, this is one regular primitive call.

### 5.7 Memory descriptor and reorder strategy

oneDNN expresses memory layouts via `memory_desc_t` (format tag, dims, dtype, blocking). ZenDNN expects specific layouts depending on the chosen kernel. Two strategies, evolving across phases:

- **Strategy A &mdash; ZenDNN-internal reorder + cache** *(PoC-current).* The user calls `dnnl_matmul`; `zen64::matmul_t::execute()` hands the user-supplied weight to ZenDNN, which prepacks internally on first use and caches the prepacked tensor in its own LRU cache (keyed on weight pointer + shape). Pros: simple, no oneDNN-side reorder primitive needed. Cons: prepack cost paid on first call; opaque to oneDNN's primitive cache.
- **Strategy B &mdash; oneDNN-side reorder ahead of time** *(in progress).* The PD requests a specific weight layout (the one ZenDNN's chosen kernel wants), letting the user (or a framework integration) issue a oneDNN reorder ahead of inference. Then `execute()` passes already-packed weights directly. Pros: oneDNN-style "prepare once, run many"; reorder is itself a oneDNN primitive that the framework can place where it wants. Cons: more PD-side plumbing to negotiate the layout, especially for `format_kind::any` inputs.

Phase-1 production ships Strategy A. Strategy B is the in-progress next step (slide deck: "Using OneDNN Reorder ahead of time, followed by ZenDNN MatMul (InProgress)") and is the target for the BMM / GroupGEMM / SDPA follow-ups in Q3.

For cases where the user-supplied weight has a layout ZenDNN cannot consume directly and where ahead-of-time reorder is not possible, the impl rejects in `pd_t::init()` and the standard oneDNN path runs.

### 5.8 No public API changes

Vertical&nbsp;1 introduces:
- One CMake option (`DNNL_ENABLE_ZENDNN`, default OFF).
- One optional secondary option (`DNNL_ENABLE_ZENDNN_STATIC`, default OFF).
- A handful of env vars (`DNNL_ENABLE_ZENDNN`, `DNNL_ZENDNN_<PRIM>`).
- New impl classes in `src/cpu/x64/zen64/` and one extra entry in each existing `cpu_<prim>_list.cpp`.

It introduces **no** new headers in `include/oneapi/dnnl/`, no new C entry points, no new C++ classes, no new `dnnl_*_t` enum values. A user program built against today's `libdnnl.so` continues to link and run unchanged when `libdnnl.so` is rebuilt with `DNNL_ENABLE_ZENDNN=ON`.

## 6. Vertical 1 PoC: MatMul fp32 / bf16 (+ Reorder)

A working PoC is already running locally. It exercises the full integration path end-to-end through the oneDNN `dnnl_matmul` primitive on a Zen-class AMD CPU, with ZenDNN selected as the impl and ZenDNN's LOWOHA `matmul_direct` doing the actual compute.

### 6.1 Scope

| Item | PoC | Notes |
|---|---|---|
| Operator | `dnnl_matmul` (+ fused via post-ops) | Same primitive interface as today |
| Datatypes | `f32`, `bf16` | Both src/wei/dst sides; bias handled where present |
| Layouts | `ab` (plain 2D) | Strategy A (ZenDNN-internal reorder + LRU cache) |
| Reorder | Internal (Strategy A) | Strategy B (oneDNN-side ahead-of-time reorder) is the next milestone |
| Fallback | Verified | Forcing `DNNL_ENABLE_ZENDNN=0` reverts to oneDNN's existing impls without rebuild |
| Vendor / uArch gate | Verified | Running on Intel correctly rejects ZenDNN backend at PD init time and falls through to BRGEMM matmul |

### 6.2 Build instructions

```bash
cmake -B build -S .                  \
      -DCMAKE_BUILD_TYPE=Release     \
      -DONEDNN_BUILD_TESTS=ON        \
      -DDNNL_ENABLE_ZENDNN=ON        \
      -DZENDNN_ROOT=/path/to/zendnn  # optional; falls back to find_package
cmake --build build -j
```

### 6.3 Run instructions

```bash
# Master switch (defaults ON when DNNL_ENABLE_ZENDNN=ON at build time, but
# explicit env vars are useful for A/B switching at runtime).
export DNNL_ENABLE_ZENDNN=1
export DNNL_ZENDNN_MATMUL=1

# Standard oneDNN verbose level.
export ONEDNN_VERBOSE=1

# Run any oneDNN matmul (benchdnn, an example, a framework call, ...).
./build/tests/benchdnn/benchdnn --matmul --dt=bf16 --stag=ab --wtag=ab \
        --dtag=ab 256x256:256x256
```

### 6.4 Verbose-log evidence (from current PoC run)

```
onednn_verbose,v1,info,oneDNN v3.12.0 (commit a7c2a8e269c216e7457251e93f75b00b8dc3f797)
onednn_verbose,v1,info,cpu,runtime:OpenMP,nthr:192
onednn_verbose,v1,info,cpu,isa:Intel AVX-512 with Intel DL Boost and bfloat16 support
onednn_verbose,v1,info,gpu,runtime:none
onednn_verbose,v1,info,graph,backend,0:dnnl_backend

*********** ZenDNN INIT (AMD-only, f32|bf16) ***********
*********** I am inside ZenDNN EXECUTE (BF16) ***********
*********** ZenDNN LowOHA MatMul (BF16) ***********

[PROF ][verbose][0.021464]: LOWOHA matmul_direct:
   M=256, N=256, K=256, alpha=1, beta=0, lda=256, ldb=256, ldc=256,
   transA=false, transB=false,
   input_dtype=bf16, output_dtype=bf16, bias=false, is_weights_const=true,
   post_op=[none], post_op_dtype=[none], Batch_A=1, Batch_B=1,
   kernel=aocl_dlp_blocked,
   weight_address=0x7f47ff200000, time=16.0136ms

onednn_verbose,v1,primitive,exec,cpu,matmul,zendnn:matmul:f32|bf16:amd,undef,
    src:bf16::blocked:ab::f0 wei:bf16::blocked:ab::f0 dst:bf16::blocked:ab::f0,,,
    256x256:256x256, 21.5581
```

What this shows:

- The standard oneDNN verbose header (commit, runtime, ISA) is unchanged &mdash; no new top-level info lines.
- The new `zen64::matmul_t` is the impl reported in the per-primitive line: `cpu,matmul,zendnn:matmul:f32|bf16:amd`. This matches oneDNN's existing convention `<engine>,<prim>,<impl_name>` and is what reviewers will look for.
- The shape, dtype, and layout descriptors are oneDNN's standard `memory_desc_t` rendering &mdash; no ZenDNN-specific encoding leaks out.
- ZenDNN's internal Auto Tuner picked `kernel=aocl_dlp_blocked` for this shape; this is the ZenDNN-side selection invisible to oneDNN.
- `time=16.0136ms` is ZenDNN's per-call profiler; the trailing `21.5581` is oneDNN's primitive-level execute time (microseconds), confirming the call did go through the standard primitive pipeline.

### 6.5 What we will measure for the production PR

The PoC numbers above are useful for plumbing validation, **not** for the performance argument that the upstream PR has to make under [`CONTRIBUTING.md` Performance gate](../../../CONTRIBUTING.md#library-functionality-guidelines). For the production PR (end June) we will publish:

- benchdnn perf runs on a representative shape set (small / medium / large MatMul, square / skinny K / skinny M, batched); ZenDNN backend vs stock oneDNN, both on the same Zen CPU.
- Model-level numbers from vLLM running BF16 / FP32 inference on a wide model set on AMD and Intel hosts; baseline = `vLLM + zentorch`, candidate = `vLLM + oneDNN(ZenDNN backend)`.
- Cold-cache and warm-cache splits so the reorder strategy can be evaluated independently of compute.
- A `--skip-impl=zendnn` rerun to confirm the stock oneDNN path is unaffected on the same host.

Numbers go in the PR cover letter and in the corresponding section of the production-PR commit body.

## 7. Vertical 2: Native Upstream Contributions (no `zen64` dependency)

These contributions go directly into oneDNN's standard tree. They have **no** runtime, link-time, or build-time dependency on ZenDNN; they do not require `DNNL_ENABLE_ZENDNN`. ZenDNN's internal kernels are referenced as the implementation source while porting, then disappear from the picture once the code is in oneDNN.

Each contribution has its own RFC and PR series. This section is the umbrella overview; details live in the per-contribution RFC.

### 7.1 Embedding + Embedding-Bag primitive

A new oneDNN primitive `dnnl_embedding_bag` (with `dnnl_embedding_lookup` algorithm for the no-reduction case), implemented natively in `src/cpu/x64/embedding_bag.{hpp,cpp}` as AVX-512 / AVX-512-FP16 intrinsics. No JIT, no Xbyak. Detailed design in the companion RFC.

- **Companion RFC:** [`doc/rfcs/embedding_bag/README.md`](../embedding_bag/README.md) (already drafted, on this branch).
- **GitHub draft:** [PR #10](https://github.com/AMD-Zenai/oneDNN-ZenDNN/pull/10) on `AMD-Zenai/oneDNN-ZenDNN`.
- **Phasing:** RFC June mid (done); first PR (BF16 + FP32) July mid; follow-up PRs Q3 (INT8, INT4, Group Embedding Bag with Grouped Memory Format).
- **Why Vertical 2, not Vertical 1.** oneDNN does not have an `embedding_bag` primitive today. We cannot register a backend for an API that does not exist; we have to add the primitive itself. Once the primitive lands, a future Vertical-1 entry could in principle register a `zen64::embedding_bag_t` impl above the native intrinsic one, but the companion RFC (§12.E) explicitly recommends *not* doing that: the kernel is simple enough that the native intrinsic impl is sufficient and avoids carrying ZenDNN as a runtime dep for a primitive that doesn't need it.

### 7.2 Dynamic Quantization support

A new oneDNN primitive (or attribute extension) for fused scale + zero-point computation followed by quantize. The primitive computes per-tensor / per-channel / per-group scale and zp from input data, then quantizes in a single AVX-512 pass.

- **Scope:**
  - Symmetric, asymmetric, and group quantization.
  - Fused AVX-512 intrinsic kernel.
  - Multiple threading strategies (mirroring ZenDNN's parallel primitive).
- **API extension required.** oneDNN's existing `attr.scales` / `attr.zero_points` machinery is parametric (caller provides scales/zps). Dynamic Quant needs an API extension where the library *computes* the scales and zps from input data; this requires either a new primitive or a new attribute mode.
- **Companion RFC:** to be authored. Target: end of June.
- **Phasing:** RFC June end; first PR (BF16 + FP32) July mid; follow-up PRs Q3 (INT8 path, integration with quantized MatMul).

### 7.3 oneDNN MatMul / BMM kernel optimisation for AMD CPUs

Tiling and threading heuristics inside the existing oneDNN MatMul / BMM JIT kernels, tuned for Zen-class CPUs. These are oneDNN-internal optimisations &mdash; no ZenDNN code is added; the changes live in the existing `src/cpu/x64/matmul/` tree.

- **Scope:**
  - Adapting MatMul tile sizes to Zen cache hierarchy (L1/L2/L3, CCD-aware where applicable).
  - Adapting threading heuristics (`balance211` / `parallel_nd_ext` distribution) for Zen CCDs.
- **No public API change.** Changes are dispatch-internal; benchmarks confirm the same kernels remain selected on Intel.
- **Phasing:** Q3/Q4 2026, after Vertical&nbsp;1 is in production and we have benchmark coverage to drive the heuristic changes responsibly.

### 7.4 AutoTuner (runtime kernel/backend selection)

A oneDNN-internal AutoTuner pass that picks among multiple registered impls for a given primitive based on runtime profiling, mirroring ZenDNN's TBP + Decision Tree. Today, oneDNN's impl-list ordering is static; first-match wins. AutoTuner adds a runtime profiler that re-ranks candidates per shape on first call, caching the winner.

- **Scope:**
  - One-time per-shape probing of the top-N impl candidates.
  - Persistent cache across primitive_desc_create calls within a process.
  - Optional warm-up hook for frameworks that want to force the first-call cost out of the inference window.
- **No public API change.** Internal infra; user code is unaffected.
- **Phasing:** Q3 2026 alongside the `zen64` follow-ups (BMM, SDPA), since AutoTuner is most valuable when there are multiple competing impls.

### 7.5 Low-Overhead APIs

Generalising ZenDNN's LOWOHA pattern (kernel-pointer dispatch, no per-call validation) into a oneDNN-internal helper used by JIT and intrinsic primitives. Today, oneDNN already does most of this through `primitive_t::execute()` &mdash; the gap is in the BRGEMM microkernel API, where small-shape callers pay measurable per-call overhead.

- **Scope:** A LOWOHA hook in the BRGEMM microkernel API path. Cooperative target with Vertical&nbsp;1's BRGEMM-microkernel hook in §9.
- **Phasing:** Q3 2026, paired with the BRGEMM-microkernel hook in Vertical&nbsp;1.

## 8. End-to-End Lifecycle

### 8.1 Vertical 1 &mdash; `zen64` backend dispatch (e.g. MatMul)

```
user code (PyTorch / TF / vLLM / app)
            │
            ▼
   dnnl::matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr)
            │                                ⟵ public API, unchanged
            ▼
   dnnl_matmul_primitive_desc_create(...)    ⟵ src/common/matmul.cpp
            │
            ▼   CASE(matmul) ⟶ get_matmul_impl_list   ⟵ src/cpu/cpu_engine.hpp
            │
            ▼   impl_list walked in order
   ┌─────────────────────────────────────────────────────────────────┐
   │ zen64::matmul_t::pd_t::init()   (only if DNNL_ENABLE_ZENDNN=ON) │
   │   1. runtime kill-switch    (DNNL_ENABLE_ZENDNN, per-prim)      │
   │   2. CPU vendor / uArch     (AuthenticAMD, Zen4/Zen5/Zen6)      │
   │   3. ISA features           (AVX-512, BF16, FP16, VNNI)         │
   │   4. shape / dtype / layout / attr / post-op                    │
   │   5. set_default_formats(); book_scratchpad()                   │
   │   ▶ success → selected                                          │
   │   ▶ unimplemented → iterator advances                           │
   └─────────────────────────────────────────────────────────────────┘
            │ unimpl
            ▼
   ┌─────────────────────────────────────────┐
   │ brgemm_matmul_t  (existing oneDNN JIT)  │
   │ + other existing entries                │
   │ + ref_matmul_t   (terminal fallback)    │
   └─────────────────────────────────────────┘
            │ success
            ▼
   primitive_desc ready  ▶  dnnl::matmul prim(pd);
            │
            ▼
   zen64::matmul_t::execute(ctx):
       src/wei/bias/dst pointers from CTX_*_MEM
       params resolved at PD time (no re-validation)
       zendnnl::lowoha::matmul_direct(params)   ← ZenDNN AutoTuner picks
                                                  ZenDNN-native vs AOCL-DLP
                                                  vs FBGEMM internally
            │
            ▼
   dst memory written
```

### 8.2 Vertical 2 &mdash; native primitive dispatch (e.g. embedding_bag)

For new primitives added natively, dispatch is identical to any other oneDNN primitive (`softmax`, `reduction`). The companion RFC §13 shows the full lifecycle for `embedding_bag`. Summary:

```
user → dnnl::embedding_bag::primitive_desc(...)
     → dnnl_embedding_bag_primitive_desc_create(...)
     → cpu_engine → CASE(embedding_bag) → impl_list
            │
            ▼
   ┌────────────────────────────────────────┐
   │ embedding_bag_t::pd_t::init()  (x64)   │
   │   mayiuse(avx512_core_*) → isa_        │
   │   VDISPATCH dtype/format/attr          │   ⟵ src/cpu/x64/embedding_bag.cpp
   └────────────────────────────────────────┘
            │ unimpl → ref_embedding_bag_t (terminal)
            ▼ success
   primitive_desc ready  ▶  prim(pd);
            │
            ▼
   embedding_bag_t::init(engine):
       kernel_fn_ = pick_kernel(...);   ← function-pointer resolution,
                                          NO Xbyak / NO JIT / NO ZenDNN
            │
            ▼   prim.execute(...)
   parallel_nd_ext fans out, kernel_fn_ called per work item
            │
            ▼
   dst memory written
```

The two flows look intentionally similar &mdash; both end in a primitive-cache-amortised PD selection plus an allocation-free hot path. The difference is *where the kernel code lives*: in `zendnnl::lowoha::matmul_direct(...)` (Vertical&nbsp;1) vs in the oneDNN tree under `src/cpu/x64/embedding_bag.cpp` (Vertical&nbsp;2).

## 9. Phasing and Delivery

The two verticals run in parallel with overlapping milestones, but have independent PR tracks. Each PR follows oneDNN's commit-message format `<scope>[: <subscope>...]: <imperative summary>` and runs `ONEDNN_TEST_SET=NIGHTLY` before merge, per [`CONTRIBUTING.md`](../../../CONTRIBUTING.md).

### 9.1 Vertical 1 timeline (`zen64` module)

| Milestone | Target | Scope | Done-when |
|---|---|---|---|
| Step 1 &mdash; PoC | In progress | MatMul (+ fused) with reorder, BF16 + FP32, Strategy A (ZenDNN-internal cache); end-to-end on AMD; gating verified on Intel | Verbose log shows `zendnn:matmul:f32\|bf16:amd` impl on Zen, falls through to `brgemm_matmul_t` on Intel; reorder ahead-of-time variant in flight |
| Step 2 &mdash; vLLM framework validation | In progress | Run `vLLM + oneDNN(zen64 enabled)` on production-relevant LLM workloads; baseline = `vLLM + zentorch` | Quantified perf lift on a wide model set on AMD and Intel hosts (Intel = no regression confirmation) |
| Step 3 &mdash; Upstream RFC | Mid June | This document, submitted to `uxlfoundation/oneDNN` `rfcs` branch alongside companion RFC | Formal alignment with oneDNN maintainers before production PR; minimises upstream risk and review churn |
| First PR | End June | ZenDNN library integration in build infra; `zen64::matmul_t` (+ fused) + `zen64::reorder_t`; BF16 + FP32; fallback to native oneDNN for non-supported features; AMD + Intel CPU validation | benchdnn correctness + perf published; no regression on Intel; clean CI on `uxlfoundation/oneDNN` `main` |
| Follow-up PR &mdash; Quantized MatMul | Q3 | WoQ + Static + Dynamic Quant via existing `attr.scales` / `attr.zero_points` for static, plus the new Dynamic-Quant API from V2 §7.2 | Quantized inference path validated end-to-end |
| Follow-up PR &mdash; Other operators | Q3 | BMM, GroupGEMM (Grouped Memory Format for variable-size batching), SDPA, Reorder (Quant/DeQuant) | Each op a separate PR with its own benchdnn evidence |
| Follow-up PR &mdash; BRGEMM microkernel hook | Q3 | ZenDNN LOWOHA integration in oneDNN's BRGEMM microkernel API (Vertical 2 §7.5 partner) | Small-shape GEMM-heavy workloads (SDPA, low-batch decode) show measurable improvement |

### 9.2 Vertical 2 timeline (native upstream contributions)

| Milestone | Target | Scope | Companion RFC |
|---|---|---|---|
| Embedding+Bag &mdash; RFC | June mid (done) | Native `embedding_bag` primitive with AVX-512 intrinsic kernel | [`doc/rfcs/embedding_bag/`](../embedding_bag/README.md) (this branch) |
| Embedding+Bag &mdash; First PR | July mid | API + ref impl + JIT-free intrinsic impl (BF16 + FP32) | per companion RFC §10 |
| Embedding+Bag &mdash; Follow-up | Q3 | INT8, INT4; Group Embedding Bag (Grouped Memory Format); per-group threading strategies | per companion RFC §10 |
| Dynamic Quant &mdash; RFC | June end | API extension for compute-scale + quantize fused primitive | new RFC: `doc/rfcs/dynamic_quant/` (to be created) |
| Dynamic Quant &mdash; First PR | July mid | BF16 + FP32, fused AVX-512 intrinsic kernel, multiple threading strategies | per future RFC |
| Dynamic Quant &mdash; Follow-up | Q3 | INT8, integration with quantized MatMul path | per future RFC |
| AMD-CPU MatMul/BMM tuning | Q3/Q4 | Tiling and threading heuristics inside existing oneDNN MatMul/BMM JIT | per future RFC |
| AutoTuner | Q3 | Internal pass for runtime kernel/backend selection | per future RFC |
| Low-Overhead APIs | Q3 | LOWOHA hook in BRGEMM microkernel API (paired with V1) | per future RFC |

### 9.3 Combined view

```
                   May    Jun mid     Jun end     Jul mid     Aug-Sep (Q3)        Q4
V1 (zen64):        PoC ── RFC ─────── First PR ──────────── Follow-ups ─────────────
                                     (MatMul+R)            (BMM/SDPA/QMM/uK)
V2 embedding_bag:        RFC ───────── (review) ─────── First PR ── Follow-ups ─────
V2 dynamic quant:               ────── RFC ─────────── First PR ── Follow-ups ─────
V2 AMD-CPU tuning:                                                  ──── design ──── First PRs
V2 AutoTuner:                                                         ─── RFC ─── First PR
V2 LOW-OHA / uK:                                                      ─── RFC ─── First PR
```

## 10. Risks and Open Questions

1. **Acceptance of a vendor-specific backend in upstream oneDNN.** oneDNN does not, today, link any third-party CPU library at build time other than what its existing JIT depends on. We need explicit maintainer alignment that `DNNL_ENABLE_ZENDNN=ON` is an acceptable optional path. Mitigation: gate behind a default-OFF flag, no public API changes, full fallback semantics, no impact on Intel hosts.

2. **ZenDNN ABI / version coupling.** A given oneDNN release built with ZenDNN integration is pinned to a ZenDNN version range. If ZenDNN's API changes, oneDNN's adapter has to change. Mitigation: keep the adapter surface narrow (one `lowoha::*_direct` call per primitive); pin a known-good ZenDNN commit in oneDNN's CMake; coordinate ZenDNN releases with oneDNN release cadence.

3. **Static vs dynamic linking of ZenDNN.** Static link bloats `libdnnl.so` size; dynamic link adds a runtime dependency. Default is dynamic with a CMake option to flip. Mitigation: document both, recommend dynamic for distribution packages, allow static for embedded / single-binary deployments.

4. **Reorder strategy choice.** Strategy A (ZenDNN-internal cache, PoC-current) is simple but opaque to oneDNN's primitive cache. Strategy B (oneDNN-side reorder ahead of time, in progress) is more idiomatic but more PD-side plumbing. Open question: do we ship Vertical&nbsp;1 first PR with Strategy A only, or wait for Strategy B? Current plan: ship Strategy A, document Strategy B as a follow-up PR.

5. **Maintenance burden.** Vertical&nbsp;1 means oneDNN reviewers see ZenDNN-specific commits going forward. Mitigation: AMD-Zenai team owns the `zen64` directory and the per-primitive adapter classes; oneDNN maintainers review the integration plumbing (impl-list registration, build option, fallback semantics) but not the kernels themselves.

6. **Performance gate at upstream.** Per `CONTRIBUTING.md`, every change must show "material workload-level impact." We must ship benchdnn perf and vLLM model-level numbers with each production PR. Risk: if Strategy A's first-call reorder cost dominates a particular benchmark, the perf argument is weaker; mitigation is Strategy B (ahead-of-time reorder).

7. **Cross-vertical interaction.** Once Vertical&nbsp;2's Embedding+Bag primitive lands, a future contributor might be tempted to add a `zen64::embedding_bag_t` impl for it. The companion RFC explicitly recommends *not* doing that for the embedding-bag case (the kernel is simple enough that the native intrinsic impl is sufficient and avoids carrying ZenDNN as a dep for a primitive that doesn't need it). For other ops where Vertical&nbsp;1 *is* warranted, the same impl-list mechanism makes the addition straightforward.

8. **Verbose / profile log evolution.** The PoC log uses a free-form `*********** ZenDNN INIT ... ***********` format inside ZenDNN's body; the production version should standardise on oneDNN's verbose pattern (`onednn_verbose,...`). The `[PROF]` line and the friendly banners stay only when `ONEDNN_VERBOSE=2` or higher.

9. **Test coverage on non-AMD hardware.** Vertical&nbsp;1 must not regress Intel-CPU performance; we explicitly run the same benchdnn shape sets on Intel with `DNNL_ENABLE_ZENDNN=ON` to verify the dispatcher correctly falls through. CI matrix needs an Intel host running with the build flag ON to catch any accidental Intel-side dispatch.

10. **Build dependency on platforms without ZenDNN.** ZenDNN ships Linux x86_64 only today. We must guard the build option so that `DNNL_ENABLE_ZENDNN=ON` on macOS / Windows / Arm fails fast with a clear error rather than producing a half-built tree.

## 11. Alternatives Considered

### A. Single vertical &mdash; everything via `zen64` (rejected)

Push every AMD-tuned operation, including ones oneDNN does not yet have (embedding bag, dynamic quant), through the `zen64` runtime backend.

- **Pros:** One mechanism end-to-end; zero native-upstream porting effort.
- **Cons:** Every consumer of those operations would need `DNNL_ENABLE_ZENDNN=ON` builds, which means frameworks that consume only stock oneDNN never see embedding-bag or dynamic-quant. Worse, it leaks ZenDNN-specific primitive shapes into oneDNN's "shadow API" surface (users have to know to call which paths only work with the flag on). Embedding-bag and dynamic-quant are general enough that they belong as standard oneDNN primitives.

### B. Single vertical &mdash; everything native (rejected)

Upstream every ZenDNN kernel as native oneDNN code, even for primitives oneDNN already has (MatMul, BMM, SDPA).

- **Pros:** Cleanest end-state &mdash; oneDNN owns every kernel.
- **Cons:** Years of porting work for kernels that already exist and ship in ZenDNN today. The `zen64` mechanism delivers the same end-user benefit (frameworks see better AMD-CPU numbers through stock oneDNN APIs) on a much faster timeline, with the door open to gradually replacing wrapped kernels with native-upstreamed ones over time if it ever makes sense.

### C. Upstream the AutoTuner only, leave kernels in ZenDNN forever (rejected)

Keep `zen64` permanently, never upstream native-side.

- **Pros:** Lowest oneDNN-side surface area.
- **Cons:** oneDNN never gets the embedding-bag / dynamic-quant operators that frameworks need; the integration story stays vendor-flag-gated forever. Vertical&nbsp;2 is essential to eventually absorb the operators that are general enough to deserve a place in oneDNN's standard surface.

### D. Two-vertical strategy (chosen)

Split the contributions by the rule in §4: wrap if oneDNN already has the API, upstream natively if it doesn't. Vertical&nbsp;1 ships fast with low risk; Vertical&nbsp;2 grows oneDNN's capability.

- **Pros:** Best of both: immediate AMD-CPU performance for existing primitives, plus permanent expansion of oneDNN's primitive set for the gaps. Each vertical is independently scoped and reviewable.
- **Cons:** Two narratives to communicate to upstream maintainers. This RFC is the umbrella that prevents the verticals from being reviewed in isolation and confused for one another.

## 12. Relationship to the `embedding_bag` Primitive RFC

The companion RFC at [`doc/rfcs/embedding_bag/`](../embedding_bag/README.md) is **the first concrete deliverable of Vertical&nbsp;2**. It is intentionally written and reviewed independently of this umbrella RFC because:

1. **Scope clarity.** The embedding_bag RFC focuses on a single new oneDNN primitive: the API shape, the algorithm semantics, the validation rules, the file layout, the test harness. None of those depend on whether Vertical&nbsp;1 exists.
2. **Reviewability.** Two narrow RFCs are easier to review than one wide one. oneDNN maintainers can approve / reject each independently. If Vertical&nbsp;1 stalls in upstream review, Vertical&nbsp;2's embedding_bag PR series can still proceed.
3. **Implementation independence.** The embedding_bag impl is a pure native AVX-512 intrinsic kernel inside `src/cpu/x64/embedding_bag.{hpp,cpp}`. It does not include any ZenDNN headers and does not depend on `DNNL_ENABLE_ZENDNN`. The two RFCs do not share a single line of code.

That said, the two RFCs **are** consistent:

- The companion RFC's §12.C (Alternatives Considered &mdash; "Wrap ZenDNN as a runtime backend") explicitly considers and rejects the `zen64`-style wrapping for embedding_bag specifically, on the grounds that the kernel is simple enough that native intrinsics are sufficient. This rejection applies *only to embedding_bag*; it is **not** a rejection of the `zen64` mechanism for primitives that need it (MatMul, BMM, SDPA, etc.).
- This umbrella RFC's §4 explicitly says "wrap if oneDNN already has the API, upstream natively if it doesn't." Embedding_bag is a "doesn't" case, hence native. MatMul is a "does" case, hence wrapped via `zen64`. Both rules are consistent.
- This umbrella RFC's §7.1 lists embedding_bag as Vertical&nbsp;2's first deliverable and links to the companion RFC for technical details. It does not duplicate the design content; it positions it within the broader strategy.

**For reviewers reading both RFCs:** start here for strategy and context, then read the embedding_bag RFC for the technical design of that specific primitive. The two RFCs together describe AMD's complete Q2&ndash;Q4 contribution plan to oneDNN.

## Appendix A. GitHub References

| Resource | URL |
|---|---|
| oneDNN upstream (uxlfoundation) | https://github.com/uxlfoundation/oneDNN |
| oneDNN `rfcs` branch (where production RFCs live) | https://github.com/uxlfoundation/oneDNN/tree/rfcs |
| AMD-Zenai oneDNN-ZenDNN fork (this work) | https://github.com/AMD-Zenai/oneDNN-ZenDNN |
| Companion RFC: `embedding_bag` (PR #10 draft) | https://github.com/AMD-Zenai/oneDNN-ZenDNN/pull/10 |
| ZenDNN library (source / docs) | https://github.com/amd/ZenDNN |
| AOCL-DLP (one of ZenDNN's backends) | https://github.com/amd/aocl-dlp |
| zentorch (PyTorch consumer of ZenDNN, today's path) | https://github.com/amd/ZenDNN-pytorch-plugin |

## Appendix B. PoC File Touchpoints (Vertical 1, MatMul + Reorder)

The PoC introduces or modifies the following files. This list is illustrative; the production PR will refine.

```
CMakeLists.txt                                    [M]  add DNNL_ENABLE_ZENDNN option
cmake/options.cmake                               [M]  documentation entry
cmake/FindZenDNN.cmake                            [+]  find_package module (or thin shim)

src/cpu/cpu_engine.hpp                            [M]  add DNNL_X64_ONLY_IF macro

src/cpu/x64/zen64/                                [+]  new directory
  zen64_common.hpp                                [+]  is_runtime_enabled / cpu_supported / isa_supported helpers
  zen64_common.cpp                                [+]
  zen64_matmul.hpp                                [+]  zen64::matmul_t : public primitive_t
  zen64_matmul.cpp                                [+]  pd_t::init() + execute() body, calls into zendnnl::lowoha
  zen64_reorder.hpp                               [+]  zen64::reorder_t : public primitive_t
  zen64_reorder.cpp                               [+]

src/cpu/matmul/cpu_matmul_list.cpp                [M]  add CPU_INSTANCE(zen64::matmul_t) ahead of brgemm_matmul_t
src/cpu/reorder/cpu_reorder_list.cpp              [M]  add zen64::reorder_t entry

doc/build/build_options.md                        [M]  document DNNL_ENABLE_ZENDNN
doc/programming_model/zendnn_backend.md           [+]  user-facing doc on the optional backend

tests/benchdnn/                                   [M]  no new driver; --skip-impl=zendnn for A/B testing
```

The `src/cpu/x64/zen64/zen64_matmul.cpp` file is the only location where ZenDNN headers are included; its includes are guarded by `#if DNNL_X64 && DNNL_ENABLE_ZENDNN` so a build with the flag OFF compiles even with no ZenDNN install on the system.

## Appendix C. Glossary

| Term | Meaning |
|---|---|
| **`zen64`** | The new optional CPU sub-target proposed by this RFC. Lives at `src/cpu/x64/zen64/`. Gated by `DNNL_ENABLE_ZENDNN`. |
| **LOWOHA** | "Low-Overhead API," ZenDNN's kernel-direct surface (e.g. `zendnnl::lowoha::matmul_direct`). |
| **TBP** | "Time-Based Profiling," ZenDNN's per-shape on-the-first-call backend probe used by the AutoTuner. |
| **DT (Decision Tree)** | The static-heuristic complement to TBP: shape / dtype / layout discriminators that pick a backend without profiling. |
| **AOCL-DLP** | AMD's optimised dense-linear-algebra primitive library; one of ZenDNN's candidate backends. |
| **FBGEMM** | Facebook's GEMM library; another candidate ZenDNN backend, particularly for small-shape paths. |
| **CCD** | Core Complex Die (Zen). Threading strategies that group threads by CCD reduce cross-CCD memory traffic. |
| **WoQ** | Weight-only Quantization &mdash; weights are int4/int8, activations stay floating-point. |
| **Static Quant** | All scales / zps known at calibration time, baked into the model. |
| **Dynamic Quant** | Scales / zps computed at inference time per tensor / per channel / per group. |
| **Strategy A / B** (reorder) | Section 5.7 &mdash; ZenDNN-internal cache (A) vs oneDNN-side ahead-of-time reorder (B). |

## Appendix D. References

**Project guidelines.**
- [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) &mdash; library functionality criteria, RFC process, commit-message format, unit-test policy.
- [`CODING_STANDARDS.md`](../../../CODING_STANDARDS.md) &mdash; clang-tidy, clang-format, naming conventions.

**Companion documents.**
- [`doc/rfcs/embedding_bag/README.md`](../embedding_bag/README.md) &mdash; Vertical 2's first deliverable.
- (Future) `doc/rfcs/dynamic_quant/README.md` &mdash; Dynamic Quant API + primitive (target end of June).

**Existing oneDNN patterns referenced by this design.**
- `src/cpu/cpu_engine.hpp` &mdash; `CPU_INSTANCE`, `CPU_INSTANCE_X64`, `DNNL_X64_ONLY`, `DECLARE_IMPL_LIST`, `CASE`.
- `src/common/impl_registration.hpp` &mdash; `REG_<PRIM>_P` macros.
- `src/common/primitive_desc_iterator.hpp` &mdash; impl-list walking semantics (`unimplemented` &rarr; advance).
- `src/cpu/matmul/cpu_matmul_list.cpp` &mdash; the file Vertical 1 modifies first.
- `src/cpu/x64/jit_brgemm_matmul.{hpp,cpp}` &mdash; the existing Intel-tuned MatMul JIT path that `zen64::matmul_t` registers ahead of.

**External references.**
- ZenDNN public documentation and source: https://github.com/amd/ZenDNN
- vLLM (intended primary consumer for Vertical 1 validation): https://github.com/vllm-project/vllm
- PyTorch native oneDNN integration: https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/onednn

