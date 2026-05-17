# RFC: ZenDNN Integration in oneDNN — `zen64` Module + Native Upstream Contributions

## Status
**Draft** &nbsp;|&nbsp; AMD-Zenai/oneDNN-ZenDNN &nbsp;|&nbsp; Branch: `rfc/embedding-bag` &nbsp;|&nbsp; Companion RFC: [`doc/rfcs/embedding_bag/`](../embedding_bag/README.md)

## Authors
- AMD-Zenai team

## Summary

This RFC describes the AMD plan to bring ZenDNN-class CPU performance into oneDNN through **two complementary verticals**. The two verticals share strategic intent (production-grade AMD-CPU performance for every framework that already consumes oneDNN) but use different integration mechanisms by design:

1. **Vertical 1 &mdash; `zen64` module (the focus of this RFC).** A new ISA-flavoured CPU sub-target under `src/cpu/x64/zen64/` that registers ZenDNN as one entry in oneDNN's existing per-primitive `impl_list`, gated at build time by `DNNL_ENABLE_ZENDNN=ON` and at runtime by `DNNL_ENABLE_ZENDNN` plus per-primitive env vars. Selected on AMD CPUs with the right uArch and ISA, falls through to oneDNN's native impls otherwise. Initial scope: MatMul (+ fused) and Reorder for fp32 / bf16, validated end-to-end with a working PoC (§6). Follow-up scope: BMM, GroupGEMM (Grouped Memory Format for variable-size batching), SDPA, Quantized MatMul (WoQ + Static + Dynamic), and a ZenDNN LOWOHA hook into oneDNN's BRGEMM microkernel API. Sections 5 and 6 of this document cover Vertical&nbsp;1 in depth.

2. **Vertical 2 &mdash; native upstream contributions (high-level overview only here).** New oneDNN primitives and primitive optimisations implemented natively in oneDNN's tree, with no `zen64` dependency, no `DNNL_ENABLE_ZENDNN` gating, and no link-time / runtime / build-time dependency on ZenDNN. Items in scope include the new `embedding_bag` primitive, Dynamic Quantization support, MatMul / BMM tiling and threading heuristics tuned for AMD CPUs, an internal AutoTuner pass for runtime kernel/backend selection, and Low-Overhead API hooks. **Each Vertical&nbsp;2 item has its own RFC and PR series** &mdash; the `embedding_bag` companion RFC at [`doc/rfcs/embedding_bag/`](../embedding_bag/README.md) is the first of those. This document gives only a high-level overview of Vertical&nbsp;2 (§7) and points reviewers to the per-item RFCs for details.

The two verticals are **not** alternatives. Vertical&nbsp;1 ships ZenDNN's existing kernels behind a gated backend so AMD users get production performance on already-existing oneDNN primitives. Vertical&nbsp;2 invests in new oneDNN-native code so the operations that ZenDNN provides today but oneDNN does not yet have (embedding bag, dynamic quant, etc.) become permanently part of oneDNN's standard surface, available to every framework on every CPU. §4 explains the rationale for the split, §12 describes how the `embedding_bag` companion RFC fits in.

A working PoC of Vertical&nbsp;1 (MatMul fp32/bf16 with Reorder) is already running locally; verbose logs are reproduced in §6. vLLM-level framework validation is in progress.

---

> ## :pushpin: First PR &mdash; the immediate ask
>
> Once this RFC is accepted, the **first** PR to upstream `uxlfoundation/oneDNN` carries exactly the following scope. Everything else is follow-up.
>
> 1. **ZenDNN library integration in build infra** &mdash; new `DNNL_ENABLE_ZENDNN` CMake option (default OFF), `find_package(ZenDNN)` plumbing, optional `DNNL_ENABLE_ZENDNN_STATIC` for link-mode control. No effect on the default build.
> 2. **Add ZenDNN MatMul (+ fused) and Reorder (for format conversion) in the `zen64` module** &mdash; new `src/cpu/x64/zen64/{zen64_matmul,zen64_reorder}.{hpp,cpp}` registered into `cpu_matmul_list.cpp` and `cpu_reorder_list.cpp` ahead of the existing entries.
>     - **BF16 and FP32 datatypes** &mdash; the two dtypes that cover the broadest set of LLM inference workloads today.
>     - **Fallback to the native oneDNN kernel for non-supported features** &mdash; staged `VDISPATCH_*` checks in `pd_t::init()` (vendor / uArch / ISA / shape / dtype / layout / post-op) return `status::unimplemented` cleanly so the dispatcher falls through to `brgemm_matmul_t` / existing reorder impls. No silent failures, no compute regressions.
>     - **Validation with AMD and Intel CPUs** &mdash; benchdnn correctness + perf evidence on a Zen host (impl selected = `zendnn:matmul:f32\|bf16:amd`) and on an Intel host with the same binary (`DNNL_ENABLE_ZENDNN=ON` build, ZenDNN code path *not* taken; existing impl runs unchanged; `--skip-impl=zendnn` rerun confirms no regression).
>
> Detailed acceptance criteria, file-level scope, and the surrounding milestone gates are in [§9.1](#91-first-pr--scope-and-acceptance-criteria-the-immediate-ask). Vertical&nbsp;1 follow-ups (BMM, GroupGEMM, SDPA, quantised MatMul, BRGEMM-microkernel hook) are in [§9.2](#92-follow-up-prs-vertical-1).

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

**ZenDNN delivers the best CPU performance on AMD CPUs today** &mdash; it is AMD's purpose-built primitive library for Zen3 / Zen4 / Zen5 / Zen6 and ships in production through `zentorch` and other framework integrations. Every capability below exists to extract that performance on AMD silicon; bringing it into oneDNN through the `zen64` module is what gives oneDNN consumers (PyTorch, TensorFlow, ONNX Runtime, vLLM, &hellip;) the same AMD-CPU performance they would otherwise have to reach for `zentorch` to get.

The capabilities below are what differentiates ZenDNN's AMD-tuned path from a stock JIT-only impl on Zen-class CPUs.

### 3.1 Multi-backend Auto Tuner (TBP + Decision Tree)

ZenDNN's MatMul / BMM / GEMV path runs an internal Auto Tuner that picks between several candidate backends per problem size: native ZenDNN kernels, **oneDNN**, AOCL-DLP, libxsmm, and FBGEMM. The presence of oneDNN itself in the candidate set is important &mdash; for shapes where oneDNN's existing JIT path is already best on Zen, the Auto Tuner picks oneDNN, so the `zen64` module is never a regression even when ZenDNN's other backends would lose to oneDNN. Selection is driven by:
- **Time-Based Profiling (TBP)** &mdash; the first call for a given shape probes a small set of candidates and caches the winner.
- **Decision Tree** &mdash; static heuristics (problem dimensions, dtype, layout) that resolve without profiling for shapes the tree already covers.

Vertical&nbsp;1 makes this Auto Tuner the dispatch core inside the `zen64::matmul_t::execute()` body; oneDNN sees a single primitive that picks the best backend per call. When the Auto Tuner picks "oneDNN" as its inner backend, `zen64::matmul_t` cleanly returns `status::unimplemented` from PD `init()` for that shape so that oneDNN's own dispatcher continues with `brgemm_matmul_t` &mdash; avoiding any double-dispatch and keeping the JIT path hot.

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

Critical for small-shape GEMM-heavy workloads &mdash; SDPA, BMM, low-batch decode &mdash; where API overhead is a measurable fraction of kernel time. Vertical&nbsp;1 surfaces LOWOHA at the `zen64::execute()` boundary; a broader oneDNN-side hook into BRGEMM microkernel APIs is a follow-up deliverable.

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
| Quantized MatMul (WoQ / static / dynamic) | Partial via `attr.scales` / `attr.zero_points` | 1 + 2 | Backend in V1; new oneDNN API for *Dynamic* Quant in V2 (separate RFC) |
| BRGEMM microkernel | Yes | 1 (LOWOHA hook) | Existing microkernel API; ZenDNN plugs in a low-overhead path |
| **Embedding (+ Bag)** | **No** | **2** | New primitive entirely &mdash; native AVX-512 intrinsic kernel; companion RFC |
| **Dynamic Quant (compute scale + quantize)** | **No standalone primitive** | **2** | New primitive + companion API extension; separate RFC |
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
| `DNNL_ZENDNN_BMM` | `1` | Per-primitive override for BMM (added in follow-up). |
| `DNNL_ZENDNN_SDPA` | `1` | Per-primitive override for SDPA (added in follow-up). |

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

The first production PR ships Strategy A. Strategy B is the in-progress next step ("Using OneDNN Reorder ahead of time, followed by ZenDNN MatMul") and is the target for the BMM / GroupGEMM / SDPA follow-ups.

For cases where the user-supplied weight has a layout ZenDNN cannot consume directly and where ahead-of-time reorder is not possible, the impl rejects in `pd_t::init()` and the standard oneDNN path runs.

### 5.8 No public API changes

Vertical&nbsp;1 introduces:
- One CMake option (`DNNL_ENABLE_ZENDNN`, default OFF).
- One optional secondary option (`DNNL_ENABLE_ZENDNN_STATIC`, default OFF).
- A handful of env vars (`DNNL_ENABLE_ZENDNN`, `DNNL_ZENDNN_<PRIM>`).
- New impl classes in `src/cpu/x64/zen64/` and one extra entry in each existing `cpu_<prim>_list.cpp`.

It introduces **no** new headers in `include/oneapi/dnnl/`, no new C entry points, no new C++ classes, no new `dnnl_*_t` enum values. A user program built against today's `libdnnl.so` continues to link and run unchanged when `libdnnl.so` is rebuilt with `DNNL_ENABLE_ZENDNN=ON`.

### 5.9 Module file layout

The `zen64` sub-tree under `src/cpu/x64/zen64/` is small and follows oneDNN's per-primitive file conventions, so reviewers can map each file to a concept they already know:

```
src/cpu/x64/zen64/
├── zen64_common.hpp           runtime-gating helpers, vendor / uArch / ISA detectors
├── zen64_common.cpp
├── zen64_status.hpp           ZenDNN status -> oneDNN status mapping (translate_status)
├── zen64_status.cpp
├── zen64_md_translate.hpp     memory_desc_t <-> ZenDNN layout descriptor translation
├── zen64_md_translate.cpp
├── zen64_matmul.hpp           zen64::matmul_t : public primitive_t, with nested pd_t
├── zen64_matmul.cpp           pd_t::init() staged checks + execute() body
├── zen64_reorder.hpp          zen64::reorder_t (handles ZenDNN-preferred wei layouts)
└── zen64_reorder.cpp
```

The split keeps cross-cutting code (env-var reading, vendor detection, MD translation, status mapping) out of the per-primitive `*_matmul.cpp` and `*_reorder.cpp` files, so each primitive class stays small and focused on its `init()` checks plus a one-line `execute()` body that calls into `zendnnl::lowoha::*`.

`zen64_common` is the only file that includes ZenDNN environment / CPU-detection headers; per-primitive files include just `zen64_common.hpp` and the relevant ZenDNN entry-point header (e.g. `lowoha/matmul.hpp`).

### 5.10 Verbose and status reporting

The `zen64` impls follow oneDNN's existing `onednn_verbose` convention exactly. The PoC log already shows the canonical line format:

```
onednn_verbose,v1,primitive,exec,cpu,matmul,zendnn:matmul:f32|bf16:amd,...
```

The impl-name string `zendnn:matmul:f32|bf16:amd` is produced by a small helper inside each `pd_t::name()` (analogous to `JIT_IMPL_NAME_HELPER` for JIT impls) and follows the convention `<backend>:<prim>:<dtype-mask>:<vendor>`. Every existing oneDNN tool that parses verbose logs (`benchdnn --skip-impl=...`, `parse-log` scripts, tracing dashboards) will continue to work without change &mdash; the impl-name is just one more string in the same field.

For dispatch failures, each `VDISPATCH_*` call inside `pd_t::init()` (§5.5) emits a verbose-mode reason string when `ONEDNN_VERBOSE=2` or higher is set, so reviewers diagnosing why a particular shape fell through to the native impl get a single, attributable line. PoC banner-style messages like `*********** ZenDNN INIT (AMD-only, f32|bf16) ***********` seen in the current verbose dump (§6.4) are PoC-only and will be removed in the production PR; only the standard `onednn_verbose,...` lines remain.

### 5.11 Threading model interaction

The `zen64` adapter respects oneDNN's threading runtime (OpenMP / TBB / threadpool, selected at build time) without taking ownership of it:

- `zen64::matmul_t::execute()` does not open its own parallel region. It calls `zendnnl::lowoha::matmul_direct(...)`, which internally uses ZenDNN's parallel primitive (cache-blocking, dynamic tiles, threading strategies).
- ZenDNN sees the calling thread limit through `dnnl_get_max_threads()` and through ZenDNN's `thread_guard` &mdash; this is the same convention ZenDNN's stand-alone API already follows in zentorch.
- The threadpool runtime path needs an extra check that ZenDNN's parallel implementation does not assume an OpenMP runtime; if there is a mismatch on a given build (e.g. oneDNN built with threadpool, ZenDNN with OMP), `pd_t::init()` rejects on a runtime-compatibility check and the native impl runs.

No new public threading knob is exposed; users tune threading exactly as they do today (`dnnl_set_max_threads`, `OMP_NUM_THREADS`, threadpool API).

### 5.12 Primitive cache and scratchpad

- **Primitive cache.** `zen64::matmul_t::pd_t` participates in oneDNN's primitive cache like every other PD; the cache key is the existing `(op_desc, attr, engine_kind)` tuple plus the `pd_t::name()` string (which already includes `zendnn:...:amd`, so a cached PD never gets reused on a non-AMD host). No changes to oneDNN's caching infrastructure are required.
- **Scratchpad.** When the chosen ZenDNN kernel needs an intermediate buffer (e.g. accumulators when running with mixed input/output dtypes, or aligned packing space), the size is booked in `pd_t::init_scratchpad()` via the existing `scratchpad_registry().registrar().book(...)` and consumed in `execute()` via `ctx.get_scratchpad_grantor().get<...>(key)`. **No allocation in `execute()`.** ZenDNN's internal LRU cache for prepacked weights (Strategy A in §5.7) sits *outside* the scratchpad; it is bounded and ZenDNN-managed, but does not appear on oneDNN's hot path beyond a pointer lookup.

### 5.13 Test and validation strategy

Three layers, each running in CI:

1. **Correctness &mdash; benchdnn against the native impl.** For every operator covered by `zen64` (PoC: MatMul, Reorder), benchdnn runs both the `zendnn:*` and the native impl on the same problem set with `--skip-impl` toggling. Results are diffed within standard oneDNN tolerance per dtype; any mismatch fails the test. This is the same gating used for any new optimised impl in oneDNN today.
2. **Fallback &mdash; dispatch verification.** A targeted test set forces unsupported configurations (non-AMD vendor on Intel host, runtime kill-switch via env var, unsupported dtype/layout/post-op combos) and asserts that:
   - `pd_t::init()` returns `status::unimplemented` with a verbose-mode reason string.
   - The next entry in `cpu_*_list.cpp` (the existing oneDNN impl) is selected.
   - The user-visible result is byte-identical to a build with `DNNL_ENABLE_ZENDNN=OFF`.
3. **Performance &mdash; benchdnn perf + model-level.** benchdnn perf runs on a representative shape set on AMD and Intel CPUs; on Intel, the requirement is **no regression** vs. the same build with `DNNL_ENABLE_ZENDNN=OFF`. Model-level numbers come from vLLM and are reported in each production-PR cover letter.

The test additions land in the same PR as the impl (no separate test-only PR), per oneDNN's contribution guidelines.

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

### 6.5 What is validated end-to-end today

The PoC has confirmed the following end-to-end behaviour against the native oneDNN MatMul path:

- **Build.** `cmake -DDNNL_ENABLE_ZENDNN=ON ...` succeeds and produces a `libdnnl.so` that links against ZenDNN; `cmake -DDNNL_ENABLE_ZENDNN=OFF ...` produces a binary byte-identical to today's stock oneDNN.
- **Dispatch on AMD.** With the build flag ON and the runtime env vars set, `dnnl::matmul` for fp32 and bf16 selects `zendnn:matmul:f32|bf16:amd` as the impl; the per-primitive `onednn_verbose` line confirms this.
- **Dispatch on Intel.** Running the same binary on an Intel CPU exercises the vendor / uArch check inside `zen64::matmul_t::pd_t::init()`, which returns `status::unimplemented`. The dispatcher advances and `brgemm_matmul_t` runs &mdash; the verbose log shows the existing impl name. No code path inside ZenDNN is invoked on Intel.
- **Runtime kill-switch.** Setting `DNNL_ENABLE_ZENDNN=0` (or `DNNL_ZENDNN_MATMUL=0`) in the same binary on the same AMD host reverts to `brgemm_matmul_t`; no rebuild needed.
- **Compute correctness.** For the shape tried in the PoC log (`M=N=K=256`, BF16, plain `ab` layout, no bias, no post-ops), the output produced by `zendnn:matmul:f32|bf16:amd` matches the output produced by `brgemm_matmul_t` within oneDNN's standard BF16 tolerance.
- **ZenDNN AutoTuner is honoured.** The PoC log line `kernel=aocl_dlp_blocked` shows ZenDNN picked the AOCL-DLP backend internally for this shape; this is invisible to oneDNN but confirms that ZenDNN's multi-backend selection is in fact reaching its decision tree, not being short-circuited.
- **Reorder Strategy A is functional.** The first call to a given weight pointer triggers ZenDNN's internal prepack; subsequent calls hit the LRU cache (`weight_address=0x7f47ff200000` is reported in the per-call profiler line).

### 6.6 What is in flight to graduate from PoC to first PR

| Item | State | Required for first PR |
|---|---|---|
| **Reorder Strategy B** &mdash; oneDNN-side reorder ahead of time | In progress | First PR includes `zen64::reorder_t` so frameworks can prepack outside the inference loop and avoid the first-call cost |
| **Fused MatMul post-ops** &mdash; bias add, eltwise, binary on the result | In progress | Required for parity with `brgemm_matmul_t` post-ops; otherwise dispatch must reject any post-op and fall through |
| **Dimension `N` constraint** &mdash; ZenDNN-side path requires `N % 64 == 0` on some shapes | Documented | Outside that constraint, `pd_t::init()` rejects and the native impl handles the shape |
| **No-matching-tag fallback** &mdash; cases where the user-supplied weight layout cannot be consumed directly by ZenDNN's chosen kernel | In progress (Strategy B partial coverage) | Either Strategy B handles via reorder, or `pd_t::init()` rejects and the native impl runs |
| **Intel CPU validation pass** &mdash; benchdnn full sweep on Intel host with `DNNL_ENABLE_ZENDNN=ON` | Pending | First-PR gate is "no regression on Intel" |
| **PoC verbose banners removed** &mdash; the `*********** ZenDNN INIT ***********` lines are PoC-only | To do | Production code emits only the standard `onednn_verbose,...` lines |

### 6.7 What we will measure for the first production PR

The PoC numbers above are useful for plumbing validation, **not** for the performance argument that the upstream PR has to make under [`CONTRIBUTING.md` Performance gate](../../../CONTRIBUTING.md#library-functionality-guidelines). For the first production PR we will publish:

- benchdnn perf runs on a representative shape set (small / medium / large MatMul, square / skinny K / skinny M, batched); `zendnn:matmul:f32|bf16:amd` vs `brgemm_matmul_t`, both on the same Zen CPU.
- Model-level numbers from vLLM running BF16 / FP32 inference on a wide model set on AMD and Intel hosts; baseline = `vLLM + zentorch`, candidate = `vLLM + oneDNN(zen64 enabled)`.
- Cold-cache and warm-cache splits so the reorder strategy (A vs B) can be evaluated independently of compute time.
- A `--skip-impl=zendnn` rerun on the same hosts to confirm the stock oneDNN path is unaffected on Intel and matches today's numbers on AMD.

Numbers go in the production-PR cover letter and in the relevant commit body.

## 7. Vertical 2: Native Upstream Contributions (high-level overview)

Vertical&nbsp;2 covers contributions that go directly into oneDNN's standard tree, with **no** runtime, link-time, or build-time dependency on ZenDNN. They do not require `DNNL_ENABLE_ZENDNN`. Where ZenDNN has an existing implementation, that code is referenced only as a porting reference; once the code is in oneDNN, ZenDNN disappears from the picture for these primitives.

**Each Vertical&nbsp;2 item below has its own RFC and PR series.** This section is intentionally short &mdash; it gives reviewers enough context to see how the items fit together without re-stating the technical detail of each. Reviewers interested in a specific item should follow the link to its dedicated RFC.

| Item | One-line scope | Companion RFC |
|---|---|---|
| **Embedding + Embedding-Bag primitive** | New oneDNN primitive (`dnnl_embedding_bag` with a `lookup` algorithm for the no-reduction case), implemented natively in `src/cpu/x64/embedding_bag.{hpp,cpp}` as AVX-512 / AVX-512-FP16 intrinsics. No JIT, no Xbyak. Includes Group Embedding Bag (Grouped Memory Format) for variable-size batching as a follow-up. | [`doc/rfcs/embedding_bag/`](../embedding_bag/README.md) (drafted on this branch) |
| **Dynamic Quantization support** | New oneDNN primitive (or attribute-mode extension) for fused scale + zero-point computation followed by quantize. Symmetric / asymmetric / group quantization. Fused AVX-512 intrinsic kernel, multiple threading strategies. Requires an API extension because today's `attr.scales` / `attr.zero_points` is caller-supplied; Dynamic Quant has the library compute scales / zps from input data. | `doc/rfcs/dynamic_quant/` (separate RFC, planned) |
| **AMD-CPU MatMul / BMM kernel optimisations** | Tiling and threading heuristics inside the existing oneDNN MatMul / BMM JIT kernels, tuned for Zen-class cache hierarchies and CCD-aware threading. Lives entirely in `src/cpu/x64/matmul/`. No public API change. Verified to leave Intel-side dispatch identical. | Separate RFC (planned) |
| **AutoTuner (runtime kernel/backend selection)** | Internal pass that re-ranks `impl_list` candidates per shape on first call and caches the winner, mirroring ZenDNN's TBP + Decision Tree. Today oneDNN's impl-list ordering is static; AutoTuner adds runtime profiling. No public API change. | Separate RFC (planned) |
| **Low-Overhead API hook in BRGEMM microkernel** | Generalises ZenDNN's LOWOHA pattern (kernel-pointer dispatch, no per-call validation) into a oneDNN-internal helper, plugged into the existing BRGEMM microkernel API path. Targeted at small-shape GEMM-heavy workloads (SDPA, low-batch decode). Pairs with the Vertical&nbsp;1 BRGEMM-microkernel hook. | Separate RFC (planned) |

**Why these are Vertical&nbsp;2 and not Vertical&nbsp;1.** Each of the above is either a primitive that does not exist in oneDNN today (Embedding+Bag, Dynamic Quant) or an internal mechanism that is not visible to users (AMD-CPU kernel tuning, AutoTuner, LOWOHA hook). Wrapping them via the `zen64` mechanism would either leak ZenDNN-specific surface into oneDNN's public API (for the primitives) or solve nothing (for the internal mechanisms, which need to be in oneDNN's tree to do their job). The two-vertical rule from §4 places them in Vertical&nbsp;2.

**Status of the embedding_bag RFC.** The companion RFC at [`doc/rfcs/embedding_bag/`](../embedding_bag/README.md) is the first concrete Vertical&nbsp;2 deliverable and is already drafted on this branch. It is the canonical example of how the rest of Vertical&nbsp;2 will be authored: each item gets a self-contained RFC describing its API, internal layout, validation, testing, and phased PR series, reviewed independently of this umbrella RFC.

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

## 9. Delivery Plan (Milestone Gates)

The two verticals run in parallel with independent PR tracks. The plan is described as **gates** rather than dates &mdash; each gate has a concrete done-when criterion and the next gate cannot open until the previous one is closed. Each PR follows oneDNN's commit-message format `<scope>[: <subscope>...]: <imperative summary>` and runs `ONEDNN_TEST_SET=NIGHTLY` before merge, per [`CONTRIBUTING.md`](../../../CONTRIBUTING.md). The First PR (§9.1) is the immediate ask of this RFC; the surrounding gates (§9.3) and the Vertical&nbsp;2 work (§9.4) are tracked here for context.

### 9.1 First PR &mdash; scope and acceptance criteria (the immediate ask)

The first PR upstreamed under this RFC delivers exactly the four items below. Each line is a checkable acceptance criterion that must be green for the PR to merge.

#### 9.1.1 ZenDNN library integration in build infra

- New CMake option `DNNL_ENABLE_ZENDNN` &mdash; default `OFF` so the standard build is byte-identical to today's oneDNN. Documented in `doc/build/build_options.md`.
- Optional companion option `DNNL_ENABLE_ZENDNN_STATIC` &mdash; default `OFF` (dynamic link); flip to static for embedded / single-binary deployments. Both options are user-facing and validated in CI.
- New `cmake/FindZenDNN.cmake` module so `find_package(ZenDNN)` resolves a system-installed or vendored ZenDNN; ZenDNN version pinned to a known-good commit range with a clear error message on mismatch.
- Build matrix gated to Linux x86_64 only when the option is `ON`; macOS / Windows / Arm fail fast at configure time with an explanatory error rather than producing a half-built tree.
- **Done-when:** clean configure + build of `cmake -DDNNL_ENABLE_ZENDNN=ON` and `cmake -DDNNL_ENABLE_ZENDNN=OFF` on the supported toolchains; the OFF build's binary `libdnnl.so` is byte-identical to today's oneDNN.

#### 9.1.2 Add ZenDNN MatMul (+ fused) and Reorder (format-conversion) in the `zen64` module

- New `src/cpu/x64/zen64/` directory with the file layout from §5.9 (common helpers, status / MD translation, per-primitive adapter classes).
- New `zen64::matmul_t` registered in `src/cpu/matmul/cpu_matmul_list.cpp` ahead of the existing entries via `DNNL_X64_ONLY_IF(DNNL_ENABLE_ZENDNN, CPU_INSTANCE(zen64::matmul_t))`.
- New `zen64::reorder_t` registered in `src/cpu/reorder/cpu_reorder_list.cpp` covering the format-conversion cases ZenDNN's MatMul prefers (input layout &rarr; ZenDNN-blocked weights). Enables Strategy B (ahead-of-time reorder) so frameworks can prepack outside the inference loop.
- Standard fused MatMul post-ops supported (bias add, eltwise, simple binary on the result) at parity with `brgemm_matmul_t`; combinations ZenDNN cannot serve are rejected in `pd_t::init()` and the existing impl runs.
- **Done-when:** `dnnl::matmul` for the supported dtypes / layouts / post-ops on a Zen host shows `zendnn:matmul:f32\|bf16:amd` as the impl in `onednn_verbose`; `dnnl::reorder` for the format-conversion cases shows `zendnn:reorder:...:amd`.

#### 9.1.3 BF16 and FP32 datatypes

- BF16 path: src / wei / dst all BF16; bias optional FP32 per oneDNN convention; activation accumulator FP32 internal to ZenDNN.
- FP32 path: src / wei / dst all FP32; bias optional FP32.
- Both dtypes covered by benchdnn correctness sweeps against `brgemm_matmul_t` within oneDNN's standard tolerance per dtype.
- Other dtypes (FP16, INT8, INT4, etc.) explicitly out of scope for this PR &mdash; `pd_t::init()` rejects them so the existing impl handles those cases unchanged. They are picked up in §9.2.
- **Done-when:** benchdnn diff vs. `--skip-impl=zendnn` is within tolerance on the published BF16 and FP32 shape sets; verbose log confirms `zen64` is the impl reached on the supported dtypes only.

#### 9.1.4 Fallback to the native oneDNN kernel for non-supported features

- Staged `VDISPATCH_*` checks in `zen64::matmul_t::pd_t::init()` (§5.5) cover: runtime kill-switch &rarr; vendor / uArch / ISA &rarr; shape / dtype / layout / post-op / attribute &rarr; format-default + scratchpad. The first failing stage returns `status::unimplemented` with a verbose-mode reason string.
- `primitive_desc_iterator_t::operator++` advances on `unimplemented`, so any unsupported configuration silently exercises the existing oneDNN impl. **No silent failures, no compute regressions.**
- A targeted dispatch-test set (part of this PR) forces unsupported configurations and asserts the next entry runs.
- **Done-when:** the dispatch-test set passes; running the same binary with `DNNL_ENABLE_ZENDNN=0` (runtime kill-switch) on the same Zen host produces verbose logs identical to a build with `DNNL_ENABLE_ZENDNN=OFF`.

#### 9.1.5 Validation with AMD and Intel CPUs

- **AMD validation.** benchdnn correctness + perf sweeps on a Zen-class host with the build flag ON, asserting `zen64` is the selected impl on supported configurations and reporting per-shape speed-up vs `brgemm_matmul_t`.
- **Intel validation (no regression gate).** Same binary on an Intel host: benchdnn full sweep with `DNNL_ENABLE_ZENDNN=ON` must produce identical impl selection (`brgemm_matmul_t` etc.) and identical results to a `DNNL_ENABLE_ZENDNN=OFF` build. This is a hard merge gate.
- **Model-level validation.** vLLM running BF16 / FP32 inference on a representative LLM set; candidate = `vLLM + oneDNN(zen64 enabled)`, baseline = `vLLM + zentorch` (existing AMD path) and `vLLM + stock oneDNN` (current Intel-tuned path).
- **CI matrix update.** Add a Zen-host CI lane and an Intel-host-with-flag-on lane to the upstream CI; both must be green before merge.
- **Done-when:** all three validation reports are attached to the PR cover letter; the CI matrix update lands in the same PR; reviewers can see the impl name in every verbose log line attached to the perf evidence.

### 9.2 Follow-up PRs (Vertical 1)

After the First PR lands, Vertical&nbsp;1 continues with one PR per area below, each carrying its own RFC delta or scoped design note. None of these are part of the First PR's review surface.

- **Quantization support** &mdash; Weight-only Quantization (WoQ), Static Quant (per-tensor / per-channel), Dynamic Quant. The static / WoQ paths use oneDNN's existing `attr.scales` / `attr.zero_points`. The Dynamic Quant path depends on the Vertical&nbsp;2 Dynamic-Quant API extension landing first.
- **Other operator support** &mdash; one PR per operator:
    - **BMM** &mdash; the batched MatMul path (uses oneDNN's `matmul` with a batch dim today; ZenDNN registers a separate impl entry for the batched dispatch).
    - **GroupGEMM** &mdash; through ZenDNN's Grouped Memory Format for variable-size batching (no padding overhead).
    - **SDPA** &mdash; the existing `dnnl_sdpa` primitive, with ZenDNN's AMD-tuned attention dispatch.
    - **Reorder (Quant / DeQuant)** &mdash; quant / dequant variants of the format-conversion path opened in the First PR (§9.1.2).
- **ZenDNN LOWOHA integration in BRGEMM microkernel** &mdash; opens a low-overhead path through oneDNN's BRGEMM microkernel API, paired with the Vertical&nbsp;2 LOWOHA hook (§7).

Each follow-up PR is its own design note + impl + benchdnn perf + model-level evidence, reviewed independently.

### 9.3 Vertical 1 milestone gates (context for the First PR)

The First PR sits in the middle of a longer chain of milestones. The full set is tracked here so reviewers can see what came before and what's planned after.

| Gate | Scope | Done-when |
|---|---|---|
| **G1.1 &mdash; PoC** *(done)* | MatMul (+ fused) with Reorder, BF16 + FP32, Strategy A (ZenDNN-internal cache); end-to-end on AMD; gating verified on Intel | Verbose log shows `zendnn:matmul:f32\|bf16:amd` impl on Zen, falls through to `brgemm_matmul_t` on Intel |
| **G1.2 &mdash; Reorder Strategy B** *(in progress)* | oneDNN-side reorder ahead of time, then ZenDNN MatMul on already-packed weights | Strategy B variant working on the same shapes G1.1 covers, with measurable amortisation on warm-cache runs |
| **G1.3 &mdash; vLLM framework validation** *(in progress)* | `vLLM + oneDNN(zen64 enabled)` on production-relevant LLM workloads; baseline = `vLLM + zentorch` | Quantified perf lift on a wide model set on AMD hosts; **no regression on Intel hosts** |
| **G1.4 &mdash; Upstream RFC alignment** | This document submitted to `uxlfoundation/oneDNN` `rfcs` branch alongside the companion RFC | Formal sign-off from oneDNN maintainers on the build option, dispatch mechanism, fallback semantics, and reorder strategy |
| **G1.5 &mdash; First PR** | The exact scope spelled out in §9.1 | All §9.1 done-when criteria green; CI green on `uxlfoundation/oneDNN` `main` |
| **G1.6 &mdash; Follow-up: Quantized MatMul** | WoQ + Static via `attr.scales` / `attr.zero_points`; Dynamic Quant integration with the V2 Dynamic-Quant primitive | Quantized inference path validated end-to-end |
| **G1.7 &mdash; Follow-up: Other operators** | BMM, GroupGEMM, SDPA, Reorder (Quant/DeQuant) per §9.2 | Each operator is a separate PR with its own benchdnn evidence |
| **G1.8 &mdash; Follow-up: BRGEMM microkernel hook** | ZenDNN LOWOHA integration in oneDNN's BRGEMM microkernel API path | Small-shape GEMM-heavy workloads (SDPA, low-batch decode) show measurable improvement |

### 9.4 Vertical 2 milestones (per companion RFC)

Vertical&nbsp;2 items are scoped, planned, and delivered through their own per-item RFCs. This umbrella RFC tracks only the umbrella commitment that each item ships:

| Item | Scope reference |
|---|---|
| Embedding + Embedding-Bag primitive | [`doc/rfcs/embedding_bag/`](../embedding_bag/README.md) (drafted on this branch); first PR adds API + ref impl + native intrinsic impl, follow-up adds Group Embedding Bag and additional dtypes |
| Dynamic Quantization | Separate RFC (planned) covering the API extension, the new primitive, the fused intrinsic kernel, and the integration with quantized MatMul |
| AMD-CPU MatMul / BMM kernel tuning | Separate RFC (planned) covering Zen-tuned tiling and threading heuristics inside existing oneDNN JIT kernels |
| AutoTuner | Separate RFC (planned) covering runtime kernel/backend selection across `impl_list` candidates |
| Low-Overhead APIs in BRGEMM microkernel | Separate RFC (planned), paired with V1 G1.8 |

The gating between Vertical&nbsp;1 and Vertical&nbsp;2 is loose: V2 items can begin their RFC and PR cycle independently of V1 progress. The only hard cross-vertical dependency is between V1 G1.6 (Quantized MatMul follow-up) and the Dynamic Quantization RFC's API extension &mdash; the V1 follow-up cannot ship a Dynamic-Quant code path until the V2 API extension is in.

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

**For reviewers reading both RFCs:** start here for strategy and context, then read the embedding_bag RFC for the technical design of that specific primitive. The two RFCs together describe the architectural direction of AMD's contribution to oneDNN; subsequent Vertical&nbsp;2 RFCs (Dynamic Quant, AMD-CPU MatMul/BMM tuning, AutoTuner, Low-Overhead APIs) follow the same shape as the embedding_bag RFC.

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
- (Planned) `doc/rfcs/dynamic_quant/README.md` &mdash; Dynamic Quant API + primitive.

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

