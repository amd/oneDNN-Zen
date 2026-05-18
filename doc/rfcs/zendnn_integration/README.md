# RFC: ZenDNN Integration in oneDNN &mdash; `zen64` Module

## Status
**Draft** &nbsp;|&nbsp; AMD-Zenai/oneDNN-ZenDNN &nbsp;|&nbsp; Branch: `rfc/zendnn-integration`

## Authors
- AMD ZenDNN team

## Summary

This RFC proposes registering ZenDNN as a CPU backend in oneDNN through a new opt-in module under `src/cpu/x64/zen64/`. Build-time gating: `DNNL_ENABLE_ZENDNN=ON` (default OFF). Runtime gating: `DNNL_ENABLE_ZENDNN` plus per-primitive env vars. ZenDNN registers as one entry in oneDNN's existing per-primitive `impl_list` and is selected on AMD CPUs (Zen4 / Zen5 / Zen6) with the right ISA features; on every other system, or for any unsupported configuration, the existing oneDNN impls run through the standard `status::unimplemented` fall-through. **No public API changes.** A working PoC of MatMul fp32 / bf16 with Reorder is already running locally; verbose-log evidence is reproduced in §5.

The motivation is straightforward: **ZenDNN delivers the best CPU performance on AMD CPUs today** and ships in production through `zentorch`. Bringing it into oneDNN through this module is what gives every oneDNN consumer (PyTorch, TensorFlow, ONNX Runtime, vLLM, &hellip;) the same AMD-CPU performance they would otherwise have to integrate `zentorch` separately to get.

---

> ## :pushpin: First PR &mdash; the immediate ask
>
> Once this RFC is accepted, the **first** PR to upstream `uxlfoundation/oneDNN` carries exactly the following scope. Everything else is follow-up.
>
> 1. **ZenDNN library integration in build infra** &mdash; new `DNNL_ENABLE_ZENDNN` CMake option (default OFF), `find_package(ZenDNN)` plumbing, optional `DNNL_ENABLE_ZENDNN_STATIC` for link-mode control. No effect on the default build.
> 2. **Add ZenDNN MatMul (+ fused) and Reorder (for format conversion) in the `zen64` module** &mdash; new `src/cpu/x64/zen64/{zen64_matmul,zen64_reorder}.{hpp,cpp}` registered ahead of the existing entries.
>     - **BF16 and FP32 datatypes** &mdash; the two dtypes covering the broadest LLM inference workloads.
>     - **Fallback to the native oneDNN kernel for non-supported features** &mdash; staged `VDISPATCH_*` checks return `status::unimplemented` cleanly so the dispatcher falls through to `brgemm_matmul_t` / existing reorder impls.
>     - **Validation with AMD and Intel CPUs** &mdash; benchdnn correctness + perf evidence on Zen, no-regression on Intel as a hard merge gate.
>
> Detailed acceptance criteria are in [§7](#7-first-pr--scope-and-acceptance-criteria).

---

## 1. Motivation

oneDNN is the de-facto CPU primitive library that frameworks (PyTorch, TensorFlow, ONNX Runtime, vLLM) depend on. On Intel CPUs it is well-tuned; on AMD CPUs (Zen3 / Zen4 / Zen5 / Zen6) the same primitives leave performance on the table when kernel selection, memory layout, or threading don't match Zen-microarchitecture preferences. AMD's ZenDNN library closes that gap and ships in production today through `zentorch`.

**Problem.** Every framework that wants competitive AMD-CPU performance must consume ZenDNN directly, bypassing oneDNN. That fragments the integration story &mdash; improvements reach each framework on its own cadence, and frameworks that consume only stock oneDNN never see them.

**Proposal.** Register ZenDNN as a runtime-selectable CPU backend (the `zen64` module) for primitives oneDNN already provides (MatMul, BMM, SDPA, Reorder, GroupGEMM). This gives every oneDNN consumer the same AMD-tuned path under existing primitive APIs &mdash; no framework-side change required.

The contribution must satisfy the three [Library Functionality Guidelines](../../../CONTRIBUTING.md#library-functionality-guidelines):

| Criterion | How |
|---|---|
| **Performance** | The PoC already brings ZenDNN's optimised dispatch under oneDNN's MatMul primitive. Performance is validated at model level via vLLM and benchdnn before each production PR. |
| **Generality** | Works through oneDNN's standard primitive API. PyTorch, TensorFlow, ONNX Runtime, vLLM &mdash; every consumer that already calls oneDNN benefits with no integration work. |
| **Complexity** | High-quality CPU MatMul / BMM / SDPA kernels are several engineer-years of work. Centralising them in oneDNN avoids each framework re-implementing them. |

## 2. Goals and Non-Goals

### Goals
- **No public API change.** Invisible to users of `dnnl::matmul`, `dnnl::reorder`, etc.
- **No mandatory build-time dependency.** Gated by `DNNL_ENABLE_ZENDNN` (default OFF). A standard `cmake` build with no extra flags produces a oneDNN that is byte-for-byte unchanged from today.
- **Graceful fallback.** Any time the `zen64` impl rejects a configuration, the existing `impl_list` walks to the next candidate and stock oneDNN runs.
- **Validation evidence in every PR.** benchdnn correctness, benchdnn perf, and a representative model-level run (vLLM for LLM workloads).

### Non-Goals
- **GPU paths.** `zen64` is CPU-only.
- **Replacing oneDNN's existing CPU kernels.** The `zen64` impl only *registers ahead* of the existing kernels in the impl list on AMD systems; it does not delete or modify any existing kernel.

## 3. ZenDNN Value Add (Background)

**ZenDNN gets its best CPU performance on AMD CPUs from AMD's AOCL-DLP library, which is optimised for Zen microarchitecture.** AOCL-DLP provides hand-tuned compute kernels (MatMul, conv, reduction) that exploit Zen-specific features (FMA throughput, cache hierarchy, AVX-512 BF16 / FP16 lanes) more effectively than a generic JIT path can. ZenDNN composes AOCL-DLP with its own dispatch, parallel-primitive, and low-overhead-API machinery to deliver production CPU performance on AMD CPUs (Zen3 / Zen4 / Zen5 / Zen6); it ships today through `zentorch`. The capabilities below are what differentiates ZenDNN's AMD-tuned path from a stock JIT-only impl on Zen-class CPUs.

### 3.1 Multi-backend Auto Tuner (TBP + Decision Tree)

ZenDNN's MatMul / BMM / GEMV path runs an internal Auto Tuner that picks between several candidate backends per problem size: native ZenDNN kernels, **oneDNN**, AOCL-DLP, libxsmm, and FBGEMM. The presence of oneDNN itself in the candidate set is important &mdash; for shapes where oneDNN's existing JIT path is already best on Zen, the Auto Tuner picks oneDNN, so the `zen64` module is never a regression. Selection is driven by:

- **Time-Based Profiling (TBP)** &mdash; the first call for a given shape probes a small set of candidates and caches the winner.
- **Decision Tree** &mdash; static heuristics resolve without profiling for shapes the tree already covers.

When the Auto Tuner picks "oneDNN" as its inner backend, `zen64::matmul_t` cleanly returns `status::unimplemented` from PD `init()` for that shape so oneDNN's own dispatcher continues with `brgemm_matmul_t` &mdash; avoiding any double-dispatch.

### 3.2 Parallel primitive

Cache blocking, dynamic tile selection, and threading strategies (Inner / Outer dimension, Batch-threaded, Hybrid CCD-aware) tuned for AMD silicon. Brought to MatMul / BMM via `zen64`.

### 3.3 Low-Overhead API (LOWOHA)

Kernel-direct surface that minimises per-call book-keeping. Critical for small-shape GEMM-heavy workloads (SDPA, BMM, low-batch decode) where API overhead is a measurable fraction of kernel time. Surfaced at the `zen64::execute()` boundary.

### 3.4 Operator coverage today

ZenDNN ships in production: MatMul (+ fused), BMM, GroupMatMul, SDPA, Reorder (Quant / Dequant / Dynamic Quant), Normalization. All are candidates for `zen64` registration over time.

## 4. Proposed Solution: `zen64` Module Design

### 4.1 Architecture overview

`zen64` is a new, opt-in CPU sub-target inside oneDNN's existing `src/cpu/x64/` tree. The diagram below shows where it sits in the library hierarchy and how it relates to the (optional, externally linked) ZenDNN library:

```
   user code  (PyTorch · TensorFlow · ONNX Runtime · vLLM · llama.cpp · …)
                                  │
                                  ▼
   ┌──────────────────────── oneDNN Library ───────────────────────────┐
   │                                                                    │
   │   Primitive APIs    dnnl::matmul · dnnl::reorder · dnnl::sdpa · …  │
   │                                  │                                 │
   │   Engines                CPU · GPU · XPU · Graph                   │
   │                                  │                                 │
   │   Architectures      x64 · aarch64 · riscv64 · ppc64 · s390x       │
   │                                  │                                 │
   │   x64 impl space     src/cpu/x64/                                  │
   │                ┌──────────────────────────────────────────────┐    │
   │                │  BRGEMM · GEMM · jit_uni_* · ref · …         │    │
   │                │                                              │    │
   │                │   ╔══════════════════════════════════════╗   │    │
   │                │   ║   zen64    (NEW, opt-in)             ║   │    │
   │                │   ║                                      ║   │    │
   │                │   ║     build:    DNNL_ENABLE_ZENDNN=ON  ║   │    │
   │                │   ║     runtime:  AMD vendor + uArch +   ║   │    │
   │                │   ║                ISA checks pass       ║   │    │
   │                │   ║                                      ║   │    │
   │                │   ║   • registers ahead in cpu_*_list    ║   │    │
   │                │   ║   • PD::init() validation gate       ║   │    │
   │                │   ║   • execute() → ZenDNN               ║   │    │
   │                │   ╚════════════════╤═════════════════════╝   │    │
   │                └────────────────────┼────────────────────────┘    │
   └─────────────────────────────────────┼──────────────────────────────┘
                                         │  zendnnl::lowoha::*_direct(…)
                                         ▼
                            ┌──────────────────────────┐
                            │      ZenDNN library      │
                            │   (linked when build     │
                            │    flag is ON; default   │
                            │    OFF)                  │
                            └──────────────────────────┘
```

The new `src/cpu/x64/zen64/` directory contributes one impl class per primitive (e.g. `zen64::matmul_t`, `zen64::reorder_t`) registered in `src/cpu/<prim>/cpu_<prim>_list.cpp` ahead of the existing entries. When the build flag is OFF (default), the entire `zen64` source set is excluded from compilation and `libdnnl.so` is byte-identical to today's oneDNN.

### 4.2 Build-time gating

```cmake
option(DNNL_ENABLE_ZENDNN
       "Build with ZenDNN as an opt-in AMD-CPU backend (Linux x86_64 only)."
       OFF)
```

When `OFF` (default): zero ZenDNN headers referenced, zero ZenDNN library linked, behaviour byte-identical to today's oneDNN.

When `ON`: `find_package(ZenDNN REQUIRED)`, compile `src/cpu/x64/zen64/*.cpp`, link `libzendnn`, conditionally register impls via `DNNL_X64_ONLY_IF(DNNL_ENABLE_ZENDNN, ...)`. A companion option `DNNL_ENABLE_ZENDNN_STATIC` (default OFF, i.e. dynamic link) controls link mode for embedded / single-binary deployments.

### 4.3 Runtime gating

| Env var | Default | Semantics |
|---|---|---|
| `DNNL_ENABLE_ZENDNN` | `1` | Master switch &mdash; `0` makes every `zen64::*_t::pd_t::init()` return `unimplemented`. |
| `DNNL_ZENDNN_MATMUL` | `1` | Per-primitive override for MatMul. |
| `DNNL_ZENDNN_REORDER` | `1` | Per-primitive override for Reorder. |

Read once during oneDNN initialisation, cached, queried inside each `pd_t::init()` before any other check. Setting any to `0` is equivalent to a non-AMD CPU from the dispatcher's point of view &mdash; the next entry in the impl list runs.

### 4.4 Per-primitive registration

Existing `cpu_<prim>_list.cpp` files gain one new entry. For MatMul:

```cpp
static constexpr impl_list_item_t impl_list[] = REG_MATMUL_P({
    DNNL_X64_ONLY_IF(DNNL_ENABLE_ZENDNN, CPU_INSTANCE(zen64::matmul_t))  // ahead of stock
    CPU_INSTANCE_X64(brgemm_matmul_t)
    // ...existing entries...
    CPU_INSTANCE(ref_matmul_t)
    nullptr,
});
```

`DNNL_X64_ONLY_IF` is a small new macro mirroring the existing `DNNL_X64_ONLY` &mdash; expands to the entry only when both `DNNL_X64` and `DNNL_ENABLE_ZENDNN` are defined.

### 4.5 PD `init()` validation flow

Each `zen64` impl's `pd_t::init()` runs a staged check. The first failing stage returns `status::unimplemented` and the impl-list iterator advances:

```cpp
status_t zen64::matmul_t::pd_t::init(engine_t *engine) {
    // Stage 1 - runtime kill-switch
    VDISPATCH_MATMUL(zen64::is_runtime_enabled("matmul"), ...);
    // Stage 2 - CPU vendor / uArch / ISA
    VDISPATCH_MATMUL(zen64::cpu_supported(),       ...);  // AuthenticAMD, Zen4+
    VDISPATCH_MATMUL(zen64::isa_supported(desc()), ...);  // BF16 / FP16 / VNNI as needed
    // Stage 3 - shapes / dtypes / layouts / attributes / post-ops
    VDISPATCH_MATMUL(zen64::shapes_supported(desc()), ...);
    VDISPATCH_MATMUL(zen64::dtypes_supported(desc()), ...);
    VDISPATCH_MATMUL(zen64::attrs_supported(attr()),  ...);
    // Stage 4 - oneDNN bookkeeping
    CHECK(set_default_formats());
    CHECK(zen64::book_scratchpad(this));
    return status::success;
}
```

### 4.6 `execute()` flow

```cpp
status_t zen64::matmul_t::execute(const exec_ctx_t &ctx) const {
    const auto *src   = CTX_IN_MEM (const char *, DNNL_ARG_SRC);
    const auto *wei   = CTX_IN_MEM (const char *, DNNL_ARG_WEIGHTS);
    const auto *bias  = CTX_IN_MEM (const char *, DNNL_ARG_BIAS);
    auto       *dst   = CTX_OUT_MEM(char *,        DNNL_ARG_DST);

    zen64::matmul_params p = pd()->kernel_params();   // resolved at PD time
    p.src = src; p.wei = wei; p.bias = bias; p.dst = dst;

    return zen64::translate_status(zendnnl::lowoha::matmul_direct(p));
}
```

Inside `matmul_direct`, ZenDNN's Auto Tuner picks the actual kernel (ZenDNN-native, AOCL-DLP, libxsmm, FBGEMM, or oneDNN), the parallel primitive picks the threading strategy and tile size, and LOWOHA avoids re-validation on hot calls. From oneDNN's dispatcher's view this is one regular primitive call.

### 4.7 Memory descriptor and reorder strategy

Two strategies, evolving across PRs:

- **Strategy A &mdash; ZenDNN-internal reorder + cache** *(PoC-current).* User calls `dnnl_matmul`; `zen64::matmul_t::execute()` hands the user-supplied weight to ZenDNN, which prepacks internally on first use and caches the prepacked tensor.
- **Strategy B &mdash; oneDNN-side reorder ahead of time** *(in progress).* The PD requests a specific weight layout; user (or framework) issues a oneDNN reorder ahead of inference; `execute()` passes already-packed weights directly. More idiomatic and cache-friendly; needed for GroupGEMM / SDPA follow-ups.

### 4.8 No public API changes

This module adds: one CMake option (default OFF), a handful of env vars, new impl classes under `src/cpu/x64/zen64/`, one extra entry per `cpu_<prim>_list.cpp`. **No new headers in `include/oneapi/dnnl/`, no new C entry points, no new C++ classes, no new enum values.** A user program built against today's `libdnnl.so` continues to link and run unchanged when `libdnnl.so` is rebuilt with `DNNL_ENABLE_ZENDNN=ON`.

## 5. PoC: MatMul fp32 / bf16 (+ Reorder)

A working PoC is running locally end-to-end through `dnnl_matmul` on a Zen-class AMD CPU.

### 5.1 Build and run

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
                    -DONEDNN_BUILD_TESTS=ON    \
                    -DDNNL_ENABLE_ZENDNN=ON
cmake --build build -j

export DNNL_ENABLE_ZENDNN=1
export DNNL_ZENDNN_MATMUL=1
export ONEDNN_VERBOSE=1

./build/tests/benchdnn/benchdnn --matmul --dt=bf16 --stag=ab --wtag=ab \
        --dtag=ab 256x256:256x256
```

### 5.2 Verbose-log evidence

```
onednn_verbose,v1,info,oneDNN v3.12.0 (commit a7c2a8e269c216e7457251e93f75b00b8dc3f797)
onednn_verbose,v1,info,cpu,runtime:OpenMP,nthr:192
onednn_verbose,v1,info,cpu,isa:Intel AVX-512 with Intel DL Boost and bfloat16 support

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

The new impl name `zendnn:matmul:f32|bf16:amd` follows oneDNN's existing convention `<engine>,<prim>,<impl_name>`. ZenDNN's internal Auto Tuner picked `kernel=aocl_dlp_blocked` for this shape (visible at `ONEDNN_VERBOSE=2`).

### 5.3 What is validated end-to-end

- **Build.** Both `DNNL_ENABLE_ZENDNN=ON` and `OFF` configure and build clean. The `OFF` binary's `libdnnl.so` is byte-identical to today's stock oneDNN.
- **Dispatch on AMD.** `zendnn:matmul:f32|bf16:amd` is the impl reached for fp32 and bf16; verbose log confirms.
- **Dispatch on Intel.** Same binary on an Intel CPU rejects in `pd_t::init()` (vendor / uArch check); `brgemm_matmul_t` runs; no ZenDNN code path invoked.
- **Runtime kill-switch.** Setting `DNNL_ENABLE_ZENDNN=0` reverts to `brgemm_matmul_t` without rebuild.
- **Correctness.** For the shape tried (M=N=K=256, BF16), output matches `brgemm_matmul_t` within oneDNN's standard BF16 tolerance.
- **AutoTuner reaching its decision tree.** Log line `kernel=aocl_dlp_blocked` confirms ZenDNN's multi-backend selection is in play.

### 5.4 What is in flight to graduate the PoC

| Item | Required for first PR |
|---|---|
| Reorder Strategy B (oneDNN-side ahead-of-time reorder) | Yes |
| Fused MatMul post-ops (bias add, eltwise, simple binary) | Yes |
| `N % 64` constraint and other shape fallbacks | Documented; rejected in `pd_t::init()` when not satisfied |
| Intel CPU benchdnn full sweep with flag ON | Yes &mdash; no-regression hard gate |
| PoC verbose banners removed in favour of `onednn_verbose` only | Yes |

## 6. End-to-End Lifecycle

```
user code (PyTorch / TF / vLLM / app)
            │
            ▼  Primitive APIs (unchanged)
   dnnl::matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr)
            │
            ▼   CASE(matmul) ⟶ get_matmul_impl_list
            │   impl_list walked in order
            ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ zen64::matmul_t::pd_t::init()  (only if DNNL_ENABLE_ZENDNN) │
   │   1. runtime kill-switch                                    │
   │   2. CPU vendor / uArch         (AuthenticAMD, Zen4+)       │
   │   3. ISA features               (AVX-512, BF16, FP16, VNNI) │
   │   4. shape / dtype / layout / attr / post-op                │
   │   ▶ success → selected                                      │
   │   ▶ unimplemented → iterator advances                       │
   └─────────────────────────────────────────────────────────────┘
            │ unimpl
            ▼
   ┌──────────────────────────────────────┐
   │ brgemm_matmul_t  (existing oneDNN)   │
   │ + other entries + ref_matmul_t       │
   └──────────────────────────────────────┘
            │ success
            ▼
   primitive_desc ready  ▶  prim.execute(stream, args)
            │
            ▼
   zen64::matmul_t::execute(ctx):
       params resolved at PD time (no re-validation)
       zendnnl::lowoha::matmul_direct(params)
            │
            ▼
   dst memory written
```

## 7. First PR &mdash; Scope and Acceptance Criteria

### 7.1 ZenDNN library integration in build infra

- New CMake option `DNNL_ENABLE_ZENDNN` (default OFF). Documented in `doc/build/build_options.md`.
- Optional `DNNL_ENABLE_ZENDNN_STATIC` (default OFF; dynamic link).
- New `cmake/FindZenDNN.cmake` module.
- Linux x86_64 only when ON; other platforms fail at configure time with an explanatory error.
- **Done-when:** OFF build's `libdnnl.so` byte-identical to today.

### 7.2 Add ZenDNN MatMul (+ fused) and Reorder in the `zen64` module

- `src/cpu/x64/zen64/` directory with common helpers, status mapping, MD translation, per-primitive classes.
- `zen64::matmul_t` registered in `cpu_matmul_list.cpp` ahead of `brgemm_matmul_t`.
- `zen64::reorder_t` registered in `cpu_reorder_list.cpp` covering ZenDNN-preferred weight layouts (enables Strategy B ahead-of-time reorder).
- Standard fused MatMul post-ops at parity with `brgemm_matmul_t`.
- **Done-when:** `dnnl::matmul` shows `zendnn:matmul:f32|bf16:amd` in verbose; reorder shows `zendnn:reorder:...:amd`.

### 7.3 BF16 and FP32 datatypes

- BF16 path: src / wei / dst all BF16.
- FP32 path: all FP32.
- benchdnn diff vs. `--skip-impl=zendnn` within tolerance on both dtypes.
- Other dtypes (FP16, INT8, INT4) explicitly out of scope; `pd_t::init()` rejects them so the existing impl handles those cases unchanged.

### 7.4 Fallback to native oneDNN kernel

- Staged `VDISPATCH_*` checks in `pd_t::init()` (§4.5).
- Targeted dispatch-test set forces unsupported configurations and asserts the next impl runs.
- **Done-when:** dispatch-test set passes; runtime kill-switch produces logs identical to a `DNNL_ENABLE_ZENDNN=OFF` build.

### 7.5 Validation with AMD and Intel CPUs

- benchdnn correctness + perf on Zen.
- benchdnn full sweep on Intel with flag ON: identical impl selection and results to OFF build (**hard merge gate**).
- vLLM model-level numbers, BF16 / FP32, on a representative LLM set.
- `--skip-impl=zendnn` rerun on the same hosts confirms no regression.

### 7.6 Follow-up PRs

After the first PR lands, follow-ups (each its own PR with its own design note):

- Quantization support (WoQ, Static, Dynamic Quant).
- BMM, GroupGEMM (Grouped Memory Format), SDPA, Reorder (Quant / DeQuant).
- ZenDNN LOWOHA integration in oneDNN's BRGEMM microkernel API.

## 8. Alternatives Considered

### A. Native upstream of every ZenDNN kernel (rejected)

Upstream every ZenDNN kernel as native oneDNN code, even for primitives oneDNN already has (MatMul, BMM, SDPA).

- **Pros:** Cleanest end-state &mdash; oneDNN owns every kernel.
- **Cons:** Years of porting work for kernels that already exist and ship in ZenDNN today. The `zen64` mechanism delivers the same end-user benefit on a much faster timeline, with the door open to gradually replacing wrapped kernels with native-upstreamed ones over time.

### B. Single big PR vs phased PR series (chosen)

Phased PR series: first PR per §7, follow-ups per §7.6. `CONTRIBUTING.md` asks for linear history and one logical concern per PR; this keeps each PR focused and reviewable.

## Appendix A. GitHub References

| Resource | URL |
|---|---|
| oneDNN upstream | https://github.com/uxlfoundation/oneDNN |
| oneDNN `rfcs` branch | https://github.com/uxlfoundation/oneDNN/tree/rfcs |
| AMD-Zenai oneDNN-ZenDNN fork (this work) | https://github.com/AMD-Zenai/oneDNN-ZenDNN |
| ZenDNN library | https://github.com/amd/ZenDNN |
| AOCL-DLP | https://github.com/amd/aocl-dlp |
| zentorch | https://github.com/amd/ZenDNN-pytorch-plugin |

## Appendix B. PoC File Touchpoints

```
CMakeLists.txt                              [M]  add DNNL_ENABLE_ZENDNN option
cmake/options.cmake                         [M]  documentation entry
cmake/FindZenDNN.cmake                      [+]  find_package module

src/cpu/cpu_engine.hpp                      [M]  add DNNL_X64_ONLY_IF macro

src/cpu/x64/zen64/                          [+]  new directory
  zen64_common.{hpp,cpp}                    [+]  is_runtime_enabled / cpu_supported / isa_supported
  zen64_status.{hpp,cpp}                    [+]  ZenDNN status -> oneDNN status mapping
  zen64_md_translate.{hpp,cpp}              [+]  memory_desc_t <-> ZenDNN layout
  zen64_matmul.{hpp,cpp}                    [+]  zen64::matmul_t : public primitive_t
  zen64_reorder.{hpp,cpp}                   [+]  zen64::reorder_t : public primitive_t

src/cpu/matmul/cpu_matmul_list.cpp          [M]  add CPU_INSTANCE(zen64::matmul_t)
src/cpu/reorder/cpu_reorder_list.cpp        [M]  add zen64::reorder_t entry

doc/build/build_options.md                  [M]  document DNNL_ENABLE_ZENDNN
doc/programming_model/zendnn_backend.md     [+]  user-facing doc

tests/benchdnn/                             [M]  --skip-impl=zendnn for A/B testing
```

The `src/cpu/x64/zen64/zen64_matmul.cpp` file is the only location where ZenDNN headers are included; its includes are guarded by `#if DNNL_X64 && DNNL_ENABLE_ZENDNN` so a build with the flag OFF compiles even with no ZenDNN install on the system.

## Appendix C. Glossary

| Term | Meaning |
|---|---|
| **`zen64`** | The new optional CPU sub-target proposed by this RFC. Lives at `src/cpu/x64/zen64/`. Gated by `DNNL_ENABLE_ZENDNN`. |
| **LOWOHA** | "Low-Overhead API," ZenDNN's kernel-direct surface (e.g. `zendnnl::lowoha::matmul_direct`). |
| **TBP** | "Time-Based Profiling," ZenDNN's per-shape on-the-first-call backend probe. |
| **DT (Decision Tree)** | The static-heuristic complement to TBP. |
| **AOCL-DLP** | AMD's optimised dense-linear-algebra primitive library; one of ZenDNN's candidate backends. |
| **CCD** | Core Complex Die (Zen). |
| **Strategy A / B** (reorder) | §4.7 &mdash; ZenDNN-internal cache (A) vs oneDNN-side ahead-of-time reorder (B). |

## Appendix D. References

- [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) &mdash; library functionality criteria, RFC process, commit-message format.
- [`CODING_STANDARDS.md`](../../../CODING_STANDARDS.md) &mdash; clang-tidy, clang-format, naming conventions.
- `src/cpu/cpu_engine.hpp` &mdash; `CPU_INSTANCE`, `CPU_INSTANCE_X64`, `DECLARE_IMPL_LIST`, `CASE`.
- `src/common/impl_registration.hpp` &mdash; `REG_<PRIM>_P` macros.
- `src/common/primitive_desc_iterator.hpp` &mdash; impl-list walking semantics (`unimplemented` &rarr; advance).
- `src/cpu/matmul/cpu_matmul_list.cpp` &mdash; the file the first PR modifies.
- ZenDNN: https://github.com/amd/ZenDNN
- vLLM (intended primary consumer for validation): https://github.com/vllm-project/vllm



