# RFC: ZenDNN-backed Grouped MatMul (MoE) in oneDNN

## Authors
- AMD ZenDNN team

## Introduction

This RFC adds an optimized CPU implementation of grouped matmul (the
Mixture-of-Experts / variable-`M` batched GEMM) backed by ZenDNN, registered
behind oneDNN's existing matmul primitive API.

Grouped matmul is already reachable through a public API: `dnnl::matmul` with
grouped memory descriptors (`memory::desc::grouped`, gated by
`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY`). What is missing is an optimized CPU
kernel. On CPU the only implementation today is the correctness-oriented
reference (`ref_grouped_t`); on GPU there is an optimized Intel path
(`grouped_micro_gemm_t`). A framework that routes an MoE layer to grouped matmul
on CPU therefore gets correctness but not performance.

The first PR adds `zen_grouped_matmul_t`, a ZenDNN-backed CPU implementation
registered ahead of `ref_grouped_t` in the matmul implementation list. It routes
each expert through ZenDNN's `group_matmul_direct`, declines configurations it does not support, and keeps the reference as a guaranteed fall-through. There are no public API changes.

## 1. Motivation

Mixture-of-Experts FFNs are now standard in large models (Mixtral, DeepSeek,
Qwen-MoE). Their core compute is a grouped matmul: many independent GEMMs that
share `K`/`N` but have a different, routing-dependent `M` per expert. oneDNN
exposes this through grouped memory on `dnnl::matmul`, but on CPU only the
reference kernel exists, which walks experts and elements scalar-wise.

The reason to own an optimized CPU path is performance. AMD `zentorch` today
routes MoE grouped GEMM through ZenDNN's tuned `group_matmul_direct` outside of
oneDNN for exactly this reason. Registering that kernel as a CPU implementation
behind the existing grouped-matmul API gives every oneDNN consumer an optimized
AMD-CPU MoE path with no framework-side integration change and no new public
surface. This mirrors how ACL plugs in on AArch64 and how the `zen64` module
plugs in for regular matmul and reorder.

The value proposition is a measurable x86 MoE speedup over the current CPU
grouped path (reference), delivered through ZenDNN's per-expert kernels.
Benchmark evidence is to be supplied with the PR (see §8).

## 2. Non-Goals

- Do not replace `ref_grouped_t`; it stays as the portable fallback and the
  coverage for configurations ZenDNN declines.
- Do not add a new public API or new primitive; this is an implementation behind
  the existing grouped matmul, and the grouped-memory API is unchanged.
- Do not add f16, quantization (scales / zero-points / WoQ), fused gated
  activations, or MoE weighted-reduce in the first PR; these are declined at
  dispatch and left to follow-ups.
- Do not change the Intel GPU grouped path (`grouped_micro_gemm_t`).

## 3. Proposal - ZenDNN-backed grouped matmul

The design reuses the existing grouped-matmul API and the `zen64` module. It adds
one CPU implementation-list entry and the CPU kernel; consumers keep calling
`dnnl::matmul` with grouped memory and are unaffected when the module is off.

### 3.1 Architecture overview

```
   framework  (PyTorch · Zentorch · vLLM · …)
                                  │  MoE layer → dnnl::matmul (grouped memory)
                                  ▼
┌───────────────────────── oneDNN Library ──────────────────────────┐
│                                                                   │
│  Primitive APIs   dnnl::matmul (grouped)                          │
│  Engines          CPU · GPU · XPU · Graph                         │
│  Architectures    x64 · aarch64 · riscv64 · ppc64 · s390x         │
│                              |                                    │
│                      matmul impl_list                             │
│                              |                                    │
│         ╔═══════════════════════════════════════════════╗         │
│         ║ zen_grouped_matmul_t   (NEW, opt-in)          ║         │
│         ║   src/cpu/x64/zen64/matmul/                   ║         │
│         ║                                               ║         │
│         ║   build:    DNNL_X64_USE_ZEN=ON               ║         │
│         ║             + EXPERIMENTAL_GROUPED_MEMORY     ║         │
│         ║   runtime:  AMD vendor + AVX-512              ║         │
│         ║                                               ║         │
│         ║   • registered ahead of ref_grouped_t         ║         │
│         ║   • PD::init() validation gate                ║         │
│         ║   • grouped src/dst · dense 3D [G,K,N]        ║         │
│         ║   • f32 / bf16 / bf16→f32 · post-ops          ║         │
│         ╚═══════════════════════╤═══════════════════════╝         │
│                      ┌──────────┴──────────┐                      │
│                      │ success             │ unimplemented        │
│                      |                     ▼                      │
│                      |             ┌───────────────┐              │
│                      │             │ ref_grouped_t │              │
│                      │             │  (reference)  │              │
│                      │             └───────────────┘              │
└──────────────────────┼────────────────────────────────────────────┘
                       │  group_matmul_direct(…) per expert
                       ▼
             ┌──────────────────────────┐
             │      ZenDNN library      │
             │  (linked when build      │
             │   flag is ON, default    │
             │   OFF)                   │
             └──────────────────────────┘
```

Gating is build-time (`DNNL_X64_USE_ZEN=ON`, default OFF) plus the existing
`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY`. Runtime dispatch is decided in
`pd_t::init()` by CPU detection (AMD gate), with no environment variable.

### 3.2 Operation semantics

For `g = 0 .. G-1`, with per-expert row count `M_g` derived from the offsets:

```
dst_g[M_g, N] = src_g[M_g, K] · W_g[K, N]  (+ bias_g[N])  (+ post-ops)
```

`src`/`dst` are slices of the concatenated buffers; `W_g` is `weights[g]`.

### 3.3 CPU registration and kernel

`zen_grouped_matmul_t` is added under
`src/cpu/x64/zen64/matmul/zen_grouped_matmul.{hpp,cpp}` (namespace
`dnnl::impl::cpu::x64::zen::matmul`), including ZenDNN headers under
`#if DNNL_X64_USE_ZEN`, following the existing `zen_matmul_t` convention. A new
macro `CPU_INSTANCE_X64_ZEN_GROUPED(...)` in `src/cpu/cpu_engine.hpp` is
`CPU_INSTANCE_X64_ZEN(...)` additionally gated on
`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY`, and it is registered in
`src/cpu/matmul/cpu_matmul_list.cpp` immediately before
`CPU_INSTANCE_GROUPED(ref_grouped_t)`. Build wiring reuses the `zen64` module's
`cmake/ZenDNN.cmake`; `zen64` sources compile with ZenDNN include dirs scoped
per-target, with no global include changes. Unsupported cases return
`unimplemented` so dispatch falls through to `ref_grouped_t`.

### 3.4 Data-model bridge

oneDNN stores grouped tensors concatenated with offsets; ZenDNN's
`group_matmul_direct` takes per-group pointer/shape vectors. `execute()`:

- reads `src` values (buffer 0) and `s32` offsets (buffer 1), `dst` values and
  offsets, dense weights, and optional bias;
- resolves weight orientation from strides: `abc` → `transB=false, ldb=N`;
  `acb` (stored `[N,K]`) → `transB=true, ldb=K`;
- for each expert `g`, computes `M_g = offsets[g] - offsets[g-1]` and base
  pointers into the concatenated buffers, skipping empty experts (`M_g == 0`);
- fills the per-group vectors and calls `group_matmul_direct` once.

### 3.5 Data-type and post-op support

Aligned with `zen_matmul_t`: uniform f32, uniform bf16, and bf16→f32 (bf16
src/wei, f32 dst). f16 is excluded (requires AVX-512-FP16, absent on the Zen
target); all other dtypes (int8 / WoQ) are declined.

Supported post-ops, validated per entry (unsupported chains fall back to the
reference):

- eltwise: relu, gelu_tanh, gelu_erf, tanh, sigmoid (logistic), swish;
- binary: add, mul with concrete `[1, N]` src1 (broadcast over `M`);
- sum, mapped to ZenDNN `beta` accumulation.

The chain is pre-built once per primitive (mirroring `zen_matmul_t`'s
`postop_template_`); binary buffer pointers are patched per call.

## 4. PoC - end-to-end and accuracy

The path was validated end-to-end via `dnnl::matmul` with grouped memory,
dispatching to ZenDNN on AMD x86 with the module enabled.

### 4.1 Verbose evidence

```
onednn_verbose,v1,primitive,exec,cpu,matmul,zen:grouped:f32|bf16:amd,undef,
  src:f32::sparse:grouped::f0 wei:f32::blocked:abc::f0 dst:f32::sparse:grouped::f0,
  attr-post-ops:eltwise_relu,,32x64:4x64x32, ...
```

`zen:grouped:f32|bf16:amd` confirms the ZenDNN impl ran; the grouped `src`/`dst`,
dense `abc` weights, and the applied post-op are visible.

### 4.2 Accuracy
TODO

## 5. Framework-Side Changes

None required beyond what already routes to grouped matmul. A framework that
issues an MoE layer as `dnnl::matmul` with grouped memory automatically gets the
ZenDNN path when the module is enabled on AMD x86, and the reference otherwise.
Applications need no source changes.

## 6. First PR - Scope and Acceptance Criteria

The first PR is complete when the following are in place and passing on CPU:

- `zen_grouped_matmul_t` and the `CPU_INSTANCE_X64_ZEN_GROUPED` registration are
  in place ahead of `ref_grouped_t`; with `DNNL_X64_USE_ZEN=ON` and
  `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON` on AMD, a grouped f32/bf16 matmul
  appears in verbose as `zen:grouped:...`.
- Coverage of f32 / bf16 / bf16→f32, `abc`/`acb` weights, per-expert bias, and
  the eltwise/binary/sum post-ops of §3.5 passes benchdnn correctness across the
  sweep in §4.2.
- The default build (`DNNL_X64_USE_ZEN=OFF`) is unaffected with no public header
  changes; existing `test_iface_grouped` and the grouped benchdnn inputs pass
  with the module off.
- gtests cover the Zen grouped path (guarded, AMD-only, single-threaded),
  comparing against an in-test reference and asserting the impl name.

Follow-up work includes multi-threaded enablement, benchmark-driven perf
validation, f16, quantization, and fused gated activations.

## 7. Alternatives Considered

- Keep only `ref_grouped_t` on CPU. No new backend and simplest, but no optimized
  CPU MoE path; frameworks keep integrating ZenDNN separately (today's state).
- Serve the grouped problem with regular matmul (runtime dimensions) or BRGEMM
  microkernels per expert.
- Register a ZenDNN-backed grouped impl behind the existing API (this RFC). The
  smallest change that gives an optimized AMD-CPU MoE path with no public-API or
  framework changes, reusing the `zen64` module with the reference as fallback.

## 8. Open Questions

- Gated activations: ZenDNN offers fused gated activations (`silu_and_mul` /
  `gelu_and_mul` / `swiglu`), but oneDNN's investigation favors extending binary
  post-op with in-place multiply over a dedicated gated attribute. Follow-up
  should align with that direction.
