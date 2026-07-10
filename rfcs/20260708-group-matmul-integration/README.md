# RFC: ZenDNN-backed Grouped MatMul (MoE) in oneDNN

## Authors
- AMD ZenDNN team

## Introduction

This RFC adds an optimized CPU implementation of grouped matmul, the
Mixture-of-Experts (MoE) variable-`M` batched GEMM, backed by ZenDNN and
registered behind oneDNN's existing matmul primitive.

Grouped matmul already has a public entry point: `dnnl::matmul` with grouped
memory descriptors (`memory::desc::grouped`, gated by
`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY`), defined in the
[grouped-GEMM RFC](../20251203-grouped-gemm-support/README.md). What is missing
is an optimized CPU kernel: today the only CPU implementation is the reference
`ref_grouped_t`, while the optimized path (`grouped_micro_gemm_t`) exists only on
Intel GPU. A framework that routes an MoE layer through grouped matmul on CPU
therefore gets correctness but not performance.

The proposal is `zen_grouped_matmul_t`, a ZenDNN-backed CPU implementation
registered ahead of `ref_grouped_t`. It routes each expert through ZenDNN's
`group_matmul_direct`, declines configurations it does not support, and leaves
the reference as the fall-through. It reuses the opt-in `zen64` module from the
[ZenDNN integration RFC](../20260616-zendnn-integration/README.md); there are no
public API changes.

## 1. Motivation

MoE FFNs are standard in current large models (Mixtral, DeepSeek, Qwen-MoE).
Their core compute is a grouped matmul: many independent GEMMs that share `K` and
`N` but have a different, routing-dependent `M` per expert.

### Why grouped matmul helps on CPU

The benefit on CPU is not the GPU argument (fewer kernel launches, device-side
offsets). It is core occupancy: how much independent work is visible to the
scheduler at one instant. This is the hard case during decode, where `M` is tiny,
on high-core-count parts (64-128 cores).

- Running one matmul per expert exposes a single small problem at a time. During
  decode that problem is small and memory-bound; no matter how the library
  splits it, one expert cannot fill all the cores. The result is a long series of
  small, latency-bound matmuls run back to back.
- Grouped matmul processes all experts together, so at any instant there is a
  much larger pool of independent work to spread across cores and CCD/L3
  bandwidth, which is exactly what a single small expert lacks. It also amortizes
  per-call overhead and allows the gated activation to be fused (the intermediate
  is consumed in place instead of written out and read back).

The right strategy is phase-dependent: prefill has large `M`, so a series of
matmuls already saturates the cores; decode has tiny `M`, where the grouped
N-tile path wins by pooling work across experts.

### Evidence

vLLM already carries both paths on CPU: a torch loop with one matmul per expert
(`forward_torch` to `cpu_fused_moe_torch`) and a grouped-GEMM path
(`forward_grouped_gemm` to `cpu_fused_moe`) that spreads N-tiles across all
experts through a shared task counter and fuses the gated activation. It now
defaults to grouped-GEMM whenever supported and falls back to the torch loop
otherwise. That grouped path is a hand-written, framework-side kernel.

ZenDNN provides an equivalent Group GEMM op, and our measurements show its
grouped N-tile path clearly beating the series-of-matmul loop. Decode throughput
(output tokens/s) on Turin 128-core, input/output length 128:

| Model | Batch | Series of matmul | Grouped N-tile | Speedup |
|---|---|---|---|---|
| unsloth/gpt-oss-20b (BF16) | 16 | 157.57 | 198.97 | 1.26x |
| unsloth/gpt-oss-20b (BF16) | 8  | 109.52 | 134.52 | 1.23x |
| Qwen/Qwen3-30B-A3B         | 16 | 77.36  | 136.07 | 1.76x |
| Qwen/Qwen3-30B-A3B         | 8  | 58.58  | 104.49 | 1.78x |

The gain is largest on the smaller, more skewed expert workload
(Qwen3-30B-A3B, ~1.8x), which is where a series of small matmuls leaves cores
idle.

### Why this belongs in the library

- Reuse: it plugs into oneDNN's existing microkernels, scratchpad, and threading
  instead of reimplementing tiling by hand.
- One implementation, many consumers: vLLM hand-rolls this today, and llama.cpp
  and others would each need their own. A library op lets them share one tuned
  path.
- Transparent dispatch: under `zen64` it registers ahead of the reference with a
  runtime AMD/Zen4+ check and falls back everywhere else, with no API change and
  no non-AMD regression.

The ask is narrow: enable a ZenDNN-backed grouped matmul for AMD under the
ongoing `zen64` integration, using the grouped-matmul API oneDNN already exposes.
Routing the vLLM MoE op to this path, along with end-to-end validation and
cross-vendor discussion, can proceed with the vLLM community on its own timeline.

## 2. Non-Goals

- Replacing `ref_grouped_t`. It stays as the portable fallback.
- New public API or a new primitive. This is an implementation behind the
  existing grouped matmul.
- f16, quantization (scales / zero-points / weight-only), fused gated
  activations, and MoE weighted-reduce in the first PR. These are declined at
  dispatch and left to follow-ups.
- The Intel GPU grouped path (`grouped_micro_gemm_t`), which is untouched.

## 3. Proposal

The design reuses the grouped-matmul API and the `zen64` module. It adds one CPU
implementation-list entry plus the kernel; a consumer keeps calling
`dnnl::matmul` with grouped memory and is unaffected when the module is off.

### 3.1 Architecture overview

```
   framework  (PyTorch · Zentorch · vLLM · …)
                                  │  MoE layer → grouped memory descriptor
                                  ▼
┌───────────────────────── oneDNN Library ──────────────────────────┐
│                                                                   │
│  dnnl::matmul (grouped) dispatches through the matmul impl_list — │
│                                                                   │
│         ╔═══════════════════════════════════════════════╗         │
│         ║ zen_grouped_matmul_t   (NEW, opt-in)          ║         │
│         ║   src/cpu/x64/zen64/matmul/                   ║         │
│         ║                                               ║         │
│         ║   build:  DNNL_X64_USE_ZEN=ON                 ║         │
│         ║           + ONEDNN_EXPERIMENTAL_GROUPED_MEMORY║         │
│         ║   runtime:  AMD vendor + AVX-512              ║         │
│         ║                                               ║         │
│         ║   • registered ahead of ref_grouped_t         ║         │
│         ║   • PD::init() validation gate                ║         │
│         ║   • grouped src/dst · dense 3D [G,K,N]        ║         │
│         ║   • f32 / bf16 / bf16→f32 · post-ops          ║         │
│         ╚═══════════════════════╤═══════════════════════╝         │
│                      ┌──────────┴──────────┐                      │
│                      │ success             │ unimplemented        │
│                      ▼                     ▼                      │
│             group_matmul_direct    ┌───────────────┐              │
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

Gating follows the `zen64` module unchanged (see the ZenDNN integration RFC,
§3.2-3.5): build-time `DNNL_X64_USE_ZEN=ON` (default OFF), plus the existing
`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY`, and a runtime AMD + AVX-512 check inside
`pd_t::init()` with no environment variable. The only addition here is the
grouped feature flag on the registration macro.

### 3.2 Operation semantics

For `g = 0 .. G-1`, with per-expert row count `M_g` derived from the offsets:

```
dst_g[M_g, N] = src_g[M_g, K] · W_g[K, N]  (+ bias_g[N])  (+ post-ops)
```

`src` and `dst` are slices of the concatenated grouped buffers; `W_g` is
`weights[g]`.

### 3.3 CPU registration

`zen_grouped_matmul_t` lives at
`src/cpu/x64/zen64/matmul/zen_grouped_matmul.{hpp,cpp}` (namespace
`dnnl::impl::cpu::x64::zen::matmul`), including ZenDNN headers under
`#if DNNL_X64_USE_ZEN`, following the existing `zen_matmul_t`. A new macro
`CPU_INSTANCE_X64_ZEN_GROUPED(...)` in `src/cpu/cpu_engine.hpp` is
`CPU_INSTANCE_X64_ZEN(...)` additionally gated on
`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY`, registered in
`src/cpu/matmul/cpu_matmul_list.cpp` immediately before
`CPU_INSTANCE_GROUPED(ref_grouped_t)`. Build wiring reuses the module's
`cmake/ZenDNN.cmake` with no global include changes. Unsupported cases return
`status::unimplemented`, so dispatch falls through to `ref_grouped_t`.

### 3.4 Data-model bridge

oneDNN stores grouped tensors as concatenated values plus an `s32` offsets buffer
(see the grouped-GEMM RFC); ZenDNN's `group_matmul_direct` takes per-group
pointer and shape vectors. `execute()` bridges the two:

- reads `src` values and offsets, `dst` values and offsets, dense weights, and
  optional bias;
- resolves weight orientation from strides: `abc` gives `transB=false, ldb=N`;
  `acb` (stored `[N,K]`) gives `transB=true, ldb=K`;
- for each expert, computes `M_g = offsets[g] - offsets[g-1]` and the base
  pointers into the concatenated buffers, skipping empty experts (`M_g == 0`);
- fills the per-group vectors and calls `group_matmul_direct` once.

### 3.5 Data types and post-ops

Data-type coverage matches `zen_matmul_t`: uniform f32, uniform bf16, and bf16 to
f32 (bf16 src/weights, f32 dst). f16 is excluded (it needs AVX-512-FP16, absent
on the Zen target); int8 and weight-only quantization are declined.

Supported post-ops, validated per entry (an unsupported chain falls back to the
reference):

- eltwise: relu, gelu_tanh, gelu_erf, tanh, sigmoid, swish;
- binary: add, mul with a concrete `[1, N]` src1 broadcast over `M`;
- sum, mapped to ZenDNN `beta` accumulation.

The chain is built once per primitive (as in `zen_matmul_t`'s
`postop_template_`); binary buffer pointers are patched per call.

## 4. Proof of Concept and Validation

A working PoC dispatches `dnnl::matmul` with grouped memory to ZenDNN on AMD when
the module is enabled, and is validated for accuracy through benchdnn.

### 4.1 Verbose evidence

```
onednn_verbose,v1,primitive,exec,cpu,matmul,zen:grouped_matmul,undef,
  src:f32::sparse:grouped::f0 wei:f32::blocked:acb::f0 dst:f32::sparse:grouped::f0,
  ,,16x32:3x32x24,77.665
```

`zen:grouped_matmul` confirms the ZenDNN implementation ran. The grouped
`src`/`dst` and the dense `acb` weights are visible; `16x32:3x32x24` is 3 experts
sharing `K=32`, `N=24` over 16 total rows.

### 4.2 Accuracy

The path is validated across the benchdnn grouped inputs, benchdnn runs in
correctness mode, building an independent reference and comparing within dtype
tolerances; supported configurations dispatch to the ZenDNN implementation and
pass, while unsupported ones fall back to `ref_grouped_t` and pass.

## 5. Framework-Side Changes

None beyond what already routes to grouped matmul. A framework that issues an MoE
layer as `dnnl::matmul` with grouped memory gets the ZenDNN path automatically
when the module is enabled on AMD, and the reference otherwise. Applications need
no source changes.

## 6. First PR - Scope and Acceptance Criteria

The first PR is complete when the following hold on CPU:

- `zen_grouped_matmul_t` and the `CPU_INSTANCE_X64_ZEN_GROUPED` registration are
  in place ahead of `ref_grouped_t`. With `DNNL_X64_USE_ZEN=ON` and
  `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON` on AMD, a grouped f32/bf16 matmul shows
  `zen:grouped_matmul` in verbose.
- gtests coverage of f32, bf16, and bf16 to f32; `abc`/`acb` weights; per-expert bias;
  and the eltwise/binary/sum post-ops.
- The default build (`DNNL_X64_USE_ZEN=OFF`) is unaffected, with no public header
  changes; `test_iface_grouped` and the grouped benchdnn inputs pass with the
  module off.

Follow-up work: benchmark-driven perf validation, f16, quantization, and fused gated activations.

## 7. Alternatives Considered

- Keep only `ref_grouped_t` on CPU. No new backend and simplest, but no optimized
  CPU MoE path; frameworks keep integrating ZenDNN separately (today's state).
- Serve the grouped problem with regular matmul (runtime dimensions) or BRGEMM
  microkernels per expert.
- Register a ZenDNN-backed grouped impl behind the existing API (this RFC). The
  smallest change that gives an optimized AMD-CPU MoE path with no public-API or
  framework changes, reusing the `zen64` module with the reference as fallback.

## 8. Open Questions

- Gated activations: ZenDNN offers fused `silu_and_mul` / `gelu_and_mul` /
  `swiglu`, but oneDNN's investigation favors extending binary post-op with an
  in-place multiply over a dedicated attribute. Follow-up should align with that
  direction.
