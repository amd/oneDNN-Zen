# RFC: ZenDNN-backed Grouped MatMul (MoE) in oneDNN

## Authors
- AMD ZenDNN team

## Summary

This RFC proposes an **optimized CPU implementation of grouped matmul** (the
Mixture-of-Experts / variable-`M` batched GEMM) backed by **ZenDNN**, registered
behind oneDNN's **existing** matmul primitive API. It plugs into the `zen64`
module proposed in the [ZenDNN Integration RFC](../zendnn_integration/README.md)
as one more entry in the CPU matmul `impl_list`, gated by `DNNL_X64_USE_ZEN`
(default OFF) and the existing `DNNL_EXPERIMENTAL_GROUPED_MEMORY` feature flag.

Unlike SDPA, grouped matmul is **already reachable through a public API** in
oneDNN: `dnnl::matmul` with grouped memory descriptors (`memory::desc::grouped`,
gated by `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY`). What is missing is an
**optimized CPU kernel** &mdash; on CPU the only implementation today is the
correctness-oriented reference (`ref_grouped_t`). This RFC adds
`zen_grouped_matmul_t`, a ZenDNN-backed implementation that routes each expert
through ZenDNN's `group_matmul_direct` (AOCL-DLP / libxsmm kernels), with the
existing reference kept as a guaranteed fall-through.

**No public API changes.** Consumers keep calling `dnnl::matmul` with grouped
memory; on AMD x86 with the module enabled, the grouped problem dispatches to
ZenDNN, otherwise to the existing reference. A working PoC (f32 / bf16, with
per-expert bias and eltwise/binary post-ops) is running locally and validated
for accuracy through benchdnn; verbose evidence is in §5.

---

> ## First PR &mdash; the immediate ask
>
> Once this RFC is accepted, the first PR carries exactly the following scope;
> everything else is follow-up.
>
> 1. **ZenDNN grouped matmul impl** &mdash; `zen_grouped_matmul_t` in
>    `src/cpu/x64/zen64/matmul/zen_grouped_matmul.{hpp,cpp}`, registered in
>    `src/cpu/matmul/cpu_matmul_list.cpp` **ahead of** `ref_grouped_t` via a
>    `CPU_INSTANCE_X64_ZEN_GROUPED(...)` macro (gated by `DNNL_X64_USE_ZEN` and
>    `DNNL_EXPERIMENTAL_GROUPED_MEMORY`).
> 2. **Data-model bridge** &mdash; translate oneDNN's grouped encoding
>    (concatenated src/dst + `s32` offsets, dense 3D weights `[G, K, N]`) into
>    ZenDNN's per-group `group_matmul_direct` vectors.
> 3. **Coverage** &mdash; uniform f32 / bf16 and bf16&rarr;f32; `abc`/`acb`
>    weights; optional per-expert bias `[G, N]`; eltwise (relu, gelu_tanh,
>    gelu_erf, tanh, sigmoid, swish), binary (add, mul) and sum post-ops.
> 4. **Fall-through** &mdash; anything unsupported (f16, quantization,
>    unsupported post-ops) returns `status::unimplemented` and runs on
>    `ref_grouped_t`.
> 5. **Validation** &mdash; benchdnn accuracy against its reference across
>    shapes / dtypes / weight tags / post-ops.
>
> Detailed acceptance criteria are in
> [Section 7](#7-first-pr--scope-and-acceptance-criteria).

---

## 1. Motivation

Mixture-of-Experts (MoE) FFNs are now standard in large models (Mixtral,
DeepSeek, Qwen-MoE, &hellip;). Their core compute is a **grouped matmul**: many
independent GEMMs that share `K`/`N` but have a **different, routing-dependent
`M` per expert**. oneDNN exposes this through grouped memory descriptors on
`dnnl::matmul`, but on CPU only the **reference** kernel exists.

**Problem.** A framework that routes an MoE layer to oneDNN's grouped matmul on
CPU gets correctness but not performance &mdash; the reference walks experts and
elements scalar-wise. AMD `zentorch` today routes MoE grouped GEMM through
ZenDNN's own optimized `group_matmul_direct` for exactly this reason, outside of
oneDNN.

**Proposal.** Register ZenDNN's optimized grouped GEMM as a CPU implementation
behind the existing grouped-matmul API, so every oneDNN consumer gets an
optimized AMD-CPU MoE path with no framework-side integration change and no new
public surface. This mirrors how ACL plugs in on AArch64 and how the `zen64`
module plugs in for regular matmul/reorder.

- **Performance.** The value proposition is a measurable x86 MoE speedup over
  the current CPU grouped path (reference), delivered through ZenDNN's tuned
  per-expert kernels. Benchmark evidence is to be supplied with the PR (see §9).
- **Generality.** Standard primitive API; grouped src/dst + dense 3D weights,
  identical to what `ref_grouped_t` already accepts.

## 2. Goals and Non-Goals

### Goals
- **Optimized CPU grouped matmul** via ZenDNN, behind the existing
  `dnnl::matmul` + grouped-memory API. No public API change.
- **Drop-in dispatch**: registered in the matmul `impl_list` ahead of
  `ref_grouped_t`, with automatic CPU-based gating and guaranteed fall-through.
- **Coverage** for the common MoE cases: f32 / bf16 (+ bf16&rarr;f32),
  `abc`/`acb` weights, per-expert bias, and the eltwise/binary/sum post-ops
  ZenDNN can express.
- **Accuracy validation** through benchdnn against its independent reference.

### Non-Goals
- **Replacing `ref_grouped_t`.** It stays as the portable fallback and the
  coverage for configs ZenDNN declines.
- **New public API / new primitive.** This is an implementation behind the
  existing grouped matmul; the grouped-memory API is unchanged.
- **f16, quantization (scales/zero-points/WoQ), fused gated activations, MoE
  weighted-reduce** in the first PR &mdash; declined at dispatch and left to
  follow-ups (see §8, §9).
- **GPU.** The Intel GPU grouped path (`grouped_micro_gemm_t`) is untouched.

## 3. Background &mdash; How grouped matmul exists in oneDNN today

- **Public API** (experimental): `dnnl::matmul` with grouped memory. `src`/`dst`
  are grouped-encoded (`memory::desc::grouped`: buffer 0 = concatenated values,
  buffer 1 = `s32` cumulative offsets); weights are dense 3D `[G, K, N]`
  (`abc`/`acb`). `dnnl_matmul_primitive_desc_create` routes grouped problems
  through `grouped_matmul_desc_init` / `grouped_matmul_attr_check`
  (`src/common/matmul.cpp`), gated by `DNNL_EXPERIMENTAL_GROUPED_MEMORY`.
- **CPU implementation**: only `ref_grouped_t`
  (`src/cpu/matmul/ref_grouped_gemm.{hpp,cpp}`) &mdash; a reference kernel that
  loops experts with `parallel_nd` and computes each GEMM element-wise.
- **GPU implementation**: Intel `grouped_micro_gemm_t` (optimized microkernel)
  + `ref_grouped_t` fallback.

So on CPU the API is present and correct, but there is no optimized kernel. This
RFC fills that gap for AMD x86 via ZenDNN.

## 4. Proposal &mdash; ZenDNN-backed grouped matmul

### 4.1 Architecture overview

```
framework (PyTorch / zentorch / vLLM MoE)
        │  Primitive API (unchanged)
        ▼
dnnl::matmul(pd) with grouped src/dst + dense 3D weights
        │  → dnnl_matmul_primitive_desc_create
        │    → grouped_matmul_desc_init (common)   [DNNL_EXPERIMENTAL_GROUPED_MEMORY]
        ▼  walk CPU matmul impl_list
   ┌───────────────────────────────────────────────┐
   │ CPU_INSTANCE_X64_ZEN_GROUPED(zen_grouped_matmul_t)  │  ← new, tried first
   │   AMD CPU · grouped src/dst · dense 3D [G,K,N] │
   │   f32/bf16/bf16→f32 · supported post-ops       │
   └───────────────────────────────────────────────┘
        │ success                          │ unimplemented
        ▼                                  ▼
   zen_grouped_matmul_t                CPU_INSTANCE_GROUPED(ref_grouped_t)
        │  execute(): per-expert vectors → ZenDNN
        ▼
   zendnnl::lowoha::matmul::group_matmul_direct(...)   [AOCL-DLP / libxsmm]
```

Gating is build-time (`DNNL_X64_USE_ZEN=ON`, default OFF) plus the existing
`DNNL_EXPERIMENTAL_GROUPED_MEMORY`; runtime dispatch is decided inside
`pd_t::init()` by CPU detection (AMD gate), with no env var.

### 4.2 Operation semantics

For `g = 0 .. G-1`, with per-expert row count `M_g` derived from the offsets:

```
dst_g[M_g, N] = src_g[M_g, K] · W_g[K, N]  (+ bias_g[N])  (+ post-ops)
```

`src`/`dst` are slices of the concatenated buffers; `W_g` is `weights[g]`.

### 4.3 Integration approach (mirrors ACL / zen64)

- New impl `zen_grouped_matmul_t` in
  `src/cpu/x64/zen64/matmul/zen_grouped_matmul.{hpp,cpp}`, namespace
  `dnnl::impl::cpu::x64::zen::matmul`, including ZenDNN headers directly under
  `#if DNNL_X64_USE_ZEN` (same convention as the existing `zen_matmul_t`).
- New registration macro in `src/cpu/cpu_engine.hpp`:
  `CPU_INSTANCE_X64_ZEN_GROUPED(...)` = `CPU_INSTANCE_X64_ZEN(...)` gated
  additionally on `DNNL_EXPERIMENTAL_GROUPED_MEMORY`.
- Registered in `src/cpu/matmul/cpu_matmul_list.cpp` immediately **before**
  `CPU_INSTANCE_GROUPED(ref_grouped_t)`.
- Build wiring reuses the `zen64` module's `cmake/ZenDNN.cmake`
  (`DNNL_X64_USE_ZEN`, `ZENDNNROOT`, `find_package(zendnnl)`); the `zen64`
  sources are compiled with ZenDNN include dirs and C++17 scoped per-target.
  No global include changes.

### 4.4 The data-model bridge (correctness core)

oneDNN stores grouped tensors **concatenated + offsets**; ZenDNN's
`group_matmul_direct` takes **per-group pointer/shape vectors**. `execute()`:

- reads `src` values (buffer 0) + `s32` offsets (buffer 1), `dst` values +
  offsets, dense weights, optional bias;
- resolves weight orientation from strides: `abc` &rarr; `transB=false, ldb=N`;
  `acb` (stored `[N,K]`) &rarr; `transB=true, ldb=K`;
- for each expert `g`: `M_g = offsets[g] - offsets[g-1]`; computes base pointers
  into the concatenated buffers (`src + start·K`, `dst + start·N`,
  `weights + g·K·N`, `bias + g·N`); **skips empty experts** (`M_g == 0`);
- fills the per-group vectors (`M`, `src`/`wei`/`dst`/`bias` ptrs, `lda=K`,
  `ldb`, `ldc=N`, dtypes) and calls `group_matmul_direct` once (parallel mode).

### 4.5 Data-type support

Aligned with `zen_matmul_t`: uniform **f32**, uniform **bf16**, and
**bf16&rarr;f32** (bf16 src/wei, f32 dst). **f16 is intentionally excluded**
(requires AVX-512-FP16, absent on the Zen target) and falls back to the
reference. All other dtypes (int8/WoQ) are declined.

### 4.6 Post-op support

The subset ZenDNN can express, validated per entry (unsupported chains fall back
to the reference):
- **eltwise**: relu, gelu_tanh, gelu_erf, tanh, sigmoid (logistic), swish;
- **binary**: add, mul with concrete `[1, N]` src1 (broadcast over `M`);
- **sum** &rarr; mapped to ZenDNN `beta` accumulation.

The chain is pre-built once per primitive (mirroring `zen_matmul_t`'s
`postop_template_`); binary buffer pointers are patched per call.

### 4.7 No disruption to existing paths

The common grouped API, `ref_grouped_t`, and the Intel GPU grouped path are
untouched. With `DNNL_X64_USE_ZEN=OFF` (default) nothing changes. On non-AMD x86
or for unsupported configs, dispatch falls through to `ref_grouped_t`.

## 5. PoC &mdash; end-to-end and accuracy

### 5.1 Dispatch evidence (verbose)

```
onednn_verbose,v1,primitive,exec,cpu,matmul,zen:grouped:f32|bf16:amd,undef,
  src:f32::sparse:grouped::f0 wei:f32::blocked:abc::f0 dst:f32::sparse:grouped::f0,
  attr-post-ops:eltwise_relu,,32x64:4x64x32, ...
```

`zen:grouped:f32|bf16:amd` confirms the ZenDNN impl ran; the grouped `src`/`dst`
and dense `abc` weights and the applied post-op are visible.

### 5.2 Accuracy (benchdnn)

benchdnn runs in correctness mode &mdash; it builds an independent reference
(matmul + post-op chain) and compares within dtype tolerances. A representative
sweep (3 shapes incl. empty/unbalanced groups × {f32, bf16, bf16&rarr;f32, f16}
× {abc, acb} × {relu, gelu_tanh, gelu_erf, tanh, logistic, swish, binary-mul}):

```
TOTAL passed=168 failed=0 | dispatched zen=126 ref=42
```

126 problems computed by ZenDNN pass accuracy validation; the 42 f16 problems
correctly fall back to `ref_grouped_t` and pass. (Runs are single-threaded; see
§9 for the threading caveat.)

## 6. Framework-Side Changes

None required beyond what already routes to grouped matmul. A framework that
issues an MoE layer as `dnnl::matmul` with grouped memory automatically gets the
ZenDNN path when the module is enabled on AMD x86; otherwise the reference. No
source changes in applications.

## 7. First PR &mdash; Scope and Acceptance Criteria

### 7.1 Implementation
- `zen_grouped_matmul_t` + `CPU_INSTANCE_X64_ZEN_GROUPED` registration ahead of
  `ref_grouped_t`.
- **Acceptance:** with `DNNL_X64_USE_ZEN=ON` + `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON`
  on AMD, a grouped f32/bf16 matmul shows `zen:grouped:...` in verbose.

### 7.2 Coverage
- f32 / bf16 / bf16&rarr;f32; `abc`/`acb`; per-expert bias; eltwise/binary/sum
  post-ops as in §4.6.
- **Acceptance:** benchdnn correctness passes for these across the sweep in §5.2.

### 7.3 Fall-through
- f16, quantization, and unsupported post-ops return `unimplemented`.
- **Acceptance:** those cases dispatch to `ref_grouped:any` and still pass.

### 7.4 No-regression / no-API-change
- Default build (`DNNL_X64_USE_ZEN=OFF`) is byte-for-byte unaffected; no public
  header changes.
- **Acceptance:** existing `test_iface_grouped` and the grouped benchdnn inputs
  pass with the module off.

### 7.5 Additional
- **gtest** for the Zen grouped path (guarded, AMD-only, single-threaded),
  comparing against an in-test reference and asserting the impl name.
- **Docs** for build/enable and supported configs.

## 8. Alternatives Considered

### 8.1 Keep only `ref_grouped_t` on CPU
- **Pros:** no new backend; simplest.
- **Cons:** no optimized CPU MoE path; frameworks keep integrating ZenDNN
  separately (today's state).

### 8.2 Use regular matmul with runtime dimensions / microkernels per expert
- oneDNN's position is that on CPU the grouped problem can be served by regular
  matmul (runtime dims) or BRGEMM microkernels rather than a dedicated grouped
  kernel. This is a viable native alternative and should be benchmarked against
  the ZenDNN path (§9). The ZenDNN backend is justified only if it delivers a
  material x86 improvement over that route.

### 8.3 ZenDNN-backed grouped impl behind the existing API (this RFC)
- Smallest change that gives an optimized AMD-CPU MoE path with no public-API or
  framework changes, reusing the `zen64` module, with the reference as fallback.

## 9. Open Questions

- **Value vs. regular matmul.** oneDNN considers grouped matmul primarily a GPU
  optimization and regular matmul/microkernels sufficient on CPU. This RFC needs
  to demonstrate (a) framework demand for a CPU grouped path and (b) a material
  x86 speedup of the ZenDNN grouped kernel over the regular-matmul route. What
  benchmark set (models, expert counts, token distributions) should anchor this?
- **Opt-in vendor dependency.** A build-time ZenDNN dependency may reduce
  upstream adoption. Should the grouped kernel ultimately land as native oneDNN
  code, with ZenDNN as one backend behind the same `impl_list` entry?
- **Gated activations.** ZenDNN offers fused gated activations
  (`silu_and_mul` / `gelu_and_mul` / `swiglu`) on its grouped path. oneDNN's
  investigation found the fusion yields memory savings rather than speedup and
  favors extending **binary post-op with in-place multiply** over a dedicated
  gated attribute. Follow-up should align with that direction rather than a new
  attribute.
- **Threading robustness.** ZenDNN's `group_matmul_direct` currently exhibits a
  multi-threaded / many-call heap-corruption issue at larger group counts
  (single-threaded is clean). This is an upstream ZenDNN bug to fix before the
  grouped path can be enabled multi-threaded by default; the first PR validates
  single-threaded and tracks the fix as a blocker for perf enablement.