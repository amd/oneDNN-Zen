/*******************************************************************************
* Copyright 2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/ref_io_helper.hpp"
#include "cpu/ref_embedding_bag.hpp"

namespace {
using namespace dnnl::impl;

template <
  typename InType,
  typename IndexType,
  typename OffsetType,
  typename OutType>
void embag_ref_kernel(
  const InType *input,           // [num_embeddings, width] - embedding table
  const float *weights,          // [indsz] or nullptr if is_weights == false
  const IndexType *indices,     // [indsz] - indices into embedding table
  const OffsetType *offsets,    // [offsz] - start positions of each bag
  OutType *dst,                  // [offsz, width] with stride dst_stride - output buffer
  int64_t width,                 // embedding dimension
  int64_t indsz,                 // number of indices
  int64_t offsz,                 // number of bags
  int64_t padidx,                // padding index to skip
  bool is_weights,               // whether weights are used
  alg_kind_t algo,               // EMBEDDING_BAG_SUM, MEAN, MAX and LOOKUP
  int64_t dst_stride,            // stride between output rows
  bool include_last_offset       // whether to include the last offset
) {

  // Determine if we need type conversions
  bool input_is_bf16 = false;
  bool input_is_f16 = false;
  bool output_is_bf16 = false;
  bool output_is_f16 = false;

  // Read accumulation precision the actual backend used (FBGEMM, native
  // AVX512 F16 FMA, native AVX512 F32, AVX2, etc.). The actual kernel
  // writes this on the embag_config_t singleton just before invoking its
  // compute path; the reference kernel reads it here to bit-match. This
  // mirrors the matmul reference kernel's pattern.
  //
  // The producers already gate F16 on __GNUC__ >= 12 and
  // can_use_f16_fma_kernel() (which itself returns false when
  // ZENDNNL_EMBAG_NATIVE_F32_ACCUM is defined), so the singleton can only
  // ever hold f16 when the F16 FMA path was actually used - no extra
  // build-time check is needed here. The (input_is_f16 || output_is_f16)
  // guard ensures non-F16-touching dtypes are unaffected by any stale
  // f16 value left in the singleton by a prior invocation.

  // const data_type_t reported_accum =
  //   embag_config_t::instance().get_accum_type();
  // const bool use_f16_accum = (reported_accum == data_type_t::f16) &&
  //                            (input_is_f16 || output_is_f16);

  const bool use_f16_accum = false;

  // Temporary buffers for type conversion
  std::vector<float> temp_input_row;
  std::vector<float> temp_output_row;
  std::vector<float16_t> f16_accum_row;
  std::vector<float16_t> f16_input_row;

  if (input_is_bf16 || input_is_f16) {
    temp_input_row.resize(static_cast<size_t>(width));
  }

  if (output_is_bf16 || output_is_f16) {
    temp_output_row.resize(static_cast<size_t>(width));
  }

  if (use_f16_accum) {
    f16_accum_row.resize(static_cast<size_t>(width));
    if (!input_is_f16) {
      f16_input_row.resize(static_cast<size_t>(width));
    }
  }

  bool is_embedding = (offsets == nullptr) ? true : false;
  int outer_loop = is_embedding ? indsz :
      (include_last_offset ? offsz -1 : offsz);

  // Iterate over the offsets
  for (int oi = 0; oi < outer_loop; ++oi) {
    int64_t start = is_embedding ? oi : offsets[oi];
    int64_t end = is_embedding ? oi + 1 : (include_last_offset ? offsets[oi + 1] :
                                           (oi < offsz - 1 ? offsets[oi + 1] : indsz));

    auto dst_offset = oi * dst_stride;
    float wt_sum = 0;
    bool first_valid_index = true;

    if (input_is_f16 || output_is_f16) {
      if (use_f16_accum) {
        // ── F16 FMA accumulation path ──
        // Mirrors the native _mm512_fmadd_ph kernel: each FMA result is rounded
        // to FP16 before the next accumulation step.
        std::fill(f16_accum_row.begin(), f16_accum_row.end(), float16_t(0.0f));

        for (auto i = start; i < end; ++i) {
          if (indices[i] != padidx) {
            auto input_offset = indices[i] * width;
            float wt_f32 = is_weights ? weights[i] : 1.0f;
            float16_t wt = float16_t(wt_f32);
            wt_sum += wt_f32;

            const float16_t *f16_row;
            if (input_is_f16) {
              f16_row = (const float16_t*)(&input[input_offset]);
            }
            else {
              const float *f32_src = reinterpret_cast<const float *>(
                                       &input[input_offset]);
              for (auto j = 0; j < width; ++j) {
                f16_input_row[j] = float16_t(f32_src[j]);
              }
              f16_row = f16_input_row.data();
            }

            if (is_embedding) {
              for (auto j = 0; j < width; ++j) {
                f16_accum_row[j] = f16_row[j];
              }
            }
            else {
              if (algo == alg_kind::embedding_bag_max) {
                if (first_valid_index) {
                  for (auto j = 0; j < width; ++j) {
                    f16_accum_row[j] = f16_row[j];
                  }
                }
                else {
                  for (auto j = 0; j < width; ++j) {
                    float a = static_cast<float>(f16_accum_row[j]);
                    float b = static_cast<float>(f16_row[j]);
                    f16_accum_row[j] = float16_t(std::max(a, b));
                  }
                }
              }
              else {
                // sum / mean: fmaf with F16 truncation after each step
                for (auto j = 0; j < width; ++j) {
                  float in_f32 = static_cast<float>(f16_row[j]);
                  float wt_f32_cast = static_cast<float>(wt);
                  float acc_f32 = static_cast<float>(f16_accum_row[j]);
                  //f16_accum_row[j] = float16_t(std::fmaf(in_f32, wt_f32_cast, acc_f32));
                }
              }
              first_valid_index = false;
            }
          }
        }

        if (!is_embedding && algo == alg_kind::embedding_bag_mean && wt_sum > 0) {
          float16_t div = float16_t(wt_sum);
          for (auto j = 0; j < width; ++j) {
            float a = static_cast<float>(f16_accum_row[j]);
            float d = static_cast<float>(div);
            //f16_accum_row[j] = float16_t(a / d);
          }
        }

        // Store F16 accumulator to output (widen to F32 if needed)
        for (auto j = 0; j < width; ++j) {
          if (output_is_f16) {
            dst[dst_offset + j] = f16_accum_row[j];
          }
          else {
            dst[dst_offset + j] = static_cast<OutType>(
                                    static_cast<float>(f16_accum_row[j]));
          }
        }
      }
    }

    if (!use_f16_accum) {
      // ── Standard F32 accumulation path (original) ──
      float *output_row;
      if (output_is_bf16 || output_is_f16) {
        output_row = temp_output_row.data();
        std::fill(output_row, output_row + width, 0.0f);
      }
      else {
        output_row = reinterpret_cast<float *>(&dst[dst_offset]);
        std::fill(output_row, output_row + width, 0.0f);
      }

      // Process all indices in the current bag
      for (auto i = start; i < end; ++i) {
        if (indices[i] != padidx) {
          auto input_offset = indices[i] * width;
          auto wt = is_weights ? weights[i] : 1.0f;

          // Get input row pointer (convert from BF16/F16 if needed)
          const float *input_row;
          if (input_is_f16) {
            const uint16_t *f16_row = reinterpret_cast<const uint16_t *>
                                      (&input[input_offset]);
            //float16_t::f16_to_f32_buf(f16_row, temp_input_row.data(), width);
            input_row = temp_input_row.data();
          }
          else if (input_is_bf16) {
            const uint16_t *bf16_row = reinterpret_cast<const uint16_t *>
                                       (&input[input_offset]);
            //bfloat16_t::bf16_to_f32_buf(bf16_row, temp_input_row.data(), width);
            input_row = temp_input_row.data();
          }
          else {
            input_row = reinterpret_cast<const float *>(&input[input_offset]);
          }

          if (is_embedding) {
            for (auto j = 0; j < width; ++j) {
              output_row[j] = input_row[j];
            }
          }
          else {
            if (first_valid_index) {
              wt_sum = wt;
              // Initialize with first valid embedding
              for (auto j = 0; j < width; ++j) {
                if (algo != alg_kind::embedding_bag_max) {
                  output_row[j] = wt * input_row[j];
                }
                else {
                  output_row[j] = input_row[j];
                }
              }
              first_valid_index = false;
            }
            else {
              // Compute embedding bags as per the algorithm
              if (algo == alg_kind::embedding_bag_max) {
                for (auto j = 0; j < width; ++j) {
                  if (output_row[j] < input_row[j]) {
                    output_row[j] = input_row[j];
                  }
                }
              }
              else {
                wt_sum += wt;
                for (auto j = 0; j < width; ++j) {
                  output_row[j] += wt * input_row[j];
                }
              }
            }
          }
        }
      }

      if (!is_embedding) {
        // Apply mean normalization if required
        if (algo == alg_kind::embedding_bag_mean && wt_sum > 0) {
          for (auto j = 0; j < width; ++j) {
              output_row[j] /= float(wt_sum);
          }
        }
      }

      // Convert output back to BF16/F16 if needed
      if (output_is_f16) {
        uint16_t *f16_dst = reinterpret_cast<uint16_t *>(&dst[dst_offset]);
        //float16_t::f32_to_f16(temp_output_row.data(), f16_dst, width);
      }
      else if (output_is_bf16) {
        int16_t *bf16_dst = reinterpret_cast<int16_t *>(&dst[dst_offset]);
        //bfloat16_t::f32_to_bf16(temp_output_row.data(), bf16_dst, width);
      }
    }
  }

}

} //namespace

namespace dnnl {
namespace impl {
namespace cpu {

status_t ref_embedding_bag_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    auto   table   = CTX_IN_MEM(const void *, DNNL_ARG_EMBEDDING_BAG_TABLE);
    auto   indices = CTX_IN_MEM(const int32_t *, DNNL_ARG_EMBEDDING_BAG_INDICES);
    auto   dst = CTX_OUT_MEM(void *, DNNL_ARG_EMBEDDING_BAG_DST);

    const int32_t* offsets = nullptr;

    if (alg_kind_ != alg_kind::embedding_bag_lookup)
        offsets = CTX_IN_MEM(const int32_t *, DNNL_ARG_EMBEDDING_BAG_OFFSETS);

    const float* weights = nullptr;
    if (is_weight_)
        weights = CTX_OUT_MEM(const float *, DNNL_ARG_EMBEDDING_BAG_WEIGHTS);

    const memory_desc_wrapper table_md(pd()->args_md(DNNL_ARG_EMBEDDING_BAG_TABLE));
    const memory_desc_wrapper indices_md(pd()->args_md(DNNL_ARG_EMBEDDING_BAG_INDICES));
    const memory_desc_wrapper offsets_md(pd()->args_md(DNNL_ARG_EMBEDDING_BAG_OFFSETS));
    const memory_desc_wrapper weights_md(pd()->src_md(DNNL_ARG_EMBEDDING_BAG_WEIGHTS));
    const memory_desc_wrapper dst_md(pd()->args_md(DNNL_ARG_EMBEDDING_BAG_DST));

    //const auto tbl_md = pd()->args_md(DNNL_ARG_EMBEDDING_BAG_TABLE);
    auto table_width = table_md.dims()[1];
    auto indsz = indices_md.dims()[0];
    auto offsz = offsets_md.dims()[0];
    auto dst_stride = dst_md.dims()[1];

    if (offsz > indsz) return status::runtime_error;

    embag_ref_kernel<float, int32_t, int32_t, float>((const float*)table,
                                                     (const float*)weights,
                                                     indices,
                                                     offsets,
                                                     (float*)dst,
                                                     table_width,
                                                     indsz,
                                                     offsz,
                                                     padding_idx_,
                                                     is_weight_,
                                                     alg_kind_,
                                                     dst_stride,
                                                     include_last_offset_);
    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
