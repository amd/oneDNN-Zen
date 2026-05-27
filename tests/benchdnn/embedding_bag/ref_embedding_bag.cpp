/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "utils/parallel.hpp"

#include "embedding_bag/embedding_bag.hpp"

namespace {

using namespace embedding_bag;

void embag_ref_kernel(
        const float *input,
        const float *weights,
        const int32_t *indices,
        const int32_t *offsets,
        float *dst,
        int64_t width,
        int64_t indsz,
        int64_t offsz,
        int64_t padidx,
        bool is_weights,
        alg_t algo,
        int64_t dst_stride,
        bool include_last_offset) {

    const bool is_embedding = (offsets == nullptr);
    const int outer_loop
            = is_embedding ? indsz
                           : (include_last_offset ? offsz - 1 : offsz);

    for (int oi = 0; oi < outer_loop; ++oi) {
        const int64_t start = is_embedding ? oi : offsets[oi];
        const int64_t end = is_embedding
                ? oi + 1
                : (include_last_offset
                                ? offsets[oi + 1]
                                : (oi < offsz - 1 ? offsets[oi + 1] : indsz));
        const auto dst_offset = oi * dst_stride;
        float *output_row = &dst[dst_offset];
        std::fill(output_row, output_row + width, 0.0f);

        float wt_sum = 0.f;
        bool first_valid = true;

        for (auto i = start; i < end; ++i) {
            if (indices[i] == padidx) continue;

            const float *input_row = &input[indices[i] * width];
            const float wt = is_weights ? weights[i] : 1.0f;

            if (first_valid) {
                wt_sum = wt;
                for (auto j = 0; j < width; ++j)
                    output_row[j] = (algo != EMBEDDING_BAG_MAX)
                            ? wt * input_row[j]
                            : input_row[j];
                first_valid = false;
            } else {
                if (algo == EMBEDDING_BAG_MAX) {
                    for (auto j = 0; j < width; ++j)
                        output_row[j] = std::max(output_row[j], input_row[j]);
                } else {
                    wt_sum += wt;
                    for (auto j = 0; j < width; ++j)
                        output_row[j] += wt * input_row[j];
                }
            }
        }

        if (!is_embedding && algo == EMBEDDING_BAG_MEAN && wt_sum > 0.f) {
            for (auto j = 0; j < width; ++j)
                output_row[j] /= wt_sum;
        }
    }
}

} // namespace

namespace embedding_bag {

void compute_ref_fwd(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &tbl = args.find(DNNL_ARG_EMBEDDING_BAG_TABLE);
    const dnn_mem_t &indices = args.find(DNNL_ARG_EMBEDDING_BAG_INDICES);
    const dnn_mem_t &offsets = args.find(DNNL_ARG_EMBEDDING_BAG_OFFSETS);
    const dnn_mem_t &dst = args.find(DNNL_ARG_EMBEDDING_BAG_DST);

    const bool is_weight = prb->get_is_weight();
    const bool include_last_offset = prb->get_include_last_offset();
    const alg_t alg = prb->get_alg();
    const int64_t padidx = prb->get_padding_idx();

    const int64_t width = tbl.dims()[1];
    const int64_t indsz = indices.nelems();
    const int64_t offsz = offsets.nelems();

    float *tblptr = tbl.get_mapped_pointer<float>();
    int32_t *iptr = indices.get_mapped_pointer<int32_t>();
    float *dstptr = dst.get_mapped_pointer<float>();
    int32_t *optr = (alg != EMBEDDING_BAG_LOOKUP)
            ? offsets.get_mapped_pointer<int32_t>()
            : nullptr;

    if (is_weight) {
        const dnn_mem_t &wt = args.find(DNNL_ARG_EMBEDDING_BAG_WEIGHTS);
        float *wtptr = wt.get_mapped_pointer<float>();
        embag_ref_kernel(tblptr, wtptr, iptr, optr, dstptr, width, indsz,
                offsz, padidx, is_weight, alg, width, include_last_offset);
    } else {
        embag_ref_kernel(tblptr, nullptr, iptr, optr, dstptr, width, indsz,
                offsz, padidx, is_weight, alg, width, include_last_offset);
    }
}

void compute_ref(const base_prb_t *base_prb, dir_t dir, const args_t &args,
        dnnl_primitive_t prim_ref) {
    compute_ref_fwd(prb_t::from(base_prb), args);
}

} // namespace embedding_bag
