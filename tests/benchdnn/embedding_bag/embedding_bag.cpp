/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2024 Arm Ltd. and affiliates
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
#include <algorithm>
#include <random>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "utils/fill.hpp"
#include "utils/memory.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "embedding_bag/embedding_bag.hpp"

// Here only for internal `embedding_bag_accurate_inf_as_zero` alg_kind. Must be
// removed once alg_kind gets converted to a public value.
#include "src/common/c_types_map.hpp"

namespace embedding_bag {

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb = this;
    skip_unimplemented_data_type({prb->tbldt, prb->dstdt, prb->wtdt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res, dnnl_embedding_bag, prb->tbldt, prb->dstdt);
    skip_unimplemented_binary_po(prb->attr, res);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_embedding_bag);
}

void prb_t::skip_invalid(res_t *res) const {
    const prb_t *prb = this;
    // Indices and offsets must be s32; other dtypes are not supported.
    if (prb->idt != dnnl_s32 || prb->odt != dnnl_s32) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }

    // offsets size must not exceed indices size (primitive returns runtime_error).
    const int64_t indsz = prb->vdims[1][0];
    const int64_t offsz = prb->vdims[2][0];
    if (offsz > indsz) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }

    // padding_idx must be within the table row range when set.
    const int64_t tbl_rows = prb->vdims[0][0];
    if (prb->padding_idx >= 0 && prb->padding_idx >= tbl_rows) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }

    // Weights are only meaningful for SUM and MEAN; MAX and LOOKUP ignore them.
    if (prb->is_weight
            && (prb->alg == EMBEDDING_BAG_MAX
                    || prb->alg == EMBEDDING_BAG_LOOKUP)) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }
}

benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> create_md(const prb_t *prb,
        data_kind_t kind) {
    dnnl_memory_desc_t md {};

    if (kind == EMBAG_TABLE) {
        dnnl_data_type_t dt = prb->get_dt(EMBAG_TABLE);
        std::string tag = prb->get_tag(EMBAG_TABLE);
        dims_t dims = prb->get_dims(EMBAG_TABLE);
        return dnn_mem_t::init_md(dims.size(), dims.data(), dt, tag);
    } else if (kind == EMBAG_DST) {
        dnnl_data_type_t dt = prb->get_dt(EMBAG_DST);
        std::string tag = prb->get_tag(EMBAG_DST);
        dims_t dims = prb->get_dims(EMBAG_DST);
        return dnn_mem_t::init_md(dims.size(), dims.data(), dt, tag);
    } else if (kind == EMBAG_INDICES) {
        dnnl_data_type_t dt = prb->get_dt(EMBAG_INDICES);
        std::string tag = prb->get_tag(EMBAG_INDICES);
        dims_t dims = prb->get_dims(EMBAG_INDICES);
        return dnn_mem_t::init_md(dims.size(), dims.data(), dt, tag);
    } else if (kind == EMBAG_OFFSETS) {
        dnnl_data_type_t dt = prb->get_dt(EMBAG_OFFSETS);
        std::string tag = prb->get_tag(EMBAG_OFFSETS);
        dims_t dims = prb->get_dims(EMBAG_OFFSETS);
        return dnn_mem_t::init_md(dims.size(), dims.data(), dt, tag);
    } else if (kind == EMBAG_WT) {
        if (prb->is_weight) {
            dnnl_data_type_t dt = prb->get_dt(EMBAG_WT);
            std::string tag = prb->get_tag(EMBAG_WT);
            dims_t dims = prb->get_dims(EMBAG_WT);
            return dnn_mem_t::init_md(dims.size(), dims.data(), dt, tag);
        }
    }

    return md;
}

dnnl_status_t init_pd(init_pd_args_t  &init_pd_args) {

    const prb_t *prb = prb_t::from(init_pd_args.base_prb);
    res_t *res = init_pd_args.res;

    auto tbl_d = create_md(prb, EMBAG_TABLE);
    auto indices_d = create_md(prb, EMBAG_INDICES);
    auto offsets_d = create_md(prb, EMBAG_OFFSETS);
    auto dst_d = create_md(prb, EMBAG_DST);
    auto wt_d = create_md(prb, EMBAG_WT);
    dnnl_alg_kind_t alg_kind = prb->get_alg_kind();
    auto prop = dnnl_forward_inference;

    TIME_C_PD(DNN_SAFE_STATUS(dnnl_embedding_bag_forward_primitive_desc_create(
            &init_pd_args.pd, init_pd_args.engine, prop, alg_kind,
            tbl_d, indices_d, offsets_d, wt_d, dst_d,
            prb->padding_idx, prb->is_weight, prb->include_last_offset, nullptr)));

    return dnnl_success;
}

int fill_data_fwd_table(const prb_t *prb, dnn_mem_t &mem_fp) {

    auto nelems = mem_fp.nelems();
    float* tbl_ptr = mem_fp.get_mapped_pointer<float>();

    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(1.0, 2.0);

        for (int64_t i = idx_start; i < idx_end; ++i)
            tbl_ptr[i] = dis(gen);
    });

    return OK;
}

int fill_data_fwd_indices(const prb_t *prb, dnn_mem_t &mem_fp) {

    auto nelems = mem_fp.nelems();

    auto tblsz = prb->vdims[0][0];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, tblsz-1);

    int* iptr = mem_fp.get_mapped_pointer<int>();
    for (int64_t i = 0; i < nelems; ++i) {
        iptr[i] = dis(gen);
    }

    return OK;
}

int fill_data_fwd_offsets(const prb_t *prb, dnn_mem_t &mem_fp) {

    auto nelems = mem_fp.nelems();
    auto indsz = prb->vdims[1][0];

    if (indsz < nelems) return FAIL;

    auto incr = (indsz -2)/nelems;
    incr = (incr > 1) ? incr : 1;

    int* optr = mem_fp.get_mapped_pointer<int>();
    optr[0] = 0;
    for (int i = 1; i < nelems; ++i)
        optr[i] = optr[i-1] + incr;

    return OK;
}

int fill_data_fwd_dst(const prb_t *prb, dnn_mem_t &mem_fp) {
    auto nelems = mem_fp.nelems();
    float *ptr = mem_fp.get_mapped_pointer<float>();
    for (int64_t i = 0; i < nelems; ++i)
        ptr[i] = 0.f;
    return OK;
}

int fill_data_fwd_wt(const prb_t *prb, dnn_mem_t &mem_fp) {

    if (prb->get_is_weight()) {
        auto nelems = mem_fp.nelems();

        float* iptr = mem_fp.get_mapped_pointer<float>();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(1.0, 2.0);

        for (int i = 0; i < nelems; ++i)
            iptr[i] = dis(gen);
    }
    return OK;
}

int fill_data_fwd(
        int exec_arg, const prb_t *prb,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t* res) {

    int sts = OK;
    switch(exec_arg) {
        case DNNL_ARG_EMBEDDING_BAG_TABLE:
            sts = fill_data_fwd_table(prb, mem_fp);
            break;
        case DNNL_ARG_EMBEDDING_BAG_INDICES:
            sts = fill_data_fwd_indices(prb, mem_fp);
            break;
        case DNNL_ARG_EMBEDDING_BAG_OFFSETS:
            sts = fill_data_fwd_offsets(prb, mem_fp);
            break;
        case DNNL_ARG_EMBEDDING_BAG_DST:
            sts = fill_data_fwd_dst(prb, mem_fp);
            break;
        case DNNL_ARG_EMBEDDING_BAG_WEIGHTS:
            sts = fill_data_fwd_wt(prb, mem_fp);
            break;
    }

    SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return sts;
}


int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    const prb_t *prb = prb_t::from(base_prb);

    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();
    //const bool is_fwd_prim = is_fwd_prop_kind(query_prop_kind(query_pd(prim)));

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        // ref_mem_map.emplace(exec_arg,
        //                 dnn_mem_t(mem.md_, mem.dt(), tag::abx, ref_engine,
        //                 /* prefill = */ false));

        ref_mem_map.emplace(exec_arg,
                        dnn_mem_t(mem.md_, ref_engine,
                                  /* prefill = */ false, {true, DNNL_MEMORY_ALLOCATE}));

        auto &ref_mem = ref_mem_map[exec_arg];

        SAFE(fill_data_fwd(exec_arg, prb, mem, ref_mem, res), WARN);
    }

    // Don't keep reference memory if it is not used further.
    if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();

    return OK;
}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    const prb_t *prb = prb_t::from(base_prb);
    // SUM and MEAN accumulate over a bag, so scale the threshold by bag size.
    // MAX and LOOKUP are selection ops — no accumulation error.
    const bool is_sum_or_mean = prb->alg == EMBEDDING_BAG_SUM
            || prb->alg == EMBEDDING_BAG_MEAN;
    const int64_t bag_size = prb->vdims[1][0]; // number of indices
    const float trh_coeff = is_sum_or_mean ? static_cast<float>(bag_size) : 1.f;
    cmp.set_threshold(trh_coeff * epsilon_dt(prb->dstdt));

    // MAX can produce ties where any tied index is a valid answer, leading to
    // legitimate mismatches versus reference; allow a higher zero-trust percent.
    if (prb->alg == EMBEDDING_BAG_MAX) cmp.set_zero_trust_percent(50.f);
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb) {
    std::vector<data_kind_t> check_kinds;
    if (prb->dir & FLAG_FWD) {
        check_kinds = {EMBAG_DST};
    } else {
        assert(!"unexpected!");
        SAFE_V(FAIL);
    }
    assert(!check_kinds.empty());
    get_kinds_to_check_shared(check_kinds, prb->attr);
    return check_kinds;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    v_prim.resize(1);
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    return OK;
}

int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    if (has_bench_mode_bit(mode_bit_t::exec)) {
        SAFE(check_total_size(res), WARN);
    }
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_caches(v_prim[0], prb->ctx_init, res), WARN);
    }
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);

    set_zmalloc_max_expected_size(res->mem_size_args.zmalloc_expected_size);

    const auto &prim = v_prim[0];

    dnn_mem_map_t mem_map, ref_mem_map;

    //init_memory_args(mem_map, prb, prim, prb->supported_exec_args(false));
    init_memory_args(mem_map, prb, prim, res);

    TIME_FILL(SAFE(
            init_ref_memory_args(ref_mem_map, mem_map, prim, prb, res), WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(run_execution(prim, args, res), WARN);

    check_correctness(prb, get_kinds_to_check(prb), args, ref_args, compute_ref,
            setup_cmp, res, prb->dir);

    SAFE(check_bitwise(prim, get_kinds_to_check(prb), args, prb->attr,
                 prb->inplace, res),
            WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace embedding_bag
