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
#include "oneapi/dnnl/dnnl.h"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

#define VCHECK_EMBEDDING_BAG(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, embedding_bag, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_EMBEDDING_BAG_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, embedding_bag, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

namespace {
status_t embedding_bag_desc_init(embedding_bag_desc_t *embedding_bag_desc,
        prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *table_desc,
        const memory_desc_t *indices_desc, const memory_desc_t *offsets_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *dst_desc,
        int is_weight, int include_last_offset, int64_t padding_idx) {

    //const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    if (alg_kind != dnnl_embedding_bag_lookup) {
        VCHECK_EMBEDDING_BAG(!any_null(table_desc, indices_desc, offsets_desc, dst_desc),
                         VERBOSE_NULL_ARG);
    } else {
        VCHECK_EMBEDDING_BAG(!any_null(table_desc, indices_desc, dst_desc),
                         VERBOSE_NULL_ARG);
    }

    if (is_weight) {
        VCHECK_EMBEDDING_BAG(weights_desc != nullptr, VERBOSE_NULL_ARG);
    }

    VCHECK_EMBEDDING_BAG(one_of(alg_kind, embedding_bag_sum, embedding_bag_mean,
                         embedding_bag_max, embedding_bag_lookup),
                         VERBOSE_BAD_ALGORITHM);

    // bool cond = is_weight && (weights_desc != nullptr);
    // VCHECK_EMBEDDING_BAG(!cond, VERBOSE_NULL_ARG);

    auto embd = embedding_bag_desc_t();
    embd.primitive_kind = primitive_kind::embedding_bag;
    embd.prop_kind = prop_kind;
    embd.alg_kind = alg_kind;
    embd.table_desc = *table_desc;
    embd.indices_desc = *indices_desc;
    //embd.offsets_desc = *offsets_desc;
    embd.dst_desc = *dst_desc;
    embd.padding_idx = padding_idx;
    embd.is_weight = is_weight;
    embd.include_last_offset = include_last_offset;

    if (alg_kind != dnnl_embedding_bag_lookup)
        embd.offsets_desc = *offsets_desc;
    else
        embd.offsets_desc = dnnl::impl::types::zero_md();

    if (is_weight)
        embd.weights_desc = *weights_desc;
    else
        embd.weights_desc = dnnl::impl::types::zero_md();

    *embedding_bag_desc = embd;
    return success;
}

} // namespace

status_t dnnl_embedding_bag_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *table_desc, const memory_desc_t *indices_desc,
        const memory_desc_t *offsets_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *dst_desc, int64_t padding_idx,
        int is_weight, int include_last_offset,
        const primitive_attr_t *attr) {
    if (!one_of(prop_kind, forward_inference, forward_training))
        return invalid_arguments;

    auto embedding_bag_desc = embedding_bag_desc_t();
    CHECK(embedding_bag_desc_init(&embedding_bag_desc, prop_kind, alg_kind,
            table_desc, indices_desc, offsets_desc, weights_desc, dst_desc,
            is_weight, include_last_offset, padding_idx));

    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&embedding_bag_desc, nullptr, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
