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

#ifndef COMMON_EMBEDDING_BAG_PD_HPP
#define COMMON_EMBEDDING_BAG_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"

#define VDISPATCH_EMBEDDING_BAG(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, embedding_bag, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_EMBEDDING_BAG_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, embedding_bag, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

#define VDISPATCH_EMBEDDING_BAG_IC(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, embedding_bag, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {

struct embedding_bag_fwd_pd_t;

struct embedding_bag_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::embedding_bag;

    const embedding_bag_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    const memory_desc_t *args_md(
          int arg, bool user_input = false) const {
      switch (arg) {
          case DNNL_ARG_EMBEDDING_BAG_TABLE:
              return &table_md_;
          case DNNL_ARG_EMBEDDING_BAG_INDICES:
              return &indices_md_;
          case DNNL_ARG_EMBEDDING_BAG_OFFSETS:
              return &offsets_md_;
          case DNNL_ARG_EMBEDDING_BAG_WEIGHTS:
              return &weights_md_;
          case DNNL_ARG_EMBEDDING_BAG_DST:
              return &dst_md_;
          default: return nullptr;
      }
    }

    const memory_desc_t *arg_md(
          int arg, bool user_input = false) const override {
      switch (arg) {
          case DNNL_ARG_EMBEDDING_BAG_TABLE:
              return &table_md_;
          case DNNL_ARG_EMBEDDING_BAG_INDICES:
              return &indices_md_;
          case DNNL_ARG_EMBEDDING_BAG_OFFSETS:
              return &offsets_md_;
          case DNNL_ARG_EMBEDDING_BAG_WEIGHTS:
              return &weights_md_;
          case DNNL_ARG_EMBEDDING_BAG_DST:
              return &dst_md_;
          default: return nullptr;
      }
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            case query::primitive_kind:
                *(primitive_kind_t *)result = desc()->primitive_kind;
                break;
            case query::alg_kind:
                *(alg_kind_t *)result = desc()->alg_kind;
                break;
            case query::exec_arg_md:
              *(const memory_desc_t **)result = this->args_md(idx);
              break;
            default:
              return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common embedding_bag aux functions */
    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    alg_kind_t alg_kind() const { return desc()->alg_kind; }

    bool       is_weight() const {return is_weight_;}
    bool       include_last_offset() const {return include_last_offset_;}
    int64_t    padding_idx() const {return padding_idx_;}

protected:
    embedding_bag_desc_t desc_;
    const embedding_bag_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t table_md_;
    memory_desc_t indices_md_;
    memory_desc_t offsets_md_;
    memory_desc_t weights_md_;
    memory_desc_t dst_md_;

    bool          is_weight_;
    bool          include_last_offset_;
    int64_t       padding_idx_;

    embedding_bag_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const embedding_bag_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*op_desc_t::to_desc<embedding_bag_desc_t>(adesc))
        , hint_fwd_pd_(hint_fwd_pd)
        , table_md_(desc_.table_desc)
        , indices_md_(desc_.indices_desc)
        , offsets_md_(desc_.offsets_desc)
        , weights_md_(desc_.weights_desc)
        , dst_md_(desc_.dst_desc)
        , is_weight_(desc_.is_weight)
        , include_last_offset_(desc_.include_last_offset)
        , padding_idx_(desc_.padding_idx) {}
};

// NOLINTBEGIN(google-default-arguments)
struct embedding_bag_fwd_pd_t : public embedding_bag_pd_t {
    using base_class = embedding_bag_fwd_pd_t;
    using hint_class = embedding_bag_fwd_pd_t;

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_EMBEDDING_BAG_TABLE) return arg_usage_t::input;
        if (arg == DNNL_ARG_EMBEDDING_BAG_INDICES) return arg_usage_t::input;
        if (arg == DNNL_ARG_EMBEDDING_BAG_OFFSETS) return arg_usage_t::input;
        if (arg == DNNL_ARG_EMBEDDING_BAG_WEIGHTS) return arg_usage_t::input;

        if (arg == DNNL_ARG_EMBEDDING_BAG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_EMBEDDING_BAG_TABLE: return src_md(0);
            case DNNL_ARG_EMBEDDING_BAG_INDICES: return src_md(1);
            case DNNL_ARG_EMBEDDING_BAG_OFFSETS: return src_md(2);
            case DNNL_ARG_EMBEDDING_BAG_WEIGHTS: return src_md(3);
            case DNNL_ARG_EMBEDDING_BAG_DST: return dst_md(0);
            default: return embedding_bag_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = DNNL_ARG_EMBEDDING_BAG_TABLE, bool user_input = false) const override {
        if (index == DNNL_ARG_EMBEDDING_BAG_TABLE) return user_input ? &desc()->table_desc : &table_md_;
        if (index == DNNL_ARG_EMBEDDING_BAG_INDICES) return user_input ? &desc()->indices_desc : &indices_md_;
        if (index == DNNL_ARG_EMBEDDING_BAG_OFFSETS) return user_input ? &desc()->offsets_desc : &offsets_md_;
        if ((index == DNNL_ARG_EMBEDDING_BAG_WEIGHTS) && is_weight_) return user_input ? &desc()->weights_desc : &weights_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *dst_md(
            int index = DNNL_ARG_EMBEDDING_BAG_DST, bool user_input = false) const override {
        if (index == DNNL_ARG_EMBEDDING_BAG_DST) return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return is_weight_ ? 4 : 3; }
    int n_outputs() const override { return 1; }

protected:
    embedding_bag_fwd_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const embedding_bag_fwd_pd_t *hint_fwd_pd)
        : embedding_bag_pd_t(adesc, attr, hint_fwd_pd) {}
};
// NOLINTEND(google-default-arguments)

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
