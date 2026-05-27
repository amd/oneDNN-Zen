/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef EMBEDDING_BAG_HPP
#define EMBEDDING_BAG_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/perf_report.hpp"
#include "utils/prb.hpp"
#include "utils/settings.hpp"

namespace embedding_bag {

enum alg_t {
    UNDEF,
    EMBEDDING_BAG_SUM,
    EMBEDDING_BAG_MEAN,
    EMBEDDING_BAG_MAX,
    EMBEDDING_BAG_LOOKUP,
};
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2alg_kind(alg_t alg);

struct settings_t : public base_settings_t {
    using base_settings_t::base_settings_t;

    prb_vdims_t prb_vdims;

    std::vector<dir_t> dir {FWD_I};
    std::vector<dnnl_data_type_t> tbldt {dnnl_f32}, dstdt {dnnl_f32}, wtdt {dnnl_f32};
    std::vector<dnnl_data_type_t> idt {dnnl_s32}, odt {dnnl_s32};
    std::vector<alg_t> alg {EMBEDDING_BAG_SUM};
    std::vector<int> padding_idx {-1};
    std::vector<bool> is_weight {false};
    std::vector<bool> include_last_offset {false};

    const char *perf_template_csv() const {
        static const std::string args
                = "%dir%,%sdt%,%ddt%,%alg%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
         return dir.size() == 1 && tbldt.size() == 1 && dstdt.size() == 1 && wtdt.size() == 1
             && idt.size() == 1 && odt.size() == 1 && alg.size() == 1
             && padding_idx.size() == 1 && is_weight.size() == 1 && include_last_offset.size() == 1
             && base_settings_t::has_single_setup();
    }
};

struct prb_t : public prb_vdims_t, public base_prb_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.prb_vdims, s.dir[0], s.tbldt[0], s.dstdt[0], s.wtdt[0],
                  s.idt[0], s.odt[0],
                  s.alg[0], s.padding_idx[0], s.is_weight[0],
                  s.include_last_offset[0], s.attributes.front(),
                  s.ctx_init[0], s.ctx_exe[0], s.impl_filter) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const prb_vdims_t &prb_vdims, dir_t dir, dnnl_data_type_t tbldt,
            dnnl_data_type_t dstdt, dnnl_data_type_t wtdt,
            dnnl_data_type_t idt, dnnl_data_type_t odt,
            alg_t alg, int padding_idx, bool is_weight,
            bool include_last_offset, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe,
            const impl_filter_t &impl_filter)
        : prb_vdims_t(prb_vdims)
        , base_prb_t(dir, false, attr, ctx_init, ctx_exe, impl_filter)
        , tbldt(tbldt)
        , dstdt(dstdt)
        , wtdt(wtdt)
        , idt(idt)
        , odt(odt)
        , tbltag(tag::abx)
        , dsttag(tag::abx)
        , wttag(tag::abx)
        , itag(tag::abx)
        , otag(tag::abx)
        , alg(alg)
        , padding_idx(padding_idx)
        , is_weight(is_weight)
        , include_last_offset(include_last_offset) {

        dims_t table_dims = vdims[0];
        dims_t indices_dims = vdims[1];
        dims_t offset_dims = vdims[2];

        dst_dims.clear();
        if (alg != EMBEDDING_BAG_LOOKUP) {
            dst_dims.push_back(offset_dims[0]);
            dst_dims.push_back(table_dims[1]);
        } else {
            dst_dims.push_back(indices_dims[0]);
            dst_dims.push_back(table_dims[1]);
        }
        repro = set_repro_line(); // must be last in ctor to collect right info
    }

    dnnl_data_type_t get_dt(data_kind_t data_kind) const {
      switch (data_kind) {
          case EMBAG_TABLE: return tbldt;
          case EMBAG_DST: return dstdt;
          case EMBAG_INDICES: return idt;
          case EMBAG_OFFSETS: return odt;
          case EMBAG_WT: return wtdt;
          default: assert(!"unexpected"); return dnnl_data_type_undef;
      }
    }

    std::string get_tag(data_kind_t data_kind) const {
      switch (data_kind) {
          case EMBAG_TABLE: return tbltag;
          case EMBAG_DST: return dsttag;
          case EMBAG_INDICES: return itag;
          case EMBAG_OFFSETS: return otag;
          case EMBAG_WT: return wttag;
          default: assert(!"unexpected"); return std::string();
      }
    }

    dims_t get_dims(data_kind_t data_kind) const {
      switch (data_kind) {
          case EMBAG_TABLE: return vdims[0];
          case EMBAG_DST: return dst_dims;
          case EMBAG_INDICES: return vdims[1];
          case EMBAG_OFFSETS: return vdims[2];
          case EMBAG_WT: return vdims[1];
          default: assert(!"unexpected"); return {};
      }
    }

    bool get_is_weight() const {
        return is_weight;
    }

    bool get_include_last_offset() const {
        return include_last_offset;
    }

    int get_padding_idx() const {
        return padding_idx;
    }

    dnnl_alg_kind_t get_alg_kind() const {
        return alg2alg_kind(alg);
    }

    alg_t get_alg() const {
        return alg;
    }

    dnnl_data_type_t tbldt, dstdt, wtdt, idt, odt;
    std::string tbltag, dsttag, wttag, itag, otag;
    alg_t alg;
    int padding_idx;
    bool is_weight;
    bool include_last_offset;

    static const prb_t *from(const base_prb_t *base_prb) {
        return downcast<const prb_t *>(base_prb);
    }

    void skip_unimplemented(res_t *res) const override;
    void skip_invalid(res_t *res) const override;
    std::vector<int> supported_exec_args(
            bool override_dir_with_fwd) const override;

private:
    std::string set_repro_line() override;
};

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const base_prb_t *base_prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb_t::from(base_prb))
        , sdt_({p_->tbldt, p_->dstdt, p_->wtdt}) {}

    void dump_alg(std::ostream &s) const override { s << alg2str(p_->alg); }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_vdims_t &>(*p_);
    }

    const attr_t *attr() const override { return &p_->attr; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }
    const std::string *name() const override { return &p_->name; }
    const dir_t *dir() const override { return &p_->dir; }
    // %sdt%: tbldt, dstdt, wtdt (the floating-point tensors)
    const std::vector<dnnl_data_type_t> *sdt() const override { return &sdt_; }
    // %ddt%: idt (indices and offsets are always s32, report once)
    const dnnl_data_type_t *ddt() const override { return &p_->idt; }

private:
    const prb_t *p_;
    std::vector<dnnl_data_type_t> sdt_;
};

dnnl_status_t init_pd(init_pd_args_t &init_pd_args);
void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args);
int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref = nullptr);

void compute_ref(const base_prb_t *base_prb, dir_t dir, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res);
int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res);
int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res);

int bench(int argc, char **argv);

} // namespace embedding_bag

#endif
