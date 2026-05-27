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

#include <sstream>

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "utils/stringstream.hpp"

#include "embedding_bag/embedding_bag.hpp"

namespace embedding_bag {

alg_t str2alg(const char *str) {
#define CASE(_alg) \
    if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(EMBEDDING_BAG_SUM);
    CASE(EMBEDDING_BAG_MEAN);
    CASE(EMBEDDING_BAG_MAX);
    CASE(EMBEDDING_BAG_LOOKUP);
#undef CASE
    assert(!"unknown algorithm");
    return UNDEF;
}

const char *alg2str(alg_t alg) {
    if (alg == EMBEDDING_BAG_SUM) return "EMBEDDING_BAG_SUM";
    if (alg == EMBEDDING_BAG_MEAN) return "EMBEDDING_BAG_MEAN";
    if (alg == EMBEDDING_BAG_MAX) return "EMBEDDING_BAG_MAX";
    if (alg == EMBEDDING_BAG_LOOKUP) return "EMBEDDING_BAG_LOOKUP";
    assert(!"unknown algorithm");
    return "UNDEF";
}

dnnl_alg_kind_t alg2alg_kind(alg_t alg) {
       if (alg == EMBEDDING_BAG_SUM) return dnnl_embedding_bag_sum;
       if (alg == EMBEDDING_BAG_MEAN) return dnnl_embedding_bag_mean;
       if (alg == EMBEDDING_BAG_MAX) return dnnl_embedding_bag_max;
       if (alg == EMBEDDING_BAG_LOOKUP) return dnnl_embedding_bag_lookup;
       assert(!"unknown algorithm");
       return dnnl_alg_kind_undef;
}

std::vector<int> prb_t::supported_exec_args(bool override_dir_with_fwd) const {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_EMBEDDING_BAG_TABLE,
            DNNL_ARG_EMBEDDING_BAG_INDICES,
            DNNL_ARG_EMBEDDING_BAG_OFFSETS,
            DNNL_ARG_EMBEDDING_BAG_DST,
            DNNL_ARG_EMBEDDING_BAG_WEIGHTS,
    };
    return exec_fwd_args;
}

std::string prb_t::set_repro_line() {
    stringstream_t s;
    dump_global_params(s);
    settings_t def;

    if (canonical || dir != def.dir[0]) s << "--dir=" << dir << " ";
    if (canonical || tbldt != def.tbldt[0]) s << "--tbldt=" << tbldt << " ";
    if (canonical || dstdt != def.dstdt[0]) s << "--dstdt=" << dstdt << " ";
    if (canonical || wtdt != def.wtdt[0]) s << "--wtdt=" << wtdt << " ";
    if (canonical || idt != def.idt[0]) s << "--idt=" << idt << " ";
    if (canonical || odt != def.odt[0]) s << "--odt=" << odt << " ";
    if (canonical || alg != def.alg[0]) s << "--alg=" << alg2str(alg) << " ";
    if (canonical || padding_idx != def.padding_idx[0])
        s << "--padding_idx=" << padding_idx << " ";
    if (canonical || is_weight != def.is_weight[0])
        s << "--is_weight=" << bool2str(is_weight) << " ";
    if (canonical || include_last_offset != def.include_last_offset[0])
        s << "--include_last_offset=" << bool2str(include_last_offset) << " ";

    s << attr;
    if (canonical || ctx_init != def.ctx_init[0])
        s << "--ctx-init=" << ctx_init << " ";
    if (canonical || ctx_exe != def.ctx_exe[0])
        s << "--ctx-exe=" << ctx_exe << " ";
    if (canonical || !impl_filter.is_def() || !global_impl_filter.is_def())
        s << impl_filter;

    s << static_cast<const prb_vdims_t &>(*this);
    return s.str();
}

} // namespace embedding_bag
