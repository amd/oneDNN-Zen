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

#include <stdio.h>
#include <stdlib.h>

#include "dnnl_common.hpp"
#include "utils/parser.hpp"
#include "utils/task_executor.hpp"
#include "utils/stringstream.hpp"
#include "embedding_bag/embedding_bag.hpp"

namespace embedding_bag {

TASK_EXECUTOR_DECL_TYPES;

void check_correctness(
        const settings_t &s, driver_task_executor_t &task_executor) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_tbldt : s.tbldt)
    for_(const auto &i_dstdt : s.dstdt)
    for_(const auto &i_wtdt : s.wtdt)
    for_(const auto &i_idt : s.idt)
    for_(const auto &i_odt : s.odt)
    for_(const auto &i_alg : s.alg)
    for_(const auto &i_padding_idx : s.padding_idx)
    for_(const auto &i_is_weight : s.is_weight)
    for_(const auto &i_include_last_offset : s.include_last_offset)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for (const auto &i_ctx_exe : s.ctx_exe) {
        auto prb = std::make_shared<prb_t>(s.prb_vdims, i_dir, i_tbldt,
                i_dstdt, i_wtdt, i_idt, i_odt, i_alg, i_padding_idx,
                i_is_weight, i_include_last_offset, i_attr, i_ctx_init,
                i_ctx_exe, s.impl_filter);
        if (s.pattern && !match_regex(prb->str(), s.pattern)) return;

        task_executor.submit(prb, s.perf_template, createit, checkit, doit);
    }
}

int verify_input(const settings_t &s) {
    // Expect exactly 3 inputs: table, indices, offsets.
    static constexpr int n_inputs = 3;
    if (s.prb_vdims.n_inputs() != n_inputs) {
        BENCHDNN_PRINT(0, "%s\n",
                "ERROR: embedding_bag driver: expect problem dimensions in "
                "format `TABLExDIM:INDICESxDIM:OFFSETSxDIM`.");
        SAFE_V(FAIL);
    }

    for (const auto &i_tbldt : s.tbldt) {
        bool ok = i_tbldt == dnnl_f32;
        if (!ok) {
            stringstream_t ss;
            ss << i_tbldt;
            BENCHDNN_PRINT(0, "%s%s%s\n",
                    "ERROR: embedding_bag driver: unsupported table data type `",
                    ss.str().c_str(), "`. Supported: f32.");
            SAFE_V(FAIL);
        }
    }

    for (const auto &i_dstdt : s.dstdt) {
        bool ok = i_dstdt == dnnl_f32;
        if (!ok) {
            stringstream_t ss;
            ss << i_dstdt;
            BENCHDNN_PRINT(0, "%s%s%s\n",
                    "ERROR: embedding_bag driver: unsupported dst data type `",
                    ss.str().c_str(), "`. Supported: f32.");
            SAFE_V(FAIL);
        }
    }

    for (const auto &i_idt : s.idt) {
        if (i_idt != dnnl_s32) {
            stringstream_t ss;
            ss << i_idt;
            BENCHDNN_PRINT(0, "%s%s%s\n",
                    "ERROR: embedding_bag driver: unsupported indices data "
                    "type `",
                    ss.str().c_str(), "`. Supported: s32.");
            SAFE_V(FAIL);
        }
    }

    for (const auto &i_odt : s.odt) {
        if (i_odt != dnnl_s32) {
            stringstream_t ss;
            ss << i_odt;
            BENCHDNN_PRINT(0, "%s%s%s\n",
                    "ERROR: embedding_bag driver: unsupported offsets data "
                    "type `",
                    ss.str().c_str(), "`. Supported: s32.");
            SAFE_V(FAIL);
        }
    }

    for (const auto &i_alg : s.alg) {
        bool ok = i_alg == EMBEDDING_BAG_SUM || i_alg == EMBEDDING_BAG_MEAN
                || i_alg == EMBEDDING_BAG_MAX
                || i_alg == EMBEDDING_BAG_LOOKUP;
        if (!ok) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: embedding_bag driver: unknown algorithm. "
                    "Supported: sum, mean, max, lookup.");
            SAFE_V(FAIL);
        }
    }

    for_(const auto &i_alg : s.alg)
    for (const auto &i_is_weight : s.is_weight) {
        if (i_is_weight && i_alg != EMBEDDING_BAG_SUM
                && i_alg != EMBEDDING_BAG_MEAN) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: embedding_bag driver: `is_weight` is only "
                    "supported with the `sum` and `mean` algorithm.");
            SAFE_V(FAIL);
        }
    }

    return OK;
}

static const std::string help_padding_idx
        = "INT    (Default: `-1`)\n    Index to be ignored during "
          "aggregation; negative value disables padding.\n";
static const std::string help_is_weight
        = "BOOL    (Default: `false`)\n    Enable per-sample weights "
          "(SUM and MEAN only).\n";
static const std::string help_include_last_offset
        = "BOOL    (Default: `false`)\n    Treat the last entry in offsets "
          "as the end of the last bag.\n";

int bench(int argc, char **argv) {
    driver_name = "embeddingbag";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    static driver_task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.dir, def.dir, argv[0])
                || parse_dt(s.tbldt, def.tbldt, argv[0], "tbldt")
                || parse_dt(s.dstdt, def.dstdt, argv[0], "dstdt")
                || parse_dt(s.wtdt, def.wtdt, argv[0], "wtdt")
                || parse_dt(s.idt, def.idt, argv[0], "idt")
                || parse_dt(s.odt, def.odt, argv[0], "odt")
                || parse_vector_option(s.padding_idx, def.padding_idx, atoi,
                        argv[0], "padding_idx", help_padding_idx)
                || parse_vector_option(s.is_weight, def.is_weight,
                        parsers::str2bool, argv[0], "is_weight", help_is_weight)
                || parse_vector_option(s.include_last_offset,
                        def.include_last_offset, parsers::str2bool, argv[0],
                        "include_last_offset", help_include_last_offset)
                || parse_alg(s.alg, def.alg, str2alg, argv[0])
                || parse_driver_shared_settings(s, def, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_prb_vdims(s.prb_vdims, argv[0]);

            SAFE(verify_input(s), WARN);
            s.finalize();
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace embedding_bag
