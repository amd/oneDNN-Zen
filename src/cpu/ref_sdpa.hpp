/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef CPU_REF_SDPA_HPP
#define CPU_REF_SDPA_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/sdpa_pd.hpp"
#include "common/sdpa_types.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// Dummy CPU SDPA forward primitive (architecture PoC).
// Validates dispatch + API plumbing. Kernel body is a stub that prints
// and returns success without computing anything.
struct ref_sdpa_fwd_t : public primitive_t {
    struct pd_t : public sdpa_fwd_pd_t {
        using sdpa_fwd_pd_t::sdpa_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_sdpa_fwd_t);

        status_t init(engine_t *engine);
    };

    ref_sdpa_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override { return status::success; }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
