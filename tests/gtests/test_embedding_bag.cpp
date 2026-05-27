/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

using tag = memory::format_tag;
using dt = memory::data_type;

struct embag_test_params_t {
    prop_kind aprop_kind;
    algorithm aalgorithm;
    dt tbl_dt;
    dt indices_dt;
    dt offsets_dt;
    dt weights_dt;
    dt dst_dt;
    memory::dims tbl_dims;
    memory::dims indices_dims;
    memory::dims offsets_dims;
    bool is_weight;
    bool include_last_offset;
    int  padding_idx;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

class embag_test_t : public ::testing::TestWithParam<embag_test_params_t> {
private:
    embag_test_params_t p;
    //memory tbl, indices, offsets, dst, wt;

protected:
    void SetUp() override {
        p = ::testing::TestWithParam<embag_test_params_t>::GetParam();

        SKIP_IF_CUDA(true, "GPU implementation is unsupported");
        SKIP_IF_HIP(true, "GPU implementation is unsupported");

        SKIP_IF(!is_fwd(p.aprop_kind), "Unsupported prop kind");

        SKIP_IF(unsupported_data_type(p.tbl_dt)
                        || unsupported_data_type(p.indices_dt)
                        || unsupported_data_type(p.offsets_dt)
                        || unsupported_data_type(p.weights_dt)
                        || unsupported_data_type(p.dst_dt),
                "Engine does not support this data type.");

        SKIP_IF(p.tbl_dt != dt::f32,
                "Only f32 tables are supported");

        SKIP_IF(p.indices_dt != dt::s32
                        || p.offsets_dt != dt::s32,
                "Only s32 indices and offsets are supported");

        SKIP_IF(p.tbl_dt != p.dst_dt && p.tbl_dt != dt::undef
                        && p.dst_dt != dt::undef,
                "Unsupported different data types for table and "
                "destination");

        if (p.is_weight) {
            SKIP_IF(p.weights_dt != dt::f32,
                    "Only f32 weights are supported");
        }

        catch_expected_failures(
                [&]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Forward() {

        using pd_t = embedding_bag_forward::primitive_desc;

        auto eng = get_test_engine();
        auto strm = make_stream(eng);
        prop_kind pk = p.aprop_kind;

        auto dst_width = p.tbl_dims[1];

        auto dst_rows  = p.aalgorithm == dnnl::algorithm::embedding_bag_lookup ?
            p.indices_dims[0] :
            p.include_last_offset ? p.offsets_dims[0] -1:
            p.offsets_dims[0];
        memory::dims dst_dims = {dst_rows, dst_width};

        auto tbl_md = memory::desc(p.tbl_dims, p.tbl_dt, tag::ab);
        auto indices_md = memory::desc(p.indices_dims, p.indices_dt, tag::a);
        auto offsets_md = memory::desc(p.offsets_dims, p.offsets_dt, tag::a);
        auto dst_md = memory::desc(dst_dims, p.dst_dt, tag::ab);

        // default pd ctor
        auto pd = pd_t();

        // regular pd ctor
        auto wt_md = memory::desc();
        if (p.is_weight) {
            wt_md = memory::desc(p.indices_dims, p.weights_dt, tag::a);
            pd = pd_t(eng, pk, p.aalgorithm, tbl_md, indices_md,
                      offsets_md, wt_md, dst_md, p.padding_idx,
                      p.include_last_offset);

        } else {
            pd = pd_t(eng, pk, p.aalgorithm, tbl_md, indices_md,
                      offsets_md, dst_md, p.padding_idx,
                      p.include_last_offset);
        }

        // default primitive ctor
        auto embedding_bag = embedding_bag_forward();
        // regular primitive ctor
        embedding_bag = embedding_bag_forward(pd);

        // check primitive kind is softmax
        ASSERT_TRUE(embedding_bag.get_kind() == primitive::kind::embedding_bag);

        // query primitive parameters
        ASSERT_EQ(pd.get_prop_kind(), pk);
        ASSERT_EQ(pd.get_algorithm(), p.aalgorithm);

        // query the parameters
        // ASSERT_EQ(pd.is_weight(), p.is_weight);
        // ASSERT_EQ(pd.include_last_offset(), p.include_last_offset);
        // ASSERT_EQ(pd.padding_idx(), p.padding_idx);

        memory tbl, indices, offsets, dst, wt;
        tbl = test::make_memory(tbl_md, eng);
        dst = test::make_memory(dst_md, eng);
        indices = test::make_memory(indices_md, eng);
        offsets = test::make_memory(offsets_md, eng);

        fill_data(p.tbl_dt, tbl, 1, 1);
        fill_mem(p.indices_dt, indices, 1, (p.tbl_dims[0] -1));
        fill_mem(p.offsets_dt, offsets, 1, (p.indices_dims[0] -2));

        // test execution
        if (p.is_weight) {
            wt = test::make_memory(wt_md, eng);
            fill_data(p.weights_dt, wt, 1, 1);
            embedding_bag.execute(strm,
                                  {{DNNL_ARG_EMBEDDING_BAG_TABLE, tbl},
                                   {DNNL_ARG_EMBEDDING_BAG_INDICES, indices},
                                   {DNNL_ARG_EMBEDDING_BAG_OFFSETS, offsets},
                                   {DNNL_ARG_EMBEDDING_BAG_WEIGHTS, wt},
                                   {DNNL_ARG_EMBEDDING_BAG_DST, dst}});
        } else {
            embedding_bag.execute(strm,
                                  {{DNNL_ARG_EMBEDDING_BAG_TABLE, tbl},
                                   {DNNL_ARG_EMBEDDING_BAG_INDICES, indices},
                                   {DNNL_ARG_EMBEDDING_BAG_OFFSETS, offsets},
                                   {DNNL_ARG_EMBEDDING_BAG_DST, dst}});
        }

        strm.wait();
    }

    void Test() {
        Forward();
    }

    bool is_fwd(prop_kind pk) const {
        return pk == prop_kind::forward_inference;
    }

    void fill_mem(memory::data_type dtype, const memory& mem,
                    int32_t start, int32_t stop, int32_t gap = 0) {

        auto dims = mem.get_desc().get_dims()[0];

        switch (dtype) {
            case memory::data_type::s32:
                {
                    auto data_ptr = map_memory<int32_t>(mem);
                    if (!gap) gap = (stop - start)/dims;
                    data_ptr[0] = start;
                    for (int32_t i = 1; i < dims; ++i) {
                        data_ptr[i] = stop < (data_ptr[i-1] + gap) ?
                            stop : (data_ptr[i-1] + gap);
                    }
                }
                break;
            default: assert("unsupported data type"); break;
        }
    }
};

using tp = embag_test_params_t;

static const auto training = prop_kind::forward_training;
static const auto inference = prop_kind::forward_inference;
static const auto backward = prop_kind::backward_data;
static const auto alg_embag_add = algorithm::embedding_bag_sum;
static const auto alg_embag_mean = algorithm::embedding_bag_mean;
static const auto alg_embag_max = algorithm::embedding_bag_max;
static const auto alg_embag_lookup = algorithm::embedding_bag_lookup;

TEST_P(embag_test_t, TestsEmbag) {}

// ---------------------------------------------------------------------------
// Sum algorithm — no weights
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(Test_Embag_Sum_NW, embag_test_t,
        ::testing::Values(
                // baseline: small table, few indices/offsets
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {10}, {4},
                    false, false, -1},
                // wider embedding dimension
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 128}, {32}, {8},
                    false, false, -1},
                // large table (many embeddings)
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {1000, 64}, {200}, {16},
                    false, false, -1},
                // single bag (one offset)
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {50, 16}, {5}, {1},
                    false, false, -1},
                // many bags, one index per bag
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {64, 32}, {8}, {8},
                    false, false, -1},
                // include_last_offset=true: offsets[last] acts as sentinel end
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {10}, {5},
                    false, true, -1},
                // padding_idx=0: first row of table should be skipped
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {10}, {4},
                    false, false, 0},
                // padding_idx mid-range
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {20}, {4},
                    false, false, 5}));

// ---------------------------------------------------------------------------
// Mean algorithm — no weights
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(Test_Embag_Mean_NW, embag_test_t,
        ::testing::Values(
                // baseline
                tp {inference, alg_embag_mean, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {10}, {4},
                    false, false, -1},
                // wide embedding
                tp {inference, alg_embag_mean, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {256, 64}, {64}, {16},
                    false, false, -1},
                // include_last_offset
                tp {inference, alg_embag_mean, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {10}, {5},
                    false, true, -1},
                // padding_idx active
                tp {inference, alg_embag_mean, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 16}, {12}, {4},
                    false, false, 3}));

// ---------------------------------------------------------------------------
// Max algorithm — no weights
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(Test_Embag_Max_NW, embag_test_t,
        ::testing::Values(
                // baseline
                tp {inference, alg_embag_max, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {10}, {4},
                    false, false, -1},
                // larger dims
                tp {inference, alg_embag_max, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {512, 32}, {128}, {8},
                    false, false, -1},
                // include_last_offset
                tp {inference, alg_embag_max, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {10}, {5},
                    false, true, -1},
                // padding_idx
                tp {inference, alg_embag_max, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {15}, {4},
                    false, false, 2}));

// ---------------------------------------------------------------------------
// Lookup algorithm — no weights (behaves like embedding, one index per bag)
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(Test_Embag_Lookup_NW, embag_test_t,
        ::testing::Values(
                // baseline
                tp {inference, alg_embag_lookup, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {10}, {4},
                    false, false, -1},
                // wide embedding
                tp {inference, alg_embag_lookup, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {128, 128}, {16}, {8},
                    false, false, -1}));

// ---------------------------------------------------------------------------
// With weights (sum and mean only — max and lookup ignore weights)
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(Test_Embag_Weighted, embag_test_t,
        ::testing::Values(
                // sum with weights, baseline
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {10}, {4},
                    true, false, -1},
                // sum with weights, wider embedding
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {256, 64}, {32}, {8},
                    true, false, -1},
                // sum with weights + include_last_offset
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {10}, {5},
                    true, true, -1},
                // sum with weights + padding_idx
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 16}, {20}, {4},
                    true, false, 4},
                // mean with weights, baseline
                tp {inference, alg_embag_mean, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {10}, {4},
                    true, false, -1},
                // mean with weights, large table
                tp {inference, alg_embag_mean, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {1000, 32}, {100}, {16},
                    true, false, -1}));

// ---------------------------------------------------------------------------
// Unsupported / invalid configurations — must fail
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(Test_Embag_Invalid, embag_test_t,
        ::testing::Values(
                // backward prop is not supported
                tp {backward, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {10}, {4},
                    false, false, -1,
                    true, dnnl_invalid_arguments},
                // non-f32 table (bf16) is not supported
                tp {inference, alg_embag_add, dt::bf16, dt::s32, dt::s32,
                    dt::f32, dt::bf16, {100, 8}, {10}, {4},
                    false, false, -1,
                    true, dnnl_invalid_arguments},
                // f32 indices are not supported (must be s32)
                tp {inference, alg_embag_add, dt::f32, dt::f32, dt::s32,
                    dt::f32, dt::f32, {100, 8}, {10}, {4},
                    false, false, -1,
                    true, dnnl_invalid_arguments},
                // f32 offsets are not supported (must be s32)
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::f32,
                    dt::f32, dt::f32, {100, 8}, {10}, {4},
                    false, false, -1,
                    true, dnnl_invalid_arguments},
                // mismatched table/dst dtypes
                tp {inference, alg_embag_add, dt::f32, dt::s32, dt::s32,
                    dt::f32, dt::bf16, {100, 8}, {10}, {4},
                    false, false, -1,
                    true, dnnl_invalid_arguments}));
} // namespace dnnl
