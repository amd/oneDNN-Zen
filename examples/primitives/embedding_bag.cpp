/*******************************************************************************
* Copyright 2020 Intel Corporation
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

/// @example embedding_bag.cpp
/// > Annotated version: @ref embedding_bag_example_cpp

/// @page embedding_bag_example_cpp_brief
/// @brief This C++ API example demonstrates how to create and execute a
/// [Embedding_Bag](@ref dev_guide_embedding_bag) primitive in forward
/// inference propagation mode.
/// @page embedding_bag_example_cpp Embedding_Bag Primitive Example
/// \copybrief embedding_bag_example_cpp_brief
///
/// @include embedding_bag.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

void embedding_bag_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Create table
    const memory::dim table_width = 8, //table width
                      table_rows  = 1000; //table rows
    memory::dims table_dims = {table_rows, table_width};
    std::vector<float> table_data(product(table_dims));

    std::generate(table_data.begin(), table_data.end(), []() {
        static int i = 0;
        //return std::cos(i++ / 10.f);
        return (float)(i++);
    });

    auto table_md = memory::desc(
            table_dims, memory::data_type::f32, memory::format_tag::ab);
    auto table_mem = memory(table_md, engine);

    write_to_dnnl_memory(table_data.data(), table_mem);

    // Create indices
    const memory::dim indices_len = 10; // indices length
    memory::dims indices_dims = {indices_len};
    std::vector<int32_t> indices_data(indices_len);

    std::generate(indices_data.begin(), indices_data.end(), []() {
        static int32_t i = 0;
        return i++;
    });

    auto indices_md = memory::desc(
            indices_dims, memory::data_type::s32, memory::format_tag::a);
    auto indices_mem = memory(indices_md, engine);

    write_to_dnnl_memory(indices_data.data(), indices_mem);

    // Create offsets
    const memory::dim offsets_len = 4; // offset length
    memory::dims offsets_dims = {offsets_len};
    std::vector<uint32_t> offsets_data(offsets_len);

    std::generate(offsets_data.begin(), offsets_data.end(), []() {
        static int32_t i = 0;
        int32_t j = i;
        i += 2;
        return j;
    });

    auto offsets_md = memory::desc(
            offsets_dims, memory::data_type::s32, memory::format_tag::a);
    auto offsets_mem = memory(offsets_md, engine);

    write_to_dnnl_memory(offsets_data.data(), offsets_mem);

    // create dst memory
    const memory::dim dst_width = table_width, //dst width
                      dst_rows  = offsets_len + 1; //dst rows
    memory::dims dst_dims = {dst_rows, dst_width};

    std::vector<float> dst_data(product(dst_dims));
    std::fill(dst_data.begin(), dst_data.end(), 0.0f);

    auto dst_md = memory::desc(
            dst_dims, memory::data_type::f32, memory::format_tag::ab);
    auto dst_mem = memory(dst_md, engine);

    // Create primitive descriptor.
    auto embedding_bag_pd = embedding_bag_forward::primitive_desc(engine,
            prop_kind::forward_inference, algorithm::embedding_bag_sum,
            table_md, indices_md, offsets_md, dst_md, -1, false);

    // Create the primitive.
    auto embedding_bag_prim = embedding_bag_forward(embedding_bag_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> embedding_bag_args;
    embedding_bag_args.insert({DNNL_ARG_EMBEDDING_BAG_TABLE, table_mem});
    embedding_bag_args.insert({DNNL_ARG_EMBEDDING_BAG_INDICES, indices_mem});
    embedding_bag_args.insert({DNNL_ARG_EMBEDDING_BAG_OFFSETS, offsets_mem});
    embedding_bag_args.insert({DNNL_ARG_EMBEDDING_BAG_DST, dst_mem});

    // Primitive execution.
    embedding_bag_prim.execute(engine_stream, embedding_bag_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), dst_mem);

    // Check correctness
    for (uint64_t i = 0; i < dst_data.size(); ++i)
      std::cout << "i:" << i << ":" << dst_data[i] << std::endl;
}

int main(int argc, char **argv) {
    return handle_example_errors(
            embedding_bag_example, parse_engine_kind(argc, argv));
}
