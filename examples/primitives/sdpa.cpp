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

/// @example sdpa.cpp
/// SDPA as a first-class CPU primitive.
///
/// User code is one primitive_desc + one primitive + one execute call.
/// No bmm/softmax/bmm orchestration on the user side.

#include <iostream>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

void sdpa_example() {
    // --- engine + stream ---
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    // --- shapes: 4D (N, H, S, D) ---
    const memory::dim N = 2, H = 4, Sq = 16, Skv = 16, D = 32;
    const memory::dims q_dims = {N, H, Sq, D}; // (N, H, Sq,  D)
    const memory::dims k_dims
            = {N, H, D, Skv}; // (N, H, D,  Skv) -- K logical (transposed)
    const memory::dims v_dims = {N, H, Skv, D}; // (N, H, Skv, D)
    const memory::dims o_dims = {N, H, Sq, D}; // (N, H, Sq,  D)

    // --- describe Q, K, V, dst ---
    // K is described as a TRANSPOSE VIEW: logical dims (N, H, D, Skv) but with
    // format tag abdc, meaning the bytes are actually a natural (N, H, Skv, D)
    // buffer (Skv-major, D contiguous). This is exactly what a framework like
    // PyTorch hands over after K.transpose(-2, -1) -- a strided view, no copy.
    // Q/V/O are plain dense abcd.
    memory::desc q_md(q_dims, memory::data_type::f32, memory::format_tag::abcd);
    memory::desc k_md(k_dims, memory::data_type::f32, memory::format_tag::abdc);
    memory::desc v_md(v_dims, memory::data_type::f32, memory::format_tag::abcd);
    memory::desc o_md(o_dims, memory::data_type::f32, memory::format_tag::abcd);

    std::cout << "[example] Creating dnnl::sdpa::primitive_desc on CPU...\n";

    // --- one primitive_desc, one primitive ---
    auto sdpa_pd = sdpa::primitive_desc(eng, q_md, k_md, v_md, o_md);
    auto sdpa_prim = sdpa(sdpa_pd);

    // --- buffers ---
    auto m_q = memory(q_md, eng);
    auto m_k = memory(k_md, eng);
    auto m_v = memory(v_md, eng);
    auto m_o = memory(o_md, eng);

    // K data is laid out in its natural (N, H, Skv, D) order -- same bytes a
    // framework would already have. The abdc descriptor above reinterprets it
    // as the transposed (N, H, D, Skv) logical tensor without moving anything.
    std::vector<float> q_data(product(q_dims), 0.1f);
    std::vector<float> k_data(
            product(k_dims), 0.2f); // natural (N,H,Skv,D) bytes
    std::vector<float> v_data(product(v_dims), 0.3f);
    std::vector<float> o_data(product(o_dims), 0.0f);

    write_to_dnnl_memory(q_data.data(), m_q);
    write_to_dnnl_memory(k_data.data(), m_k);
    write_to_dnnl_memory(v_data.data(), m_v);

    std::cout << "[example] Executing dnnl::sdpa primitive on CPU...\n";

    // --- single execute call (no bmm/softmax/bmm at the user side) ---
    sdpa_prim.execute(s,
            {{DNNL_ARG_SRC_0, m_q}, // QUERIES
                    {DNNL_ARG_SRC_1, m_k}, // KEYS
                    {DNNL_ARG_SRC_2, m_v}, // VALUES
                    {DNNL_ARG_DST, m_o}});
    s.wait();

    std::cout << "[example] SDPA primitive call returned successfully.\n";
}

int main(int argc, char **argv) {
    return handle_example_errors({engine::kind::cpu}, sdpa_example);
}
