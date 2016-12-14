#pragma once

struct kernel_properties {
  int tile_m;
  int tile_n;
  int tile_k;
  int vA;
  int vB;
  int vC;
  int div;
  bool op;
  int threads;
  std::vector<int> shared_sizes;
  std::string tile_string;
};

kernel_properties kernel_properties_[6] = {
  {128, 128, 8, 4, 4, 1, 2, false, 256, {0}, "128x128x8"},
  {32, 32,  32, 4, 4, 1, 4, false, 128, {0, 1 << 14}, "32x32x32"},
  {32, 64,  32, 8, 4, 4, 4, true,  128, {0, 1 << 13}, "32x64x32"},
  {32, 32,  64, 8, 8, 4, 4, true,  128, {0}, "32x32x64"},
  {16, 64,  64, 8, 4, 4, 4, true,  128, {0}, "16x64x64"},
  {16, 64,  64, 8, 8, 4, 4, true,  128, {0}, "16x64x64"},
};

std::unordered_map<std::string, std::unordered_map<std::string, std::vector<kernel_properties>>>
  selections = {
    {"s", {
                {"TN", {kernel_properties_[0], kernel_properties_[1]}},
                {"NN", {kernel_properties_[0], kernel_properties_[1]}},
                {"NT", {kernel_properties_[0], kernel_properties_[1]}},
                {"TT", {kernel_properties_[0], kernel_properties_[1]}}
                }
    },
    {"h", {
                {"TN", {kernel_properties_[0], kernel_properties_[1]}},
                {"NN", {kernel_properties_[0], kernel_properties_[1], kernel_properties_[2], kernel_properties_[4]}},
                {"NT", {kernel_properties_[0], kernel_properties_[1], kernel_properties_[3], kernel_properties_[5]}},
                {"TT", {kernel_properties_[0], kernel_properties_[1]}}
                }
    }
  };

std::unordered_map<std::string, const uint8_t*> kernels_60 = {
                 {"hgemm_16x64x64_NN_sm_60", hgemm_16x64x64_NN_sm_60},
                 {"hgemm_16x64x64_NN_vec_sm_60", hgemm_16x64x64_NN_vec_sm_60},
                 {"hgemm_16x64x64_NT_sm_60", hgemm_16x64x64_NT_sm_60},
                 {"hgemm_16x64x64_NT_vec_sm_60", hgemm_16x64x64_NT_vec_sm_60},
                 {"hgemm_32x32x64_NT_sm_60", hgemm_32x32x64_NT_sm_60},
                 {"hgemm_32x32x64_NT_vec_sm_60", hgemm_32x32x64_NT_vec_sm_60},
                 {"hgemm_32x64x32_NN_sm_60", hgemm_32x64x32_NN_sm_60},
                 {"hgemm_32x64x32_NN_vec_sm_60", hgemm_32x64x32_NN_vec_sm_60},
                 {"hgemm_128x128x8_NN_sm_60", hgemm_128x128x8_NN_sm_60},
                 {"hgemm_128x128x8_TN_sm_60", hgemm_128x128x8_TN_sm_60},
                 {"hgemm_128x128x8_NT_sm_60", hgemm_128x128x8_NT_sm_60},
                 {"hgemm_128x128x8_TT_sm_60", hgemm_128x128x8_TT_sm_60},
                 {"hgemm_128x128x8_NN_vec_sm_60", hgemm_128x128x8_NN_vec_sm_60},
                 {"hgemm_128x128x8_TN_vec_sm_60", hgemm_128x128x8_TN_vec_sm_60},
                 {"hgemm_128x128x8_NT_vec_sm_60", hgemm_128x128x8_NT_vec_sm_60},
                 {"hgemm_128x128x8_TT_vec_sm_60", hgemm_128x128x8_TT_vec_sm_60},
                 {"hgemm_32x32x32_NN_sm_60", hgemm_32x32x32_NN_sm_60},
                 {"hgemm_32x32x32_TN_sm_60", hgemm_32x32x32_TN_sm_60},
                 {"hgemm_32x32x32_NT_sm_60", hgemm_32x32x32_NT_sm_60},
                 {"hgemm_32x32x32_TT_sm_60", hgemm_32x32x32_TT_sm_60},
                 {"hgemm_32x32x32_NN_vec_sm_60", hgemm_32x32x32_NN_vec_sm_60},
                 {"hgemm_32x32x32_TN_vec_sm_60", hgemm_32x32x32_TN_vec_sm_60},
                 {"hgemm_32x32x32_NT_vec_sm_60", hgemm_32x32x32_NT_vec_sm_60},
                 {"hgemm_32x32x32_TT_vec_sm_60", hgemm_32x32x32_TT_vec_sm_60},
                 {"sgemm_128x128x8_NN_sm_60", sgemm_128x128x8_NN_sm_60},
                 {"sgemm_128x128x8_TN_sm_60", sgemm_128x128x8_TN_sm_60},
                 {"sgemm_128x128x8_NT_sm_60", sgemm_128x128x8_NT_sm_60},
                 {"sgemm_128x128x8_TT_sm_60", sgemm_128x128x8_TT_sm_60},
                 {"sgemm_128x128x8_NN_vec_sm_60", sgemm_128x128x8_NN_vec_sm_60},
                 {"sgemm_128x128x8_TN_vec_sm_60", sgemm_128x128x8_TN_vec_sm_60},
                 {"sgemm_128x128x8_NT_vec_sm_60", sgemm_128x128x8_NT_vec_sm_60},
                 {"sgemm_128x128x8_TT_vec_sm_60", sgemm_128x128x8_TT_vec_sm_60},
                 {"sgemm_32x32x32_NN_sm_60", sgemm_32x32x32_NN_sm_60},
                 {"sgemm_32x32x32_TN_sm_60", sgemm_32x32x32_TN_sm_60},
                 {"sgemm_32x32x32_NT_sm_60", sgemm_32x32x32_NT_sm_60},
                 {"sgemm_32x32x32_TT_sm_60", sgemm_32x32x32_TT_sm_60},
                 {"sgemm_32x32x32_NN_vec_sm_60", sgemm_32x32x32_NN_vec_sm_60},
                 {"sgemm_32x32x32_TN_vec_sm_60", sgemm_32x32x32_TN_vec_sm_60},
                 {"sgemm_32x32x32_NT_vec_sm_60", sgemm_32x32x32_NT_vec_sm_60},
                 {"sgemm_32x32x32_TT_vec_sm_60", sgemm_32x32x32_TT_vec_sm_60},
               };

std::unordered_map<std::string, const uint8_t*> kernels_50 = {
                 {"hgemm_16x64x64_NN_sm_50", hgemm_16x64x64_NN_sm_50},
                 {"hgemm_16x64x64_NN_vec_sm_50", hgemm_16x64x64_NN_vec_sm_50},
                 {"hgemm_16x64x64_NT_sm_50", hgemm_16x64x64_NT_sm_50},
                 {"hgemm_16x64x64_NT_vec_sm_50", hgemm_16x64x64_NT_vec_sm_50},
                 {"hgemm_32x32x64_NT_sm_50", hgemm_32x32x64_NT_sm_50},
                 {"hgemm_32x32x64_NT_vec_sm_50", hgemm_32x32x64_NT_vec_sm_50},
                 {"hgemm_32x64x32_NN_sm_50", hgemm_32x64x32_NN_sm_50},
                 {"hgemm_32x64x32_NN_vec_sm_50", hgemm_32x64x32_NN_vec_sm_50},
                 {"hgemm_128x128x8_NN_sm_50", hgemm_128x128x8_NN_sm_50},
                 {"hgemm_128x128x8_TN_sm_50", hgemm_128x128x8_TN_sm_50},
                 {"hgemm_128x128x8_NT_sm_50", hgemm_128x128x8_NT_sm_50},
                 {"hgemm_128x128x8_TT_sm_50", hgemm_128x128x8_TT_sm_50},
                 {"hgemm_128x128x8_NN_vec_sm_50", hgemm_128x128x8_NN_vec_sm_50},
                 {"hgemm_128x128x8_TN_vec_sm_50", hgemm_128x128x8_TN_vec_sm_50},
                 {"hgemm_128x128x8_NT_vec_sm_50", hgemm_128x128x8_NT_vec_sm_50},
                 {"hgemm_128x128x8_TT_vec_sm_50", hgemm_128x128x8_TT_vec_sm_50},
                 {"hgemm_32x32x32_NN_sm_50", hgemm_32x32x32_NN_sm_50},
                 {"hgemm_32x32x32_TN_sm_50", hgemm_32x32x32_TN_sm_50},
                 {"hgemm_32x32x32_NT_sm_50", hgemm_32x32x32_NT_sm_50},
                 {"hgemm_32x32x32_TT_sm_50", hgemm_32x32x32_TT_sm_50},
                 {"hgemm_32x32x32_NN_vec_sm_50", hgemm_32x32x32_NN_vec_sm_50},
                 {"hgemm_32x32x32_TN_vec_sm_50", hgemm_32x32x32_TN_vec_sm_50},
                 {"hgemm_32x32x32_NT_vec_sm_50", hgemm_32x32x32_NT_vec_sm_50},
                 {"hgemm_32x32x32_TT_vec_sm_50", hgemm_32x32x32_TT_vec_sm_50},
                 {"sgemm_128x128x8_NN_sm_50", sgemm_128x128x8_NN_sm_50},
                 {"sgemm_128x128x8_TN_sm_50", sgemm_128x128x8_TN_sm_50},
                 {"sgemm_128x128x8_NT_sm_50", sgemm_128x128x8_NT_sm_50},
                 {"sgemm_128x128x8_TT_sm_50", sgemm_128x128x8_TT_sm_50},
                 {"sgemm_128x128x8_NN_vec_sm_50", sgemm_128x128x8_NN_vec_sm_50},
                 {"sgemm_128x128x8_TN_vec_sm_50", sgemm_128x128x8_TN_vec_sm_50},
                 {"sgemm_128x128x8_NT_vec_sm_50", sgemm_128x128x8_NT_vec_sm_50},
                 {"sgemm_128x128x8_TT_vec_sm_50", sgemm_128x128x8_TT_vec_sm_50},
                 {"sgemm_32x32x32_NN_sm_50", sgemm_32x32x32_NN_sm_50},
                 {"sgemm_32x32x32_TN_sm_50", sgemm_32x32x32_TN_sm_50},
                 {"sgemm_32x32x32_NT_sm_50", sgemm_32x32x32_NT_sm_50},
                 {"sgemm_32x32x32_TT_sm_50", sgemm_32x32x32_TT_sm_50},
                 {"sgemm_32x32x32_NN_vec_sm_50", sgemm_32x32x32_NN_vec_sm_50},
                 {"sgemm_32x32x32_TN_vec_sm_50", sgemm_32x32x32_TN_vec_sm_50},
                 {"sgemm_32x32x32_NT_vec_sm_50", sgemm_32x32x32_NT_vec_sm_50},
                 {"sgemm_32x32x32_TT_vec_sm_50", sgemm_32x32x32_TT_vec_sm_50},
               };
