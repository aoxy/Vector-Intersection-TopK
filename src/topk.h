#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <cuda.h>

#define MAX_DOC_SIZE 128
#define MAX_QUERY_SIZE 4096
#define N_THREADS_IN_ONE_BLOCK 512
#define TOPK 100
#define MAX_ID 50000
#define N_THREADS_CPU 16
#define QUERY_GROUP_SIZE (sizeof(uint16_t) / sizeof(uint8_t))

#define GET_BYTE_0(WORD) (WORD & 255u)
#define GET_BYTE_1(WORD) ((WORD >> 8) & 255u)
#define GET_BYTE_2(WORD) ((WORD >> 16) & 255u)
#define GET_BYTE_3(WORD) ((WORD >> 24) & 255u)
#define GET_BYTE_4(WORD) ((WORD >> 32) & 255u)
#define GET_BYTE_5(WORD) ((WORD >> 40) & 255u)
#define GET_BYTE_6(WORD) ((WORD >> 48) & 255u)
#define GET_BYTE_7(WORD) ((WORD >> 56) & 255u)

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &query,
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices);

void doc_query_scoring_cpu_function(std::vector<std::vector<uint16_t>> &query,
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<std::vector<int>> &indices);

int compare(std::vector<std::vector<int>> &indices_1, std::vector<std::vector<int>> &indices_2);