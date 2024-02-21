#pragma once
#include <cuda.h>
#include <algorithm>
#include <iostream>
#include <vector>

#define MAX_DOC_SIZE 128
#define MAX_QUERY_SIZE 4096
#define N_THREADS_IN_ONE_BLOCK 512
#define TOPK 100
#define QUERY_MAX_VALUE 50000
#define LONG_SCORES_SCALE 1381200000000
#define QUERY_MASK_SIZE 1568
#define MAX_BATCH_SIZE 4
#define WINDOW_SIZE 10

#define CHECK(call)                                                          \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSuccess) {                                          \
            printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
            printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                         \
        }                                                                    \
    }

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>>& query,
                                    std::vector<std::vector<uint16_t>>& docs,
                                    std::vector<uint16_t>& lens,
                                    std::vector<std::vector<int>>& indices);
