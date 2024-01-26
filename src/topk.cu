
#include <assert.h>
#include <numeric>
#include <string>
#include <thread>
#include "topk.h"

typedef uint4 group_t;  // uint32_t

template <typename BatchType, int BatchSize>
void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(const __restrict__ uint16_t* docs,
                                                                 const int* doc_lens,
                                                                 const size_t n_docs,
                                                                 uint16_t* query_len,
                                                                 const uint32_t* d_query_batch,
                                                                 short* batch_scores) {
    // each thread process one batch doc-querys pair scoring task
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x, tnum = gridDim.x * blockDim.x;

    if (tid >= n_docs) {
        return;
    }

    __shared__ uint32_t query_mask[BatchSize * QUERY_MASK_SIZE];
#pragma unroll
    for (auto q = threadIdx.x; q < BatchSize * QUERY_MASK_SIZE; q += blockDim.x) {
        query_mask[q] = __ldg(d_query_batch + q);
    }

    __syncthreads();

    for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
        unsigned char tmp_scores[BatchSize] = {0};
        register bool no_more_load = false;
#pragma unroll
        for (auto i = 0; i < MAX_DOC_SIZE / (sizeof(group_t) / sizeof(uint16_t)); i++) {
            if (no_more_load) {
                break;
            }
            register group_t loaded = ((group_t*)docs)[i * n_docs + doc_id];
            register uint16_t* doc_segment = (uint16_t*)(&loaded);
#pragma unroll
            for (auto j = 0; j < sizeof(group_t) / sizeof(uint16_t); j++) {
                if (doc_segment[j] == 0) {
                    no_more_load = true;
                    break;
                }
                uint16_t tindex = doc_segment[j] >> 5;
                uint16_t tpos = doc_segment[j] & 31;
                BatchType mask = reinterpret_cast<BatchType*>(query_mask)[tindex];
                uint32_t* mask_ptr = reinterpret_cast<uint32_t*>(&mask);
#pragma unroll
                for (auto q = 0; q < BatchSize; q++) {
                    bool find = (mask_ptr[q] >> tpos) & static_cast<uint32_t>(1);
                    if (find) {
                        tmp_scores[q]++;
                    }
                }
            }
        }
        for (auto q = 0; q < BatchSize; q++) {
            batch_scores[n_docs * q + doc_id] = static_cast<short>(16384u * tmp_scores[q] / max(query_len[q], doc_lens[doc_id]));
        }
    }
}

__forceinline__ size_t AlignSize(size_t size, size_t align = 128) {
    if (size == 0) {
        return align;
    }
    return (size + align - 1) / align * align;
}

template <typename BatchType>
void PackQuerys(const std::vector<std::vector<uint16_t>>& querys,
                const std::vector<int>& query_idx,
                int batch_size,
                int total_query_len_bytes,
                int batch_query_bytes,
                int n_batches,
                char** d_querys_data) {
    int batch_query_size = batch_query_bytes / sizeof(BatchType);
    CHECK(cudaMalloc(d_querys_data, total_query_len_bytes + batch_query_bytes * n_batches));
    uint16_t* d_query_len = reinterpret_cast<uint16_t*>(*d_querys_data);
    BatchType* d_query_batch = reinterpret_cast<BatchType*>(*d_querys_data + total_query_len_bytes);
    std::vector<uint16_t> h_query_len(querys.size());
    for (int i = 0; i < querys.size(); i++) {
        h_query_len[i] = querys[query_idx[i]].size();
    }
    CHECK(cudaMemcpy(d_query_len, h_query_len.data(), total_query_len_bytes, cudaMemcpyHostToDevice));
    std::vector<BatchType> h_query_batch(batch_query_size * n_batches, 0);
    for (size_t idx = 0; idx < querys.size(); idx += batch_size) {
        BatchType* batch_querys_ptr = h_query_batch.data() + idx / batch_size * batch_query_size;
        if (idx + batch_size > querys.size()) {
            batch_size = querys.size() - idx;
        }
        for (size_t j = 0; j < batch_size; j++) {
            for (auto& q : querys[query_idx[j + idx]]) {
                uint16_t index = q >> 5;
                uint16_t postion = q & 31;
                batch_querys_ptr[batch_size * index + j] |= static_cast<BatchType>(1) << postion;
            }
        }
    }
    CHECK(cudaMemcpy(d_query_batch, h_query_batch.data(), batch_query_bytes * n_batches, cudaMemcpyHostToDevice));
}

std::chrono::time_point<std::chrono::high_resolution_clock> start_time() {
    return std::chrono::high_resolution_clock::now();
}

void end_time(std::chrono::time_point<std::chrono::high_resolution_clock>& t1, std::string message) {
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << message << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms " << std::endl;
}

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>>& querys,
                                    std::vector<std::vector<uint16_t>>& docs,
                                    std::vector<uint16_t>& lens,
                                    std::vector<std::vector<int>>& indices  // shape [querys.size(), TOPK]
) {
    auto n_querys = querys.size();
    auto n_docs = docs.size();
    std::vector<int> s_indices(n_docs);

    int batch_size = 3;
    int total_query_len_bytes = AlignSize(sizeof(uint16_t) * n_querys);
    int batch_query_bytes = AlignSize(batch_size * QUERY_MAX_VALUE / sizeof(uint32_t));
    int n_batches = (n_querys + batch_size - 1) / batch_size;

    std::vector<int> query_idx(n_querys);
    std::iota(query_idx.begin(), query_idx.end(), 0);
    std::sort(query_idx.begin(), query_idx.end(), [&querys](int a, int b) {
        if (querys[a].size() != querys[b].size()) {
            return querys[a].size() < querys[b].size();
        }
        return querys[a].back() < querys[b].back();
    });

    char* d_querys_data = nullptr;
    std::thread* pack_thread;
    pack_thread = new std::thread(PackQuerys<uint32_t>, std::ref(querys), std::ref(query_idx), batch_size, total_query_len_bytes, batch_query_bytes,
                                  n_batches, &d_querys_data);

    uint16_t* d_docs = nullptr;
    int* d_doc_lens = nullptr;

    // copy to device
    CHECK(cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs));
    CHECK(cudaMalloc(&d_doc_lens, sizeof(int) * n_docs));

    uint16_t* h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
    memset(h_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
    std::vector<int> h_doc_lens_vec(n_docs);
    for (int i = 0; i < docs.size(); i++) {
        for (int j = 0; j < docs[i].size(); j++) {
            auto group_sz = sizeof(group_t) / sizeof(uint16_t);
            auto layer_0_offset = j / group_sz;
            auto layer_0_stride = n_docs * group_sz;
            auto layer_1_offset = i;
            auto layer_1_stride = group_sz;
            auto layer_2_offset = j % group_sz;
            auto final_offset = layer_0_offset * layer_0_stride + layer_1_offset * layer_1_stride + layer_2_offset;
            h_docs[final_offset] = docs[i][j];
        }
        h_doc_lens_vec[i] = docs[i].size();
    }

    CHECK(cudaMemcpy(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_doc_lens, h_doc_lens_vec.data(), sizeof(int) * n_docs, cudaMemcpyHostToDevice));

    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);

    cudaSetDevice(0);

    pack_thread->join();
    delete pack_thread;

    for (int i = 0; i < n_docs; ++i) {
        s_indices[i] = i;
    }
    assert(std::is_sorted(lens.begin(), lens.end()));
    int lens_freq[129] = {0};
    int lens_offset[130] = {0};
    for (int i = 0; i < lens.size(); ++i) {
        ++lens_freq[lens[i]];
    }
    for (int i = 0; i < 129; ++i) {
        lens_offset[i + 1] = lens_offset[i] + lens_freq[i];
    }

    std::vector<short> batch_scores(n_docs * batch_size);
    short* d_batch_scores = nullptr;
    CHECK(cudaMalloc(&d_batch_scores, sizeof(short) * n_docs * batch_size));
    indices.resize(n_querys);
    for (size_t idx = 0; idx < n_querys; idx += batch_size) {
        auto t1 = start_time();
        uint16_t* d_query_len = reinterpret_cast<uint16_t*>(d_querys_data) + idx;
        char* d_query_batch = d_querys_data + total_query_len_bytes + idx / batch_size * batch_query_bytes;
        if (idx + batch_size > n_querys) {
            batch_size = n_querys - idx;
        }
        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (n_docs + block - 1) / block;
        if (batch_size == 1) {
            docQueryScoringCoalescedMemoryAccessSampleKernel<uint32_t, 1>
                <<<grid, block>>>(d_docs, d_doc_lens, n_docs, d_query_len, reinterpret_cast<uint32_t*>(d_query_batch), d_batch_scores);
        } else if (batch_size == 2) {
            docQueryScoringCoalescedMemoryAccessSampleKernel<uint2, 2>
                <<<grid, block>>>(d_docs, d_doc_lens, n_docs, d_query_len, reinterpret_cast<uint32_t*>(d_query_batch), d_batch_scores);
        } else if (batch_size == 3) {
            docQueryScoringCoalescedMemoryAccessSampleKernel<uint3, 3>
                <<<grid, block>>>(d_docs, d_doc_lens, n_docs, d_query_len, reinterpret_cast<uint32_t*>(d_query_batch), d_batch_scores);
        } else {
            docQueryScoringCoalescedMemoryAccessSampleKernel<uint4, 4>
                <<<grid, block>>>(d_docs, d_doc_lens, n_docs, d_query_len, reinterpret_cast<uint32_t*>(d_query_batch), d_batch_scores);
        }
        cudaDeviceSynchronize();

        end_time(t1, "QueryScoring kernel cost: ");
        t1 = start_time();

        CHECK(cudaMemcpy(batch_scores.data(), d_batch_scores, sizeof(short) * n_docs * batch_size, cudaMemcpyDeviceToHost));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR: %s:%d,", __FILE__, __LINE__);
            printf("CUDA Error[%d]: %s\n", err, cudaGetErrorString(err));
            exit(1);
        }

        end_time(t1, "Copy scores cost: ");

        t1 = start_time();
        for (int q = 0; q < batch_size; q++) {
            short* scores = batch_scores.data() + q * n_docs;

            std::vector<int> temp_indices(s_indices);
            std::partial_sort(temp_indices.begin(), temp_indices.begin() + TOPK, temp_indices.end(), [&scores](const int& a, const int& b) {
                if (scores[a] != scores[b]) {
                    return scores[a] > scores[b];
                }
                return a < b;
            });
            std::vector<int> s_ans(temp_indices.begin(), temp_indices.begin() + TOPK);
            indices[query_idx[idx + q]].swap(s_ans);
        }
        end_time(t1, "Partial sort cost: ");
    }

    // deallocation
    cudaFree(d_docs);
    cudaFree(d_doc_lens);
    cudaFree(d_querys_data);
    cudaFree(d_batch_scores);
    free(h_docs);
}
