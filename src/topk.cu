
#include <thread>
#include <string>
#include "topk.h"

typedef uint4 group_t;  // uint32_t

template <typename BatchType, int BatchSize>
void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(const __restrict__ uint16_t* docs,
                                                                 const int* doc_lens,
                                                                 const size_t n_docs,
                                                                 uint16_t* query_len,
                                                                 const BatchType* d_query_batch,
                                                                 float* batch_scores) {
    // each thread process one batch doc-querys pair scoring task
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x, tnum = gridDim.x * blockDim.x;

    if (tid >= n_docs) {
        return;
    }

    __shared__ uint16_t s_query_lens[BatchSize];
#pragma unroll
    for (auto q = threadIdx.x; q < BatchSize; q += blockDim.x) {
        s_query_lens[q] = query_len[q];
    }

    __syncthreads();

    for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
        float tmp_scores[BatchSize] = {0.};
        register bool no_more_load = false;

        for (auto i = 0; i < MAX_DOC_SIZE / (sizeof(group_t) / sizeof(uint16_t)); i++) {
            if (no_more_load) {
                break;
            }
            register group_t loaded = ((group_t*)docs)[i * n_docs + doc_id];
            register uint16_t* doc_segment = (uint16_t*)(&loaded);
            for (auto j = 0; j < sizeof(group_t) / sizeof(uint16_t); j++) {
                if (doc_segment[j] == 0) {
                    no_more_load = true;
                    break;
                }
                BatchType querys = d_query_batch[doc_segment[j]];
                if (querys == 0) {
                    continue;
                }
                for (auto q = 0; q < BatchSize; q++) {
                    bool find = querys & static_cast<BatchType>(1) << q;
                    if (find) {
                        tmp_scores[q]++;
                    }
                }
            }
        }
        for (auto q = 0; q < BatchSize; q++) {
            batch_scores[n_docs * q + doc_id] = tmp_scores[q] / max(s_query_lens[q], doc_lens[doc_id]);
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
void PackQuerys(std::vector<std::vector<uint16_t>>& querys,
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
        h_query_len[i] = querys[i].size();
    }
    CHECK(cudaMemcpy(d_query_len, h_query_len.data(), total_query_len_bytes, cudaMemcpyHostToDevice));
    std::vector<BatchType> h_query_batch(batch_query_size * n_batches, 0);
    for (size_t idx = 0; idx < querys.size(); idx += batch_size) {
        BatchType* batch_querys_ptr = h_query_batch.data() + idx / batch_size * batch_query_size;
        for (size_t j = 0; j < batch_size; j++) {
            size_t query_idx = j + idx;
            if (query_idx >= querys.size())
                break;
            for (size_t q = 0; q < querys[query_idx].size(); q++) {
                batch_querys_ptr[querys[query_idx][q]] |= static_cast<BatchType>(1) << j;
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
    std::vector<float> scores(n_docs);
    std::vector<int> s_indices(n_docs);

    int batch_size;
    if (n_querys <= sizeof(unsigned short) * 8) {
        batch_size = sizeof(unsigned short) * 8;
    } else if (n_querys <= sizeof(unsigned int) * 8) {
        batch_size = sizeof(unsigned int) * 8;
    } else {
        batch_size = sizeof(unsigned long) * 8;
    }
    int total_query_len_bytes = AlignSize(sizeof(uint16_t) * n_querys);
    int batch_query_bytes = AlignSize(batch_size / 8 * QUERY_MAX_VALUE);
    int n_batches = (n_querys + batch_size - 1) / batch_size;

    char* d_querys_data = nullptr;
    std::thread* pack_thread;
    if (batch_size == sizeof(unsigned short) * 8) {
        pack_thread = new std::thread(PackQuerys<unsigned short>, std::ref(querys), batch_size, total_query_len_bytes, batch_query_bytes, n_batches,
                                      &d_querys_data);
    } else if (batch_size == sizeof(unsigned int) * 8) {
        pack_thread = new std::thread(PackQuerys<unsigned int>, std::ref(querys), batch_size, total_query_len_bytes, batch_query_bytes, n_batches,
                                      &d_querys_data);
    } else {
        pack_thread = new std::thread(PackQuerys<unsigned long>, std::ref(querys), batch_size, total_query_len_bytes, batch_query_bytes, n_batches,
                                      &d_querys_data);
    }

    float* d_scores = nullptr;
    uint16_t* d_docs = nullptr;
    int* d_doc_lens = nullptr;

    // copy to device
    CHECK(cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs));
    CHECK(cudaMalloc(&d_scores, sizeof(float) * n_docs));
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
    std::vector<float> batch_scores(n_docs * batch_size);
    float* d_batch_scores = nullptr;
    CHECK(cudaMalloc(&d_batch_scores, sizeof(float) * n_docs * batch_size));
    for (size_t query_idx = 0; query_idx < n_querys; query_idx += batch_size) {
        auto t1 = start_time();
        uint16_t* d_query_len = reinterpret_cast<uint16_t*>(d_querys_data) + query_idx;
        char* d_query_batch = d_querys_data + total_query_len_bytes + query_idx / batch_size * batch_query_bytes;
        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (n_docs + block - 1) / block;
        if (batch_size == sizeof(unsigned short) * 8) {
            docQueryScoringCoalescedMemoryAccessSampleKernel<unsigned short, sizeof(unsigned short) * 8>
                <<<grid, block>>>(d_docs, d_doc_lens, n_docs, d_query_len, reinterpret_cast<unsigned short*>(d_query_batch), d_batch_scores);
        } else if (batch_size == sizeof(unsigned int) * 8) {
            docQueryScoringCoalescedMemoryAccessSampleKernel<unsigned int, sizeof(unsigned int) * 8>
                <<<grid, block>>>(d_docs, d_doc_lens, n_docs, d_query_len, reinterpret_cast<unsigned int*>(d_query_batch), d_batch_scores);
        } else {
            docQueryScoringCoalescedMemoryAccessSampleKernel<unsigned long, sizeof(unsigned long) * 8>
                <<<grid, block>>>(d_docs, d_doc_lens, n_docs, d_query_len, reinterpret_cast<unsigned long*>(d_query_batch), d_batch_scores);
        }
        cudaDeviceSynchronize();

        end_time(t1, "QueryScoring kernel cost: ");
        t1 = start_time();

        CHECK(cudaMemcpy(batch_scores.data(), d_batch_scores, sizeof(float) * n_docs * batch_size, cudaMemcpyDeviceToHost));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR: %s:%d,", __FILE__, __LINE__);
            printf("CUDA Error[%d]: %s\n", err, cudaGetErrorString(err));
            exit(1);
        }

        end_time(t1, "Copy scores cost: ");

        t1 = start_time();
        for (int q = 0; q < batch_size; q++) {
            if (query_idx + q >= n_querys) {
                break;
            }
            float* scores = batch_scores.data() + q * n_docs;

            std::vector<int> temp_indices(s_indices);
            std::partial_sort(temp_indices.begin(), temp_indices.begin() + TOPK, temp_indices.end(), [&scores](const int& a, const int& b) {
                if (scores[a] != scores[b]) {
                    return scores[a] > scores[b];
                }
                return a < b;
            });
            std::vector<int> s_ans(temp_indices.begin(), temp_indices.begin() + TOPK);
            indices.push_back(s_ans);
        }
        end_time(t1, "TopK cost: ");
    }

    // deallocation
    cudaFree(d_docs);
    cudaFree(d_scores);
    cudaFree(d_doc_lens);
    cudaFree(d_querys_data);
    cudaFree(d_batch_scores);
    free(h_docs);
}
