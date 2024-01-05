
#include <string>
#include <thread>
#include "topk.h"

typedef uint4 group_t;  // uint32_t

template <typename BatchType, int BatchSize>
void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(const __restrict__ uint16_t* docs,
                                                                 const int* acc_doc_lens,
                                                                 const size_t n_docs,
                                                                 const uint16_t* query_lens,
                                                                 const BatchType* d_query_batch,
                                                                 long* batch_scores) {
    // each thread process one batch doc-querys pair scoring task
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto doc_start_id = tid * DOCS_PER_THREAD;

    __shared__ uint16_t s_query_lens[BatchSize];

#pragma unroll
    for (auto i = threadIdx.x; i < BatchSize; i += blockDim.x)
        s_query_lens[i] = query_lens[i];
    __syncthreads();

    auto last_alin_len = acc_doc_lens[doc_start_id];
    auto docs_ptr = docs + last_alin_len;
#pragma unroll
    for (auto d = 0; d < DOCS_PER_THREAD; d++) {
        auto doc_id = doc_start_id + d;
        if (doc_id >= n_docs)
            return;

        unsigned char tmp_scores[BatchSize] = {0};

        auto doc_alin_len = acc_doc_lens[doc_id + 1] - last_alin_len;
        last_alin_len = doc_alin_len + last_alin_len;
        auto doc_len = doc_alin_len;
        bool no_more_load = false;

#pragma unroll
        for (int i = 0; i < doc_alin_len; i += DOC_IN_GROUP) {
            if (no_more_load)
                break;
            const group_t loaded = ((group_t*)(docs_ptr + i))[0];
            const uint16_t* doc_segment = (uint16_t*)(&loaded);

            for (char j = 0; j < DOC_IN_GROUP; j++) {
                const auto target_doc = doc_segment[j];
                if (target_doc == 0) {
                    doc_len = i + j;
                    no_more_load = true;
                    break;
                }
                BatchType packed_query = d_query_batch[target_doc];
                if (packed_query == 0)
                    continue;
                for (int q = 0; q < BatchSize; q++) {
                    auto find = packed_query & ((BatchType)1) << q;
                    if (find)
                        tmp_scores[q]++;
                }
            }
        }
        for (auto q = 0; q < BatchSize; q++) {
            batch_scores[n_docs * q + tid] = LONG_SCORES_SCALE * tmp_scores[q] / max(s_query_lens[q], doc_len) - doc_id;
        }

        docs_ptr += doc_alin_len;
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

void MergeDoc(std::vector<std::vector<uint16_t>>& docs,
              std::vector<uint16_t>& doc_lens,
              int* acc_doc_lens,
              uint16_t* h_docs,
              int start_index,
              int copy_docs_count) {
    int last_end_alin = acc_doc_lens[start_index];
    h_docs += last_end_alin;
    for (int i = start_index; i < copy_docs_count + start_index; i++) {
        int cur_doc_len = doc_lens[i];
        int cur_alien_len = acc_doc_lens[i + 1] - last_end_alin;
        memcpy(h_docs, docs[i].data(), cur_doc_len * sizeof(uint16_t));
        memset(h_docs + cur_doc_len, 0, (cur_alien_len - cur_doc_len) * sizeof(uint16_t));
        h_docs += cur_alien_len;
        last_end_alin += cur_alien_len;
    }
}

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>>& querys,
                                    std::vector<std::vector<uint16_t>>& docs,
                                    std::vector<uint16_t>& lens,
                                    std::vector<std::vector<int>>& indices  // shape [querys.size(), TOPK]
) {
    auto n_querys = querys.size();
    auto n_docs = docs.size();
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

    int* acc_doc_lens = new int[n_docs + 1];
    int last_sum_len = 0;
    acc_doc_lens[0] = last_sum_len;
    for (int i = 0; i < n_docs; i++) {
        int cur_align_size = AlignSize(lens[i], DOC_IN_GROUP);
        last_sum_len += cur_align_size;
        acc_doc_lens[i + 1] = last_sum_len;
    }
    uint16_t* h_docs = new uint16_t[last_sum_len];
    size_t padding_docs_bytes = AlignSize(sizeof(uint16_t) * last_sum_len);
    size_t acc_doc_lens_bytes = AlignSize(sizeof(int) * (n_docs + 1));
    int num_merge_threads = 16;
    int per_thread_count = (n_docs + num_merge_threads - 1) / num_merge_threads;
    std::thread doc_merge_threads[num_merge_threads];
    for (int i = 0; i < num_merge_threads - 1; i++) {
        doc_merge_threads[i] = std::thread(MergeDoc, std::ref(docs), std::ref(lens), acc_doc_lens, h_docs, i * per_thread_count, per_thread_count);
    }
    doc_merge_threads[num_merge_threads - 1] =
        std::thread(MergeDoc, std::ref(docs), std::ref(lens), acc_doc_lens, h_docs, (num_merge_threads - 1) * per_thread_count,
                    n_docs - (num_merge_threads - 1) * per_thread_count);
    for (int i = 0; i < num_merge_threads; i++) {
        doc_merge_threads[i].join();
    }

    uint16_t* d_docs = nullptr;
    int* d_acc_doc_lens = nullptr;

    // copy to device
    CHECK(cudaMalloc(&d_docs, padding_docs_bytes));
    CHECK(cudaMalloc(&d_acc_doc_lens, acc_doc_lens_bytes));

    CHECK(cudaMemcpy(d_docs, h_docs, padding_docs_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_acc_doc_lens, acc_doc_lens, acc_doc_lens_bytes, cudaMemcpyHostToDevice));

    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);

    cudaSetDevice(0);

    pack_thread->join();
    delete pack_thread;

    for (int i = 0; i < n_docs; ++i) {
        s_indices[i] = i;
    }
    std::vector<long> batch_scores(n_docs * batch_size);
    long* d_batch_scores = nullptr;
    CHECK(cudaMalloc(&d_batch_scores, sizeof(long) * n_docs * batch_size));
    for (size_t query_idx = 0; query_idx < n_querys; query_idx += batch_size) {
        auto t1 = start_time();
        uint16_t* d_query_len = reinterpret_cast<uint16_t*>(d_querys_data) + query_idx;
        char* d_query_batch = d_querys_data + total_query_len_bytes + query_idx / batch_size * batch_query_bytes;
        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (n_docs + block - 1) / block;
        if (batch_size == sizeof(unsigned short) * 8) {
            docQueryScoringCoalescedMemoryAccessSampleKernel<unsigned short, sizeof(unsigned short) * 8>
                <<<grid, block>>>(d_docs, d_acc_doc_lens, n_docs, d_query_len, reinterpret_cast<unsigned short*>(d_query_batch), d_batch_scores);
        } else if (batch_size == sizeof(unsigned int) * 8) {
            docQueryScoringCoalescedMemoryAccessSampleKernel<unsigned int, sizeof(unsigned int) * 8>
                <<<grid, block>>>(d_docs, d_acc_doc_lens, n_docs, d_query_len, reinterpret_cast<unsigned int*>(d_query_batch), d_batch_scores);
        } else {
            docQueryScoringCoalescedMemoryAccessSampleKernel<unsigned long, sizeof(unsigned long) * 8>
                <<<grid, block>>>(d_docs, d_acc_doc_lens, n_docs, d_query_len, reinterpret_cast<unsigned long*>(d_query_batch), d_batch_scores);
        }
        cudaDeviceSynchronize();

        end_time(t1, "QueryScoring kernel cost: ");
        t1 = start_time();

        CHECK(cudaMemcpy(batch_scores.data(), d_batch_scores, sizeof(long) * n_docs * batch_size, cudaMemcpyDeviceToHost));

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
            long* scores = batch_scores.data() + q * n_docs;

            std::vector<int> temp_indices(s_indices);
            std::partial_sort(temp_indices.begin(), temp_indices.begin() + TOPK, temp_indices.end(),
                              [&scores](const int& a, const int& b) { return scores[a] > scores[b]; });
            std::vector<int> s_ans(temp_indices.begin(), temp_indices.begin() + TOPK);
            indices.push_back(s_ans);
        }
        end_time(t1, "Partial sort cost: ");
    }

    // deallocation
    cudaFree(d_docs);
    cudaFree(d_acc_doc_lens);
    cudaFree(d_querys_data);
    cudaFree(d_batch_scores);
    free(h_docs);
}
