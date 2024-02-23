
#include <assert.h>
#include <numeric>
#include <string>
#include <thread>
#include "fast_topk.cuh"
#include "topk.h"

typedef uint4 group_t;  // uint32_t

template <typename BatchType, int BatchSize>
void __global__ docQueryScoringInWindowKernel(const __restrict__ uint16_t* docs,
                                              const int* doc_lens,
                                              const size_t win_docs,
                                              const size_t n_docs,
                                              const size_t doc_start,
                                              const size_t doc_end,
                                              uint16_t* query_len,
                                              uint32_t* d_query_mask,
                                              const uint16_t max_query_token,
                                              short* batch_scores) {
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    register auto doc_id = doc_start + tid;

    __shared__ uint32_t query_mask[BatchSize * QUERY_MASK_SIZE];

#pragma unroll
    for (auto q = threadIdx.x; q < BatchSize * QUERY_MASK_SIZE; q += N_THREADS_IN_ONE_BLOCK) {
        query_mask[q] = __ldg(d_query_mask + q);
    }

    __syncthreads();

    if (tid >= win_docs) {
        return;
    }

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
            if (doc_segment[j] == 0 || doc_segment[j] > max_query_token) {
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
        batch_scores[win_docs * q + tid] = static_cast<short>(16384u * tmp_scores[q] / max(query_len[q], doc_lens[doc_id]));
    }
}

template <typename BatchType, int BatchSize>
void __global__ docQueryScoringThreshKernel(const __restrict__ uint16_t* docs,
                                            const int* doc_lens,
                                            const int doc_offset1,
                                            const int doc_num1,
                                            const int doc_offset2,
                                            const int doc_num2,
                                            const size_t n_docs,
                                            uint16_t* query_len,
                                            const uint32_t* d_query_mask,
                                            const int max_query_len,
                                            const uint16_t max_query_token,
                                            const float thresh,
                                            short* batch_scores) {
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    int doc_num = doc_num1 + doc_num2;
    register auto doc_id = tid < doc_num1 ? (doc_offset1 + tid) : (doc_offset2 + tid - doc_num1);

    __shared__ uint32_t query_mask[BatchSize * QUERY_MASK_SIZE];
#pragma unroll
    for (auto q = threadIdx.x; q < BatchSize * QUERY_MASK_SIZE; q += N_THREADS_IN_ONE_BLOCK) {
        query_mask[q] = __ldg(d_query_mask + q);
    }

    __syncthreads();

    if (tid >= doc_num) {
        return;
    }

    int doc_len = doc_lens[doc_id];
    uint32_t tmp_score_thresh = static_cast<uint32_t>(thresh * max(doc_len, max_query_len));
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
            if (doc_segment[j] == 0 || doc_segment[j] > max_query_token) {
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
        unsigned char max_tmp_score = tmp_scores[0];

#pragma unroll
        for (auto q = 1; q < BatchSize; q++) {
            max_tmp_score = max(max_tmp_score, tmp_scores[q]);
        }
        if (max_tmp_score + (doc_len - (i + 1) * 8) < tmp_score_thresh) {
            break;
        }
    }
    for (auto q = 0; q < BatchSize; q++) {
        batch_scores[doc_num * q + tid] = static_cast<short>(16384u * tmp_scores[q] / max(query_len[q], doc_len));
    }
}

__forceinline__ size_t AlignSize(size_t size, size_t align = 128) {
    if (size == 0) {
        return align;
    }
    return (size + align - 1) / align * align;
}

__forceinline__ void PackQuerys(const std::vector<std::vector<uint16_t>>& querys,
                                const std::vector<int>& query_idx,
                                int batch_size,
                                char* h_querys_data,
                                char* d_querys_data) {
    int batch_query_bytes = AlignSize(MAX_BATCH_SIZE * sizeof(uint16_t) + MAX_BATCH_SIZE * QUERY_MAX_VALUE / sizeof(uint32_t));
    memset(h_querys_data, 0, batch_query_bytes);
    uint16_t* h_query_len = reinterpret_cast<uint16_t*>(h_querys_data);
    uint32_t* h_query_mask = reinterpret_cast<uint32_t*>(h_querys_data + MAX_BATCH_SIZE * sizeof(uint16_t));
    for (size_t j = 0; j < batch_size; j++) {
        const std::vector<uint16_t>& query = querys[query_idx[j]];
        for (auto& q : query) {
            uint16_t index = q >> 5;
            uint16_t postion = q & 31;
            h_query_mask[batch_size * index + j] |= static_cast<uint32_t>(1) << postion;
        }
        h_query_len[j] = query.size();
    }
    CHECK(cudaMemcpy(d_querys_data, h_querys_data, batch_query_bytes, cudaMemcpyHostToDevice));
}

std::chrono::time_point<std::chrono::high_resolution_clock> start_time() {
    return std::chrono::high_resolution_clock::now();
}

void end_time(std::chrono::time_point<std::chrono::high_resolution_clock>& t1, std::string message) {
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << message << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms " << std::endl;
}

void end_time(std::chrono::time_point<std::chrono::high_resolution_clock>& t1, std::string message, int batch_idx) {
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Batch[" << batch_idx << "] " << message << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms "
              << std::endl;
}

void do_pack_docs(const std::vector<std::vector<uint16_t>>& docs,
                  uint16_t* h_docs,
                  std::vector<int>& h_doc_lens_vec,
                  const size_t from,
                  const size_t to) {
    auto n_docs = docs.size();
    for (int i = from; i < to; i++) {
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
}
// 3683 ms

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>>& querys,
                                    std::vector<std::vector<uint16_t>>& docs,
                                    std::vector<uint16_t>& lens,
                                    std::vector<std::vector<int>>& indices  // shape [querys.size(), TOPK]
) {
    auto t0 = start_time();
    auto n_querys = querys.size();
    auto n_docs = docs.size();
    std::vector<int> s_indices(n_docs);

    int batch_size;
    const int init_batch_size = 4;
    int total_query_len_bytes = AlignSize(sizeof(uint16_t) * n_querys);
    int max_batch_query_bytes = AlignSize(MAX_BATCH_SIZE * sizeof(uint16_t) + MAX_BATCH_SIZE * QUERY_MAX_VALUE / sizeof(uint32_t));

    std::vector<int> query_idx(n_querys);
    std::iota(query_idx.begin(), query_idx.end(), 0);
    std::sort(query_idx.begin(), query_idx.end(), [&querys](int a, int b) {
        if (querys[a].size() != querys[b].size()) {
            return querys[a].size() < querys[b].size();
        }
        return querys[a].back() < querys[b].back();
    });

    std::iota(s_indices.begin(), s_indices.end(), 0);
    size_t lens_freq[129] = {0ul};
    size_t lens_offset[130] = {0ul};
    for (int i = 0; i < lens.size(); ++i) {
        ++lens_freq[lens[i]];
    }
    for (int i = 0; i < 129; ++i) {
        lens_offset[i + 1] = lens_offset[i] + lens_freq[i];
    }
    end_time(t0, "Init cost: ");

    t0 = start_time();
    uint16_t* d_docs = nullptr;
    int* d_doc_lens = nullptr;
    CHECK(cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs));
    CHECK(cudaMalloc(&d_doc_lens, sizeof(int) * n_docs));
    std::vector<short> batch_scores(n_docs * init_batch_size);
    short* d_batch_scores = nullptr;
    char* d_querys_data = nullptr;
    CHECK(cudaMalloc(&d_batch_scores, sizeof(short) * n_docs * init_batch_size));
    CHECK(cudaMalloc(&d_querys_data, max_batch_query_bytes));
    char* h_querys_data = new char[max_batch_query_bytes];
    int topk_bytes = AlignSize(sizeof(Pair) * MAX_BATCH_SIZE * TOPK);
    char* h_topk_pool = new char[topk_bytes];
    Pair* h_topk = reinterpret_cast<Pair*>(h_topk_pool);
    Pair* d_topk;
    CHECK(cudaMalloc(&d_topk, topk_bytes));
    int8_t* workspace;  // 64 * 1024 * 1024
    CHECK(cudaMalloc(&workspace, AlignSize(64 * 1024 * 1024)));
    indices.resize(n_querys);
    uint16_t* d_query_len = reinterpret_cast<uint16_t*>(d_querys_data);
    uint32_t* d_query_mask = reinterpret_cast<uint32_t*>(d_querys_data + MAX_BATCH_SIZE * sizeof(uint16_t));
    uint16_t* h_query_len = reinterpret_cast<uint16_t*>(h_querys_data);
    end_time(t0, "Malloc cost: ");

    t0 = start_time();
    size_t n_threads = 16;
    if (n_threads > n_docs) {
        n_threads = n_docs;
    }
    size_t n_docs_per_thread = n_docs / n_threads;
    size_t n_onemore_doc_thread = n_docs - n_docs_per_thread * n_threads;
    std::vector<size_t> docs_from(n_threads);
    std::vector<size_t> docs_to(n_threads);
    for (size_t i = 0; i < n_threads; i++) {
        if (i < n_onemore_doc_thread) {
            docs_from[i] = i * (n_docs_per_thread + 1);
            docs_to[i] = (i + 1) * (n_docs_per_thread + 1);
        } else {
            docs_from[i] = i * (n_docs_per_thread) + n_onemore_doc_thread;
            docs_to[i] = (i + 1) * (n_docs_per_thread) + n_onemore_doc_thread;
        }
    }
    std::vector<std::thread> pack_docs_threads(n_threads - 1);
    uint16_t* h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
    memset(h_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
    std::vector<int> h_doc_lens_vec(n_docs);
    for (size_t i = 0; i < n_threads - 1; i++) {
        pack_docs_threads[i] = std::thread([&, i]() { do_pack_docs(docs, h_docs, h_doc_lens_vec, docs_from[i], docs_to[i]); });
    }
    do_pack_docs(docs, h_docs, h_doc_lens_vec, docs_from[n_threads - 1], docs_to[n_threads - 1]);
    for (auto& t : pack_docs_threads) {
        t.join();
    }
    end_time(t0, "Pack docs cost: ");

    t0 = start_time();
    CHECK(cudaMemcpy(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_doc_lens, h_doc_lens_vec.data(), sizeof(int) * n_docs, cudaMemcpyHostToDevice));
    end_time(t0, "Copy docs cost: ");

    cudaDeviceProp device_props;
    cudaStream_t stream;
    cudaGetDeviceProperties(&device_props, 0);

    cudaSetDevice(0);

    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    for (size_t idx = 0; idx < n_querys; idx += init_batch_size) {
        auto t1 = start_time();
        batch_size = init_batch_size;
        if (idx + batch_size > n_querys) {
            batch_size = n_querys - idx;
        }
        // 1. 去除空query
        std::vector<int> cur_query_idx;
        for (int j = 0; j < batch_size; ++j) {
            int q_index = query_idx[idx + j];
            if (querys[q_index].empty()) {
                std::vector<int> s_ans(TOPK);
                std::iota(s_ans.begin(), s_ans.end(), 0);
                indices[q_index] = std::move(s_ans);
                continue;
            }
            cur_query_idx.push_back(q_index);
        }
        batch_size = cur_query_idx.size();
        if (batch_size == 0) {
            continue;
        }

        // 2. 打包query
        PackQuerys(querys, cur_query_idx, batch_size, h_querys_data, d_querys_data);

        // 3. 查找范围
        int max_query_len = 0;
        int min_query_len = MAX_DOC_SIZE;
        uint16_t max_query_token = 0;
        for (int q = 0; q < batch_size; ++q) {
            auto& query = querys[cur_query_idx[q]];
            min_query_len = std::min(min_query_len, static_cast<int>(query.size()));
            max_query_len = std::max(max_query_len, static_cast<int>(query.size()));
            max_query_token = std::max(max_query_token, query.back());
        }
        int doc_len_start = std::max(0, min_query_len - WINDOW_SIZE);
        int doc_len_end = std::min(MAX_DOC_SIZE, max_query_len + 2 * WINDOW_SIZE) + 1;
        int win_docs = lens_offset[doc_len_end] - lens_offset[doc_len_start];
        while (win_docs < TOPK) {
            doc_len_start = std::max(0, doc_len_start - 1);
            doc_len_end = std::min(129, doc_len_end + 1);
            win_docs = lens_offset[doc_len_end] - lens_offset[doc_len_start];
        }
        size_t doc_start = lens_offset[doc_len_start];
        size_t doc_end = lens_offset[doc_len_end];
        // std::cout << "Docs len from " << doc_len_start << " to " << doc_len_end << std::endl;
        // std::cout << "Docs offset from " << doc_start << " to " << doc_end << " || " << win_docs << std::endl;

        // 4. 查询窗口
        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (win_docs + block - 1) / block;
        if (batch_size == 1) {
            docQueryScoringInWindowKernel<uint32_t, 1><<<grid, block>>>(d_docs, d_doc_lens, win_docs, n_docs, doc_start, doc_end, d_query_len,
                                                                        d_query_mask, max_query_token, d_batch_scores);
        } else if (batch_size == 2) {
            docQueryScoringInWindowKernel<uint2, 2><<<grid, block>>>(d_docs, d_doc_lens, win_docs, n_docs, doc_start, doc_end, d_query_len,
                                                                     d_query_mask, max_query_token, d_batch_scores);
        } else if (batch_size == 3) {
            docQueryScoringInWindowKernel<uint3, 3><<<grid, block>>>(d_docs, d_doc_lens, win_docs, n_docs, doc_start, doc_end, d_query_len,
                                                                     d_query_mask, max_query_token, d_batch_scores);
        } else {
            docQueryScoringInWindowKernel<uint4, 4><<<grid, block>>>(d_docs, d_doc_lens, win_docs, n_docs, doc_start, doc_end, d_query_len,
                                                                     d_query_mask, max_query_token, d_batch_scores);
        }
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR: %s:%d,", __FILE__, __LINE__);
            printf("CUDA Error[%d]: %s\n", err, cudaGetErrorString(err));
            exit(1);
        }
        end_time(t1, "QueryScoringInWindow kernel cost: ", idx / init_batch_size);

        t1 = start_time();
        launch_gather_topk_kernel(d_batch_scores, d_topk, workspace, TOPK, batch_size, win_docs, stream);
        CHECK(cudaMemcpyAsync(h_topk, d_topk, TOPK * batch_size * sizeof(Pair), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        end_time(t1, "CopyWindowTopK cost: ", idx / init_batch_size);

        t1 = start_time();
        std::vector<int> unfinished_querys;
        float cur_score_thresh = 1.0f;
        std::vector<std::vector<int>> win_topk_indices(MAX_BATCH_SIZE);
        std::vector<std::vector<short>> win_topk_scores(MAX_BATCH_SIZE);
        int cur_min_start = MAX_DOC_SIZE;
        int cur_max_end = 0;
        int cur_doc_len_start;
        int cur_doc_len_end;
        std::vector<short> window_scores(TOPK);
        std::vector<int> window_topk(TOPK);
        for (int q = 0; q < batch_size; q++) {
            Pair* topk = h_topk + q * TOPK;
            std::sort(topk, topk + TOPK, [](const Pair& a, const Pair& b) {
                if (a.score != b.score) {
                    return a.score > b.score;
                }
                return a.index < b.index;
            });

            for (int k = 0; k < TOPK; k++) {
                window_scores[k] = topk[k].score;
                window_topk[k] = topk[k].index + doc_start;
            }

            float score_thresh = window_scores.back() / 16384.0f;

            bool finished = score_thresh > 0.f && static_cast<int>(h_query_len[q] * score_thresh) >= doc_len_start &&
                            std::min(128, static_cast<int>(h_query_len[q] / score_thresh)) < doc_len_end;
            finished = finished || (doc_len_start == 0 && doc_len_end == 129);

            if (finished) {
                indices[cur_query_idx[q]] = window_topk;
            } else {
                win_topk_indices[unfinished_querys.size()] = window_topk;
                win_topk_scores[unfinished_querys.size()] = window_scores;

                unfinished_querys.push_back(cur_query_idx[q]);
                cur_score_thresh = std::min(cur_score_thresh, score_thresh);
                if (score_thresh == 0.f) {
                    cur_min_start = 0;
                    cur_max_end = MAX_DOC_SIZE + 1;
                } else {
                    int min_start = static_cast<int>(h_query_len[q] * score_thresh);
                    int max_end = std::min(MAX_DOC_SIZE + 1, static_cast<int>(h_query_len[q] / score_thresh + 1));
                    cur_min_start = std::min(cur_min_start, min_start);
                    cur_max_end = std::max(cur_max_end, max_end);
                }
            }
        }
        cur_query_idx = unfinished_querys;
        if (cur_query_idx.empty()) {
            continue;
        }
        end_time(t1, "Window sort cost: ", idx / init_batch_size);

        t1 = start_time();

        // 6. 剩余部分
        if (batch_size != cur_query_idx.size()) {
            batch_size = cur_query_idx.size();
            PackQuerys(querys, cur_query_idx, batch_size, h_querys_data, d_querys_data);
        }

        cur_doc_len_start = doc_len_start;
        cur_doc_len_end = doc_len_end;
        doc_len_start = std::min(cur_min_start, doc_len_start);
        doc_len_end = std::max(cur_max_end, doc_len_end);

        // 搜索范围为 [doc_len_start, cur_doc_len_start) 及 [cur_doc_len_end, doc_len_end)
        int doc_num1 = lens_offset[cur_doc_len_start] - lens_offset[doc_len_start];
        int doc_offset1 = lens_offset[doc_len_start];
        int doc_num2 = lens_offset[doc_len_end] - lens_offset[cur_doc_len_end];
        int doc_offset2 = lens_offset[cur_doc_len_end];
        int doc_num = doc_num1 + doc_num2;
        int new_topk = std::min(doc_num, TOPK);

        assert(doc_num1 >= 0);
        assert(doc_num2 >= 0);
        assert(doc_num > 0);

        // std::cout << "Docs len1 from " << doc_len_start << " to " << cur_doc_len_start << std::endl;
        // std::cout << "Docs len2 from " << cur_doc_len_end << " to " << doc_len_end << std::endl;
        // std::cout << "Docs offset1 from " << doc_offset1 << " to " << lens_offset[cur_doc_len_start] << " || " << doc_num1 << std::endl;
        // std::cout << "Docs offset2 from " << doc_offset2 << " to " << lens_offset[doc_len_end] << " || " << doc_num2 << std::endl;
        // std::cout << "doc_num = " << doc_num << std::endl;
        // std::cout << "batch_size = " << batch_size << std::endl;

        block = N_THREADS_IN_ONE_BLOCK;
        grid = (doc_num + N_THREADS_IN_ONE_BLOCK - 1) / N_THREADS_IN_ONE_BLOCK;
        if (batch_size == 1) {
            docQueryScoringThreshKernel<uint32_t, 1><<<grid, block>>>(d_docs, d_doc_lens, doc_offset1, doc_num1, doc_offset2, doc_num2, n_docs,
                                                                      d_query_len, d_query_mask, max_query_len, max_query_token, cur_score_thresh,
                                                                      d_batch_scores);
        } else if (batch_size == 2) {
            docQueryScoringThreshKernel<uint2, 2><<<grid, block>>>(d_docs, d_doc_lens, doc_offset1, doc_num1, doc_offset2, doc_num2, n_docs,
                                                                   d_query_len, d_query_mask, max_query_len, max_query_token, cur_score_thresh,
                                                                   d_batch_scores);
        } else if (batch_size == 3) {
            docQueryScoringThreshKernel<uint3, 3><<<grid, block>>>(d_docs, d_doc_lens, doc_offset1, doc_num1, doc_offset2, doc_num2, n_docs,
                                                                   d_query_len, d_query_mask, max_query_len, max_query_token, cur_score_thresh,
                                                                   d_batch_scores);
        } else {
            docQueryScoringThreshKernel<uint4, 4><<<grid, block>>>(d_docs, d_doc_lens, doc_offset1, doc_num1, doc_offset2, doc_num2, n_docs,
                                                                   d_query_len, d_query_mask, max_query_len, max_query_token, cur_score_thresh,
                                                                   d_batch_scores);
        }
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR: %s:%d,", __FILE__, __LINE__);
            printf("CUDA Error[%d]: %s\n", err, cudaGetErrorString(err));
            exit(1);
        }
        end_time(t1, "QueryScoringThreshKernel kernel cost: ", idx / init_batch_size);

        t1 = start_time();
        launch_gather_topk_kernel(d_batch_scores, d_topk, workspace, new_topk, batch_size, doc_num, stream);
        CHECK(cudaMemcpyAsync(h_topk, d_topk, new_topk * batch_size * sizeof(Pair), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        end_time(t1, "CopyThreshTopk scores cost: ", idx / init_batch_size);

        // 7. 排序剩余部分并合并
        t1 = start_time();

        std::vector<short> other_scores(new_topk);
        std::vector<int> other_topk(new_topk);

        for (int q = 0; q < batch_size; q++) {
            Pair* topk = h_topk + q * TOPK;
            std::sort(topk, topk + TOPK, [](const Pair& a, const Pair& b) {
                if (a.score != b.score) {
                    return a.score > b.score;
                }
                return a.index < b.index;
            });

            for (int k = 0; k < TOPK; k++) {
                other_scores[k] = topk[k].score;
                int did = topk[k].index;
                other_topk[k] = did < doc_num1 ? (doc_offset1 + did) : (doc_offset2 + did - doc_num1);
            }

            std::vector<short>& win_scores = win_topk_scores[q];
            std::vector<int>& win_topk = win_topk_indices[q];
            std::vector<int> s_ans(TOPK);
            int win_idx = 0;
            int other_idx = 0;

            for (int k = 0; k < TOPK; ++k) {
                if (win_scores[win_idx] > other_scores[other_idx] || other_idx >= new_topk) {
                    s_ans[k] = win_topk[win_idx];
                    win_idx++;
                } else if (win_scores[win_idx] < other_scores[other_idx]) {
                    s_ans[k] = other_topk[other_idx];
                    other_idx++;
                } else {
                    if (win_topk[win_idx] < other_topk[other_idx]) {
                        s_ans[k] = win_topk[win_idx];
                        win_idx++;
                    } else {
                        s_ans[k] = other_topk[other_idx];
                        other_idx++;
                    }
                }
            }
            indices[cur_query_idx[q]].swap(s_ans);
        }
        end_time(t1, "Merge result cost: ", idx / init_batch_size);
    }

    // deallocation
    cudaFree(d_docs);
    cudaFree(d_doc_lens);
    cudaFree(d_querys_data);
    cudaFree(d_batch_scores);
    cudaFree(d_topk);
    cudaFree(workspace);
    free(h_docs);
    free(h_querys_data);
    free(h_topk_pool);
}
