
#include "topk.h"
#include <thread>
#include <cassert>

typedef uint4 group_t; // uint32_t

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
        const __restrict__ uint16_t *docs, 
        const int *doc_lens, const size_t n_docs, 
        uint16_t *query, const int query_len, float *scores) {
    // each thread process one doc-query pair scoring task
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x, tnum = gridDim.x * blockDim.x;

    if (tid >= n_docs) {
        return;
    }

    __shared__ uint16_t query_on_shm[MAX_QUERY_SIZE];
#pragma unroll
    for (auto i = threadIdx.x; i < query_len; i += blockDim.x) {
        query_on_shm[i] = query[i]; // not very efficient query loading temporally, as assuming its not hotspot
    }

    __syncthreads();

    for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
        register int query_idx = 0;

        register float tmp_score = 0.;

        register bool no_more_load = false;

        for (auto i = 0; i < MAX_DOC_SIZE / (sizeof(group_t) / sizeof(uint16_t)); i++) {
            if (no_more_load) {
                break;
            }
            register group_t loaded = ((group_t *)docs)[i * n_docs + doc_id]; // tid
            register uint16_t *doc_segment = (uint16_t*)(&loaded);
            for (auto j = 0; j < sizeof(group_t) / sizeof(uint16_t); j++) {
                if (doc_segment[j] == 0) {
                    no_more_load = true;
                    break;
                    // return;
                }
                while (query_idx < query_len && query_on_shm[query_idx] < doc_segment[j]) {
                    ++query_idx;
                }
                if (query_idx < query_len) {
                    tmp_score += (query_on_shm[query_idx] == doc_segment[j]);
                }
            }
            __syncwarp();
        }
        scores[doc_id] = tmp_score / max(query_len, doc_lens[doc_id]); // tid
    }
}

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices //shape [querys.size(), TOPK]
    ) {

    auto n_docs = docs.size();
    std::vector<float> scores(n_docs);
    std::vector<int> s_indices(n_docs);

    float *d_scores = nullptr;
    uint16_t *d_docs = nullptr, *d_query = nullptr;
    int *d_doc_lens = nullptr;

    // copy to device
    cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
    cudaMalloc(&d_scores, sizeof(float) * n_docs);
    cudaMalloc(&d_doc_lens, sizeof(int) * n_docs);

    uint16_t *h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
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

    cudaMemcpy(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_doc_lens, h_doc_lens_vec.data(), sizeof(int) * n_docs, cudaMemcpyHostToDevice);

    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);

    cudaSetDevice(0);

    for(auto& query : querys) {
        //init indices
        for (int i = 0; i < n_docs; ++i) {
            s_indices[i] = i;
        }

        const size_t query_len = query.size();
        cudaMalloc(&d_query, sizeof(uint16_t) * query_len);
        cudaMemcpy(d_query, query.data(), sizeof(uint16_t) * query_len, cudaMemcpyHostToDevice);

        // launch kernel
        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (n_docs + block - 1) / block;
        docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block>>>(d_docs,
            d_doc_lens, n_docs, d_query, query_len, d_scores);
        cudaDeviceSynchronize();

        cudaMemcpy(scores.data(), d_scores, sizeof(float) * n_docs, cudaMemcpyDeviceToHost);

        // sort scores
        std::partial_sort(s_indices.begin(), s_indices.begin() + TOPK, s_indices.end(),
                        [&scores](const int& a, const int& b) {
                            if (scores[a] != scores[b]) {
                                return scores[a] > scores[b];  // 按照分数降序排序
                            }
                            return a < b;  // 如果分数相同，按索引从小到大排序
                    });
        std::vector<int> s_ans(s_indices.begin(), s_indices.begin() + TOPK);
        indices.push_back(s_ans);

        cudaFree(d_query);
    }

    // deallocation
    cudaFree(d_docs);
    //cudaFree(d_query);
    cudaFree(d_scores);
    cudaFree(d_doc_lens);
    free(h_docs);

}

void do_doc_query_scoring(const std::vector<std::vector<uint16_t>> &docs,
                          const std::vector<std::vector<uint64_t>> &querys_map,
                          const std::vector<size_t> &querys_len,
                          std::vector<std::vector<float>> &scores,
                          const size_t from, const size_t to)
{
    #pragma omp parallel for
    for (size_t d = from; d < to; d++)
    {
        #pragma omp parallel for
        for (size_t q = 0; q < querys_map[0].size(); q++)
        {
            size_t start_q = q * QUERY_GROUP_SIZE;
            register uint64_t inter = 0;
            for (const uint16_t id : docs[d])
            {
                inter += querys_map[id][q];
            }
            // uint8_t *inters = reinterpret_cast<uint8_t*>(&inter);
            // scores[start_q][d] = inters[0] * 1.0 / std::max(querys_len[start_q], docs[d].size());
            // scores[start_q + 1][d] = inters[1] * 1.0 / std::max(querys_len[start_q + 1], docs[d].size());
            // scores[start_q + 2][d] = inters[2] * 1.0 / std::max(querys_len[start_q + 2], docs[d].size());
            // scores[start_q + 3][d] = inters[3] * 1.0 / std::max(querys_len[start_q + 3], docs[d].size());
            // scores[start_q + 4][d] = inters[4] * 1.0 / std::max(querys_len[start_q + 4], docs[d].size());
            // scores[start_q + 5][d] = inters[5] * 1.0 / std::max(querys_len[start_q + 5], docs[d].size());
            // scores[start_q + 6][d] = inters[6] * 1.0 / std::max(querys_len[start_q + 6], docs[d].size());
            // scores[start_q + 7][d] = inters[7] * 1.0 / std::max(querys_len[start_q + 7], docs[d].size());

            scores[start_q][d] = GET_BYTE_0(inter) * 1.0 / std::max(querys_len[start_q], docs[d].size());
            scores[start_q + 1][d] = GET_BYTE_1(inter) * 1.0 / std::max(querys_len[start_q + 1], docs[d].size());
            scores[start_q + 2][d] = GET_BYTE_2(inter) * 1.0 / std::max(querys_len[start_q + 2], docs[d].size());
            scores[start_q + 3][d] = GET_BYTE_3(inter) * 1.0 / std::max(querys_len[start_q + 3], docs[d].size());
            scores[start_q + 4][d] = GET_BYTE_4(inter) * 1.0 / std::max(querys_len[start_q + 4], docs[d].size());
            scores[start_q + 5][d] = GET_BYTE_5(inter) * 1.0 / std::max(querys_len[start_q + 5], docs[d].size());
            scores[start_q + 6][d] = GET_BYTE_6(inter) * 1.0 / std::max(querys_len[start_q + 6], docs[d].size());
            scores[start_q + 7][d] = GET_BYTE_7(inter) * 1.0 / std::max(querys_len[start_q + 7], docs[d].size());
        }
    }
}

void do_scoring_topk(const std::vector<std::vector<float>> &scores, const std::vector<int> &s_indices, std::vector<std::vector<int>> &indices, const size_t from, const size_t to)
{
    #pragma omp parallel for
    for (size_t q = from; q < to; q++)
    {
        std::vector<int> new_indices = s_indices;
        const std::vector<float> &query_scores = scores[q];
        std::partial_sort(new_indices.begin(), new_indices.begin() + TOPK, new_indices.end(),
                          [&query_scores](const int &a, const int &b)
                          {
                              if (query_scores[a] != query_scores[b])
                              {
                                  return query_scores[a] > query_scores[b];
                              }
                              return a < b;
                          });
        std::vector<int> s_ans(new_indices.begin(), new_indices.begin() + TOPK);
        indices[q].swap(s_ans);
    }
}

void doc_query_scoring_cpu_function(std::vector<std::vector<uint16_t>> &querys,
                                    std::vector<std::vector<uint16_t>> &docs,
                                    std::vector<std::vector<int>> &indices // shape [querys.size(), TOPK]
)
{
    const uint64_t bit_map[8] = {0x0000000000000001ull, 0x0000000000000100ull, 0x0000000000010000ull, 0x0000000001000000ull, 0x0000000100000000ull, 0x0000010000000000ull, 0x0001000000000000ull, 0x0100000000000000ull};
    size_t n_docs = docs.size();
    size_t n_querys = querys.size();
    size_t n_querys_group = (n_querys + QUERY_GROUP_SIZE - 1) / QUERY_GROUP_SIZE;
    size_t n_threads = N_THREADS_CPU;
    if (n_threads > n_docs)
    {
        n_threads = n_docs;
    }
    size_t n_docs_per_thread = n_docs / n_threads;
    size_t n_onemore_doc_thread = n_docs - n_docs_per_thread * n_threads;
    std::vector<size_t> docs_from(n_threads);
    std::vector<size_t> docs_to(n_threads);
    for (size_t i = 0; i < n_threads; i++)
    {
        if (i < n_onemore_doc_thread)
        {
            docs_from[i] = i * (n_docs_per_thread + 1);
            docs_to[i] = (i + 1) * (n_docs_per_thread + 1);
        }
        else
        {
            docs_from[i] = i * (n_docs_per_thread) + n_onemore_doc_thread;
            docs_to[i] = (i + 1) * (n_docs_per_thread) + n_onemore_doc_thread;
        }
    }

    std::vector<std::vector<float>> scores(n_querys_group * QUERY_GROUP_SIZE, std::vector<float>(n_docs, 0.0));
    std::vector<std::vector<uint64_t>> querys_map(MAX_ID, std::vector<uint64_t>(n_querys_group, 0));
    std::vector<size_t> querys_len(n_querys_group * QUERY_GROUP_SIZE, 1);
    for (size_t q = 0; q < n_querys_group; q++)
    {
        size_t end_q = std::min((q + 1) * QUERY_GROUP_SIZE, n_querys);
        for (size_t sq = q * QUERY_GROUP_SIZE; sq < end_q; sq++)
        {
            std::for_each(querys[sq].begin(), querys[sq].end(), [&](const uint16_t &id)
                      { querys_map[id][q] |= bit_map[sq & 7]; });
            querys_len[sq] = querys[sq].size();
        }
    }

    std::vector<std::thread> scoring_threads(n_threads - 1);
    for (size_t i = 0; i < n_threads - 1; i++)
    {
        scoring_threads[i] = std::thread([&, i]()
                                         { do_doc_query_scoring(docs, querys_map, querys_len, scores, docs_from[i], docs_to[i]); });
    }
    do_doc_query_scoring(docs, querys_map, querys_len, scores, docs_from[n_threads - 1], docs_to[n_threads - 1]);
    for (auto &t : scoring_threads)
    {
        t.join();
    }

    std::cout << "scoring ok!" << std::endl;

    // Top K
    indices.resize(n_querys);
    n_threads = N_THREADS_CPU;
    if (n_threads > n_querys)
    {
        n_threads = n_querys;
    }
    size_t n_querys_per_thread = n_querys / n_threads;
    size_t n_onemore_query_thread = n_querys - n_querys_per_thread * n_threads;
    std::vector<size_t> querys_from(n_threads);
    std::vector<size_t> querys_to(n_threads);
    for (size_t i = 0; i < n_threads; i++)
    {
        if (i < n_onemore_query_thread)
        {
            querys_from[i] = i * (n_querys_per_thread + 1);
            querys_to[i] = (i + 1) * (n_querys_per_thread + 1);
        }
        else
        {
            querys_from[i] = i * (n_querys_per_thread) + n_onemore_query_thread;
            querys_to[i] = (i + 1) * (n_querys_per_thread) + n_onemore_query_thread;
        }
    }

    std::vector<int> s_indices(n_docs);
    for (int i = 0; i < n_docs; ++i)
    {
        s_indices[i] = i;
    }
    std::vector<std::thread> sorting_threads(n_threads - 1);
    for (size_t i = 0; i < n_threads - 1; i++)
    {
        sorting_threads[i] = std::thread([&, i]()
                                         { do_scoring_topk(scores, s_indices, indices, querys_from[i], querys_to[i]); });
    }
    do_scoring_topk(scores, s_indices, indices, querys_from[n_threads - 1], querys_to[n_threads - 1]);
    for (auto &t : sorting_threads)
    {
        t.join();
    }
}

int compare(std::vector<std::vector<int>> &indices_1, std::vector<std::vector<int>> &indices_2) {
    assert(indices_1.size() == indices_2.size());
    for (int i = 0; i < indices_1.size(); i++) {
        assert(indices_1[i].size() == indices_2[i].size());
        for (int j = 0; j < indices_1[i].size(); j++) {
            if (indices_1[i][j] != indices_2[i][j]) {
                printf("r=%d, c=%d, indices(%d) != indices_baseline(%d)\n", i, j, indices_1[i][j], indices_2[i][j]);
                return 0;
            }
        }
    }
    printf("compare done!\n");

    return 0;
}