#include "ffm.hpp"

#include "../util/model.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

#include <cstring>


class state {
public:
    std::vector<uint64_t> dropout_mask;
    float dropout_mult;
public:
    void init_train_dropout_mask(int len) {
        dropout_mult = 2.0;
        dropout_mask.resize((len + 63) / 64);

        for (uint64_t * p = dropout_mask.data(); p != dropout_mask.data() + dropout_mask.size(); ++ p)
            if (_rdrand64_step((unsigned long long *)p) != 1)
                throw std::runtime_error("Error generating random number!");

    }

    void init_test_dropout_mask(int len) {
        dropout_mult = 1.0;
        dropout_mask.resize((len + 63) / 64);

        memset(dropout_mask.data(), 0xFF, dropout_mask.size() * sizeof(uint64_t));
    }
};

static thread_local state local_state;


template <typename D>
static void init_ffm_weights(float * weights, uint64_t n, uint32_t n_dim, uint32_t n_dim_aligned, D gen, std::default_random_engine & rnd) {
    float * w = weights;

    for(uint64_t i = 0; i < n; i++) {
        for (uint d = 0; d < n_dim; d++, w++)
            *w = gen(rnd);

        for (uint d = n_dim; d < n_dim_aligned; d++, w++)
            *w = 0;

        for (uint d = n_dim_aligned; d < 2*n_dim_aligned; d++, w++)
            *w = 1;
    }
}


static void init_lin_weights(float * weights, uint64_t n) {
    float * w = weights;

    for(uint64_t i = 0; i < n; i++) {
        *w++ = 0;
        *w++ = 1;
    }
}


ffm_model::ffm_model(uint32_t n_fields, uint32_t n_indices, uint32_t n_index_bits, uint32_t n_dim, int seed, float eta, float lambda) {
    this->n_fields = n_fields;
    this->n_indices = n_indices;
    this->n_index_bits = n_index_bits;
    this->n_dim = n_dim;
    this->eta = eta;
    this->lambda = lambda;

    n_dim_aligned = ((n_dim - 1) / align_floats + 1) * align_floats;

    index_stride = n_fields * n_dim_aligned * 2;
    field_stride = n_dim_aligned * 2;
    index_mask = (1ul << n_index_bits) - 1;

    std::default_random_engine rnd(seed);

    bias_w = 0;
    bias_wg = 1;

    try {
        uint64_t total_weights = size_t(n_indices) * n_fields * n_dim_aligned * 2 + n_indices * 2;

        std::cout << "Allocating " << (total_weights * sizeof(float) / 1024 / 1024) << " MB memory for model weights... ";
        std::cout.flush();

        ffm_weights = malloc_aligned<float>(uint64_t(n_indices) * n_fields * n_dim_aligned * 2);
        lin_weights = malloc_aligned<float>(n_indices * 2);

        std::cout << "done." << std::endl;
    } catch (std::bad_alloc & e) {
        throw std::runtime_error("Can't allocate weights memory");
    }

    std::cout << "Initializing weights... ";
    std::cout.flush();

    init_ffm_weights(ffm_weights, size_t(n_indices) * n_fields, n_dim, n_dim_aligned, std::uniform_real_distribution<float>(0.0, 1.0/sqrt(n_dim)), rnd);
    init_lin_weights(lin_weights, n_indices);

    std::cout << "done." << std::endl;
}


ffm_model::~ffm_model() {
    free(ffm_weights);
    free(lin_weights);
}


float ffm_model::predict(const batch_learn::feature * start, const batch_learn::feature * end, float norm, bool train) {
    uint feature_count = end - start;
    uint interaction_count = feature_count * (feature_count + 1) / 2;

    if (train)
        local_state.init_train_dropout_mask(interaction_count);
    else
        local_state.init_test_dropout_mask(interaction_count);

    auto dropout_mask = local_state.dropout_mask.data();
    float dropout_mult = local_state.dropout_mult;


    float linear_total = bias_w;
    float linear_norm = end - start;

    __m256 xmm_total = _mm256_set1_ps(0);

    uint i = 0;

    for (const batch_learn::feature * fa = start; fa != end; ++ fa) {
        uint index_a = fa->index &  index_mask;
        uint field_a = fa->index >> n_index_bits;
        float value_a = fa->value;

        // Check index/field bounds
        if (index_a >= n_indices || field_a >= n_fields)
            continue;

        linear_total += value_a * lin_weights[index_a*2] / linear_norm;

        for (const batch_learn::feature * fb = start; fb != fa; ++ fb, ++ i) {
            uint index_b = fb->index &  index_mask;
            uint field_b = fb->index >> n_index_bits;
            float value_b = fb->value;

            // Check index/field bounds
            if (index_b >= n_indices || field_b >= n_fields)
                continue;

            if (test_mask_bit(dropout_mask, i) == 0)
                continue;

            float * wa = ffm_weights + index_a * index_stride + field_b * field_stride;
            float * wb = ffm_weights + index_b * index_stride + field_a * field_stride;

            __m256 xmm_val = _mm256_set1_ps(dropout_mult * value_a * value_b / norm);

            for(uint d = 0; d < n_dim; d += 8) {
                __m256 xmm_wa = _mm256_load_ps(wa + d);
                __m256 xmm_wb = _mm256_load_ps(wb + d);

                xmm_total = _mm256_add_ps(xmm_total, _mm256_mul_ps(_mm256_mul_ps(xmm_wa, xmm_wb), xmm_val));
            }
        }
    }

    return sum(xmm_total) + linear_total;
}


void ffm_model::update(const batch_learn::feature * start, const batch_learn::feature * end, float norm, float kappa) {
    auto dropout_mask = local_state.dropout_mask.data();
    float dropout_mult = local_state.dropout_mult;

    float linear_norm = end - start;

    __m256 xmm_eta = _mm256_set1_ps(eta);
    __m256 xmm_lambda = _mm256_set1_ps(lambda);

    uint i = 0;

    for (const batch_learn::feature * fa = start; fa != end; ++ fa) {
        uint index_a = fa->index &  index_mask;
        uint field_a = fa->index >> n_index_bits;
        float value_a = fa->value;

        // Check index/field bounds
        if (index_a >= n_indices || field_a >= n_fields)
            continue;

        float g = lambda * lin_weights[index_a*2] + kappa * value_a / linear_norm;
        float wg = lin_weights[index_a*2 + 1] + g*g;

        lin_weights[index_a*2] -= eta * g / sqrt(wg);
        lin_weights[index_a*2 + 1] = wg;

        for (const batch_learn::feature * fb = start; fb != fa; ++ fb, ++ i) {
            uint index_b = fb->index &  index_mask;
            uint field_b = fb->index >> n_index_bits;
            float value_b = fb->value;

            // Check index/field bounds
            if (index_b >= n_indices || field_b >= n_fields)
                continue;

            if (test_mask_bit(dropout_mask, i) == 0)
                continue;

            float * wa = ffm_weights + index_a * index_stride + field_b * field_stride;
            float * wb = ffm_weights + index_b * index_stride + field_a * field_stride;

            float * wga = wa + n_dim_aligned;
            float * wgb = wb + n_dim_aligned;

            __m256 xmm_kappa_val = _mm256_set1_ps(kappa * dropout_mult * value_a * value_b / norm);

            for(uint d = 0; d < n_dim; d += 8) {
                // Load weights
                __m256 xmm_wa = _mm256_load_ps(wa + d);
                __m256 xmm_wb = _mm256_load_ps(wb + d);

                __m256 xmm_wga = _mm256_load_ps(wga + d);
                __m256 xmm_wgb = _mm256_load_ps(wgb + d);

                // Compute gradient values
                __m256 xmm_ga = _mm256_add_ps(_mm256_mul_ps(xmm_lambda, xmm_wa), _mm256_mul_ps(xmm_kappa_val, xmm_wb));
                __m256 xmm_gb = _mm256_add_ps(_mm256_mul_ps(xmm_lambda, xmm_wb), _mm256_mul_ps(xmm_kappa_val, xmm_wa));

                // Update weights
                xmm_wga = _mm256_add_ps(xmm_wga, _mm256_mul_ps(xmm_ga, xmm_ga));
                xmm_wgb = _mm256_add_ps(xmm_wgb, _mm256_mul_ps(xmm_gb, xmm_gb));

                xmm_wa  = _mm256_sub_ps(xmm_wa, _mm256_mul_ps(xmm_eta, _mm256_mul_ps(_mm256_rsqrt_ps(xmm_wga), xmm_ga)));
                xmm_wb  = _mm256_sub_ps(xmm_wb, _mm256_mul_ps(xmm_eta, _mm256_mul_ps(_mm256_rsqrt_ps(xmm_wgb), xmm_gb)));

                // Store weights
                _mm256_store_ps(wa + d, xmm_wa);
                _mm256_store_ps(wb + d, xmm_wb);

                _mm256_store_ps(wga + d, xmm_wga);
                _mm256_store_ps(wgb + d, xmm_wgb);
            }
        }
    }

    // Update bias
    bias_wg += kappa;
    bias_w -= eta * kappa / sqrt(bias_wg);
}
