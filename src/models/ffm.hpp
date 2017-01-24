#pragma once

#include "model.hpp"


class ffm_model : public model {
    uint32_t n_fields, n_indices, n_index_bits, n_dim;

    uint32_t n_dim_aligned, index_stride, field_stride, index_mask;

    float * ffm_weights;
    float * lin_weights;

    float bias_w;
    float bias_wg;

    float eta;
    float lambda;
public:
    ffm_model(uint32_t n_fields, uint32_t n_indices, uint32_t n_index_bits, uint32_t n_dim, int seed, float eta, float lambda);
    virtual ~ffm_model();

    virtual float predict(const batch_learn::feature * start, const batch_learn::feature * end, float norm, bool train);
    virtual void update(const batch_learn::feature * start, const batch_learn::feature * end, float norm, float kappa);
};
