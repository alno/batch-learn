#pragma once

#include "model.hpp"


class nn_model : public model {
    float * lin_w;
    float * lin_wg;

    float * l1_w;
    float * l1_wg;

    float * l2_w;
    float * l2_wg;

    float * l3_w;
    float * l3_wg;

    float eta;
    float lambda;

    uint32_t n_indices, n_index_bits, index_mask;
public:
    nn_model(uint32_t n_indices, uint32_t n_index_bits, int seed, float eta, float lambda);
    virtual ~nn_model();

    virtual float predict(const batch_learn::feature * start, const batch_learn::feature * end, float norm, uint64_t * dropout_mask, float dropout_mult);
    virtual void update(const batch_learn::feature * start, const batch_learn::feature * end, float norm, float kappa, uint64_t * dropout_mask, float dropout_mult);

    virtual uint get_dropout_mask_size(const batch_learn::feature * start, const batch_learn::feature * end);
};
