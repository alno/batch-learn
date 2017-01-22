#pragma once

#include <batch_learn.hpp>

class model {
public:
    model() {}
    virtual ~model() {}

    virtual float predict(const batch_learn::feature * start, const batch_learn::feature * end, float norm, uint64_t * dropout_mask, float dropout_mult) = 0;
    virtual void update(const batch_learn::feature * start, const batch_learn::feature * end, float norm, float kappa, uint64_t * dropout_mask, float dropout_mult) = 0;

    virtual uint get_dropout_mask_size(const batch_learn::feature * start, const batch_learn::feature * end) = 0;
};
