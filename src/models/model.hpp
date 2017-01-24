#pragma once

#include <batch_learn.hpp>

class model {
public:
    model() {}
    virtual ~model() {}

    virtual float predict(const batch_learn::feature * start, const batch_learn::feature * end, float norm, bool train) = 0;
    virtual void update(const batch_learn::feature * start, const batch_learn::feature * end, float norm, float kappa) = 0;
};
