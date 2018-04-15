#pragma once

#include "model.hpp"
#include "../models/ffm.hpp"


class ffm_command : public model_command {
protected:
    uint n_dim;
    float eta, lambda;
    bool dropout;
public:
    ffm_command() {
        using namespace boost::program_options;

        options_desc.add_options()
            ("dim,k", value<uint>(&n_dim)->default_value(4), "dimensions")
            ("eta", value<float>(&eta)->default_value(0.2), "learning rate")
            ("lambda", value<float>(&lambda)->default_value(0.00002), "l2 regularization coeff")
            ("dropout", value<bool>(&dropout)->default_value(true), "dropout");
    }

    virtual std::string name() { return "ffm"; }
    virtual std::string description() { return "train and apply ffm model"; }

    virtual std::unique_ptr<model> create_model(uint32_t n_fields, uint32_t n_indices, uint32_t n_index_bits) {
        return std::unique_ptr<model>(new ffm_model(n_fields, n_indices, n_index_bits, n_dim, seed, eta, lambda, dropout));
    }
};
