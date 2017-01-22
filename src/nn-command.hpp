#pragma once

#include "model-command.hpp"
#include "nn-model.hpp"


class nn_command : public model_command {
protected:
    float eta, lambda;
public:
    nn_command() {
        using namespace boost::program_options;

        options_desc.add_options()
            ("eta", value<float>(&eta)->default_value(0.02), "learning rate")
            ("lambda", value<float>(&lambda)->default_value(0.00002), "l2 regularization coeff");
    }

    virtual std::string name() { return "nn"; }
    virtual std::string description() { return "train and apply nn model"; }

    virtual std::unique_ptr<model> create_model(batch_learn::file_index & index) {
        return std::unique_ptr<model>(new nn_model(index.n_indices, index.n_index_bits, seed, eta, lambda));
    }
};
