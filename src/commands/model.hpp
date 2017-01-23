#pragma once

#include "command.hpp"
#include "../models/model.hpp"


class model_command : public command {
protected:
    std::string train_file_name, val_file_name, test_file_name, pred_file_name;
    uint n_epochs, n_threads, seed;
public:
    model_command(): seed(0) {
        using namespace boost::program_options;

        options_desc.add_options()
            ("train", value<std::string>(&train_file_name)->required(), "train dataset file")
            ("val", value<std::string>(&val_file_name), "validation dataset file")
            ("test", value<std::string>(&test_file_name), "test dataset file")
            ("pred", value<std::string>(&pred_file_name), "file to save predictions")
            ("seed,s", value<uint>(&seed), "random seed")
            ("epochs", value<uint>(&n_epochs)->default_value(10), "number of epochs")
            ("threads,t", value<uint>(&n_threads)->default_value(4), "number of threads");

        positional_options_desc.add("test", 1).add("pred", 1);
    }

    virtual int run();
    virtual std::unique_ptr<model> create_model(uint32_t n_fields, uint32_t n_indices, uint32_t n_index_bits) = 0;
};
