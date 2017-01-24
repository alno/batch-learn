#pragma once

#include "common.hpp"

#include <batch_learn.hpp>


class batch_learn_dataset {
public:
    batch_learn::file_index index;
    std::string data_file_name;

    batch_learn_dataset(const std::string & file_name) {
        std::cout << "Loading " << file_name << ".index... ";
        std::cout.flush();

        index = batch_learn::read_index(file_name + ".index");
        data_file_name = file_name + ".data";

        std::cout << index.n_examples << " examples" << std::endl;
    }

    std::vector<std::pair<uint64_t, uint64_t>> generate_batches(uint64_t batch_size) const {
        std::vector<std::pair<uint64_t, uint64_t>> batches;

        for (uint64_t batch_start = 0; batch_start < index.n_examples; batch_start += batch_size)
            batches.push_back(std::make_pair(batch_start, min(batch_start + batch_size, index.n_examples)));

        return batches;
    }
};
