#include "model.hpp"

#include "../util/dataset.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <algorithm>

#include <immintrin.h>
#include <omp.h>

#include <boost/program_options.hpp>

// Batch configuration
const uint32_t batch_size = 20000;
const uint32_t mini_batch_size = 24;


static std::default_random_engine rnd(2017);


std::vector<std::pair<uint64_t, uint64_t>> generate_mini_batches(uint64_t begin, uint64_t end) {
    std::vector<std::pair<uint64_t, uint64_t>> batches;

    for (uint64_t mini_batch_start = begin; mini_batch_start < end; mini_batch_start += mini_batch_size)
        batches.push_back(std::make_pair(mini_batch_start, min(mini_batch_start + mini_batch_size, end)));

    return batches;
}


float compute_norm(batch_learn::feature * fa, batch_learn::feature * fb) {
    float norm = 0;

    for (batch_learn::feature * f = fa; f != fb; ++ f)
        norm += f->value * f->value;

    return norm;
}


double train_on_dataset(model & m, const batch_learn_dataset & dataset) {
    time_t start_time = time(nullptr);

    std::cout << "  Training... ";
    std::cout.flush();

    auto batches = dataset.generate_batches(batch_size);

    std::shuffle(batches.begin(), batches.end(), rnd);

    double loss = 0.0;
    uint64_t cnt = 0;

    // Iterate over batches, read each and then iterate over examples
    #pragma omp parallel for schedule(dynamic, 1) reduction(+: loss) reduction(+: cnt)
    for (uint64_t bi = 0; bi < batches.size(); ++ bi) {
        auto batch_start_index = batches[bi].first;
        auto batch_end_index = batches[bi].second;

        auto batch_start_offset = dataset.index.offsets[batch_start_index];
        auto batch_end_offset = dataset.index.offsets[batch_end_index];

        auto mini_batches = generate_mini_batches(batch_start_index, batch_end_index);

        std::vector<batch_learn::feature> batch_features = batch_learn::read_batch(dataset.data_file_name, batch_start_offset, batch_end_offset);
        batch_learn::feature * batch_features_data = batch_features.data();

        std::shuffle(mini_batches.begin(), mini_batches.end(), rnd);

        for (auto mb = mini_batches.begin(); mb != mini_batches.end(); ++ mb) {
            for (auto ei = mb->first; ei < mb->second; ++ ei) {
                float y = dataset.index.labels[ei];

                auto start_offset = dataset.index.offsets[ei] - batch_start_offset;
                auto end_offset = dataset.index.offsets[ei+1] - batch_start_offset;

                float norm = compute_norm(batch_features_data + start_offset, batch_features_data + end_offset);

                float t = m.predict(batch_features_data + start_offset, batch_features_data + end_offset, norm, true);
                float expnyt = exp(-y*t);

                m.update(batch_features_data + start_offset, batch_features_data + end_offset, norm, -y * expnyt / (1+expnyt));

                loss += log(1+exp(-y*t));
            }
        }

        cnt += batch_end_index - batch_start_index;
    }

    std::cout << cnt << " examples processed in " << (time(nullptr) - start_time) << " seconds, loss = " << std::fixed << std::setprecision(5) << (loss / cnt) << std::endl;

    return loss;
}


double evaluate_on_dataset(model & m, const batch_learn_dataset & dataset) {
    time_t start_time = time(nullptr);

    std::cout << "  Evaluating... ";
    std::cout.flush();

    auto batches = dataset.generate_batches(batch_size);

    double loss = 0.0;
    uint32_t cnt = 0;

    std::vector<float> predictions(dataset.index.n_examples);

    // Iterate over batches, read each and then iterate over examples
    #pragma omp parallel for schedule(dynamic, 1) reduction(+: loss) reduction(+: cnt)
    for (uint32_t bi = 0; bi < batches.size(); ++ bi) {
        auto batch_start_index = batches[bi].first;
        auto batch_end_index = batches[bi].second;

        auto batch_start_offset = dataset.index.offsets[batch_start_index];
        auto batch_end_offset = dataset.index.offsets[batch_end_index];

        std::vector<batch_learn::feature> batch_features = batch_learn::read_batch(dataset.data_file_name, batch_start_offset, batch_end_offset);
        batch_learn::feature * batch_features_data = batch_features.data();

        for (auto ei = batch_start_index; ei < batch_end_index; ++ ei) {
            float y = dataset.index.labels[ei];

            auto start_offset = dataset.index.offsets[ei] - batch_start_offset;
            auto end_offset = dataset.index.offsets[ei+1] - batch_start_offset;

            float norm = compute_norm(batch_features_data + start_offset, batch_features_data + end_offset);
            float t = m.predict(batch_features_data + start_offset, batch_features_data + end_offset, norm, false);

            loss += log(1+exp(-y*t));
            predictions[ei] = 1 / (1+exp(-t));
        }

        cnt += batch_end_index - batch_start_index;
    }

    std::cout << cnt << " examples processed in " << (time(nullptr) - start_time) << " seconds, loss = " << std::fixed << std::setprecision(5) << (loss / cnt) << std::endl;

    return loss;
}

void predict_on_dataset(model & m, const batch_learn_dataset & dataset, std::ostream & out) {
    time_t start_time = time(nullptr);

    std::cout << "  Predicting... ";
    std::cout.flush();

    auto batches = dataset.generate_batches(batch_size);

    uint64_t cnt = 0;

    // Iterate over batches, read each and then iterate over examples
    for (uint64_t bi = 0; bi < batches.size(); ++ bi) {
        auto batch_start_index = batches[bi].first;
        auto batch_end_index = batches[bi].second;

        auto batch_start_offset = dataset.index.offsets[batch_start_index];
        auto batch_end_offset = dataset.index.offsets[batch_end_index];

        std::vector<batch_learn::feature> batch_features = batch_learn::read_batch(dataset.data_file_name, batch_start_offset, batch_end_offset);
        batch_learn::feature * batch_features_data = batch_features.data();

        for (auto ei = batch_start_index; ei < batch_end_index; ++ ei) {
            auto start_offset = dataset.index.offsets[ei] - batch_start_offset;
            auto end_offset = dataset.index.offsets[ei+1] - batch_start_offset;

            float norm = compute_norm(batch_features_data + start_offset, batch_features_data + end_offset);
            float t = m.predict(batch_features_data + start_offset, batch_features_data + end_offset, norm, false);

            out << 1/(1+exp(-t)) << std::endl;
        }

        cnt += batch_end_index - batch_start_index;
    }

    std::cout << cnt << " examples processed in " << (time(nullptr) - start_time) << " seconds" << std::endl;
}


int model_command::run() {
    using namespace std;

    omp_set_num_threads(n_threads);
    rnd.seed(seed);

    auto ds_train = batch_learn_dataset(train_file_name);

    auto model = create_model(ds_train.index.n_fields, ds_train.index.n_indices, ds_train.index.n_index_bits);

    if (val_file_name.empty()) { // No validation set given, just train
        for (uint epoch = 0; epoch < n_epochs; ++ epoch) {
            cout << "Epoch " << epoch << "..." << endl;

            train_on_dataset(*model, ds_train);
        }
    } else { // Train with validation each epoch
        auto ds_val = batch_learn_dataset(val_file_name);

        if (ds_val.index.n_index_bits != ds_train.index.n_index_bits)
            throw std::runtime_error("Mismatching index bits in train and val");

        for (uint epoch = 0; epoch < n_epochs; ++ epoch) {
            cout << "Epoch " << epoch << "..." << endl;

            train_on_dataset(*model, ds_train);
            evaluate_on_dataset(*model, ds_val);
        }
    }

    // Predict on test if given
    if (!test_file_name.empty() && !pred_file_name.empty()) {
        auto ds_test = batch_learn_dataset(test_file_name);

        if (ds_test.index.n_index_bits != ds_train.index.n_index_bits)
            throw std::runtime_error("Mismatching index bits in train and test");

        ofstream out(pred_file_name);
        predict_on_dataset(*model, ds_test, out);
    }

    return 0;
}
