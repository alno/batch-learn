#include "convert.hpp"

#include <batch_learn.hpp>

#include <fstream>

#include <boost/format.hpp>

constexpr uint max_line_size = 100000;

inline uint32_t h(uint32_t x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

int convert_command::run() {
    if (input_format_name == std::string("ffm")) {
        convert_from_ffm();
    } else {
        std::cout << "Error: unknown input format, supported formats: ffm" << std::endl;
        return -1;
    }

    return 0;
}

void convert_command::convert_from_ffm() {
    using namespace std;
    using namespace batch_learn;
    using boost::format;

    cout << "Converting " << input_file_name << " to " << output_file_name << " using " << index_bits << " index bits... ";
    cout.flush();

    FILE * input_file = fopen(input_file_name.c_str(), "r");
    if (input_file == nullptr)
        throw runtime_error("Error opening input file");

    file_index output_index;
    output_index.n_examples = 0;
    output_index.n_fields = 0;
    output_index.n_indices = 0;
    output_index.n_index_bits = index_bits;
    output_index.offsets.push_back(0);

    stream_data_writer output_data_writer(output_file_name + ".data");
    vector<feature> features;
    char line[max_line_size];

    uint64_t i = 0;
    while (fgets(line, max_line_size, input_file) != nullptr) {
        features.clear();

        ++ i;

        char *y_char = strtok(line, " \t");
        float y = (atoi(y_char) > 0) ? 1.0f : -1.0f;

        while (true) {
            char * feature_char = strtok(nullptr, " \t");

            if (feature_char == nullptr || *feature_char == '\n')
                break;


            char * index_delim = strpbrk(feature_char, ":");

            if (index_delim == nullptr)
                throw std::runtime_error(str(format("Invalid feature spec '%s' at line %d") % feature_char % i));

            char * value_delim = strpbrk(index_delim + 1, ":");

            if (value_delim == nullptr)
                throw std::runtime_error(str(format("Invalid feature spec '%s' at line %d") % feature_char % i));

            if (feature_char[0] == ':' || feature_char[0] == 0)
                throw std::runtime_error(str(format("Empty field in '%s' at line %d") % feature_char % i));

            if (index_delim[1] == ':' || index_delim[1] == 0)
                throw std::runtime_error(str(format("Empty index in '%s' at line %d") % feature_char % i));

            if (value_delim[1] == 0)
                throw std::runtime_error(str(format("Empty value in '%s' at line %d") % feature_char % i));

            index_delim[0] = 0;
            value_delim[0] = 0;

            uint field = atoi(feature_char);
            uint index = atoi(index_delim + 1);
            float value = atof(value_delim + 1);

            if (rehash_indexes > 0)
                index = h(index) % rehash_indexes;

            if (field >= output_index.n_fields)
                output_index.n_fields = field + 1;

            if (index >= output_index.n_indices)
                output_index.n_indices = index + 1;

            feature f;
            f.index = (field << index_bits) | index;
            f.value = value;

            features.push_back(f);
        }

        output_index.n_examples ++;
        output_index.labels.push_back(y);
        output_index.groups.push_back(0); // No group support in ffm format
        output_index.offsets.push_back(output_data_writer.write(features));

        if (output_index.n_examples % progress_step == 0) {
            uint progress = output_index.n_examples;
            std::string unit;

            if (progress_step % 1000000 == 0) {
                progress /= 1000000;
                unit = "M";
            } else if (progress_step % 1000 == 0) {
                progress /= 1000;
                unit = "K";
            }

            cout << progress << unit << "... ";
            cout.flush();
        }
    }

    fclose(input_file);

    write_index(output_file_name + ".index", output_index);

    cout << "Done." << endl;
}
