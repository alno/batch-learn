#pragma once

#include <vector>
#include <stdexcept>
#include <cstdio>

namespace batch_learn {

const uint32_t file_format_version = 1;

struct feature {
    uint32_t index; // Feature index consists of two parts: field (in high bits) and in-field index (in low bits), number of bits to store index specified in file header
    float value;
};

struct file_index {
    uint64_t n_examples; // Number of examples;
    uint32_t n_fields; // Number of feature fields (max + 1)
    uint32_t n_indices; // Number of feature in-field indices (max + 1)

    uint32_t n_index_bits; // Number of bits used to store index part (should be enough to include max index)

    std::vector<float> labels; // Target values of examples (size N)
    std::vector<uint64_t> offsets; // Offsets of example data (size N +1) in number of features
    std::vector<uint64_t> groups; // Group identifiers for MAP calculation
};

// Index IO functions

inline void write_index(const std::string & file_name, const file_index & index) {
    using namespace std;

    if (index.labels.size() != index.n_examples)
        throw runtime_error("Invalid index labels size");

    if (index.offsets.size() != index.n_examples + 1)
        throw runtime_error("Invalid index offsets size");

    if (index.groups.size() != index.n_examples)
        throw runtime_error("Invalid index groups size");

    if ((1ul << index.n_index_bits) < index.n_indices)
        throw runtime_error("Not enough index bits allocated to store max index");

    if ((1ul << (32 - index.n_index_bits)) < index.n_fields)
        throw runtime_error("Not enough field bits allocated to store max field");

    FILE * file = fopen(file_name.c_str(), "wb");

    if(file == nullptr)
        throw runtime_error(string("Can't open index file ") + file_name);

    if (fwrite(&file_format_version, sizeof(uint32_t), 1, file) != 1)
        throw runtime_error("Error writing format version");

    // Header

    if (fwrite(&index.n_examples, sizeof(uint64_t), 1, file) != 1)
        throw runtime_error("Error writing example count");

    if (fwrite(&index.n_fields, sizeof(uint32_t), 1, file) != 1)
        throw runtime_error("Error writing field count");

    if (fwrite(&index.n_indices, sizeof(uint32_t), 1, file) != 1)
        throw runtime_error("Error writing index count");

    if (fwrite(&index.n_index_bits, sizeof(uint32_t), 1, file) != 1)
        throw runtime_error("Error writing index bit count");

    // Index itself

    if (fwrite(index.labels.data(), sizeof(float), index.labels.size(), file) != index.labels.size())
        throw runtime_error("Error writing labels");

    if (fwrite(index.offsets.data(), sizeof(uint64_t), index.offsets.size(), file) != index.offsets.size())
        throw runtime_error("Error writing offsets");

    if (fwrite(index.groups.data(), sizeof(uint64_t), index.groups.size(), file) != index.groups.size())
        throw runtime_error("Error writing groups");

    fclose(file);
};

inline file_index read_index(const std::string & file_name) {
    using namespace std;

    file_index index;
    FILE * file = fopen(file_name.c_str(), "rb");

    if(file == nullptr)
        throw runtime_error(string("Can't open index file ") + file_name);

    uint32_t version;

    if (fread(&version, sizeof(uint32_t), 1, file) != 1)
        throw runtime_error("Error reading version");

    if (version != file_format_version)
        throw runtime_error("File format version mismatch");

    // Header

    if (fread(&index.n_examples, sizeof(uint64_t), 1, file) != 1)
        throw runtime_error("Error reading example count");

    if (fread(&index.n_fields, sizeof(uint32_t), 1, file) != 1)
        throw runtime_error("Error reading field count");

    if (fread(&index.n_indices, sizeof(uint32_t), 1, file) != 1)
        throw runtime_error("Error reading index count");

    if (fread(&index.n_index_bits, sizeof(uint32_t), 1, file) != 1)
        throw runtime_error("Error reading index bit count");

    // Reserve space for y and offsets
    index.labels.resize(index.n_examples, 0);
    index.offsets.resize(index.n_examples + 1, 0);
    index.groups.resize(index.n_examples, 0);

    if (fread(index.labels.data(), sizeof(float), index.labels.size(), file) != index.labels.size())
        throw runtime_error("Error reading labels");

    if (fread(index.offsets.data(), sizeof(uint64_t), index.offsets.size(), file) != index.offsets.size())
        throw runtime_error("Error reading offsets");

    if (fread(index.groups.data(), sizeof(uint64_t), index.groups.size(), file) != index.groups.size())
        throw runtime_error("Error reading groups");

    fclose(file);

    return index;
};

// Batch IO functions

inline void read_batch(const std::string & file_name, uint64_t from, uint64_t to, std::vector<feature> & features) {
    using namespace std;

    if (to < from)
        throw runtime_error("Wrong range");

    features.resize(to - from);

    // Empty range, no need to read
    if (to == from)
        return;

    FILE * file = fopen(file_name.c_str(), "rb");

    if (file == nullptr)
        throw runtime_error(string("Can't open data file ") + file_name);

    if (fseek((FILE *)file, from * sizeof(feature), SEEK_SET) != 0)
        throw new runtime_error("Can't set file pos");

    if (fread(features.data(), sizeof(feature), features.size(), (FILE *)file) != features.size())
        throw new runtime_error("Can't read data");

    fclose(file);
}

inline std::vector<feature> read_batch(const std::string & file_name, uint64_t from, uint64_t to) {
    std::vector<feature> features(to - from);
    read_batch(file_name, from, to, features);
    return features;
}

// Data file writer

class stream_data_writer {
    FILE * file;
    uint64_t offset;
public:
    stream_data_writer(const std::string & file_name): offset(0) {
        using namespace std;

        file = fopen(file_name.c_str(), "wb");

        if (file == nullptr)
            throw runtime_error(string("Can't open data file ") + file_name);
    }

    ~stream_data_writer() {
        fclose((FILE *)file);
    }

    uint64_t write(const std::vector<feature> & features) {
        if (fwrite(features.data(), sizeof(feature), features.size(), (FILE *)file) != features.size())
            throw std::runtime_error("Error writing example count");

        offset += features.size();

        return offset;
    }
};

};
