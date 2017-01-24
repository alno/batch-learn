#pragma once

#include "command.hpp"


class convert_command : public command {
protected:
    std::string input_file_name, output_file_name, input_format_name;
    uint index_bits, progress_step, rehash_indexes;
public:
    convert_command(): rehash_indexes(0) {
        using namespace boost::program_options;

        options_desc.add_options()
            ("bits,b", value<uint>(&index_bits)->default_value(24), "number of bits to store feature indices")
            ("rehash", value<uint>(&rehash_indexes), "rehash feature indices to given max")
            ("progress,p", value<uint>(&progress_step)->default_value(1000000), "print progress every N examples")
            ("format,f", value<std::string>(&input_format_name)->required(), "input format name (only ffm supported for now)")
            ("input-file,I", value<std::string>(&input_file_name)->required(), "input file name")
            ("output-file,O", value<std::string>(&output_file_name)->required(), "output file name");

        positional_options_desc.add("input-file", 1).add("output-file", 1);
    }

    virtual std::string name() { return "convert"; }
    virtual std::string description() { return "convert file to batch-learn binary format"; }

    virtual int run();
private:
    void convert_from_ffm();
};
