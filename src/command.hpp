#pragma once

#include <iostream>

#include <boost/program_options.hpp>


class command {
protected:
    boost::program_options::options_description options_desc;
    boost::program_options::positional_options_description positional_options_desc;
public:
    boost::program_options::variables_map options_vm;
public:
    command() {
        options_desc.add_options()("help", "show command help");
    }

    virtual ~command() {}

    virtual std::string name() = 0;
    virtual std::string description() = 0;

    virtual void parse_options(int ac, char * av[]) {
        using namespace boost::program_options;

        store(command_line_parser(ac, av).options(options_desc).positional(positional_options_desc).run(), options_vm);
        notify(options_vm);
    }

    virtual void print_help() {
        std::cout << "Supported command options:" << std::endl;
        std::cout << options_desc << std::endl;
    }

    virtual int run() = 0;
};
