#include "command.hpp"
#include "convert-command.hpp"
#include "ffm-command.hpp"
#include "nn-command.hpp"

#include <unordered_map>
#include <iostream>
#include <string>
#include <memory>


static std::unordered_map<std::string, std::unique_ptr<command>> commands;


class help_command : public command {
protected:
    std::string command_name;
public:
    help_command() {
        using namespace boost::program_options;

        options_desc.add_options()
            ("command", value<std::string>(&command_name), "command name to get help about");

        positional_options_desc.add("command", -1);
    }

    virtual std::string name() { return "help"; }
    virtual std::string description() { return "get the help"; }

    virtual int run() {
        using namespace std;

        if (commands.count(command_name) == 0) {
            cout << "Supported commands:" << endl;

            for (auto it = commands.begin(); it != commands.end(); ++ it) {
                cout << "  ";
                cout.width(10);
                cout << left << it->first << it->second->description() << endl;
              }
        } else {
            commands[command_name]->print_help();
        }

        return 0;
    }
};



int main(int ac, char* av[]) {
    using namespace std;

    // Prepare commands
    commands.insert(make_pair("help", new help_command()));
    commands.insert(make_pair("convert", new convert_command()));
    commands.insert(make_pair("ffm", new ffm_command()));
    commands.insert(make_pair("nn", new nn_command()));

    // Check if command specified
    if (ac <= 1) {
        cout << "No command specified" << endl;

        commands["help"]->run();

        return -1;
    }

    // Extract command and check if it's supported
    string cmd_name(av[1]);

    if (commands.count(cmd_name) == 0) {
        cout << "Unknown command " << cmd_name << " specified" << endl;

        commands["help"]->run();

        return -2;
    }

    auto & cmd = commands[cmd_name];

    // Try to parse command options
    try {
        cmd->parse_options(ac - 1, av + 1);
    } catch (const std::exception & e) {
        cout << "Error: " << e.what() << endl;
        cmd->print_help();
        return -3;
    }

    // Print help if required
    if (cmd->options_vm.count("help") > 0) {
        cmd->print_help();
        return 0;
    }

    // Run command
    return cmd->run();
}
