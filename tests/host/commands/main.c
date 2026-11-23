#include <assert.h>
#include <stdio.h>
#include "command_args.h"
int main(void) {
    char *bad[][3] = {{"vad", "-1"}, {"vad", "abc"}, {"vad", "0"}, {"hold", "60001"},
        {"hold", "1e3"}, {"hold", "-1"}, {"grab", "0"}, {"grab", "4"}, {"tau", "NaN"}, {"tau", "1"}, {"ns", "auto", "maybe"},
        {"ns", "alpha", "1x"}, {"ns", "floor", "inf"}, {"ns", "beta", "0"}};
    for (unsigned i = 0; i < sizeof bad / sizeof bad[0]; i++)
        assert(!command_args_valid(bad[i][2] ? 3 : 2, bad[i]));
    char *good[][3] = {{"vad", "300"}, {"hold", "0"}, {"grab", "2"}, {"tau", ".017"}, {"ns", "auto", "off"}, {"ns", "floor", "0"}};
    for (unsigned i = 0; i < sizeof good / sizeof good[0]; i++)
        assert(command_args_valid(good[i][2] ? 3 : 2, good[i]));
    puts("PASS command validation");
}
