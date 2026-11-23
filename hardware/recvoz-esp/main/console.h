#pragma once
#include <stdbool.h>
typedef int (*recvoz_command_fn)(int argc, char **argv);
void console_init(void);
bool console_pending(void);
void console_process(recvoz_command_fn execute);
