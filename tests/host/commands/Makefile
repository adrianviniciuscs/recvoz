CC = gcc
CFLAGS = -O2 -Wall -Wextra -Werror -std=c11
APP = ../../../hardware/recvoz-esp/main
commands: main.c $(APP)/command_args.c $(APP)/command_args.h
	$(CC) $(CFLAGS) -I$(APP) main.c $(APP)/command_args.c -lm -o $@
.PHONY: run clean
run: commands
	./commands
clean:
	rm -f commands
