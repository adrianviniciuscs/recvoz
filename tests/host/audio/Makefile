CC = gcc
CFLAGS = -O2 -Wall -Wextra -Werror -std=c11
COMP = ../../../hardware/recvoz-esp/components/audio_in
HEADERS = $(wildcard ../stubs/*.h ../stubs/driver/*.h ../stubs/freertos/*.h $(COMP)/include/*.h)
audio: main.c $(COMP)/audio_in.c $(HEADERS)
	$(CC) $(CFLAGS) -I../stubs -I$(COMP)/include main.c $(COMP)/audio_in.c -o $@
.PHONY: run clean
run: audio
	./audio
clean:
	rm -f audio
