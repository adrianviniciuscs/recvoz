CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
MFCC_DIR = ../../hardware/recvoz-esp/components/mfcc
SRC = main.c $(MFCC_DIR)/mfcc_dsp.c
INC = -I$(MFCC_DIR)/include -I$(MFCC_DIR)

all: hosttest

hosttest: $(SRC)
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) -lm

run: hosttest
	./hosttest

clean:
	rm -f hosttest

.PHONY: all run clean
