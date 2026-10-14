CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
MFCC_DIR = ../../hardware/recvoz-esp/components/mfcc
SRC = main.c $(MFCC_DIR)/mfcc_dsp.c $(MFCC_DIR)/noise_sup.c
INC = -I$(MFCC_DIR)/include -I$(MFCC_DIR)

all: ns_hosttest

ns_hosttest: $(SRC)
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) -lm

run: ns_hosttest
	./ns_hosttest

clean:
	rm -f ns_hosttest

.PHONY: all run clean
