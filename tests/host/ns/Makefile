CC = gcc
CFLAGS = -O2 -Wall -Wextra -Werror -std=c11
MFCC_DIR = ../../../hardware/recvoz-esp/components/mfcc
SRC = main.c $(MFCC_DIR)/mfcc_dsp.c $(MFCC_DIR)/noise_sup.c
INC = -I$(MFCC_DIR)/include -I$(MFCC_DIR)

HEADERS = $(wildcard $(KNN_DIR)/include/*.h $(SPK_DIR)/*.h $(MFCC_DIR)/include/*.h $(MFCC_DIR)/*.h)

all: ns_hosttest

ns_hosttest: $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) -lm

run: ns_hosttest
	./ns_hosttest

clean:
	rm -f ns_hosttest

.PHONY: all run clean
