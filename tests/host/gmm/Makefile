CC = gcc
CFLAGS = -O2 -Wall -Wextra -Werror -std=c11
COMP = ../../../hardware/recvoz-esp/components
SRC = main.c $(COMP)/speaker_db/gmm_ubm.c
HEADERS = $(wildcard $(COMP)/speaker_db/include/*.h $(COMP)/speaker_db/*.h)
gmm_hosttest: $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMP)/speaker_db/include -I$(COMP)/speaker_db $(SRC) -lm -o $@
.PHONY: run clean
run: gmm_hosttest
	./gmm_hosttest
clean:
	rm -f gmm_hosttest
