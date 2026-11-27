CC = gcc
CFLAGS = -O2 -Wall -Wextra -Werror -std=c11
COMP = ../../../hardware/recvoz-esp/components
SRC = main.c $(COMP)/gmm_ubm/gmm_ubm.c
HEADERS = $(wildcard $(COMP)/gmm_ubm/include/*.h)
gmm_hosttest: $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMP)/gmm_ubm/include $(SRC) -lm -o $@
.PHONY: run clean
run: gmm_hosttest
	./gmm_hosttest
clean:
	rm -f gmm_hosttest
