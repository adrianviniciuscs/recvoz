CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
KNN_DIR = ../../hardware/recvoz-esp/components/knn
SPK_DIR = ../../hardware/recvoz-esp/components/speaker_db
SRC = main.c $(KNN_DIR)/knn.c
INC = -I$(KNN_DIR)/include -I$(SPK_DIR)

all: knn_hosttest

knn_hosttest: $(SRC)
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) -lm

run: knn_hosttest
	./knn_hosttest

clean:
	rm -f knn_hosttest

.PHONY: all run clean
