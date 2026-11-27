CC = gcc
CFLAGS = -O2 -Wall -Wextra -Werror -std=c11
COMP = ../../../hardware/recvoz-esp/components
SRC = main.c $(COMP)/speaker_db/speaker_db.c $(COMP)/gmm_ubm/gmm_ubm.c $(COMP)/knn/knn.c
HEADERS = $(wildcard ../stubs/*.h $(COMP)/speaker_db/include/*.h $(COMP)/speaker_db/*.h $(COMP)/gmm_ubm/include/*.h $(COMP)/knn/include/*.h)
persistence: $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) -I../stubs -I$(COMP)/speaker_db/include -I$(COMP)/speaker_db -I$(COMP)/gmm_ubm/include -I$(COMP)/knn/include $(SRC) -lm -o $@
.PHONY: run clean
run: persistence
	./persistence
clean:
	rm -f persistence
