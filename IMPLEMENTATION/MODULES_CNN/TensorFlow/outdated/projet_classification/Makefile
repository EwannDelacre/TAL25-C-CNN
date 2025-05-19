CC = gcc
CFLAGS = -O2 -I/usr/local/include
LDFLAGS = -L/usr/local/lib -ltensorflow
SRC = src/main.c src/tf_infer.c
OUT = main

all:
	$(CC) $(SRC) -o $(OUT) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f main
