CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -D_XOPEN_SOURCE=700
LIBS = -lsqlite3 -lm

SOURCES = main.c kernel.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = NAIC

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

rebuild: clean $(TARGET)

.PHONY: clean rebuild