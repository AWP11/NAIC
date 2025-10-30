# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -D_XOPEN_SOURCE=700
LIBS = -lsqlite3 -lm

# Directories
SRC_DIR = .
CORE_DIR = core
OBJ_DIR = obj

# Source files
SRCS = main.c interface_AI.c $(CORE_DIR)/kernel.c $(CORE_DIR)/fractal_tensor.c

# Object files (placed in obj directory)
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/interface_AI.o $(OBJ_DIR)/kernel.o $(OBJ_DIR)/fractal_tensor.o

# Include paths
INCLUDES = -I. -I$(CORE_DIR)

# Output executable name
OUTPUT = C_AI

# Default target
$(OUTPUT): $(OBJS)
	$(CC) $(OBJS) -o $(OUTPUT) $(LIBS)

# Create obj directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Object file rules
$(OBJ_DIR)/main.o: main.c interface_AI.h $(CORE_DIR)/kernel.h $(CORE_DIR)/fractal_tensor.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c main.c -o $(OBJ_DIR)/main.o

$(OBJ_DIR)/interface_AI.o: interface_AI.c interface_AI.h $(CORE_DIR)/kernel.h $(CORE_DIR)/fractal_tensor.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c interface_AI.c -o $(OBJ_DIR)/interface_AI.o

$(OBJ_DIR)/kernel.o: $(CORE_DIR)/kernel.c $(CORE_DIR)/kernel.h $(CORE_DIR)/fractal_tensor.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $(CORE_DIR)/kernel.c -o $(OBJ_DIR)/kernel.o

$(OBJ_DIR)/fractal_tensor.o: $(CORE_DIR)/fractal_tensor.c $(CORE_DIR)/fractal_tensor.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $(CORE_DIR)/fractal_tensor.c -o $(OBJ_DIR)/fractal_tensor.o

# Clean target
clean:
	rm -rf $(OBJ_DIR) $(OUTPUT)

# Debug target to show variables
debug:
	@echo "SRCS: $(SRCS)"
	@echo "OBJS: $(OBJS)"
	@echo "INCLUDES: $(INCLUDES)"
	@echo "OUTPUT: $(OUTPUT)"

# Install dependencies (optional)
deps:
	sudo apt-get update
	sudo apt-get install build-essential

.PHONY: clean debug deps