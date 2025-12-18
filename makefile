# ===== УНИВЕРСАЛЬНЫЙ MAKEFILE =====
# Автоматически определяет ОС и компилирует соответственно

# Имя исполняемого файла
TARGET = naic

# Определяем ОС
ifeq ($(OS),Windows_NT)
    DETECTED_OS = Windows
else
    DETECTED_OS = $(shell uname -s)
endif

# Компилятор и флаги по умолчанию
CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2
LDFLAGS = 

# Настройки для разных ОС
ifeq ($(DETECTED_OS),Windows)
    # Windows
    TARGET := $(TARGET).exe
    LIBS = 
    RM = del /Q
    ECHO = echo
else ifeq ($(DETECTED_OS),Linux)
    # Linux
    LIBS = -lm
    RM = rm -f
    ECHO = echo
else ifeq ($(DETECTED_OS),Darwin)
    # macOS
    LIBS = -lm
    RM = rm -f
    ECHO = echo
else
    $(error ОС не поддерживается: $(DETECTED_OS))
endif

# Исходные файлы
SRCS = main.c core.c
OBJS = $(SRCS:.c=.o)

# Правила сборки
all: $(TARGET)

$(TARGET): $(OBJS)
	@$(ECHO) "Сборка для $(DETECTED_OS)..."
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Очистка
clean:
	@$(ECHO) "Очистка..."
ifeq ($(DETECTED_OS),Windows)
	$(RM) $(subst /,\,$(OBJS) $(TARGET) *.exe 2>nul)
else
	$(RM) $(OBJS) $(TARGET)
endif

# Отладочная сборка
debug: CFLAGS += -g -DDEBUG
debug: clean $(TARGET)

# Статическая сборка (только Linux/macOS)
static: CFLAGS += -static
static: clean $(TARGET)

# Показать информацию
info:
	@$(ECHO) "Операционная система: $(DETECTED_OS)"
	@$(ECHO) "Компилятор: $(CC)"
	@$(ECHO) "Целевой файл: $(TARGET)"
	@$(ECHO) "Флаги компиляции: $(CFLAGS)"
	@$(ECHO) "Библиотеки: $(LIBS)"

# Помощь
help:
	@$(ECHO) "Доступные команды:"
	@$(ECHO) "  make all     - сборка проекта"
	@$(ECHO) "  make clean   - очистка"
	@$(ECHO) "  make debug   - сборка с отладочной информацией"
	@$(ECHO) "  make static  - статическая сборка (Linux/macOS)"
	@$(ECHO) "  make info    - показать информацию о сборке"
	@$(ECHO) "  make help    - эта справка"

.PHONY: all clean debug static info help 