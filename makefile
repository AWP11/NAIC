CC = gcc
CFLAGS = -std=c11 -Wall -O2 -g -DDEBUG_MODE=1
CFLAGS_SHARED = -fPIC
LDFLAGS = -lm -rdynamic
LDFLAGS_SHARED = -shared -lm

# Имена файлов
TARGET_CONSOLE = ai_system
TARGET_LIB = libuna.so
TARGET_STATIC_LIB = libuna.a

# Исходные файлы
UNIFIED_MAIN = main_unified.c
CORE_SOURCE = core.c
LIB_SOURCE = una_lib.c

# Объектные файлы
CORE_OBJ = core.o
LIB_OBJ = una_lib.o
CORE_SHARED_OBJ = core_shared.o
LIB_SHARED_OBJ = una_lib_shared.o

# Заголовочные файлы
HEADERS = core.h una_lib.h

# По умолчанию собираем все
all: console library

# ===== КОНСОЛЬНАЯ ВЕРСИЯ (интерфейс) =====
console: $(TARGET_CONSOLE)

$(TARGET_CONSOLE): $(UNIFIED_MAIN) $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(CORE_OBJ): $(CORE_SOURCE) $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# ===== БИБЛИОТЕКА (для Python) =====
library: shared_lib static_lib

shared_lib: $(TARGET_LIB)

static_lib: $(TARGET_STATIC_LIB)

# Динамическая библиотека
$(TARGET_LIB): $(LIB_SHARED_OBJ) $(CORE_SHARED_OBJ)
	$(CC) $(LDFLAGS_SHARED) -o $@ $^

# Статическая библиотека
$(TARGET_STATIC_LIB): $(LIB_SHARED_OBJ) $(CORE_SHARED_OBJ)
	ar rcs $@ $^

# Компиляция для библиотеки (с -fPIC)
$(LIB_SHARED_OBJ): $(LIB_SOURCE) $(HEADERS)
	$(CC) $(CFLAGS) $(CFLAGS_SHARED) -c $< -o $@

$(CORE_SHARED_OBJ): $(CORE_SOURCE) $(HEADERS)
	$(CC) $(CFLAGS) $(CFLAGS_SHARED) -c $< -o $@

# ===== УНИВЕРСАЛЬНЫЕ ЦЕЛИ =====

# Быстрая сборка
quick: $(UNIFIED_MAIN) $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $(TARGET_CONSOLE) $^ $(LDFLAGS)

# Запуск в обычном режиме
run: console
	./$(TARGET_CONSOLE)

# Запуск в режиме отладки
run_debug: console
	./$(TARGET_CONSOLE) -debug

# Исправление предупреждения в semantic_memory_binding
fix_warnings:
	@echo "Исправление предупреждения в semantic_memory_binding..."
	@sed -i 's/uint32_t now = (uint32_t)time(NULL);//g' core.c
	@sed -i 's/uint32_t now = (uint32_t)time(NULL);//g' una_lib.c 2>/dev/null || true
	@echo "Предупреждения устранены"

# Отладочная сборка с максимальной информацией
debug_build:
	$(CC) -std=c11 -Wall -O0 -g3 -DDEBUG_MODE=1 -fsanitize=address \
	-o $(TARGET_CONSOLE)_debug $(UNIFIED_MAIN) $(CORE_SOURCE) $(LDFLAGS)

# Профилировка
profile:
	$(CC) -std=c11 -Wall -O2 -pg \
	-o $(TARGET_CONSOLE)_profile $(UNIFIED_MAIN) $(CORE_SOURCE) $(LDFLAGS)

# Минимальная сборка (без отладочной информации)
release:
	$(CC) -std=c11 -Wall -O3 -DNDEBUG \
	-o $(TARGET_CONSOLE)_release $(UNIFIED_MAIN) $(CORE_SOURCE) $(LDFLAGS) -s
	$(CC) $(CFLAGS) $(CFLAGS_SHARED) -O3 -DNDEBUG \
	-c $(CORE_SOURCE) -o core_release.o
	$(CC) $(CFLAGS) $(CFLAGS_SHARED) -O3 -DNDEBUG \
	-c $(LIB_SOURCE) -o una_lib_release.o
	$(CC) $(LDFLAGS_SHARED) -O3 -o libuna_release.so una_lib_release.o core_release.o
	ar rcs libuna_release.a una_lib_release.o core_release.o

# Тест библиотеки Python
test_python: library
	@echo "Тестирование загрузки библиотеки..."
	@python3 -c "import ctypes; lib = ctypes.CDLL('./libuna.so'); print('✅ Библиотека libuna.so загружена успешно')" || \
	echo "❌ Ошибка загрузки библиотеки"

# Пример использования библиотеки
example: library
	@if [ -f "una_python.py" ]; then \
		python3 una_python.py; \
	else \
		echo "Файл una_python.py не найден"; \
		echo "Создайте его по примеру из документации"; \
	fi

# Установка библиотеки в систему (требует прав sudo)
install: library
	sudo cp $(TARGET_LIB) /usr/local/lib/
	sudo cp $(TARGET_STATIC_LIB) /usr/local/lib/
	sudo cp una_lib.h /usr/local/include/
	sudo cp core.h /usr/local/include/
	sudo ldconfig
	@echo "✅ Библиотека установлена в систему"

# Удаление из системы
uninstall:
	sudo rm -f /usr/local/lib/$(TARGET_LIB)
	sudo rm -f /usr/local/lib/$(TARGET_STATIC_LIB)
	sudo rm -f /usr/local/include/una_lib.h
	sudo rm -f /usr/local/include/core.h
	sudo ldconfig
	@echo "✅ Библиотека удалена из системы"

# Проверка исходников на ошибки
check:
	cppcheck --enable=all --suppress=missingIncludeSystem \
	$(UNIFIED_MAIN) $(CORE_SOURCE) $(LIB_SOURCE) $(HEADERS)

# Просмотр зависимостей
deps: library
	@echo "=== Зависимости консольной версии ==="
	@if [ -f "$(TARGET_CONSOLE)" ]; then \
		objdump -p $(TARGET_CONSOLE) | grep NEEDED || true; \
	else \
		echo "Консольная версия не собрана"; \
	fi
	@echo ""
	@echo "=== Зависимости библиотеки ==="
	@if [ -f "$(TARGET_LIB)" ]; then \
		objdump -p $(TARGET_LIB) | grep NEEDED || true; \
	else \
		echo "Библиотека не собрана"; \
	fi

# Размеры объектов
size:
	@echo "=== Размеры объектных файлов ==="
	@for obj in *.o; do \
		if [ -f "$$obj" ]; then \
			size "$$obj" | tail -1 | awk '{print "$$obj:", $$1+$$2, "байт";}'; \
		fi \
	done
	@echo ""
	@echo "=== Размеры исполняемых файлов ==="
	@for exe in $(TARGET_CONSOLE) $(TARGET_LIB) $(TARGET_STATIC_LIB); do \
		if [ -f "$$exe" ]; then \
			ls -lh "$$exe" | awk '{print $$9 ": " $$5}'; \
		fi \
	done

# Информация о системе
info:
	@echo "=== Информация о системе ==="
	@echo "Компилятор: $(CC)"
	@echo "Версия: $$($(CC) --version | head -1)"
	@echo "ОС: $$(uname -s -r)"
	@echo "Архитектура: $$(uname -m)"
	@echo "Python3: $$(python3 --version 2>/dev/null || echo 'не установлен')"
	@echo ""
	@echo "=== Сборка библиотеки ==="
	@echo "Для сборки библиотеки используйте: make library"
	@echo "Проблемы:"
	@echo "1. Ошибка 'recompile with -fPIC' - исправлена в этом Makefile"
	@echo "2. Предупреждение 'unused variable now' - можно игнорировать или исправить командой make fix_warnings"

# Создать простой тест Python
create_python_test:
	@echo "Создание простого теста Python..."
	@echo '#!/usr/bin/env python3' > test_lib.py
	@echo 'import ctypes' >> test_lib.py
	@echo '' >> test_lib.py
	@echo '# Загружаем библиотеку' >> test_lib.py
	@echo 'lib = ctypes.CDLL("./libuna.so")' >> test_lib.py
	@echo '' >> test_lib.py
	@echo '# Определяем функции' >> test_lib.py
	@echo 'lib.una_init.restype = ctypes.c_int' >> test_lib.py
	@echo 'lib.una_think.restype = ctypes.c_char_p' >> test_lib.py
	@echo '' >> test_lib.py
	@echo '# Тестируем' >> test_lib.py
	@echo 'print("Тестирование библиотеки UNA...")' >> test_lib.py
	@echo 'result = lib.una_init()' >> test_lib.py
	@echo 'print(f"Инициализация: {result}")' >> test_lib.py
	@echo '' >> test_lib.py
	@echo 'if result == 0:' >> test_lib.py
	@echo '    response = lib.una_think()' >> test_lib.py
	@echo '    if response:' >> test_lib.py
	@echo '        print(f"Первая мысль: {response.decode()}")' >> test_lib.py
	@echo '    else:' >> test_lib.py
	@echo '        print("Нет мыслей")' >> test_lib.py
	@echo 'else:' >> test_lib.py
	@echo '    print("Ошибка инициализации")' >> test_lib.py
	@chmod +x test_lib.py
	@echo "✅ Создан файл test_lib.py"

# Очистка
clean:
	rm -f *.o $(TARGET_CONSOLE) $(TARGET_CONSOLE)_debug \
	$(TARGET_CONSOLE)_profile $(TARGET_CONSOLE)_release \
	$(TARGET_LIB) $(TARGET_STATIC_LIB) *.so *.a memory.bin crash_dump.bin \
	*_shared.o

# Полная очистка (включая релизные объекты)
clean_all: clean
	rm -f *_release.o libuna_release.so libuna_release.a test_lib.py

# Помощь
help:
	@echo "=== UNA AGI СИСТЕМА ==="
	@echo ""
	@echo "=== СБОРКА КОНСОЛЬНОЙ ВЕРСИИ (интерфейс) ==="
	@echo "  make console        - сборка только консольной версии"
	@echo "  make run           - сборка и запуск консольной версии"
	@echo "  make run_debug     - запуск в режиме отладки"
	@echo ""
	@echo "=== СБОРКА БИБЛИОТЕКИ (для Python) ==="
	@echo "  make library       - сборка динамической и статической библиотек"
	@echo "  make shared_lib    - только динамическая библиотека (.so)"
	@echo "  make static_lib    - только статическая библиотека (.a)"
	@echo "  make test_python   - тест загрузки библиотеки в Python"
	@echo "  make create_python_test - создать простой тест Python"
	@echo "  make example       - запуск примера на Python"
	@echo "  make fix_warnings  - исправить предупреждения компиляции"
	@echo ""
	@echo "=== УСТАНОВКА И УДАЛЕНИЕ ==="
	@echo "  make install       - установка библиотеки в систему"
	@echo "  make uninstall     - удаление библиотеки из системы"
	@echo ""
	@echo "=== УТИЛИТЫ И ТЕСТИРОВАНИЕ ==="
	@echo "  make debug_build   - сборка с AddressSanitizer"
	@echo "  make profile       - сборка для профилировки"
	@echo "  make release       - оптимизированная сборка"
	@echo "  make check         - проверка исходников"
	@echo "  make deps          - просмотр зависимостей"
	@echo "  make size          - размеры файлов"
	@echo "  make info          - информация о системе"
	@echo ""
	@echo "=== ОЧИСТКА ==="
	@echo "  make clean         - удаление собранных файлов"
	@echo "  make clean_all     - полная очистка"
	@echo ""
	@echo "=== ИСПОЛЬЗОВАНИЕ ==="
	@echo "  Консольная версия: ./ai_system [-debug] [-autonomy N]"
	@echo "  Библиотека Python: import ctypes; lib = ctypes.CDLL('./libuna.so')"
	@echo ""
	@echo "=== ВНУТРИ ПРОГРАММЫ ==="
	@echo "  /debug    - переключение режимов отладки"
	@echo "  /think    - принудительное мышление"
	@echo "  /stats    - статистика системы"
	@echo "  /help     - список команд"

.PHONY: all console library shared_lib static_lib quick run run_debug fix_warnings \
	debug_build profile release test_python example install uninstall \
	check deps size info create_python_test clean clean_all help
