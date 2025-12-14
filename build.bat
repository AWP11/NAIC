@echo off
echo Компиляция NAIC...

set CC=gcc
set CFLAGS=-std=c99 -Wall -O2
set EXE=naic.exe

echo 1. Очистка старых файлов...
del *.exe 2>nul
del *.o 2>nul

echo 2. Компиляция основных файлов...
%CC% %CFLAGS% -c main.c -o main.o
%CC% %CFLAGS% -c kernel.c -o kernel.o

echo 3. Линковка...
%CC% %CFLAGS% -o %EXE% main.o kernel.o -lm

echo 4. Проверка...
if exist %EXE% (
    echo Успешно! Создан %EXE%
    dir %EXE%
) else (
    echo Ошибка компиляции!
    pause
)

pause