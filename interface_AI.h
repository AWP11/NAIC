// interface_AI.h
#ifndef INTERFACE_AI_H
#define INTERFACE_AI_H

#include "core/kernel.h" // Включаем kernel.h, чтобы знать о типах FractalField, NeuralMemory и т.д.

// === Объявления функций ===

// Основная функция запуска чата
void run_chat_interface(FractalField* field, NeuralMemory* memory);

// Вспомогательные функции интерфейса (если нужно использовать в main.c или других местах)
char* get_user_input(void); // Убираем зависимости от структур DialogueHistory в интерфейсе
void display_bot_response(const char* response);
void print_status(FractalField* field, NeuralMemory* memory, int message_count);

#endif // INTERFACE_AI_H