// interface_AI.h
#ifndef INTERFACE_AI_H
#define INTERFACE_AI_H

#include "kernel.h" // Включаем kernel.h, чтобы знать о типах FractalField, NeuralMemory и т.д.

// === Объявления структур ===

// Прямое объявление структуры FractalGrid (чтобы избежать циклических зависимостей)
typedef struct FractalGrid FractalGrid;

// === Объявления функций для интерфейса пользователя ===

// Основная функция запуска чата
void run_chat_interface(FractalField* field, NeuralMemory* memory);

// Вспомогательные функции интерфейса
char* get_user_input(void);
void display_bot_response(const char* response);
void print_status(FractalField* field, NeuralMemory* memory, int message_count);

// === Объявления функций для работы с фрактальной сеткой ===

// Создание и уничтожение фрактальной сетки
FractalGrid* fractal_grid_create(void);
void fractal_grid_destroy(FractalGrid* grid);

// Обучение и генерация ответов через фрактальную сетку
void fractal_grid_learn(FractalGrid* grid, const char* text, float weight);
char* fractal_grid_generate_response(FractalGrid* grid, const char* seed_text, int max_length);

// Управление резонансом
void activate_resonance(FractalGrid* grid, const char* pattern, float strength);

// Утилиты для текста
float calculate_text_complexity(const char* text);
float calculate_dialogue_quality(const char* input, const char* response, void* unused);

// Основная функция генерации ответов с использованием фрактальной сетки
char* generate_response_with_fractal_grid(FractalGrid* grid, NeuralMemory* memory, 
                                        const char* input, FractalField* field);

// === Объявления функций для сохранения/загрузки фрактальной сетки ===
void save_fractal_grid(FractalGrid* grid, const char* filename);
FractalGrid* load_fractal_grid(const char* filename);

#endif // INTERFACE_AI_H