// main.c
#include "interface_AI.h"
#include "core/kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MEMORY_FILE "memory.bin"

// Функция для инициализации начальной памяти
void init_memory(NeuralMemory* memory) {
    if (!memory) return;
    
    printf("[MAIN] Инициализация начальной памяти...\n");
    const char* initial_words[] = {
        "привет", "здравствуй", "как", "дела", "что", 
        "ты", "умеешь", "расскажи", "пока", "спасибо",
        "да", "нет", "хорошо", "плохо", "интересно",
        "мне", "тебе", "говорить", "общаться", "учиться" 
    };
    int num_initial_words = sizeof(initial_words) / sizeof(initial_words[0]);
    
    for (int i = 0; i < num_initial_words; i++) {
        char* path[] = {"initial", "word"};
        FractalSpike* neuron = create_fractal_spike(
            time(NULL), 
            0.5f + (float)rand() / RAND_MAX * 0.5f,
            initial_words[i], 
            0.3f + (float)rand() / RAND_MAX * 0.6f,
            path, 
            2
        );
        if (neuron) {
            add_neuron_to_memory(memory, neuron);
            destroy_fractal_spike(neuron);
        }
    }
    printf("[MAIN] Добавлено %d начальных слов в память\n", num_initial_words);
}

int main() {
    srand(time(NULL));
    printf("NAIC - Фрактальный AGI - Нейросимволическая генерация + R-STDP обучение\n");
    printf("==================================================================\n");
    printf("Версия с реальным сохранением памяти в файл: %s\n", MEMORY_FILE);
    printf("==================================================================\n");

    // Инициализация компонентов ядра
    printf("Загрузка памяти из файла...\n");
    NeuralMemory* memory = load_memory_from_file(MEMORY_FILE);
    FractalField* field = create_fractal_field(10, 20);

    if (!memory) {
        printf("Ошибка: не удалось создать память.\n");
        return 1;
    }
    if (!field) {
        printf("Ошибка: не удалось создать FractalField.\n");
        destroy_neural_memory(memory);
        return 1;
    }

    printf("Память успешно загружена: %d нейронов\n", memory->count);
    printf("FractalField инициализирован: %d нейронов, %d связей\n", 
           field->neuron_count, field->connection_count);

    // Инициализация памяти если она пустая
    if (memory->count == 0) {
        init_memory(memory);
        // Сразу сохраняем инициализированную память
        save_memory_to_file(memory, MEMORY_FILE);
        printf("[MAIN] Начальная память сохранена в файл\n");
    }

    // Статистика памяти при запуске
    printf("\n=== Статистика системы при запуске ===\n");
    get_memory_stats(memory);
    printf("======================================\n\n");

    // Запуск интерфейса
    run_chat_interface(field, memory);
    
    // Оптимизация и сохранение перед завершением
    printf("\nЗавершение работы...\n");
    printf("Оптимизация памяти...\n");
    
    compact_memory(memory);
    optimize_memory_structure(memory);
    
    printf("Финальная статистика памяти:\n");
    get_memory_stats(memory);
    
    printf("Сохранение памяти в файл '%s'...\n", MEMORY_FILE);
    save_memory_to_file(memory, MEMORY_FILE);

    // Очистка памяти
    printf("Очистка ресурсов...\n");
    destroy_neural_memory(memory);
    destroy_fractal_field(field);

    printf("NAIC завершил работу. Все данные сохранены.\n");
    printf("==================================================================\n");

    return 0;
}