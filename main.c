// main.c
#include "interface_AI.h"
#include "core/kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MEMORY_FILE "memory.bin"

// Дополнительные функции для улучшенной инициализации
void add_initial_phrases(NeuralMemory* memory);
void add_semantic_clusters(NeuralMemory* memory);
void add_contextual_patterns(NeuralMemory* memory);

// Функция для инициализации начальной памяти с улучшенным содержанием
void init_memory(NeuralMemory* memory) {
    if (!memory) return;
    
    printf("[MAIN] Инициализация расширенной начальной памяти...\n");
    printf("[MAIN] Initializing enhanced starting memory...\n\n");
    
    // === 1. БАЗОВЫЕ СЛОВА (50+ штук) ===
    printf("[MAIN] Добавление базового словаря...\n");
    
    const char* basic_words[] = {
        // Приветствия
        "привет", "здравствуй", "добрый", "день", "вечер", "утро", "здорово", "приветствую",
        
        // Местоимения и основные слова
        "я", "ты", "мы", "вы", "он", "она", "оно", "они",
        "это", "то", "здесь", "там", "везде", "всегда", "иногда",
        
        // Вопросы
        "как", "что", "где", "когда", "почему", "зачем", "кто", "чей",
        
        // Действия
        "делать", "говорить", "думать", "знать", "понимать", "учить", "работать", "жить",
        "идти", "стоять", "сидеть", "лежать", "бежать", "смотреть", "слушать", "читать",
        
        // Состояния и качества
        "хороший", "плохой", "интересный", "сложный", "простой", "важный", "нужный",
        "красивый", "умный", "быстрый", "медленный", "сильный", "слабый",
        
        // Эмоции
        "радость", "грусть", "злость", "страх", "удивление", "любовь", "ненависть",
        
        // Время
        "сейчас", "потом", "раньше", "позже", "вчера", "сегодня", "завтра",
        
        // Пространство
        "дом", "улица", "город", "страна", "мир", "космос", "вселенная",
        
        // Технологии (для AGI)
        "искусственный", "интеллект", "нейросеть", "алгоритм", "программа", "код",
        "обучение", "память", "мышление", "сознание", "разум", "логика"
    };
    
    int basic_count = sizeof(basic_words) / sizeof(basic_words[0]);
    printf("[MAIN] Базовых слов: %d\n", basic_count);
    
    for (int i = 0; i < basic_count; i++) {
        char* path[] = {"init", "basic", "word"};
        
        FractalSpike* neuron = create_fractal_spike(
            time(NULL) - rand() % 1000, // Разное время для разнообразия
            0.6f + (float)rand() / RAND_MAX * 0.3f, // Более высокая начальная интенсивность
            basic_words[i], 
            1.2f + (float)rand() / RAND_MAX * 0.8f, // Более высокая фрактальная размерность
            path, 
            3
        );
        if (neuron) {
            add_neuron_to_memory(memory, neuron);
            destroy_fractal_spike(neuron);
        }
    }
    
    // === 2. ФРАЗЫ И ПРЕДЛОЖЕНИЯ (тексты) ===
    add_initial_phrases(memory);
    
    // === 3. СЕМАНТИЧЕСКИЕ КЛАСТЕРЫ ===
    add_semantic_clusters(memory);
    
    // === 4. КОНТЕКСТУАЛЬНЫЕ ПАТТЕРНЫ ===
    add_contextual_patterns(memory);
    
    printf("[MAIN] Всего добавлено элементов: %d\n", memory->count);
    printf("[MAIN] Total elements added: %d\n", memory->count);
}

// Добавление начальных фраз и текстов
void add_initial_phrases(NeuralMemory* memory) {
    printf("[MAIN] Добавление начальных фраз и текстов...\n");
    
    const char* phrases[] = {
        // Базовые фразы для общения
        "привет как дела",
        "здравствуйте меня зовут",
        "как тебя зовут",
        "что ты умеешь",
        "расскажи о себе",
        "где ты живешь",
        "кто тебя создал",
        "что такое искусственный интеллект",
        "как работает нейросеть",
        "что значит машинное обучение",
        
        // Вопросы и ответы
        "давай поговорим",
        "я хочу узнать больше",
        "объясни понятными словами",
        "приведи пример",
        "в чем разница",
        "как это связано",
        "почему это важно",
        "что будет дальше",
        
        // Контекстные фразы
        "я думаю что",
        "мне кажется что",
        "по моему мнению",
        "с одной стороны",
        "с другой стороны",
        "в результате получается",
        "итак мы видим что",
        
        // Философские и абстрактные
        "что такое сознание",
        "может ли машина мыслить",
        "в чем смысл жизни",
        "что такое реальность",
        "как работает память",
        "что такое творчество",
        "как рождаются идеи",
        
        // Технические
        "фрактальная сеть это",
        "резонансные паттерны в",
        "самоорганизующаяся система",
        "адаптивное обучение с",
        "иерархическая обработка информации",
        "распределенное представление знаний"
    };
    
    int phrase_count = sizeof(phrases) / sizeof(phrases[0]);
    
    for (int i = 0; i < phrase_count; i++) {
        // Разные типы путей для разных категорий фраз
        char* path[4];
        if (i < 10) {
            path[0] = "init"; path[1] = "phrase"; path[2] = "basic"; path[3] = "greeting";
        } else if (i < 18) {
            path[0] = "init"; path[1] = "phrase"; path[2] = "question"; path[3] = "dialogue";
        } else if (i < 25) {
            path[0] = "init"; path[1] = "phrase"; path[2] = "context"; path[3] = "abstract";
        } else {
            path[0] = "init"; path[1] = "phrase"; path[2] = "technical"; path[3] = "agi";
        }
        
        // Высокая фрактальная размерность для фраз
        float fractal_dim = 1.8f + (float)rand() / RAND_MAX * 0.4f;
        
        FractalSpike* neuron = create_fractal_spike(
            time(NULL) - rand() % 5000,
            0.7f + (float)rand() / RAND_MAX * 0.2f,
            phrases[i],
            fractal_dim,
            path,
            4
        );
        
        if (neuron) {
            add_neuron_to_memory(memory, neuron);
            destroy_fractal_spike(neuron);
        }
    }
    
    printf("[MAIN] Добавлено фраз: %d\n", phrase_count);
}

// Создание семантических кластеров
void add_semantic_clusters(NeuralMemory* memory) {
    printf("[MAIN] Создание семантических кластеров...\n");
    
    // Кластер: Технологии
    const char* tech_cluster[] = {
        "компьютер процессор память жесткий диск",
        "программа алгоритм функция библиотека",
        "сеть интернет протокол сервер клиент",
        "данные информация база запрос",
        "код синтаксис компилятор отладка"
    };
    
    // Кластер: Наука
    const char* science_cluster[] = {
        "физика математика химия биология",
        "эксперимент теория гипотеза доказательство",
        "исследование открытие изобретение патент",
        "университет лаборатория ученый профессор",
        "публикация конференция журнал статья"
    };
    
    // Кластер: Искусство
    const char* art_cluster[] = {
        "живопись музыка литература театр",
        "художник композитор писатель поэт",
        "картина симфония роман поэма",
        "творчество вдохновение мастерство стиль",
        "красота гармония эмоция выражение"
    };
    
    // Кластер: Философия
    const char* philosophy_cluster[] = {
        "бытие сознание познание истина",
        "этика мораль ценности принципы",
        "логика рассуждение аргумент вывод",
        "свобода воля выбор ответственность",
        "смысл цель предназначение судьба"
    };
    
    // Добавляем все кластеры
    struct {
        const char** cluster;
        int size;
        const char* name;
    } clusters[] = {
        {tech_cluster, 5, "technology"},
        {science_cluster, 5, "science"},
        {art_cluster, 5, "art"},
        {philosophy_cluster, 5, "philosophy"}
    };
    
    for (int c = 0; c < 4; c++) {
        for (int i = 0; i < clusters[c].size; i++) {
            char path[5][50];
            snprintf(path[0], 50, "init");
            snprintf(path[1], 50, "cluster");
            snprintf(path[2], 50, "%s", clusters[c].name);
            snprintf(path[3], 50, "semantic");
            snprintf(path[4], 50, "group%d", i);
            
            char* path_ptrs[5] = {path[0], path[1], path[2], path[3], path[4]};
            
            FractalSpike* neuron = create_fractal_spike(
                time(NULL) - rand() % 3000,
                0.65f + (float)rand() / RAND_MAX * 0.25f,
                clusters[c].cluster[i],
                2.0f + (float)rand() / RAND_MAX * 0.5f, // Высокая размерность для кластеров
                path_ptrs,
                5
            );
            
            if (neuron) {
                add_neuron_to_memory(memory, neuron);
                destroy_fractal_spike(neuron);
            }
        }
    }
    
    printf("[MAIN] Создано семантических кластеров: 4 (по 5 элементов в каждом)\n");
}

// Добавление контекстуальных паттернов
void add_contextual_patterns(NeuralMemory* memory) {
    printf("[MAIN] Добавление контекстуальных паттернов...\n");
    
    // Паттерны вопрос-ответ
    const char* qa_patterns[] = {
        "что такое|это означает",
        "как работает|функционирует система",
        "почему важно|значимо это",
        "где используется|применяется технология",
        "когда появилось|возникло понятие",
        "кто создал|разработал алгоритм",
        "зачем нужно|необходимо обучение",
        "сколько времени|занимает процесс",
        "можно ли|возможно ли создать",
        "чем отличается|разнится подход"
    };
    
    // Паттерны причинно-следственные
    const char* cause_effect[] = {
        "если то тогда",
        "поскольку следовательно поэтому",
        "из за того что в результате",
        "благодаря тому что благодаря этому",
        "несмотря на тем не менее однако"
    };
    
    // Паттерны сравнения
    const char* comparison[] = {
        "с одной стороны с другой стороны",
        "по сравнению с аналогично похоже",
        "в отличие от наоборот противоположно",
        "так же как и подобно аналогично",
        "лучше чем хуже чем эффективнее чем"
    };
    
    // Добавляем все паттерны
    int total_patterns = 0;
    
    for (int i = 0; i < sizeof(qa_patterns)/sizeof(qa_patterns[0]); i++) {
        char* path[] = {"init", "pattern", "qa", "contextual"};
        
        FractalSpike* neuron = create_fractal_spike(
            time(NULL) - rand() % 2000,
            0.75f, // Высокая интенсивность для паттернов
            qa_patterns[i],
            2.2f + (float)rand() / RAND_MAX * 0.3f, // Очень высокая фрактальная размерность
            path,
            4
        );
        
        if (neuron) {
            add_neuron_to_memory(memory, neuron);
            destroy_fractal_spike(neuron);
            total_patterns++;
        }
    }
    
    for (int i = 0; i < sizeof(cause_effect)/sizeof(cause_effect[0]); i++) {
        char* path[] = {"init", "pattern", "cause", "effect"};
        
        FractalSpike* neuron = create_fractal_spike(
            time(NULL) - rand() % 2000,
            0.72f,
            cause_effect[i],
            2.1f + (float)rand() / RAND_MAX * 0.4f,
            path,
            4
        );
        
        if (neuron) {
            add_neuron_to_memory(memory, neuron);
            destroy_fractal_spike(neuron);
            total_patterns++;
        }
    }
    
    for (int i = 0; i < sizeof(comparison)/sizeof(comparison[0]); i++) {
        char* path[] = {"init", "pattern", "comparison", "contrast"};
        
        FractalSpike* neuron = create_fractal_spike(
            time(NULL) - rand() % 2000,
            0.7f,
            comparison[i],
            2.0f + (float)rand() / RAND_MAX * 0.5f,
            path,
            4
        );
        
        if (neuron) {
            add_neuron_to_memory(memory, neuron);
            destroy_fractal_spike(neuron);
            total_patterns++;
        }
    }
    
    printf("[MAIN] Добавлено контекстуальных паттернов: %d\n", total_patterns);
}

int main() {
    srand(time(NULL));
    
    printf("==============================================================\n");
    printf("    NAIC - Фрактальный AGI Система (Улучшенная версия)\n");
    printf("    NAIC - Fractal AGI System (Enhanced Version)\n");
    printf("==============================================================\n");
    printf("Особенности этой версии:\n");
    printf("1. Расширенный начальный словарь (50+ базовых слов)\n");
    printf("2. Тексты и фразы для контекстного понимания\n");
    printf("3. Семантические кластеры по темам\n");
    printf("4. Контекстуальные паттерны вопрос-ответ\n");
    printf("5. Реальное сохранение памяти в: %s\n\n", MEMORY_FILE);
    
    printf("Features of this version:\n");
    printf("1. Extended initial vocabulary (50+ basic words)\n");
    printf("2. Texts and phrases for contextual understanding\n");
    printf("3. Semantic clusters by topics\n");
    printf("4. Contextual Q&A patterns\n");
    printf("5. Real memory saving to: %s\n\n", MEMORY_FILE);

    // Инициализация компонентов ядра
    printf("Загрузка памяти из файла...\n");
    printf("Loading memory from file...\n");
    
    NeuralMemory* memory = load_memory_from_file(MEMORY_FILE);
    FractalField* field = create_fractal_field(50, 200); // Больше начальных нейронов

    if (!memory) {
        printf("Ошибка: не удалось создать память.\n");
        printf("Error: Failed to create memory structure.\n");
        return 1;
    }
    if (!field) {
        printf("Ошибка: не удалось создать FractalField.\n");
        printf("Error: Failed to create FractalField.\n");
        destroy_neural_memory(memory);
        return 1;
    }

    printf("Память успешно загружена: %d нейронов\n", memory->count);
    printf("Memory successfully loaded: %d neurons\n", memory->count);
    printf("FractalField инициализирован: %d нейронов, %d связей\n", 
           field->neuron_count, field->connection_count);
    printf("FractalField initialized: %d neurons, %d connections\n\n", 
           field->neuron_count, field->connection_count);

    // Если память пуста или слишком мала, инициализируем
    if (memory->count < 30) {
        printf("[MAIN] Память содержит мало данных (< 30 нейронов)\n");
        printf("[MAIN] Запуск расширенной инициализации...\n");
        
        init_memory(memory);
        
        // Сразу сохраняем инициализированную память
        save_memory_to_file(memory, MEMORY_FILE);
        printf("[MAIN] Расширенная память сохранена в файл\n");
        printf("[MAIN] Enhanced memory saved to file\n");
    } else {
        printf("[MAIN] Используется существующая память\n");
        printf("[MAIN] Using existing memory\n");
    }

    printf("\n=== СТАТИСТИКА СИСТЕМЫ ПРИ ЗАПУСКЕ ===\n");
    printf("=== SYSTEM STATISTICS AT STARTUP ===\n");
    get_memory_stats(memory);
    printf("======================================\n\n");

    // Запуск основного интерфейса
    run_chat_interface(field, memory);
    
    // Завершение работы
    printf("\nЗавершение работы...\n");
    printf("Shutting down...\n");
    printf("Оптимизация памяти...\n");
    printf("Optimizing memory...\n");
    
    compact_memory(memory);
    optimize_memory_structure(memory);
    
    printf("\nФИНАЛЬНАЯ СТАТИСТИКА:\n");
    printf("FINAL STATISTICS:\n");
    printf("-----------------\n");
    get_memory_stats(memory);
    printf("FractalField: нейронов=%d, связей=%d, вознаграждение=%.3f\n",
           field->neuron_count, field->connection_count, field->global_reward_signal);
    printf("FractalField: neurons=%d, connections=%d, reward=%.3f\n",
           field->neuron_count, field->connection_count, field->global_reward_signal);
    
    printf("\nСохранение памяти в файл '%s'...\n", MEMORY_FILE);
    printf("Saving memory to file '%s'...\n", MEMORY_FILE);
    save_memory_to_file(memory, MEMORY_FILE);

    printf("\nОчистка ресурсов...\n");
    printf("Cleaning up resources...\n");
    destroy_neural_memory(memory);
    destroy_fractal_field(field);

    printf("\n==============================================================\n");
    printf("    NAIC завершил работу. Все данные сохранены.\n");
    printf("    NAIC finished work. All data saved.\n");
    printf("==============================================================\n");
    
    return 0;
}