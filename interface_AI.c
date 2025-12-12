// interface_AI.c
#include "interface_AI.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "fractal_tensor.h"

#define MAX_INPUT_LENGTH 10000
#define MAX_LINES 100
#define MAX_WORDS 3000
#define MAX_WORD_LENGTH 100
#define MAX_PATTERN_SCALE 3


void save_fractal_grid(FractalGrid* grid, const char* filename);
FractalGrid* load_fractal_grid(const char* filename);
// ==================== ФРАКТАЛЬНАЯ СЕТКА ====================

typedef struct FractalNode {
    char* pattern;                     // Паттерн (например, "hello_world")
    float** connections;               // Матрица связей [слово][вес/позиция/активация]
    int connection_count;              // Количество связей
    float resonance;                   // Уровень резонанса (0..1)
    float fractal_dimension;           // Фрактальная размерность
    char** connected_words;            // Массив связанных слов
} FractalNode;

typedef struct ResonancePattern {
    char* pattern_hash;                // Хэш паттерна
    char** neurons_to_activate;        // Нейроны для активации
    int neuron_count;                  // Количество нейронов
    float resonance_strength;          // Сила резонанса
    float scale;                       // Масштаб паттерна
} ResonancePattern;

typedef struct FractalGrid {
    FractalNode** nodes;               // Фрактальные узлы
    int node_count;                    // Количество узлов
    int node_capacity;                 // Емкость массива
    
    ResonancePattern** patterns;       // Резонансные паттерны
    int pattern_count;                 // Количество паттернов
    int pattern_capacity;              // Емкость паттернов
} FractalGrid;

// Вспомогательная структура для хранения веса/позиции/активации
typedef struct {
    float weight;
    int position;
    int activation;
} ConnectionData;

// Создание фрактальной сетки
FractalGrid* fractal_grid_create() {
    FractalGrid* grid = (FractalGrid*)malloc(sizeof(FractalGrid));
    if (!grid) return NULL;
    
    grid->node_capacity = 100;
    grid->nodes = (FractalNode**)malloc(grid->node_capacity * sizeof(FractalNode*));
    grid->node_count = 0;
    
    grid->pattern_capacity = 50;
    grid->patterns = (ResonancePattern**)malloc(grid->pattern_capacity * sizeof(ResonancePattern*));
    grid->pattern_count = 0;
    
    if (!grid->nodes || !grid->patterns) {
        free(grid->nodes);
        free(grid->patterns);
        free(grid);
        return NULL;
    }
    
    return grid;
}

// Уничтожение фрактальной сетки
void fractal_grid_destroy(FractalGrid* grid) {
    if (!grid) return;
    
    for (int i = 0; i < grid->node_count; i++) {
        FractalNode* node = grid->nodes[i];
        if (node) {
            free(node->pattern);
            if (node->connections) {
                for (int j = 0; j < node->connection_count; j++) {
                    free(node->connections[j]);
                }
                free(node->connections);
            }
            if (node->connected_words) {
                for (int j = 0; j < node->connection_count; j++) {
                    free(node->connected_words[j]);
                }
                free(node->connected_words);
            }
            free(node);
        }
    }
    free(grid->nodes);
    
    for (int i = 0; i < grid->pattern_count; i++) {
        ResonancePattern* pattern = grid->patterns[i];
        if (pattern) {
            free(pattern->pattern_hash);
            for (int j = 0; j < pattern->neuron_count; j++) {
                free(pattern->neurons_to_activate[j]);
            }
            free(pattern->neurons_to_activate);
            free(pattern);
        }
    }
    free(grid->patterns);
    
    free(grid);
}

// Расчет фрактальной размерности для слова
float calculate_fractal_dimension(const char* word) {
    if (!word || strlen(word) == 0) return 1.1f;
    
    int length = strlen(word);
    int vowels = 0;
    for (int i = 0; i < length; i++) {
        char c = tolower(word[i]);
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
            vowels++;
        }
    }
    
    // Формула: 1.1 + длина*0.03 + (гласные/длина)*0.2
    float dimension = 1.1f + (length * 0.03f);
    if (length > 0) {
        dimension += ((float)vowels / length) * 0.2f;
    }
    
    return dimension;
}

// Создание фрактального узла
FractalNode* create_fractal_node(const char* pattern) {
    FractalNode* node = (FractalNode*)malloc(sizeof(FractalNode));
    if (!node) return NULL;
    
    node->pattern = strdup(pattern);
    node->connection_count = 0;
    node->connections = NULL;
    node->connected_words = NULL;
    node->resonance = 0.0f;
    node->fractal_dimension = calculate_fractal_dimension(pattern);
    
    return node;
}

// Генерация фрактального хэша
char* generate_fractal_hash(const char* pattern) {
    // Простой хэш для примера
    unsigned long hash = 5381;
    int c;
    const char* p = pattern;
    
    while ((c = *p++)) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    
    char* result = (char*)malloc(20);
    if (result) {
        snprintf(result, 20, "%lx", hash);
    }
    return result;
}

// Поиск или создание узла
FractalNode* get_or_create_node(FractalGrid* grid, const char* pattern) {
    for (int i = 0; i < grid->node_count; i++) {
        if (strcmp(grid->nodes[i]->pattern, pattern) == 0) {
            return grid->nodes[i];
        }
    }
    
    if (grid->node_count >= grid->node_capacity) {
        grid->node_capacity *= 2;
        FractalNode** temp = (FractalNode**)realloc(grid->nodes, grid->node_capacity * sizeof(FractalNode*));
        if (!temp) return NULL;
        grid->nodes = temp;
    }
    
    FractalNode* node = create_fractal_node(pattern);
    if (!node) return NULL;
    
    grid->nodes[grid->node_count++] = node;
    return node;
}

// Обучение на тексте
void fractal_grid_learn(FractalGrid* grid, const char* text, float weight) {
    if (!grid || !text) return;
    
    char* words[MAX_WORDS];
    int word_count = 0;
    
    // Разделяем текст на слова
    char* copy = strdup(text);
    char* token = strtok(copy, " ,.!?;:\t\n\r");
    while (token && word_count < MAX_WORDS) {
        if (strlen(token) > 1) {
            words[word_count++] = strdup(token);
        }
        token = strtok(NULL, " ,.!?;:\t\n\r");
    }
    free(copy);
    
    if (word_count < 2) {
        for (int i = 0; i < word_count; i++) free(words[i]);
        return;
    }
    
    // Обучение на разных масштабах (1, 2, 3 слова)
    for (int scale = 1; scale <= MAX_PATTERN_SCALE; scale++) {
        for (int i = 0; i < word_count - scale; i++) {
            // Создаем паттерн из scale слов
            char pattern[500] = "";
            for (int j = 0; j < scale; j++) {
                strcat(pattern, words[i + j]);
                if (j < scale - 1) strcat(pattern, "_");
            }
            
            char* next_word = words[i + scale];
            FractalNode* node = get_or_create_node(grid, pattern);
            if (!node) continue;
            
            // Добавляем связь
            int found = -1;
            for (int j = 0; j < node->connection_count; j++) {
                if (strcmp(node->connected_words[j], next_word) == 0) {
                    found = j;
                    break;
                }
            }
            
            if (found >= 0) {
                // Обновляем существующую связь: увеличиваем вес
                node->connections[found][0] += weight * (1.0f / scale) * node->fractal_dimension;
                node->connections[found][1] = (node->connections[found][1] * node->connections[found][2] + i) / (node->connections[found][2] + 1);
                node->connections[found][2] += 1;
            } else {
                // Добавляем новую связь
                node->connection_count++;
                float** new_conns = (float**)realloc(node->connections, node->connection_count * sizeof(float*));
                char** new_words = (char**)realloc(node->connected_words, node->connection_count * sizeof(char*));
                
                if (!new_conns || !new_words) {
                    node->connection_count--;
                    continue;
                }
                
                node->connections = new_conns;
                node->connected_words = new_words;
                
                node->connections[node->connection_count - 1] = (float*)malloc(3 * sizeof(float));
                if (!node->connections[node->connection_count - 1]) {
                    node->connection_count--;
                    continue;
                }
                
                // Инициализация: вес/позиция/активация
                node->connections[node->connection_count - 1][0] = weight * (1.0f / scale) * node->fractal_dimension; // вес
                node->connections[node->connection_count - 1][1] = i; // позиция
                node->connections[node->connection_count - 1][2] = 1; // активация
                
                node->connected_words[node->connection_count - 1] = strdup(next_word);
            }
            
            // Создаем резонансный паттерн
            char* hash = generate_fractal_hash(pattern);
            if (hash) {
                // Проверяем, существует ли паттерн
                int pattern_exists = 0;
                for (int p = 0; p < grid->pattern_count; p++) {
                    if (strcmp(grid->patterns[p]->pattern_hash, hash) == 0) {
                        pattern_exists = 1;
                        break;
                    }
                }
                
                if (!pattern_exists) {
                    if (grid->pattern_count >= grid->pattern_capacity) {
                        grid->pattern_capacity *= 2;
                        ResonancePattern** temp = (ResonancePattern**)realloc(grid->patterns, grid->pattern_capacity * sizeof(ResonancePattern*));
                        if (!temp) {
                            free(hash);
                            continue;
                        }
                        grid->patterns = temp;
                    }
                    
                    ResonancePattern* pattern = (ResonancePattern*)malloc(sizeof(ResonancePattern));
                    if (pattern) {
                        pattern->pattern_hash = hash;
                        pattern->neurons_to_activate = (char**)malloc(10 * sizeof(char*));
                        pattern->neuron_count = 0;
                        pattern->resonance_strength = 0.7f + ((float)rand() / RAND_MAX) * 0.3f;
                        pattern->scale = scale;
                        
                        // Добавляем 3 наиболее семантически близких узла
                        for (int n = 0; n < grid->node_count && pattern->neuron_count < 3; n++) {
                            if (strstr(grid->nodes[n]->pattern, words[i]) || 
                                strstr(grid->nodes[n]->pattern, words[i+scale-1])) {
                                pattern->neurons_to_activate[pattern->neuron_count++] = strdup(grid->nodes[n]->pattern);
                            }
                        }
                        
                        grid->patterns[grid->pattern_count++] = pattern;
                    } else {
                        free(hash);
                    }
                } else {
                    free(hash);
                }
            }
        }
    }
    
    // Очистка
    for (int i = 0; i < word_count; i++) free(words[i]);
}

// Активация резонанса
void activate_resonance(FractalGrid* grid, const char* pattern, float strength) {
    if (!grid || strength < 0.01f) return;
    
    char* hash = generate_fractal_hash(pattern);
    if (!hash) return;
    
    // Находим паттерн
    for (int i = 0; i < grid->pattern_count; i++) {
        if (strcmp(grid->patterns[i]->pattern_hash, hash) == 0) {
            ResonancePattern* res_pattern = grid->patterns[i];
            
            // Активируем нейроны (ограничиваем количество)
            int limit = res_pattern->neuron_count < 50 ? res_pattern->neuron_count : 50;
            for (int j = 0; j < limit; j++) {
                // Находим узел
                for (int k = 0; k < grid->node_count; k++) {
                    if (strcmp(grid->nodes[k]->pattern, res_pattern->neurons_to_activate[j]) == 0) {
                        grid->nodes[k]->resonance = fminf(1.0f, grid->nodes[k]->resonance + 
                            res_pattern->resonance_strength * strength * res_pattern->scale);
                        break;
                    }
                }
            }
            break;
        }
    }
    
    free(hash);
    
    // Затухание резонанса (с 30% вероятностью)
    if (strength > 0.1f && (rand() % 100) < 30) {
        for (int i = 0; i < grid->node_count; i++) {
            if (grid->nodes[i]->resonance > 0.01f) {
                grid->nodes[i]->resonance *= 0.9f;
            }
        }
    }
}

// Получение резонансных опций для выбора следующего слова
char* get_resonant_next_word(FractalGrid* grid, const char* current, const char* context, 
                            char** history, int history_count) {
    if (!grid || !current) return NULL;
    
    // Собираем опции на разных масштабах
    float total_weight = 0.0f;
    char* options[100];
    float weights[100];
    int option_count = 0;
    
    for (int scale = 1; scale <= MAX_PATTERN_SCALE; scale++) {
        const char* pattern_to_search = NULL;
        
        switch(scale) {
            case 1: pattern_to_search = current; break;
            case 2: pattern_to_search = context; break;
            case 3: 
                if (history_count >= 3) {
                    char pattern3[500] = "";
                    for (int i = history_count - 3; i < history_count; i++) {
                        strcat(pattern3, history[i]);
                        if (i < history_count - 1) strcat(pattern3, "_");
                    }
                    pattern_to_search = pattern3;
                }
                break;
        }
        
        if (!pattern_to_search) continue;
        
        // Ищем узел
        for (int i = 0; i < grid->node_count; i++) {
            if (strcmp(grid->nodes[i]->pattern, pattern_to_search) == 0) {
                FractalNode* node = grid->nodes[i];
                
                // Добавляем связи как опции
                for (int j = 0; j < node->connection_count && option_count < 100; j++) {
                    // Проверяем, нет ли уже этого слова в истории
                    int in_history = 0;
                    for (int k = 0; k < history_count && k < 5; k++) {
                        if (history[k] && strcmp(node->connected_words[j], history[k]) == 0) {
                            in_history = 1;
                            break;
                        }
                    }
                    
                    if (!in_history) {
                        // Усиливаем вес через резонанс
                        float resonance_boost = 0.3f + 0.7f * node->resonance;
                        float fractal_weight = node->connections[j][0] * resonance_boost * node->fractal_dimension;
                        
                        options[option_count] = node->connected_words[j];
                        weights[option_count] = fractal_weight;
                        total_weight += fractal_weight;
                        option_count++;
                    }
                }
                break;
            }
        }
    }
    
    if (option_count == 0 || total_weight <= 0) return NULL;
    
    // Фрактальный случайный выбор
    float random_val = ((float)rand() / RAND_MAX) * total_weight;
    float cumulative = 0.0f;
    
    for (int i = 0; i < option_count; i++) {
        cumulative += weights[i];
        if (random_val <= cumulative) {
            return strdup(options[i]);
        }
    }
    
    return strdup(options[0]);
}

// Генерация ответа с использованием фрактальной сетки
char* fractal_grid_generate_response(FractalGrid* grid, const char* seed_text, int max_length) {
    if (!grid || !seed_text || grid->node_count == 0) return strdup("...");
    
    // Разделяем seed на слова
    char* seed_words[MAX_WORDS];
    int seed_count = 0;
    
    char* copy = strdup(seed_text);
    char* token = strtok(copy, " ,.!?;:\t\n\r");
    while (token && seed_count < MAX_WORDS) {
        seed_words[seed_count++] = strdup(token);
        token = strtok(NULL, " ,.!?;:\t\n\r");
    }
    free(copy);
    
    if (seed_count == 0) {
        return strdup("...");
    }
    
    char* result_words[100];
    int result_count = 0;
    
    // Начальное слово
    char* current_word = strdup(seed_words[seed_count - 1]);
    result_words[result_count++] = strdup(current_word);
    
    // Начальный контекст
    char context[500] = "";
    if (seed_count >= 2) {
        snprintf(context, sizeof(context), "%s_%s", seed_words[seed_count - 2], seed_words[seed_count - 1]);
    } else {
        strcpy(context, current_word);
    }
    
    // Активируем начальный резонанс
    activate_resonance(grid, context, 1.0f);
    
    // Генерация ответа
    int attempts = 0;
    while (result_count < max_length && attempts < max_length * 2) {
        attempts++;
        
        // Получаем следующее слово через резонанс
        char* next_word = get_resonant_next_word(grid, current_word, context, 
                                                result_words, result_count);
        if (!next_word || strcmp(next_word, current_word) == 0) {
            if (next_word) free(next_word);
            break;
        }
        
        // Проверяем, нет ли повторов
        int repeated = 0;
        for (int i = 0; i < result_count; i++) {
            if (strcmp(result_words[i], next_word) == 0) {
                repeated = 1;
                break;
            }
        }
        
        if (repeated) {
            free(next_word);
            break;
        }
        
        result_words[result_count++] = next_word;
        
        // Обновляем контекст
        free(current_word);
        current_word = strdup(next_word);
        
        if (result_count >= 2) {
            snprintf(context, sizeof(context), "%s_%s", 
                    result_words[result_count - 2], result_words[result_count - 1]);
        } else {
            strcpy(context, current_word);
        }
        
        // Активируем резонанс для нового контекста
        activate_resonance(grid, context, 0.8f);
    }
    
    // Формируем финальный ответ
    char* response = (char*)malloc(MAX_INPUT_LENGTH);
    if (!response) {
        for (int i = 0; i < seed_count; i++) free(seed_words[i]);
        for (int i = 0; i < result_count; i++) free(result_words[i]);
        free(current_word);
        return strdup("...");
    }
    
    response[0] = '\0';
    for (int i = 0; i < result_count; i++) {
        if (i > 0) strcat(response, " ");
        strcat(response, result_words[i]);
        free(result_words[i]);
    }
    
    // Добавляем окончание
    if (result_count >= max_length) {
        strcat(response, "...");
    } else if ((rand() % 100) < 40) {
        strcat(response, "!");
    } else if ((rand() % 100) < 30) {
        strcat(response, "?");
    } else {
        strcat(response, ".");
    }
    
    // Очистка
    for (int i = 0; i < seed_count; i++) free(seed_words[i]);
    free(current_word);
    
    return response;
}

// ==================== ИНТЕРФЕЙС АИ ====================

// Структура для хранения диалога
typedef struct {
    char** lines;
    int count;
    int capacity;
} DialogueHistory;

// Вспомогательные функции
DialogueHistory* create_dialogue_history() {
    DialogueHistory* history = (DialogueHistory*)malloc(sizeof(DialogueHistory));
    if (!history) return NULL;
    
    history->capacity = 50;
    history->count = 0;
    history->lines = (char**)malloc(history->capacity * sizeof(char*));
    
    if (!history->lines) {
        free(history);
        return NULL;
    }
    
    return history;
}

void destroy_dialogue_history(DialogueHistory* history) {
    if (!history) return;
    
    for (int i = 0; i < history->count; i++) {
        free(history->lines[i]);
    }
    free(history->lines);
    free(history);
}

void add_to_dialogue(DialogueHistory* history, const char* line) {
    if (!history || !line) return;
    
    if (history->count >= history->capacity) {
        history->capacity *= 2;
        char** temp = (char**)realloc(history->lines, history->capacity * sizeof(char*));
        if (!temp) return;
        history->lines = temp;
    }
    
    history->lines[history->count] = strdup(line);
    if (history->lines[history->count]) {
        history->count++;
    }
}

// Основная функция получения ответа
char* generate_response_with_fractal_grid(FractalGrid* grid, NeuralMemory* memory, 
                                        const char* input, FractalField* field) {
    if (!grid || !input) return strdup("...");
    
    // Обучение на входных данных с весом 0.5
    fractal_grid_learn(grid, input, 0.5f);
    
    // Сохранение в память
    FractalSpike* input_spike = create_fractal_spike(
        time(NULL), 0.8f, input, calculate_text_complexity(input), 
        (char*[]){"input", "fractal_grid"}, 2
    );
    
    if (input_spike && memory) {
        add_neuron_to_memory(memory, input_spike);
        destroy_fractal_spike(input_spike);
    }
    
    // Генерация ответа через фрактальную сетку
    char* response = fractal_grid_generate_response(grid, input, 15);
    
    // Сохранение ответа в память
    if (response && memory && strcmp(response, "...") != 0) {
        FractalSpike* response_spike = create_fractal_spike(
            time(NULL), 0.7f, response, calculate_text_complexity(response),
            (char*[]){"output", "fractal_grid"}, 2
        );
        
        if (response_spike) {
            add_neuron_to_memory(memory, response_spike);
            destroy_fractal_spike(response_spike);
        }
        
        // Обучение на ответе с весом 0.3
        fractal_grid_learn(grid, response, 0.3f);
    }
    
    // Обновление FractalField
    if (field) {
        float quality = calculate_dialogue_quality(input, response, NULL);
        field->global_reward_signal = quality;
        propagate_fractal_field(field, quality);
        update_fractal_field(field);
    }
    
    return response;
}

// Функция для расчета качества диалога (упрощенная)
float calculate_dialogue_quality(const char* input, const char* response, void* unused) {
    if (!input || !response) return 0.0f;
    
    float quality = 0.0f;
    
    if (strcmp(response, "...") == 0 || strlen(response) < 3) {
        quality -= 0.5f;
    } else {
        quality += 0.2f;
    }
    
    // Проверяем, есть ли слова из запроса в ответе
    char* input_copy = strdup(input);
    char* response_copy = strdup(response);
    
    char* input_words[100];
    int input_count = 0;
    
    char* token = strtok(input_copy, " ,.!?;:\t\n\r");
    while (token && input_count < 100) {
        input_words[input_count++] = token;
        token = strtok(NULL, " ,.!?;:\t\n\r");
    }
    
    int matches = 0;
    for (int i = 0; i < input_count; i++) {
        if (strstr(response_copy, input_words[i]) != NULL) {
            matches++;
        }
    }
    
    if (input_count > 0) {
        quality += (float)matches / input_count * 0.5f;
    }
    
    free(input_copy);
    free(response_copy);
    
    return fmaxf(-1.0f, fminf(1.0f, quality));
}

// Основной цикл чата
void run_chat_interface(FractalField* field, NeuralMemory* memory) {
    if (!field || !memory) {
        printf("Ошибка: FractalField или NeuralMemory не инициализированы.\n");
        return;
    }
    
    printf("=== NAIC - Фрактальный AI Чат с Fractal Grid ===\n");
    printf("Система использует фрактальную сетку с резонансными связями\n");
    printf("Формат связей: вес/позиция/активация\n");
    printf("Вводите сообщения построчно. Для завершения ввода нажмите Enter на пустой строке\n");
    printf("Для выхода из чата введите пустую первую строку\n\n");
    
    // Создаем фрактальную сетку
    FractalGrid* grid = fractal_grid_create();
    if (!grid) {
        printf("Ошибка создания фрактальной сетки\n");
        return;
    }
    
    DialogueHistory* dialogue = create_dialogue_history();
    int message_count = 0;
    
    while (1) {
        printf("Вы: ");
        fflush(stdout);
        
        // Сбор многострочного ввода
        char* lines[MAX_LINES];
        int line_count = 0;
        char buffer[MAX_INPUT_LENGTH];
        
        // Читаем первую строку
        if (fgets(buffer, MAX_INPUT_LENGTH, stdin) == NULL) {
            break;
        }
        
        // Убираем символ новой строки
        buffer[strcspn(buffer, "\n")] = 0;
        
        // Если первая строка пустая - выход из чата
        if (strlen(buffer) == 0) {
            break;
        }
        
        lines[line_count++] = strdup(buffer);
        
        // Читаем дополнительные строки
        while (line_count < MAX_LINES) {
            printf("> ");  // Индикатор продолжения ввода
            fflush(stdout);
            
            if (fgets(buffer, MAX_INPUT_LENGTH, stdin) == NULL) {
                break;
            }
            
            buffer[strcspn(buffer, "\n")] = 0;
            
            // Пустая строка означает конец многострочного ввода
            if (strlen(buffer) == 0) {
                break;
            }
            
            lines[line_count++] = strdup(buffer);
        }
        
        // Объединяем все строки в одно сообщение
        int total_length = 0;
        for (int i = 0; i < line_count; i++) {
            if (lines[i]) total_length += strlen(lines[i]) + 1;
        }
        
        char* input = (char*)malloc(total_length + 1);
        if (!input) {
            // Очистка в случае ошибки
            for (int i = 0; i < line_count; i++) free(lines[i]);
            break;
        }
        
        input[0] = '\0';
        for (int i = 0; i < line_count; i++) {
            if (lines[i]) {
                if (i > 0) strcat(input, " ");
                strcat(input, lines[i]);
                free(lines[i]);
            }
        }
        
        // Проверяем, что есть хоть какой-то ввод
        if (strlen(input) == 0) {
            free(input);
            continue;
        }
        
        // Добавляем в историю диалога
        add_to_dialogue(dialogue, input);
        
        // Генерация ответа через фрактальную сетку
        char* response = generate_response_with_fractal_grid(grid, memory, input, field);
        
        // Вывод ответа
        printf("NAIC: %s\n\n", response);
        
        // Добавляем ответ в историю
        add_to_dialogue(dialogue, response);
        
        // Очистка
        free(response);
        free(input);
        
        message_count++;
        
        // Периодическое сохранение
        if (message_count % 5 == 0) {
            printf("[NAIC] Обработано %d сообщений\n", message_count);
            printf("[NAIC] FractalGrid: %d узлов, %d паттернов\n", 
                   grid->node_count, grid->pattern_count);
            printf("[NAIC] Memory: %d нейронов\n", memory->count);
            printf("[NAIC] FractalField: %d нейронов, %d связей, RWD=%.3f\n",
                   field->neuron_count, field->connection_count, field->global_reward_signal);
        }
        
        // Периодическая оптимизация
        if (message_count % 10 == 0) {
            optimize_memory_structure(memory);
            
            // Стимулируем рост FractalField при хорошем диалоге
            if (field->global_reward_signal > 0.3f && field->neuron_count < field->max_neurons * 0.7f) {
                check_growth_conditions(field);
            }
        }
    }
    
    // Очистка
    fractal_grid_destroy(grid);
    destroy_dialogue_history(dialogue);
    
    printf("\nЧат завершен. Всего сообщений: %d\n", message_count);
    printf("Финальная статистика:\n");
    printf("- FractalGrid: %d узлов, %d паттернов\n", grid ? grid->node_count : 0, grid ? grid->pattern_count : 0);
    printf("- Memory: %d нейронов\n", memory->count);
    printf("- FractalField: %d нейронов, %d связей\n", field->neuron_count, field->connection_count);
}

// Вспомогательная функция для расчета сложности текста
float calculate_text_complexity(const char* text) {
    if (!text) return 0.0f;
    
    int length = strlen(text);
    if (length == 0) return 0.0f;
    
    // Простая мера сложности
    int unique_chars = 0;
    int char_counts[256] = {0};
    
    for (int i = 0; i < length; i++) {
        unsigned char c = text[i];
        if (char_counts[c] == 0) unique_chars++;
        char_counts[c]++;
    }
    
    float complexity = (float)unique_chars / length;
    float length_factor = fminf(1.0f, (float)length / 100.0f);
    
    return 0.3f + complexity * 0.5f + length_factor * 0.2f;
}

void save_fractal_grid(FractalGrid* grid, const char* filename) {
    if (!grid || !filename) return;

    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("[FractalGrid] Ошибка создания файла '%s'\n", filename);
        return;
    }

    // Сохраняем количество узлов
    if (fwrite(&grid->node_count, sizeof(int), 1, file) != 1) {
        printf("[FractalGrid] Ошибка записи node_count\n");
        fclose(file);
        return;
    }

    // Сохраняем каждый узел
    for (int i = 0; i < grid->node_count; i++) {
        FractalNode* node = grid->nodes[i];
        if (!node) continue;

        // Длина строки pattern
        int pattern_len = strlen(node->pattern);
        if (fwrite(&pattern_len, sizeof(int), 1, file) != 1) {
            printf("[FractalGrid] Ошибка записи длины pattern\n");
            continue;
        }
        if (fwrite(node->pattern, 1, pattern_len, file) != pattern_len) {
            printf("[FractalGrid] Ошибка записи pattern\n");
            continue;
        }

        // Сохраняем фрактальную размерность
        if (fwrite(&node->fractal_dimension, sizeof(float), 1, file) != 1) {
            printf("[FractalGrid] Ошибка записи fractal_dimension\n");
            continue;
        }

        // Сохраняем резонанс
        if (fwrite(&node->resonance, sizeof(float), 1, file) != 1) {
            printf("[FractalGrid] Ошибка записи resonance\n");
            continue;
        }

        // Сохраняем количество связей
        if (fwrite(&node->connection_count, sizeof(int), 1, file) != 1) {
            printf("[FractalGrid] Ошибка записи connection_count\n");
            continue;
        }

        // Сохраняем каждую связь
        for (int j = 0; j < node->connection_count; j++) {
            // Сохраняем вес, позиция, активация
            if (fwrite(node->connections[j], sizeof(float), 3, file) != 3) {
                printf("[FractalGrid] Ошибка записи connection\n");
                continue;
            }

            // Сохраняем связанное слово
            int word_len = strlen(node->connected_words[j]);
            if (fwrite(&word_len, sizeof(int), 1, file) != 1) {
                printf("[FractalGrid] Ошибка записи длины connected_word\n");
                continue;
            }
            if (fwrite(node->connected_words[j], 1, word_len, file) != word_len) {
                printf("[FractalGrid] Ошибка записи connected_word\n");
                continue;
            }
        }
    }

    fclose(file);
    printf("[FractalGrid] Сохранено %d узлов в '%s'\n", grid->node_count, filename);
}

FractalGrid* load_fractal_grid(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("[FractalGrid] Файл '%s' не найден, создается пустая сетка\n", filename);
        return fractal_grid_create();
    }

    FractalGrid* grid = fractal_grid_create();
    if (!grid) {
        fclose(file);
        return NULL;
    }

    int node_count;
    if (fread(&node_count, sizeof(int), 1, file) != 1) {
        printf("[FractalGrid] Ошибка чтения node_count\n");
        fclose(file);
        fractal_grid_destroy(grid);
        return NULL;
    }

    printf("[FractalGrid] Загрузка %d узлов...\n", node_count);

    for (int i = 0; i < node_count; i++) {
        int pattern_len;
        if (fread(&pattern_len, sizeof(int), 1, file) != 1) {
            printf("[FractalGrid] Ошибка чтения длины pattern\n");
            break;
        }

        char* pattern = (char*)malloc(pattern_len + 1);
        if (!pattern) break;

        if (fread(pattern, 1, pattern_len, file) != pattern_len) {
            printf("[FractalGrid] Ошибка чтения pattern\n");
            free(pattern);
            break;
        }
        pattern[pattern_len] = '\0';

        FractalNode* node = create_fractal_node(pattern);
        if (!node) {
            free(pattern);
            continue;
        }

        if (fread(&node->fractal_dimension, sizeof(float), 1, file) != 1) {
            printf("[FractalGrid] Ошибка чтения fractal_dimension\n");
            free(pattern);
            free(node);
            continue;
        }

        if (fread(&node->resonance, sizeof(float), 1, file) != 1) {
            printf("[FractalGrid] Ошибка чтения resonance\n");
            free(pattern);
            free(node);
            continue;
        }

        int connection_count;
        if (fread(&connection_count, sizeof(int), 1, file) != 1) {
            printf("[FractalGrid] Ошибка чтения connection_count\n");
            free(pattern);
            free(node);
            continue;
        }

        node->connection_count = connection_count;
        node->connections = (float**)malloc(connection_count * sizeof(float*));
        node->connected_words = (char**)malloc(connection_count * sizeof(char*));

        for (int j = 0; j < connection_count; j++) {
            node->connections[j] = (float*)malloc(3 * sizeof(float));
            if (fread(node->connections[j], sizeof(float), 3, file) != 3) {
                printf("[FractalGrid] Ошибка чтения connection\n");
                free(node->connections[j]);
                free(node->connected_words[j]);
                continue;
            }

            int word_len;
            if (fread(&word_len, sizeof(int), 1, file) != 1) {
                printf("[FractalGrid] Ошибка чтения длины connected_word\n");
                continue;
            }

            node->connected_words[j] = (char*)malloc(word_len + 1);
            if (!node->connected_words[j]) continue;

            if (fread(node->connected_words[j], 1, word_len, file) != word_len) {
                printf("[FractalGrid] Ошибка чтения connected_word\n");
                free(node->connected_words[j]);
                continue;
            }
            node->connected_words[j][word_len] = '\0';
        }

        // Добавляем узел в сетку
        if (grid->node_count >= grid->node_capacity) {
            grid->node_capacity *= 2;
            grid->nodes = (FractalNode**)realloc(grid->nodes, grid->node_capacity * sizeof(FractalNode*));
        }
        grid->nodes[grid->node_count++] = node;
    }

    fclose(file);
    printf("[FractalGrid] Загружено %d узлов из '%s'\n", grid->node_count, filename);
    return grid;
}