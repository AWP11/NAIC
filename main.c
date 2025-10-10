#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MEMORY_FILE "memory.bin"
#define MAX_INPUT_LENGTH 2048
#define MAX_LINES 50
#define MAX_WORDS 100
#define MAX_WORD_LENGTH 50

// Структура для хранения диалога
typedef struct {
    char** lines;
    int count;
    int capacity;
} DialogueHistory;

// Структура для бинарной памяти нейронов
typedef struct {
    FractalSpike** neurons;
    int count;
    int capacity;
    time_t last_update;
} NeuralMemory;

// Структура для словаря слов
typedef struct {
    char** words;
    float* scores;
    int count;
    int capacity;
} WordDictionary;

// =============== ДИАЛОГОВЫЕ ФУНКЦИИ ===============

DialogueHistory* create_dialogue_history() {
    DialogueHistory* history = (DialogueHistory*)malloc(sizeof(DialogueHistory));
    history->capacity = 10;
    history->count = 0;
    history->lines = (char**)malloc(history->capacity * sizeof(char*));
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
        history->lines = (char**)realloc(history->lines, history->capacity * sizeof(char*));
    }
    
    history->lines[history->count] = strdup(line);
    history->count++;
}

// =============== СЛОВАРЬ СЛОВ ===============

WordDictionary* create_word_dictionary() {
    WordDictionary* dict = (WordDictionary*)malloc(sizeof(WordDictionary));
    dict->capacity = 100;
    dict->count = 0;
    dict->words = (char**)malloc(dict->capacity * sizeof(char*));
    dict->scores = (float*)malloc(dict->capacity * sizeof(float));
    return dict;
}

void destroy_word_dictionary(WordDictionary* dict) {
    if (!dict) return;
    for (int i = 0; i < dict->count; i++) {
        free(dict->words[i]);
    }
    free(dict->words);
    free(dict->scores);
    free(dict);
}

void add_word_to_dictionary(WordDictionary* dict, const char* word, float score) {
    if (!dict || !word) return;
    
    // Проверяем, есть ли слово уже в словаре
    for (int i = 0; i < dict->count; i++) {
        if (strcmp(dict->words[i], word) == 0) {
            dict->scores[i] += score; // Увеличиваем score существующего слова
            return;
        }
    }
    
    // Добавляем новое слово
    if (dict->count >= dict->capacity) {
        dict->capacity *= 2;
        dict->words = (char**)realloc(dict->words, dict->capacity * sizeof(char*));
        dict->scores = (float*)realloc(dict->scores, dict->capacity * sizeof(float));
    }
    
    dict->words[dict->count] = strdup(word);
    dict->scores[dict->count] = score;
    dict->count++;
}

// Разбиваем текст на слова
int split_into_words(const char* text, char** words, int max_words) {
    if (!text || !words) return 0;
    
    char buffer[MAX_INPUT_LENGTH];
    strcpy(buffer, text);
    
    int word_count = 0;
    char* token = strtok(buffer, " ,.!?;:\t\n");
    
    while (token != NULL && word_count < max_words) {
        // Пропускаем очень короткие слова
        if (strlen(token) > 1) {
            words[word_count] = strdup(token);
            word_count++;
        }
        token = strtok(NULL, " ,.!?;:\t\n");
    }
    
    return word_count;
}

// =============== БИНАРНАЯ ПАМЯТЬ НЕЙРОНОВ ===============

NeuralMemory* create_neural_memory(int capacity) {
    NeuralMemory* memory = (NeuralMemory*)malloc(sizeof(NeuralMemory));
    memory->capacity = capacity;
    memory->count = 0;
    memory->neurons = (FractalSpike**)malloc(capacity * sizeof(FractalSpike*));
    memory->last_update = time(NULL);
    return memory;
}

void destroy_neural_memory(NeuralMemory* memory) {
    if (!memory) return;
    for (int i = 0; i < memory->count; i++) {
        if (memory->neurons[i]) {
            destroy_fractal_spike(memory->neurons[i]);
        }
    }
    free(memory->neurons);
    free(memory);
}

int safe_fread(void* ptr, size_t size, size_t count, FILE* stream) {
    return fread(ptr, size, count, stream) == count;
}

void save_memory_to_file(NeuralMemory* memory, const char* filename) {
    if (!memory || !filename) return;
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        return;
    }
    
    fwrite(&memory->count, sizeof(int), 1, file);
    
    for (int i = 0; i < memory->count; i++) {
        FractalSpike* neuron = memory->neurons[i];
        if (neuron) {
            fwrite(&neuron->timestamp, sizeof(long), 1, file);
            fwrite(&neuron->intensity, sizeof(float), 1, file);
            fwrite(&neuron->fractalDimension, sizeof(float), 1, file);
            
            int source_len = strlen(neuron->source) + 1;
            fwrite(&source_len, sizeof(int), 1, file);
            fwrite(neuron->source, sizeof(char), source_len, file);
            
            fwrite(&neuron->pathSize, sizeof(int), 1, file);
            for (int j = 0; j < neuron->pathSize; j++) {
                int path_len = strlen(neuron->propagationPath[j]) + 1;
                fwrite(&path_len, sizeof(int), 1, file);
                fwrite(neuron->propagationPath[j], sizeof(char), path_len, file);
            }
        }
    }
    
    fclose(file);
}

NeuralMemory* load_memory_from_file(const char* filename) {
    if (!filename) return NULL;
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return create_neural_memory(100);
    }
    
    NeuralMemory* memory = create_neural_memory(100);
    
    int neuron_count;
    if (!safe_fread(&neuron_count, sizeof(int), 1, file)) {
        fclose(file);
        return memory;
    }
    
    for (int i = 0; i < neuron_count; i++) {
        long timestamp;
        float intensity, fractalDimension;
        
        if (!safe_fread(&timestamp, sizeof(long), 1, file)) break;
        if (!safe_fread(&intensity, sizeof(float), 1, file)) break;
        if (!safe_fread(&fractalDimension, sizeof(float), 1, file)) break;
        
        int source_len;
        if (!safe_fread(&source_len, sizeof(int), 1, file)) break;
        char* source = (char*)malloc(source_len);
        if (!safe_fread(source, sizeof(char), source_len, file)) {
            free(source);
            break;
        }
        
        int pathSize;
        if (!safe_fread(&pathSize, sizeof(int), 1, file)) {
            free(source);
            break;
        }
        
        char** path = NULL;
        if (pathSize > 0) {
            path = (char**)malloc(pathSize * sizeof(char*));
            int valid_path = 1;
            for (int j = 0; j < pathSize; j++) {
                int path_len;
                if (!safe_fread(&path_len, sizeof(int), 1, file)) {
                    valid_path = 0;
                    break;
                }
                path[j] = (char*)malloc(path_len);
                if (!safe_fread(path[j], sizeof(char), path_len, file)) {
                    free(path[j]);
                    valid_path = 0;
                    break;
                }
            }
            
            if (!valid_path) {
                for (int j = 0; j < pathSize; j++) {
                    if (path[j]) free(path[j]);
                }
                free(path);
                free(source);
                break;
            }
        }
        
        FractalSpike* neuron = create_fractal_spike(timestamp, intensity, source, fractalDimension, path, pathSize);
        if (neuron && memory->count < memory->capacity) {
            memory->neurons[memory->count++] = neuron;
        }
        
        free(source);
        if (path) {
            for (int j = 0; j < pathSize; j++) {
                free(path[j]);
            }
            free(path);
        }
    }
    
    fclose(file);
    return memory;
}

void add_neuron_to_memory(NeuralMemory* memory, FractalSpike* neuron) {
    if (!memory || !neuron) return;
    
    if (memory->count >= memory->capacity) {
        memory->capacity *= 2;
        memory->neurons = (FractalSpike**)realloc(memory->neurons, memory->capacity * sizeof(FractalSpike*));
    }
    
    FractalSpike* neuron_copy = create_fractal_spike(
        neuron->timestamp,
        neuron->intensity,
        neuron->source,
        neuron->fractalDimension,
        neuron->propagationPath,
        neuron->pathSize
    );
    
    memory->neurons[memory->count++] = neuron_copy;
    memory->last_update = time(NULL);
}

// =============== УЛУЧШЕННЫЕ ФУНКЦИИ ИЗ ЯДРА ===============

// Анализ динамики диалога
float analyze_dialogue_dynamics(DialogueHistory* history) {
    if (!history || history->count < 3) return 0.5f;
    
    float dynamics = 0.0f;
    for (int i = 1; i < history->count; i++) {
        if (history->lines[i] && history->lines[i-1]) {
            float length_ratio = (float)strlen(history->lines[i]) / 
                               (float)strlen(history->lines[i-1]);
            dynamics += fabsf(length_ratio - 1.0f);
        }
    }
    
    return dynamics / (history->count - 1);
}

// Интеллектуальная оптимизация на основе качества диалога
void intelligent_system_optimization(HierarchicalSpikeSystem* system, float coherence, float dynamics) {
    if (!system) return;
    
    // Адаптируем пороги на основе успешности диалога
    if (system->cache) {
        // Высокая когерентность = более строгая кластеризация
        float adaptive_threshold = 0.1f + coherence * 0.3f;
        hash_cache_clusterize(system->cache, adaptive_threshold);
        
        // Высокая динамика = более агрессивная оптимизация энергии
        float target_efficiency = 0.6f + (1.0f - dynamics) * 0.4f;
        optimize_hash_energy(system->cache, target_efficiency);
    }
    
    // Оптимизация весовых коэффициентов
    optimize_hierarchical_connections(system);
}

// Улучшенная семантическая когерентность с IIT
float integrated_semantic_coherence(const char** patterns, int pattern_count, 
                                   float base_coherence, float integration_level) {
    if (pattern_count == 0) return base_coherence;
    
    float coherence = base_coherence;
    int connections = 0;
    
    // Анализ взаимосвязей между паттернами
    for (int i = 0; i < pattern_count - 1; i++) {
        for (int j = i + 1; j < pattern_count; j++) {
            if (patterns[i] && patterns[j]) {
                float similarity = 0.0f;
                
                // Проверка содержательных связей
                if (strstr(patterns[i], patterns[j]) || strstr(patterns[j], patterns[i])) {
                    similarity = 0.3f;
                }
                
                // Проверка лексического сходства (упрощенная)
                int common_chars = 0;
                const char* p1 = patterns[i];
                const char* p2 = patterns[j];
                while (*p1 && *p2) {
                    if (*p1 == *p2) common_chars++;
                    p1++; p2++;
                }
                similarity += (float)common_chars / fmaxf(strlen(patterns[i]), strlen(patterns[j])) * 0.2f;
                
                coherence += similarity;
                connections++;
            }
        }
    }
    
    // Фактор интегрированной информации
    float integration_factor = 1.0f + integration_level * (connections / (float)pattern_count);
    
    return fminf(1.0f, coherence * integration_factor);
}

// =============== УЛУЧШЕННОЕ ОБУЧЕНИЕ ===============

void enhanced_adaptive_learning(DialogueHistory* history, HierarchicalSpikeSystem* system, NeuralMemory* memory) {
    if (!history || !system || !memory || history->count < 2) return;
    
    // РАСШИРЕННЫЙ АНАЛИЗ ДИАЛОГА
    const char** patterns = (const char**)malloc(history->count * sizeof(char*));
    for (int i = 0; i < history->count; i++) {
        patterns[i] = history->lines[i];
    }
    
    float integration_level = analyze_dialogue_dynamics(history);
    float coherence = integrated_semantic_coherence(patterns, history->count, 0.5f, integration_level);
    free(patterns);
    
    // БИОЛОГИЧЕСКИ ОПТИМАЛЬНОЕ ОБУЧЕНИЕ
    char learning_source[256];
    snprintf(learning_source, sizeof(learning_source), 
             "learning_coherence_%.3f_dynamics_%.3f", 
             coherence, integration_level);
    
    float learning_dimension = 0.7f + coherence * 0.2f; // Адаптивная размерность
    CLAMP(learning_dimension);
    
    char* learning_path[] = {"enhanced_learning", "adaptive_optimization", "coherence_based"};
    FractalSpike* learning_spike = create_fractal_spike(
        time(NULL),
        coherence * (0.8f + integration_level * 0.2f), // Интенсивность зависит от качества
        learning_source,
        learning_dimension,
        learning_path,
        3
    );
    
    // ПРИМЕНЕНИЕ С УЧЕТОМ ВСЕХ УЛУЧШЕНИЙ
    float activation = propagate_through_hierarchy(system, learning_spike);
    
    // КОМПЛЕКСНАЯ ОПТИМИЗАЦИЯ СИСТЕМЫ
    intelligent_system_optimization(system, coherence, integration_level);
    
    // СОХРАНЕНИЕ С ОПТИМИЗАЦИЕЙ ПАМЯТИ
    add_neuron_to_memory(memory, learning_spike);
    
    // АДАПТИВНОЕ УПРАВЛЕНИЕ РЕСУРСАМИ
    if (activation > 0.7f) {
        // Успешное обучение - усиливаем устойчивые связи
        if (system->cache) {
            update_hash_learning_rates(system->cache, activation);
        }
    }
    
    printf("[Обучение] Когерентность: %.3f, Динамика: %.3f, Активация: %.3f\n", 
           coherence, integration_level, activation);
}

// =============== ГЕНЕРАЦИЯ ОТВЕТОВ ИЗ СЛОВ ===============

float calculate_text_complexity(const char* text) {
    if (!text || strlen(text) == 0) return 0.0f;
    
    int length = strlen(text);
    int unique_chars = 0;
    int char_counts[256] = {0};
    
    for (int i = 0; i < length; i++) {
        unsigned char c = text[i];
        if (char_counts[c] == 0) unique_chars++;
        char_counts[c]++;
    }
    
    float entropy = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (char_counts[i] > 0) {
            float probability = (float)char_counts[i] / length;
            entropy -= probability * logf(probability);
        }
    }
    
    // Нормализация к биологически релевантному диапазону из PDF
    float complexity = entropy / logf(256.0f);
    float length_factor = fminf(1.0f, (float)length / 100.0f);
    
    // Сдвигаем к оптимальному диапазону фрактальной размерности
    return 0.3f + complexity * 0.5f + length_factor * 0.2f; // Диапазон ~0.3-1.0
}

// Создаем словарь из памяти
WordDictionary* build_dictionary_from_memory(NeuralMemory* memory) {
    WordDictionary* dict = create_word_dictionary();
    if (!memory) return dict;
    
    for (int i = 0; i < memory->count; i++) {
        FractalSpike* neuron = memory->neurons[i];
        if (neuron && neuron->source) {
            char* words[MAX_WORDS];
            int word_count = split_into_words(neuron->source, words, MAX_WORDS);
            
            for (int j = 0; j < word_count; j++) {
                // Score зависит от интенсивности нейрона и сложности текста
                float score = neuron->intensity * (1.0f + neuron->fractalDimension);
                add_word_to_dictionary(dict, words[j], score);
                free(words[j]);
            }
        }
    }
    
    return dict;
}

// Генерация ответа из отдельных слов
char* generate_response_from_words(WordDictionary* dict, const char* input, float activation) {
    if (!dict || dict->count == 0) return strdup("...");
    
    // Разбиваем вход на слова для контекста
    char* input_words[MAX_WORDS];
    int input_word_count = split_into_words(input, input_words, MAX_WORDS);
    
    // Выбираем слова для ответа
    char response[MAX_INPUT_LENGTH] = "";
    int words_used = 0;
    int max_words = 5 + (int)(activation * 15); // От 5 до 15 слов
    
    // Сначала пытаемся использовать слова из входа (если они есть в словаре)
    for (int i = 0; i < input_word_count && words_used < max_words; i++) {
        for (int j = 0; j < dict->count; j++) {
            if (strcmp(input_words[i], dict->words[j]) == 0) {
                if (words_used > 0) strcat(response, " ");
                strcat(response, dict->words[j]);
                words_used++;
                break;
            }
        }
        free(input_words[i]);
    }
    
    // Добавляем случайные слова из словаря
    while (words_used < max_words) {
        // Выбираем слово с учетом score (более высокий score = больше шансов)
        float total_score = 0.0f;
        for (int i = 0; i < dict->count; i++) {
            total_score += dict->scores[i];
        }
        
        float random_val = (float)rand() / RAND_MAX * total_score;
        float current_sum = 0.0f;
        int selected_index = 0;
        
        for (int i = 0; i < dict->count; i++) {
            current_sum += dict->scores[i];
            if (current_sum >= random_val) {
                selected_index = i;
                break;
            }
        }
        
        // Проверяем, нет ли этого слова уже в ответе
        char temp_response[MAX_INPUT_LENGTH];
        strcpy(temp_response, response);
        if (strlen(temp_response) > 0) strcat(temp_response, " ");
        strcat(temp_response, dict->words[selected_index]);
        
        // Проверяем, не слишком ли длинный ответ
        if (strlen(temp_response) < MAX_INPUT_LENGTH - 10) {
            strcpy(response, temp_response);
            words_used++;
        } else {
            break;
        }
    }
    
    // Если не набрали слов, используем самые популярные
    if (words_used == 0) {
        // Находим слово с максимальным score
        float max_score = 0.0f;
        int best_index = 0;
        for (int i = 0; i < dict->count; i++) {
            if (dict->scores[i] > max_score) {
                max_score = dict->scores[i];
                best_index = i;
            }
        }
        strcpy(response, dict->words[best_index]);
    }
    
    return strdup(response);
}

// Основная функция генерации ответа
char* generate_word_based_response(const char* input, HierarchicalSpikeSystem* system, NeuralMemory* memory, DialogueHistory* history) {
    if (!input || !system) return strdup("...");
    
    // Создаем спайк для текущего входа
    float input_complexity = calculate_text_complexity(input);
    char* input_path[] = {"input", "word_analysis"};
    FractalSpike* input_spike = create_fractal_spike(
        time(NULL), 
        0.8f, 
        input, 
        input_complexity, 
        input_path, 
        2
    );
    
    // Пропускаем через систему
    float activation = propagate_through_hierarchy(system, input_spike);
    
    // Строим словарь из памяти
    WordDictionary* dict = build_dictionary_from_memory(memory);
    
    // Генерируем ответ из слов
    char* response = generate_response_from_words(dict, input, activation);
    
    // Создаем спайк для ответа
    char* response_path[] = {"output", "word_generation"};
    FractalSpike* response_spike = create_fractal_spike(
        time(NULL), 
        activation, 
        response,
        input_complexity * 0.8f,
        response_path, 
        2
    );
    
    // Сохраняем в память
    add_neuron_to_memory(memory, input_spike);
    add_neuron_to_memory(memory, response_spike);
    
    // Очистка
    destroy_fractal_spike(input_spike);
    destroy_word_dictionary(dict);
    
    return response;
}

// =============== УЛУЧШЕННЫЙ МНОГОСТРОЧНЫЙ ВВОД ===============

char* get_multiline_input() {
    printf("Вы: ");
    fflush(stdout);
    
    char* lines[MAX_LINES];
    int line_count = 0;
    char buffer[MAX_INPUT_LENGTH];
    
    // Читаем первую строку
    if (fgets(buffer, MAX_INPUT_LENGTH, stdin) == NULL) {
        return NULL;
    }
    
    // Убираем символ новой строки
    buffer[strcspn(buffer, "\n")] = 0;
    
    // Если строка пустая - возвращаем NULL для выхода
    if (strlen(buffer) == 0) {
        return NULL;
    }
    
    lines[line_count++] = strdup(buffer);
    
    // Читаем дополнительные строки, если пользователь продолжает ввод
    while (line_count < MAX_LINES) {
        printf("> ");
        fflush(stdout);
        
        if (fgets(buffer, MAX_INPUT_LENGTH, stdin) == NULL) {
            break;
        }
        
        buffer[strcspn(buffer, "\n")] = 0;
        
        // Пустая строка означает конец ввода
        if (strlen(buffer) == 0) {
            break;
        }
        
        lines[line_count++] = strdup(buffer);
    }
    
    // Объединяем все строки в одну
    int total_length = 0;
    for (int i = 0; i < line_count; i++) {
        total_length += strlen(lines[i]) + 1; // +1 для пробела
    }
    
    char* result = (char*)malloc(total_length + 1);
    result[0] = '\0';
    
    for (int i = 0; i < line_count; i++) {
        if (i > 0) strcat(result, " ");
        strcat(result, lines[i]);
        free(lines[i]);
    }
    
    return result;
}

// =============== ЧАТ С ГЕНЕРАЦИЕЙ ИЗ СЛОВ ===============

void run_word_based_chat() {
    printf("=== Фрактальный AI Чат (улучшенная генерация из слов) ===\n");
    printf("Бот собирает ответы из отдельных слов памяти с адаптивным обучением\n");
    printf("Вводите сообщения построчно, пустая строка - конец ввода\n");
    printf("Пустая первая строка - выход из чата\n\n");
    
    // Инициализация систем
    HierarchicalSpikeSystem* system = create_hierarchical_spike_system(100, 50, 25);
    NeuralMemory* memory = load_memory_from_file(MEMORY_FILE);
    DialogueHistory* dialogue = create_dialogue_history();
    
    int message_count = 0;
    
    // Добавляем начальные слова в память если она пустая
    if (memory->count == 0) {
        char* initial_words[] = {
            "привет", "как", "дела", "что", "нового", "расскажи", 
            "интересно", "продолжай", "понимаю", "хорошо", "да", "нет",
            "может", "быть", "это", "очень", "хороший", "вопрос"
        };
        
        for (int i = 0; i < 18; i++) {
            char* path[] = {"initial", "word"};
            FractalSpike* neuron = create_fractal_spike(
                time(NULL), 0.5f, initial_words[i], 0.3f, path, 2
            );
            add_neuron_to_memory(memory, neuron);
        }
        save_memory_to_file(memory, MEMORY_FILE);
    }
    
    while (1) {
        // Получаем многострочный ввод
        char* input = get_multiline_input();
        if (!input) {
            break;
        }
        
        if (strlen(input) == 0) {
            free(input);
            continue;
        }
        
        // Добавляем в историю
        add_to_dialogue(dialogue, input);
        
        // Генерируем ответ ИЗ СЛОВ
        char* response = generate_word_based_response(input, system, memory, dialogue);
        
        printf("Бот: %s\n\n", response);
        add_to_dialogue(dialogue, response);
        
        free(response);
        free(input);
        message_count++;
        
        // УЛУЧШЕННОЕ АДАПТИВНОЕ ОБУЧЕНИЕ каждые 2 сообщения
        if (message_count % 2 == 0) {
            enhanced_adaptive_learning(dialogue, system, memory);
        }
        
        // Автосохранение каждые 5 сообщений
        if (message_count % 5 == 0) {
            save_memory_to_file(memory, MEMORY_FILE);
            printf("[Система] Память сохранена (%d сообщений, %d нейронов)\n", 
                   message_count, memory->count);
        }
        
        // Периодический статус системы
        if (message_count % 10 == 0) {
            printf("\n=== Статус системы ===\n");
            print_hierarchical_system_status(system);
            printf("Сообщений: %d, Нейронов в памяти: %d\n\n", 
                   message_count, memory->count);
        }
    }
    
    // Сохраняем и очищаем
    save_memory_to_file(memory, MEMORY_FILE);
    
    destroy_hierarchical_spike_system(system);
    destroy_neural_memory(memory);
    destroy_dialogue_history(dialogue);
    
    printf("Чат завершен. Сообщений: %d, Слов в памяти: %d\n", message_count, memory->count);
}

// =============== ОСНОВНАЯ ФУНКЦИЯ ===============

int main() {
    srand(time(NULL));
    
    printf("Фрактальный AGI - Улучшенная генерация из слов\n");
    printf("==============================================\n");
    printf("Бот учится и собирает ответы из отдельных слов\n");
    printf("с биологически оптимальной фрактальной архитектурой\n\n");
    
    run_word_based_chat();
    
    return 0;
}
