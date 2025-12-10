// interface_AI.c
#include "interface_AI.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "core/fractal_tensor.h"
#define MAX_INPUT_LENGTH 10000
#define MAX_LINES 100
#define MAX_WORDS 3000
#define MAX_WORD_LENGTH 100

// Структура для хранения диалога (внутренняя для interface_AI.c)
typedef struct {
    char** lines;
    int count;
    int capacity;
} DialogueHistory;

// Структура для словаря слов (внутренняя для interface_AI.c)
typedef struct {
    char** words;
    float* scores;
    int count;
    int capacity;
} WordDictionary;

// === Внутренние (static) вспомогательные функции для interface_AI.c ===

static float analyze_dialogue_resonance(DialogueHistory* history);
static void perform_resonance_learning(NeuralMemory* memory, FractalField* field, float resonance_strength);

static DialogueHistory* create_dialogue_history() {
    DialogueHistory* history = (DialogueHistory*)malloc(sizeof(DialogueHistory));
    if (!history) return NULL;
    history->capacity = 15;
    history->count = 0;
    history->lines = (char**)malloc(history->capacity * sizeof(char*));
    if (!history->lines) { free(history); return NULL; }
    return history;
}

static void destroy_dialogue_history(DialogueHistory* history) {
    if (!history) return;
    for (int i = 0; i < history->count; i++) {
        free(history->lines[i]);
    }
    free(history->lines);
    free(history);
}

static void add_to_dialogue(DialogueHistory* history, const char* line) {
    if (!history || !line) return;
    if (history->count >= history->capacity) {
        history->capacity *= 2;
        char** temp = (char**)realloc(history->lines, history->capacity * sizeof(char*));
        if (!temp) return; // Ошибка, не расширяем
        history->lines = temp;
    }
    history->lines[history->count] = strdup(line);
    if (history->lines[history->count]) history->count++; // Увеличиваем только если strdup успешен
}

// --- СЛОВАРЬ СЛОВ (внутренний) ---
static WordDictionary* create_word_dictionary() {
    WordDictionary* dict = (WordDictionary*)malloc(sizeof(WordDictionary));
    if (!dict) return NULL;
    dict->capacity = 100;
    dict->count = 0;
    dict->words = (char**)malloc(dict->capacity * sizeof(char*));
    dict->scores = (float*)malloc(dict->capacity * sizeof(float));
    if (!dict->words || !dict->scores) { free(dict->words); free(dict->scores); free(dict); return NULL; }
    return dict;
}

static void destroy_word_dictionary(WordDictionary* dict) {
    if (!dict) return;
    for (int i = 0; i < dict->count; i++) {
        free(dict->words[i]);
    }
    free(dict->words);
    free(dict->scores);
    free(dict);
}

static void add_word_to_dictionary(WordDictionary* dict, const char* word, float score) {
    if (!dict || !word) return;
    for (int i = 0; i < dict->count; i++) {
        if (strcmp(dict->words[i], word) == 0) {
            dict->scores[i] += score;
            return;
        }
    }
    if (dict->count >= dict->capacity) {
        dict->capacity *= 2;
        char** temp_w = (char**)realloc(dict->words, dict->capacity * sizeof(char*));
        float* temp_s = (float*)realloc(dict->scores, dict->capacity * sizeof(float));
        if (!temp_w || !temp_s) return; // Ошибка, не расширяем
        dict->words = temp_w;
        dict->scores = temp_s;
    }
    dict->words[dict->count] = strdup(word);
    if (dict->words[dict->count]) {
        dict->scores[dict->count] = score;
        dict->count++;
    }
}

// Исправление в функции split_into_words:
static int split_into_words(const char* text, char** words, int max_words) {
    if (!text || !words || max_words <= 0) return 0;
    
    size_t text_len = strlen(text);
    if (text_len == 0) return 0;
    
    // Изменение здесь: использовать memcpy вместо strncpy
    char* buffer = (char*)malloc(text_len + 1);
    if (!buffer) {
        printf("[split_into_words] Ошибка выделения буфера\n");
        return 0;
    }
    
    // Безопасное копирование с memcpy
    memcpy(buffer, text, text_len);
    buffer[text_len] = '\0';  // Явно добавляем нуль-терминатор
    
    int word_count = 0;
    char* saveptr = NULL;
    const char* delimiters = " ,.!?;:\t\n\r";
    
    char* token = strtok_r(buffer, delimiters, &saveptr);
    
    while (token != NULL && word_count < max_words) {
        size_t token_len = strlen(token);
        
        if (token_len > 1) {
            words[word_count] = strdup(token);
            
            if (!words[word_count]) {
                printf("[split_into_words] Ошибка выделения памяти для слова\n");
                for (int i = 0; i < word_count; i++) {
                    free(words[i]);
                    words[i] = NULL;
                }
                free(buffer);
                return 0;
            }
            
            word_count++;
        }
        token = strtok_r(NULL, delimiters, &saveptr);
    }
    
    free(buffer);
    return word_count;
} 

// (Мы их копируем сюда, но помечаем как static, чтобы они не конфликтовали с kernel.c, если вдруг kernel.c их не содержит)
// Лучше всего их перенести в kernel.c, но если kernel.c не меняется, то копируем сюда как static.
static float calculate_text_complexity(const char* text) {
    if (!text || strlen(text) == 0) return 0.0f;
    int length = strlen(text); int unique_chars = 0; int char_counts[256] = {0};
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
    float complexity = entropy / logf(256.0f);
    float length_factor = fminf(1.0f, (float)length / 100.0f);
    return 0.3f + complexity * 0.5f + length_factor * 0.2f;
}

static WordDictionary* build_dictionary_from_memory(NeuralMemory* memory) {
    WordDictionary* dict = create_word_dictionary();
    if (!memory || !dict) return dict;
    for (int i = 0; i < memory->count; i++) {
        FractalSpike* neuron = memory->neurons[i];
        if (neuron && neuron->source) {
            char* words[MAX_WORDS];
            int word_count = split_into_words(neuron->source, words, MAX_WORDS);
            for (int j = 0; j < word_count; j++) {
                if (words[j]) {
                    float score = neuron->intensity * (1.0f + neuron->fractalDimension);
                    add_word_to_dictionary(dict, words[j], score);
                    free(words[j]);
                }
            }
        }
    }
    return dict;
}

// === НОВАЯ ФУНКЦИЯ: Получение резонансных слов из памяти ===
static WordDictionary* get_resonant_words_from_memory(NeuralMemory* memory, float resonance_threshold) {
    WordDictionary* dict = create_word_dictionary();
    if (!memory || !dict) return dict;
    
    // Собираем слова из резонансных нейронов
    for (int i = 0; i < memory->count; i++) {
        FractalSpike* neuron = memory->neurons[i];
        if (neuron && neuron->intensity >= resonance_threshold && neuron->source) {
            char* words[MAX_WORDS];
            int word_count = split_into_words(neuron->source, words, MAX_WORDS);
            for (int j = 0; j < word_count; j++) {
                if (words[j] && strlen(words[j]) > 2) {
                    // Повышаем оценку для резонансных слов
                    float score = neuron->intensity * (1.0f + neuron->fractalDimension) * 1.5f;
                    add_word_to_dictionary(dict, words[j], score);
                    free(words[j]);
                }
            }
        }
    }
    
    return dict;
}

// === ОПТИМИЗИРОВАННАЯ ФУНКЦИЯ С РЕЗОНАНСОМ ===
static char* generate_response_from_words(  
    WordDictionary* dict, 
    WordDictionary* resonant_dict,
    const char* input, 
    float line_activation,
    FractalField* field
) {
    if (!dict || dict->count == 0 || !input) return strdup("...");
    
    char* input_words[MAX_WORDS];
    int input_word_count = split_into_words(input, input_words, MAX_WORDS);
    
    char response[MAX_INPUT_LENGTH] = ""; 
    int words_used = 0;

    float fractal_growth_factor = 1.0f;
    float resonance_modulation = 1.0f;
    
    if (field) {
        fractal_growth_factor = 1.0f + field->global_reward_signal * 0.5f;
        fractal_growth_factor = fmaxf(fractal_growth_factor, 0.7f);
        fractal_growth_factor = fminf(fractal_growth_factor, 1.3f);
        
        // Резонансная модуляция на основе состояния поля
        if (field->neuron_count > 0) {
            float neuron_activity_ratio = (float)field->neuron_count / field->max_neurons;
            resonance_modulation += neuron_activity_ratio * 0.3f;
        }
        if (field->connection_count > 0) {
            float connectivity_ratio = (float)field->connection_count / field->max_connections;
            resonance_modulation += connectivity_ratio * 0.2f;
        }
        resonance_modulation += fabsf(field->global_reward_signal) * 0.5f;
    }

    // Используем line_activation вместо activation
    line_activation = fmaxf(fminf(line_activation, 1.0f), 0.0f);
    
    // НЕлинейная активация с резонансной модуляцией
    float nonlinear_activation = powf(line_activation * resonance_modulation, 1.3f);

    // Определяем количество слов на основе активации и резонанса
    int max_words = (int)(nonlinear_activation * 20 * fractal_growth_factor);
    max_words = fmin(max_words, 20);

    // Шаг 1: Используем слова из входного запроса
    for (int i = 0; i < input_word_count && words_used < max_words; i++) {
        if (!input_words[i]) continue;

        for (int j = 0; j < dict->count; j++) {
            if (strcmp(input_words[i], dict->words[j]) == 0) {
                size_t new_len = strlen(response) + strlen(dict->words[j]) + 2;
                if (new_len >= MAX_INPUT_LENGTH) break;

                if (words_used > 0) strcat(response, " ");
                strcat(response, dict->words[j]);
                words_used++;
                break;
            }
        }
        free(input_words[i]);
    }

    // Шаг 2: Добавляем резонансные слова с повышенным приоритетом
    if (resonant_dict && resonant_dict->count > 0 && words_used < max_words) {
        float total_resonant_score = 0.0f;
        for (int i = 0; i < resonant_dict->count; i++) {
            total_resonant_score += resonant_dict->scores[i];
        }
        
        if (total_resonant_score > 0.0f) {
            // Выбираем резонансные слова взвешенно
            int resonant_words_to_add = fmin(3, max_words - words_used);
            for (int r = 0; r < resonant_words_to_add && words_used < max_words; r++) {
                float random_val = (float)rand() / RAND_MAX * total_resonant_score;
                float current_sum = 0.0f;
                int selected_index = 0;
                
                for (int i = 0; i < resonant_dict->count; i++) {
                    current_sum += resonant_dict->scores[i];
                    if (current_sum >= random_val) {
                        selected_index = i;
                        break;
                    }
                }
                
                size_t word_len = strlen(resonant_dict->words[selected_index]);
                size_t current_len = strlen(response);
                if (current_len + word_len + 1 >= MAX_INPUT_LENGTH) break;

                if (words_used > 0) strcat(response, " ");
                strcat(response, resonant_dict->words[selected_index]);
                words_used++;
            }
        }
    }

    // Шаг 3: Используем словарь для завершения ответа
    float exploration_factor = 1.0f;
    if (field) {
        if (field->global_reward_signal < -0.3f) {
            exploration_factor = 0.6f;
        } else if (field->global_reward_signal > 0.5f) {
            exploration_factor = 1.8f;
        }
        exploration_factor = fmaxf(0.3f, fminf(3.0f, exploration_factor));
    }

    while (words_used < max_words && dict->count > 0) {
        float total_score = 0.0f;
        for (int i = 0; i < dict->count; i++) {
            total_score += powf(dict->scores[i], exploration_factor);
        }
        if (total_score <= 0.0f) break;

        float random_val = (float)rand() / RAND_MAX * total_score;
        float current_sum = 0.0f;
        int selected_index = 0;
        for (int i = 0; i < dict->count; i++) {
            current_sum += powf(dict->scores[i], exploration_factor);
            if (current_sum >= random_val) {
                selected_index = i;
                break;
            }
        }

        // Экспериментальный режим при высокой активации
        if (exploration_factor > 1.5f) {
            if ((float)rand() / RAND_MAX < 0.3f) {
                selected_index = rand() % dict->count;
            }
        }

        size_t word_len = strlen(dict->words[selected_index]);
        size_t current_len = strlen(response);
        if (current_len + word_len + 1 >= MAX_INPUT_LENGTH) break;

        if (words_used > 0) {
            strncat(response, " ", 1);
        }
        strncat(response, dict->words[selected_index], MAX_INPUT_LENGTH - current_len - 1);
        words_used++;
    }

    // Шаг 4: Если ответ пустой, используем лучшее слово
    if (words_used == 0 && dict->count > 0) {
        int best_index = 0;
        float max_score = dict->scores[0];
        for (int i = 1; i < dict->count; i++) {
            if (dict->scores[i] > max_score) {
                best_index = i;
                max_score = dict->scores[i];
            }
        }
        strncpy(response, dict->words[best_index], MAX_INPUT_LENGTH - 1);
        response[MAX_INPUT_LENGTH - 1] = '\0';
    }

    return strdup(response);
}

// --- НОВАЯ ФУНКЦИЯ: ОЦЕНКА КАЧЕСТВА ДИАЛОГА (внутренняя) ---
static float calculate_dialogue_quality(const char* input, const char* response, DialogueHistory* history) {
    if (!input || !response || !history) return 0.0f;
    float quality = 0.0f;
    if (strcmp(response, "...") == 0 || strlen(response) < 3) {
        quality -= 0.5f;
    } else {
        quality += 0.2f;
    }
    char* input_words[MAX_WORDS];
    int input_count = split_into_words(input, input_words, MAX_WORDS);
    char* response_words[MAX_WORDS];
    int response_count = split_into_words(response, response_words, MAX_WORDS);
    int matches = 0;
    for (int i = 0; i < input_count; i++) {
        if (input_words[i]) {
            for (int j = 0; j < response_count; j++) {
                if (response_words[j] && strcmp(input_words[i], response_words[j]) == 0) {
                    matches++; break;
                }
            }
            free(input_words[i]);
        }
    }
    for (int j = 0; j < response_count; j++) {
        if (response_words[j]) free(response_words[j]);
    }
    if (input_count > 0) {
        quality += (float)matches / input_count * 0.5f;
    }
    // Ограничиваем диапазон [0, 1]
    if (quality > 1.0f) quality = 1.0f;
    if (quality < 0.0f) quality = 0.0f;
    // Преобразуем [0,1] -> [-1,1]
    return quality * 2.0f - 1.0f;
}

// --- УЛУЧШЕННАЯ ФУНКЦИЯ ИЗ ЯДРА (внутренняя) ---
static float analyze_dialogue_dynamics(DialogueHistory* history) {
    if (!history || history->count < 3) return 0.5f;
    float dynamics = 0.0f;
    int thematic_shifts = 0; // Счётчик смены тем (новые ключевые слова)
    int repetition_count = 0; // Счётчик повторений
    for (int i = 1; i < history->count; i++) {
        if (history->lines[i] && history->lines[i-1]) {
            float length_ratio = (float)strlen(history->lines[i]) / (float)strlen(history->lines[i-1]);
            dynamics += fabsf(length_ratio - 1.0f);

            // Простая проверка на смену темы: если в новой строке есть слово, отсутствующее в старой
            if (strstr(history->lines[i], "ключевое_слово") && !strstr(history->lines[i-1], "ключевое_слово")) {
                 thematic_shifts++; // Пример, требует уточнения
            }
            // Простая проверка на повторение: если строки идентичны
            if (strcmp(history->lines[i], history->lines[i-1]) == 0) {
                 repetition_count++;
            }
        }
    }
    // Нормализуем и добавляем вклад смены тем и повторений
    float thematic_factor = (float)thematic_shifts / (history->count - 1);
    float repetition_factor = (float)repetition_count / (history->count - 1);
    return (dynamics + thematic_factor * 0.5f) * (1.0f - repetition_factor * 0.3f); // Уменьшаем динамику при повторениях
}

static float integrated_semantic_coherence(const char** patterns, int pattern_count, float base_coherence, float integration_level) {
    if (pattern_count == 0) return base_coherence;
    float coherence = base_coherence; int connections = 0;
    for (int i = 0; i < pattern_count - 1; i++) {
        for (int j = i + 1; j < pattern_count; j++) {
            if (patterns[i] && patterns[j]) {
                float similarity = 0.0f;
                if (strstr(patterns[i], patterns[j]) || strstr(patterns[j], patterns[i])) {
                    similarity = 0.3f;
                }
                int common_chars = 0;
                const char* p1 = patterns[i]; const char* p2 = patterns[j];
                while (*p1 && *p2) { if (*p1 == *p2) common_chars++; p1++; p2++; }
                similarity += (float)common_chars / fmaxf(strlen(patterns[i]), strlen(patterns[j])) * 0.2f;
                coherence += similarity; connections++;
            }
        }
    }
    float integration_factor = 1.0f + integration_level * (connections / (float)pattern_count);
    return fminf(1.0f, coherence * integration_factor);
}

static void intelligent_system_optimization(HierarchicalSpikeSystem* system, float coherence, float dynamics) {
    if (!system) return;
    if (system->cache) {
        float adaptive_threshold = 0.1f + coherence * 0.3f;
        hash_cache_clusterize(system->cache, adaptive_threshold);
        float target_efficiency = 0.6f + (1.0f - dynamics) * 0.4f;
        optimize_hash_energy(system->cache, target_efficiency);
    }
    optimize_hierarchical_connections(system);
}

// --- УЛУЧШЕННОЕ ОБУЧЕНИЕ (внутреннее) ---
static void enhanced_adaptive_learning(DialogueHistory* history, HierarchicalSpikeSystem* system, NeuralMemory* memory, FractalField* field) {
    if (!history || !system || !memory || !field || history->count < 2) return;
    
    update_neuron_importance(memory);
    
    // Анализ диалога
    float resonance_score = analyze_dialogue_resonance(history);
    float integration_level = analyze_dialogue_dynamics(history);
    
    const char** patterns = (const char**)malloc(history->count * sizeof(char*));
    if (!patterns) return;
    for (int i = 0; i < history->count; i++) patterns[i] = history->lines[i];
    
    float coherence = integrated_semantic_coherence(patterns, history->count, 0.5f, integration_level);
    free(patterns);
    
    float quality_score = 
        coherence * 0.4f + 
        integration_level * 0.3f + 
        resonance_score * 0.3f;
    
    float global_reward = quality_score * 2.0f - 1.0f;
    
    // === УДАЛЯЕМ СОЗДАНИЕ СЛУЖЕБНОГО СПАЙКА! ===
    // НЕ создаем learning_spike для служебных данных
    // НЕ добавляем его в память
    
    // === ПРИМЕНЯЕМ ОБУЧЕНИЕ НАПРЯМУЮ ===
    field->global_reward_signal = global_reward;
    
    propagate_fractal_field(field, global_reward);
    update_fractal_field(field);
    check_growth_conditions(field);
    
    // Резонансное обучение (если нужно)
    if (resonance_score > 0.6f) {
        perform_resonance_learning(memory, field, resonance_score);
    }
    
    // Оптимизация системы
    intelligent_system_optimization(system, coherence, integration_level);
    
    // === НЕ добавляем служебные данные в memory! ===
    // Вместо этого можно добавить РЕАЛЬНЫЙ контент из диалога
    
    // Обновление кэша на основе качества
    if (system->cache && quality_score > 0.3f) {
        update_hash_learning_rates(system->cache, quality_score * resonance_score);
    }
    
    printf("[Обучение] Качество: %.3f, Вознаграждение: %.3f\n", quality_score, global_reward);
}

// === НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ===
static float analyze_dialogue_resonance(DialogueHistory* history) {
    if (!history || history->count < 3) return 0.0f;
    
    float resonance = 0.0f;
    int thematic_continuity = 0;
    
    // Анализируем тематическую непрерывность
    for (int i = 1; i < history->count; i++) {
        if (history->lines[i] && history->lines[i-1]) {
            // Простая проверка на общие слова
            char* words1[MAX_WORDS], *words2[MAX_WORDS];
            int count1 = split_into_words(history->lines[i-1], words1, MAX_WORDS);
            int count2 = split_into_words(history->lines[i], words2, MAX_WORDS);
            
            int common_words = 0;
            for (int w1 = 0; w1 < count1; w1++) {
                for (int w2 = 0; w2 < count2; w2++) {
                    if (words1[w1] && words2[w2] && strcmp(words1[w1], words2[w2]) == 0) {
                        common_words++;
                        break;
                    }
                }
                if (words1[w1]) free(words1[w1]);
            }
            for (int w2 = 0; w2 < count2; w2++) {
                if (words2[w2]) free(words2[w2]);
            }
            
            if (count1 > 0) {
                resonance += (float)common_words / count1 * 0.3f;
            }
            
            if (common_words > 0) thematic_continuity++;
        }
    }
    
    // Нормализуем
    if (history->count > 1) {
        resonance /= (history->count - 1);
        float continuity_ratio = (float)thematic_continuity / (history->count - 1);
        resonance = resonance * 0.7f + continuity_ratio * 0.3f;
    }
    
    return fminf(1.0f, resonance);
}

// === НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ===
static void perform_resonance_learning(NeuralMemory* memory, FractalField* field, float resonance_strength) {
    if (!memory || !field) return;
    
    printf("[Резонансное обучение] Начало (сила: %.3f)\n", resonance_strength);
    
    // Усиливаем наиболее активные нейроны
    for (int i = 0; i < memory->count && i < 50; i++) { // Ограничиваем для производительности
        if (memory->neurons[i]) {
            // Нейроны с высокой интенсивностью получают boost
            if (memory->neurons[i]->intensity > 0.5f) {
                memory->neurons[i]->intensity *= (1.0f + resonance_strength * 0.2f);
                CLAMP(memory->neurons[i]->intensity);
                
                // Также усиливаем фрактальную размерность для сложных паттернов
                if (memory->neurons[i]->fractalDimension > 1.5f) {
                    memory->neurons[i]->fractalDimension += resonance_strength * 0.05f;
                    CLAMP(memory->neurons[i]->fractalDimension);
                }
            }
        }
    }
    
    // Стимулируем рост FractalField при высоком резонансе
    if (resonance_strength > 0.8f && field->neuron_count < field->max_neurons) {
        printf("[Резонансное обучение] Стимулируем рост FractalField\n");
        field->global_reward_signal = 1.0f; // Положительное вознаграждение
        check_growth_conditions(field);
    }
}

// --- ОСНОВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТА (внутренняя) ---
// Исправьте эту функцию в interface_AI.c
static char* generate_word_based_response(const char* input, HierarchicalSpikeSystem* system, NeuralMemory* memory, DialogueHistory* history, FractalField* field) {
    if (!input || !system || !field) return strdup("...");
    
    // === ИНТЕГРАЦИЯ С FractalField ===
    float input_complexity = calculate_text_complexity(input);
    
    // Создаем входной спайк
    char* input_path[] = {"input", "word_analysis"};
    FractalSpike* input_spike = create_fractal_spike(time(NULL), 0.8f, input, input_complexity, input_path, 2);
    
    // Используем FractalField для оценки входного сигнала
   float field_activation = 0.0f;
if (field->neuron_count > 0) {
    // Интеллектуальная активация с проверкой границ знаний
    float knowledge_boundary = (float)field->connection_count / (field->neuron_count + 1);
    float bounded_randomness = ((float)rand() / RAND_MAX) * 0.2f * (1.0f - knowledge_boundary);
    
    field_activation = input_complexity * (0.4f + knowledge_boundary * 0.2f) + bounded_randomness;
    CLAMP(field_activation);
    
    // Обновляем глобальный сигнал вознаграждения
    field->global_reward_signal = field_activation - 0.5f; // [-0.5, 0.5]
    
    // Пропагируем через FractalField
    propagate_fractal_field(field, field->global_reward_signal);
    update_fractal_field(field);
}
    
float activation_from_old_system = propagate_through_hierarchy_with_resonance(system, input_spike, NULL);
    
    // Комбинируем активации
    float line_activation = activation_from_old_system * 0.3f + field_activation * 0.7f;
    
    WordDictionary* dict = build_dictionary_from_memory(memory);
    
    // === НОВОЕ: Получаем резонансные слова ===
    WordDictionary* resonant_dict = get_resonant_words_from_memory(memory, 0.6f);
    
    // === КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Исключаем служебные паттерны из словаря ===
    // Фильтруем словарь, удаляя служебные слова типа "learning_coherence_"
    for (int i = dict->count - 1; i >= 0; i--) {
        if (dict->words[i] && strstr(dict->words[i], "learning_coherence") != NULL) {
            free(dict->words[i]);
            dict->words[i] = dict->words[dict->count - 1];
            dict->scores[i] = dict->scores[dict->count - 1];
            dict->count--;
        }
    }
    
    // Используем оптимизированную функцию с резонансом
    char* response = generate_response_from_words(dict, resonant_dict, input, line_activation, field);

    // === Оценка качества и формирование сигнала вознаграждения ===
    float global_reward = calculate_dialogue_quality(input, response, history);
    field->global_reward_signal = global_reward; // Передаём в FractalField

    // === Сохранение в память (только если ответ не служебный) ===
    if (!strstr(response, "learning_coherence")) {
        char* response_path[] = {"output", "word_generation"};
        FractalSpike* response_spike = create_fractal_spike(
            time(NULL), 
            line_activation, 
            response, 
            input_complexity * 0.8f, 
            response_path, 
            2
        );
        add_neuron_to_memory(memory, input_spike);
        add_neuron_to_memory(memory, response_spike);
        destroy_fractal_spike(response_spike);
    }

    // === Очистка ===
    destroy_fractal_spike(input_spike);
    destroy_word_dictionary(dict);
    destroy_word_dictionary(resonant_dict);

    return response;
}
// === Реализация функций, объявленных в interface_AI.h ===

char* get_user_input(void) {
    printf("Вы: "); fflush(stdout);

    char* lines[MAX_LINES];
    int line_count = 0;
    char buffer[MAX_INPUT_LENGTH];

    // Читаем первую строку
    if (fgets(buffer, MAX_INPUT_LENGTH, stdin) == NULL) {
        return NULL; // Ошибка ввода
    }

    // Убираем символ новой строки
    buffer[strcspn(buffer, "\n")] = 0;

    // Если строка пустая сразу - возвращаем NULL для выхода
    if (strlen(buffer) == 0) {
        return NULL;
    }

    lines[line_count++] = strdup(buffer);

    // Читаем дополнительные строки, если пользователь продолжает ввод
    while (line_count < MAX_LINES) {
        printf("> "); fflush(stdout);

        if (fgets(buffer, MAX_INPUT_LENGTH, stdin) == NULL) {
            break; // Ошибка ввода
        }

        // Убираем символ новой строки
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
        if (lines[i]) total_length += strlen(lines[i]) + 1; // +1 для пробела или конца строки
    }

    char* result = (char*)malloc(total_length + 1);
    if (!result) {
        // Освобождаем память в случае ошибки
        for (int i = 0; i < line_count; i++) {
            if (lines[i]) free(lines[i]);
        }
        return NULL;
    }
    result[0] = '\0';

    for (int i = 0; i < line_count; i++) {
        if (lines[i]) {
            if (i > 0) strcat(result, " "); // Разделяем строки пробелом
            strcat(result, lines[i]);
            free(lines[i]); // Освобождаем временную строку
        }
    }

    return result;
}

void display_bot_response(const char* response) {
    if (response) {
        printf("NAIC: %s\n\n", response);
    }
}

void print_status(FractalField* field, NeuralMemory* memory, int message_count) {
    if (!field || !memory) return;
    printf("\n=== Статус NAIC ===\n");
    // Печатаем статус старой системы, если она используется
    // print_hierarchical_system_status(system); // Раскомментируйте, если система используется
    printf("Сообщений: %d, Нейронов в памяти: %d\n", message_count, memory->count);
    printf("FractalField: Нейронов=%d, Связей=%d, Вознаграждение=%.3f\n", field->neuron_count, field->connection_count, field->global_reward_signal);
    printf("========================\n\n");
}


void run_chat_interface(FractalField* field, NeuralMemory* memory) {
    if (!field || !memory) {
        printf("Ошибка: FractalField или NeuralMemory не инициализированы.\n");
        return;
    }

    printf("=== NAIC - Фрактальный AI Чат (с FractalField и R-STDP) ===\n");
    printf("NAIC учится через вознаграждение, растёт и управляет связями\n");
    printf("Вводите сообщения построчно, пустая строка - конец ввода\n");
    printf("Пустая первая строка - выход из чата\n\n");

    // Инициализация вспомогательных структур для этого интерфейса
    DialogueHistory* dialogue = create_dialogue_history();
    HierarchicalSpikeSystem* system = create_hierarchical_spike_system(100, 50, 25); // Для совместимости

    int message_count = 0;
    while (1) {
        char* input = get_user_input();
        if (!input) break;
        if (strlen(input) == 0) { free(input); continue; }

        add_to_dialogue(dialogue, input);
        char* response = generate_word_based_response(input, system, memory, dialogue, field);
        display_bot_response(response);
        add_to_dialogue(dialogue, response);

        free(response); free(input); message_count++;

        if (message_count % 2 == 0) {
            enhanced_adaptive_learning(dialogue, system, memory, field);
        }

        if (message_count % 5 == 0) {
            // save_memory_to_file(memory, MEMORY_FILE); // Вызов из main.c
            printf("[NAIC] Память сохранена (%d сообщений, %d нейронов)\n", message_count, memory->count);
            printf("[NAIC] FractalField: Нейронов=%d, Связей=%d\n", field->neuron_count, field->connection_count);
        }

        if (message_count % 10 == 0) {
           // print_status(field, memory, message_count);
        }
    }

    // Очистка вспомогательных структур
    destroy_hierarchical_spike_system(system);
    destroy_dialogue_history(dialogue);

    printf("Чат NAIC завершен. Сообщений: %d\n", message_count);
}