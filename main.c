#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include "kernel.h"
#include <ctype.h>

#define MAX_DATA_INPUT 3000
#define MAX_RESPONSE_LENGTH 200
#define RECURSION_DEPTH 3
#define MAX_MEMORY_ENTRIES 10000
#define MAX_SPIKES 5000
#define MAX_RESONANCE_STATES 100
#define EMBEDDING_SIZE 256
#define BINARY_SIGNATURE_SIZE 32

// === НОВАЯ СТРУКТУРА ДЛЯ ФРАКТАЛЬНО-ГРАФОВОГО ПРЕДСТАВЛЕНИЯ ===
typedef struct {
    float* activation_vector;    // Горизонтальные связи (традиционные нейроны)
    float** fractal_connections; // Вертикальные связи между уровнями [depth][connections]
    int depth;
    int* connections_per_level;
    float energy_flow;           // Поток энергии через структуру
    float coherence;             // Когерентность паттерна
} FractalGraphPattern;

// === ГИБРИДНЫЕ СТРУКТУРЫ ДАННЫХ ===

// Расширенная структура памяти с эмбеддингами
typedef struct {
    char utf8_char[5];           // UTF-8 символ
    long created_at;             // Временная метка
    float embedding[EMBEDDING_SIZE]; // Вектор эмбеддинга
    uint8_t binary_signature[BINARY_SIGNATURE_SIZE]; // Бинарная сигнатура
    int access_count;            // Счетчик обращений
    float importance;            // Важность записи (для забывания)
    FractalGraphPattern* fractal_pattern; // НОВОЕ: фрактально-графовое представление
} EnhancedMemoryEntry;

// Глобальные структуры
static FractalHashCache* global_fractal_cache = NULL;
static NeuralResonance* global_resonance = NULL;

// Глобальные массивы для хранения данных
static EnhancedMemoryEntry memory_entries[MAX_MEMORY_ENTRIES];
static int memory_count = 0;

// === НОВЫЕ ФУНКЦИИ ДЛЯ ФРАКТАЛЬНО-ГРАФОВОЙ МЕТРИКИ ===

// 1. Энерго-осознанное косинусное сходство
float energy_aware_cosine_similarity(const float* a, const float* b, 
                                    float energy_a, float energy_b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        // Корректируем с учетом энергетического потока
        float weighted_a = a[i] * (1.0f + energy_a * 0.1f);
        float weighted_b = b[i] * (1.0f + energy_b * 0.1f);
        
        dot += weighted_a * weighted_b;
        norm_a += weighted_a * weighted_a;
        norm_b += weighted_b * weighted_b;
    }
    
    if (norm_a == 0 || norm_b == 0) return 0.0f;
    
    float base_similarity = dot / (sqrtf(norm_a) * sqrtf(norm_b));
    
    // Дополнительная коррекция на энергетическую совместимость
    float energy_compatibility = 1.0f - fabsf(energy_a - energy_b) * 0.5f;
    
    return base_similarity * energy_compatibility;
}

// 2. Сравнение фрактальных структур (САМАЯ ВАЖНАЯ ЧАСТЬ!)
float compare_fractal_structures(FractalGraphPattern* a, FractalGraphPattern* b) {
    if (!a || !b) return 0.0f;
    
    float structure_similarity = 0.0f;
    int levels_compared = 0;
    
    int max_depth = (a->depth < b->depth) ? a->depth : b->depth;
    
    for (int level = 0; level < max_depth; level++) {
        float level_similarity = 0.0f;
        int connections_compared = 0;
        
        int max_connections = (a->connections_per_level[level] < b->connections_per_level[level]) 
                            ? a->connections_per_level[level] 
                            : b->connections_per_level[level];
        
        for (int conn = 0; conn < max_connections; conn++) {
            if (a->fractal_connections[level] && b->fractal_connections[level]) {
                // Сравниваем силу связей между уровнями
                float connection_strength_a = a->fractal_connections[level][conn];
                float connection_strength_b = b->fractal_connections[level][conn];
                
                float connection_sim = 1.0f - fabsf(connection_strength_a - connection_strength_b);
                level_similarity += connection_sim;
                connections_compared++;
            }
        }
        
        if (connections_compared > 0) {
            level_similarity /= connections_compared;
            
            // Вес уровня увеличивается с глубиной (более глубокие уровни важнее)
            float level_weight = (float)(level + 1) / max_depth;
            structure_similarity += level_similarity * level_weight;
            levels_compared++;
        }
    }
    
    return levels_compared > 0 ? structure_similarity / levels_compared : 0.0f;
}

// 3. Резонансная синхронизация паттернов
float calculate_resonance_sync(FractalGraphPattern* a, FractalGraphPattern* b,
                              NeuralResonance* resonance) {
    if (!resonance || !a || !b) return 0.0f;
    
    float total_sync = 0.0f;
    int sync_points = 0;
    
    // Анализируем синхронизацию на разных временных масштабах
    for (float time_scale = 0.1f; time_scale <= 1.0f; time_scale += 0.2f) {
        float resonant_a = apply_resonance(resonance, a->coherence, time_scale);
        float resonant_b = apply_resonance(resonance, b->coherence, time_scale);
        
        float phase_diff = fabsf(resonant_a - resonant_b);
        float sync_level = 1.0f - phase_diff;
        
        total_sync += sync_level;
        sync_points++;
    }
    
    return sync_points > 0 ? total_sync / sync_points : 0.0f;
}

// === ЯДРО НОВОЙ МЕТРИКИ ===
float fractal_graph_similarity(FractalGraphPattern* pattern_a, 
                              FractalGraphPattern* pattern_b,
                              NeuralResonance* resonance) {
    
    if (!pattern_a || !pattern_b) return 0.0f;
    
    float total_similarity = 0.0f;
    int components_compared = 0;
    
    // 1. СХОДСТВО АКТИВАЦИОННЫХ ВЕКТОРОВ (горизонтальный уровень)
    float activation_sim = 0.0f;
    if (pattern_a->activation_vector && pattern_b->activation_vector) {
        // Используем модифицированное косинусное сходство с энергетической коррекцией
        activation_sim = energy_aware_cosine_similarity(
            pattern_a->activation_vector, 
            pattern_b->activation_vector,
            pattern_a->energy_flow, 
            pattern_b->energy_flow
        );
    }
    total_similarity += activation_sim * 0.3f; // 30% веса
    components_compared++;
    
    // 2. ФРАКТАЛЬНАЯ СТРУКТУРНАЯ СХОДСТВО (вертикальные связи)
    float structural_sim = compare_fractal_structures(pattern_a, pattern_b);
    total_similarity += structural_sim * 0.4f; // 40% веса - самый важный аспект!
    components_compared++;
    
    // 3. РЕЗОНАНСНАЯ СИНХРОНИЗАЦИЯ
    float resonance_sim = 0.0f;
    if (resonance) {
        resonance_sim = calculate_resonance_sync(pattern_a, pattern_b, resonance);
    }
    total_similarity += resonance_sim * 0.2f; // 20% веса
    components_compared++;
    
    // 4. ЭНЕРГЕТИЧЕСКАЯ СОВМЕСТИМОСТЬ
    float energy_sim = 1.0f - fabsf(pattern_a->energy_flow - pattern_b->energy_flow);
    total_similarity += energy_sim * 0.1f; // 10% веса
    components_compared++;
    
    return total_similarity;
}

// === ФУНКЦИИ ДЛЯ РАБОТЫ С ФРАКТАЛЬНЫМИ ГРАФАМИ ===

FractalGraphPattern* create_fractal_graph_pattern(int depth, int base_connections) {
    FractalGraphPattern* pattern = (FractalGraphPattern*)malloc(sizeof(FractalGraphPattern));
    if (!pattern) return NULL;
    
    pattern->depth = depth;
    pattern->activation_vector = (float*)calloc(EMBEDDING_SIZE, sizeof(float));
    pattern->connections_per_level = (int*)malloc(depth * sizeof(int));
    pattern->fractal_connections = (float**)malloc(depth * sizeof(float*));
    
    pattern->energy_flow = 0.5f; // Начальное значение
    pattern->coherence = 0.7f;
    
    // Инициализация связей для каждого уровня
    for (int i = 0; i < depth; i++) {
        int connections = base_connections * (1 << i); // Экспоненциальный рост
        pattern->connections_per_level[i] = connections;
        pattern->fractal_connections[i] = (float*)calloc(connections, sizeof(float));
        
        // Инициализация случайными значениями
        for (int j = 0; j < connections; j++) {
            pattern->fractal_connections[i][j] = (float)rand() / RAND_MAX * 0.5f;
        }
    }
    
    return pattern;
}

void destroy_fractal_graph_pattern(FractalGraphPattern* pattern) {
    if (!pattern) return;
    
    if (pattern->activation_vector) {
        free(pattern->activation_vector);
    }
    
    if (pattern->fractal_connections) {
        for (int i = 0; i < pattern->depth; i++) {
            if (pattern->fractal_connections[i]) {
                free(pattern->fractal_connections[i]);
            }
        }
        free(pattern->fractal_connections);
    }
    
    if (pattern->connections_per_level) {
        free(pattern->connections_per_level);
    }
    
    free(pattern);
}

// Вспомогательные функции для фрактальных графов
float calculate_energy_flow(FractalGraphPattern* pattern) {
    if (!pattern) return 0.0f;
    
    float total_energy = 0.0f;
    int connections_count = 0;
    
    for (int i = 0; i < pattern->depth; i++) {
        for (int j = 0; j < pattern->connections_per_level[i]; j++) {
            total_energy += pattern->fractal_connections[i][j];
            connections_count++;
        }
    }
    
    return connections_count > 0 ? total_energy / connections_count : 0.0f;
}

float calculate_pattern_coherence(FractalGraphPattern* pattern) {
    if (!pattern || !pattern->activation_vector) return 0.0f;
    
    float variance = 0.0f;
    float mean = 0.0f;
    
    // Вычисляем среднее активационного вектора
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        mean += pattern->activation_vector[i];
    }
    mean /= EMBEDDING_SIZE;
    
    // Вычисляем дисперсию
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        float diff = pattern->activation_vector[i] - mean;
        variance += diff * diff;
    }
    variance /= EMBEDDING_SIZE;
    
    // Когерентность обратно пропорциональна дисперсии
    return 1.0f / (1.0f + sqrtf(variance));
}

void build_fractal_connections(FractalGraphPattern* pattern, const char* input) {
    if (!pattern || !input) return;
    
    int input_len = strlen(input);
    
    for (int level = 0; level < pattern->depth; level++) {
        int connections = pattern->connections_per_level[level];
        
        for (int conn = 0; conn < connections; conn++) {
            // Строим связи на основе семантики входных данных
            float connection_strength = 0.0f;
            
            for (int i = 0; i < input_len && i < EMBEDDING_SIZE; i++) {
                // Сила связи зависит от позиции символа и его значения
                connection_strength += (float)input[i] * 
                                     sinf((float)i * 0.1f + (float)conn * 0.01f) *
                                     (1.0f / (level + 1));
            }
            
            pattern->fractal_connections[level][conn] = 
                fabsf(connection_strength) / (input_len > 0 ? input_len : 1);
        }
    }
}

// === ИНТЕГРАЦИЯ В ОСНОВНУЮ СИСТЕМУ ===

// Вместо простых эмбеддингов используем фрактальные графы
void process_input_with_fractal_graph(const char* input, FractalGraphPattern* output_pattern) {
    if (!input || !output_pattern) return;
    
    // 1. Создаем активационный вектор из входных данных
    generate_text_embedding(input, output_pattern->activation_vector, EMBEDDING_SIZE);
    
    // 2. Строим фрактальные связи на основе семантики
    build_fractal_connections(output_pattern, input);
    
    // 3. Вычисляем энергетические характеристики
    output_pattern->energy_flow = calculate_energy_flow(output_pattern);
    output_pattern->coherence = calculate_pattern_coherence(output_pattern);
}

// === ФУНКЦИИ ПОСТЕПЕННОГО ЗАБЫВАНИЯ ===

// Вычисление "возраста" записи для забывания
float calculate_memory_age_factor(const EnhancedMemoryEntry* entry) {
    long current_time = time(NULL);
    long age = current_time - entry->created_at;
    return 1.0f - expf(-age / 86400.0f); // Забывание за 24 часа
}

// Постепенное забывание наименее важных записей
void gradual_memory_forgetting() {
    if (memory_count < MAX_MEMORY_ENTRIES * 0.8f) return;
    
    printf("NAIC>> Запуск постепенного забывания...\n");
    
    // Вычисляем оценки для всех записей
    float scores[MAX_MEMORY_ENTRIES];
    float total_score = 0.0f;
    
    for (int i = 0; i < memory_count; i++) {
        float age_factor = calculate_memory_age_factor(&memory_entries[i]);
        float access_factor = 1.0f - (float)memory_entries[i].access_count / 100.0f;
        float importance = memory_entries[i].importance;
        
        // Комбинированная оценка: чем выше, тем более вероятно удаление
        scores[i] = age_factor * 0.6f + access_factor * 0.3f + (1.0f - importance) * 0.1f;
        total_score += scores[i];
    }
    
    // Удаляем 10% наименее ценных записей
    int forget_count = memory_count / 10;
    int removed = 0;
    
    for (int f = 0; f < forget_count && memory_count > 0; f++) {
        float avg_score = total_score / memory_count;
        int worst_index = -1;
        float worst_score = 2.0f; // Заведомо больше максимального
        
        // Находим наихудшую запись
        for (int i = 0; i < memory_count; i++) {
            if (scores[i] < worst_score) {
                worst_score = scores[i];
                worst_index = i;
            }
        }
        
        if (worst_index != -1 && worst_score > avg_score * 0.8f) {
            // Освобождаем фрактальный паттерн перед удалением
            if (memory_entries[worst_index].fractal_pattern) {
                destroy_fractal_graph_pattern(memory_entries[worst_index].fractal_pattern);
            }
            
            // Удаляем наихудшую запись
            total_score -= scores[worst_index];
            for (int i = worst_index; i < memory_count - 1; i++) {
                memory_entries[i] = memory_entries[i + 1];
                scores[i] = scores[i + 1];
            }
            memory_count--;
            removed++;
        }
    }
    
    printf("NAIC>> Удалено %d наименее важных записей памяти\n", removed);
    save_memory_to_bin();
}

// === УТИЛИТЫ ГИБРИДНОЙ СИСТЕМЫ ===

// Генерация бинарной сигнатуры
void generate_binary_signature(const uint8_t* data, size_t len, uint8_t* signature) {
    memset(signature, 0, BINARY_SIGNATURE_SIZE);
    
    for (size_t i = 0; i < len; i++) {
        signature[i % BINARY_SIGNATURE_SIZE] ^= data[i];
    }
    
    // Добавляем хэш для уникальности
    unsigned long hash = 5381;
    for (size_t i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + data[i];
    }
    
    memcpy(signature + BINARY_SIGNATURE_SIZE - 4, &hash, 4);
}

// Косинусное сходство
float cosine_similarity(const float* a, const float* b, int size) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    
    for (int i = 0; i < size; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0 || norm_b == 0) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

// Генерация текстового эмбеддинга
void generate_text_embedding(const char* text, float* embedding, int size) {
    if (!text || !embedding) return;
    
    // Инициализация случайным вектором
    for (int i = 0; i < size; i++) {
        embedding[i] = 0.0f;
    }
    
    int len = strlen(text);
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < size; j++) {
            embedding[j] += (float)text[i] * sinf((float)i * 0.1f + j * 0.01f);
        }
    }
    
    // Нормализация
    float norm = 0.0f;
    for (int i = 0; i < size; i++) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm);
    
    if (norm > 0) {
        for (int i = 0; i < size; i++) {
            embedding[i] /= norm;
        }
    }
}

// === ИНТЕГРАЦИЯ ОБУЧЕНИЯ НА ЛЕТУ ===

// Расширенная функция онлайн-обучения с гибридными features
void enhanced_online_learning(const char* input_pattern, float actual_output, 
                            float expected_output, float dimension) {
    if (!global_fractal_cache || !global_resonance) return;
    
    // Используем функцию из kernel.c для основного обучения
    fractal_online_learning(global_fractal_cache, global_resonance,
                          input_pattern, actual_output, expected_output, dimension);
    
    // Дополнительная адаптация на основе семантики
    float error = expected_output - actual_output;
    
    // Обновляем важность в памяти на основе ошибки
    for (int i = 0; i < memory_count; i++) {
        if (strstr(memory_entries[i].utf8_char, input_pattern)) {
            // Увеличиваем важность успешных паттернов
            memory_entries[i].importance += fabsf(error) * 0.1f;
            CLAMP(memory_entries[i].importance);
            memory_entries[i].access_count++;
        }
    }
    
    // Адаптируем глобальные параметры резонанса
    if (fabsf(error) > 0.3f) {
        global_resonance->frequency *= (1.0f + error * 0.05f);
        global_resonance->amplitude *= (1.0f - error * 0.03f);
        CLAMP(global_resonance->frequency);
        CLAMP(global_resonance->amplitude);
    }
}

// === СУЩЕСТВУЮЩИЕ ФУНКЦИИ (АДАПТИРОВАНЫ) ===

// Улучшенная функция определения длины UTF-8 символа
static int utf8_char_length_safe(unsigned char byte) {
    if ((byte & 0x80) == 0) return 1;
    if ((byte & 0xE0) == 0xC0) return 2;
    if ((byte & 0xF0) == 0xE0) return 3;
    if ((byte & 0xF8) == 0xF0) return 4;
    return -1;
}

// Безопасное извлечение UTF-8 символа
static int extract_utf8_char(const char* str, int str_len, int pos, char* output, int output_size) {
    if (!str || pos < 0 || pos >= str_len || !output || output_size < 2) 
        return -1;
    
    unsigned char first_byte = (unsigned char)str[pos];
    int expected_len = utf8_char_length_safe(first_byte);
    
    if (expected_len <= 0) {
        output[0] = str[pos];
        output[1] = '\0';
        return 1;
    }
    
    if (pos + expected_len > str_len) {
        output[0] = str[pos];
        output[1] = '\0';
        return 1;
    }
    
    for (int i = 1; i < expected_len; i++) {
        if ((str[pos + i] & 0xC0) != 0x80) {
            output[0] = str[pos];
            output[1] = '\0';
            return 1;
        }
    }
    
    if (expected_len >= output_size) {
        output[0] = str[pos];
        output[1] = '\0';
        return 1;
    }
    
    memcpy(output, &str[pos], expected_len);
    output[expected_len] = '\0';
    return expected_len;
}

// Улучшенная функция сохранения в память с эмбеддингами И фрактальными графами
void save_message_to_memory_enhanced(const char *message) {
    if (!message) return;
    
    // Проверяем необходимость забывания перед добавлением
    gradual_memory_forgetting();
    
    long now = time(NULL);
    int str_len = strlen(message);
    int i = 0;
    
    // Генерация эмбеддинга для всего сообщения
    float message_embedding[EMBEDDING_SIZE];
    uint8_t binary_sig[BINARY_SIGNATURE_SIZE];
    
    generate_text_embedding(message, message_embedding, EMBEDDING_SIZE);
    generate_binary_signature((uint8_t*)message, str_len, binary_sig);
    
    // НОВОЕ: Создаем фрактальный граф для всего сообщения
    FractalGraphPattern* message_pattern = create_fractal_graph_pattern(4, 8); // 4 уровня, 8 базовых связей
    if (message_pattern) {
        process_input_with_fractal_graph(message, message_pattern);
    }
    
    while (i < str_len && memory_count < MAX_MEMORY_ENTRIES) {
        char utf8_char[5] = "";
        int char_len = extract_utf8_char(message, str_len, i, utf8_char, sizeof(utf8_char));
        
        if (char_len > 0) {
            // Проверяем, не превысили ли лимит после забывания
            if (memory_count >= MAX_MEMORY_ENTRIES) {
                gradual_memory_forgetting();
                if (memory_count >= MAX_MEMORY_ENTRIES) break;
            }
            
            // Копируем данные поэлементно вместо присвоения структур
            memcpy(memory_entries[memory_count].utf8_char, utf8_char, 5);
            memory_entries[memory_count].created_at = now;
            memory_entries[memory_count].access_count = 1;
            memory_entries[memory_count].importance = 0.5f; // Средняя важность
            
            // Сохраняем эмбеддинг и бинарную сигнатуру
            memcpy(memory_entries[memory_count].embedding, message_embedding, sizeof(float) * EMBEDDING_SIZE);
            memcpy(memory_entries[memory_count].binary_signature, binary_sig, BINARY_SIGNATURE_SIZE);
            
            // НОВОЕ: Сохраняем фрактальный паттерн (общий для всего сообщения)
            memory_entries[memory_count].fractal_pattern = message_pattern;
            
            memory_count++;
        }
        
        i += (char_len > 0) ? char_len : 1;
    }
    
    save_memory_to_bin();
}

// Бинарные функции хранения - ИСПРАВЛЕННАЯ ВЕРСИЯ
void save_memory_to_bin() {
    FILE* file = fopen("memory.bin", "wb");
    if (file) {
        // Сохраняем количество записей
        fwrite(&memory_count, sizeof(int), 1, file);
        
        // Сохраняем каждую запись отдельно, без фрактальных паттернов
        for (int i = 0; i < memory_count; i++) {
            // Сохраняем базовые поля
            fwrite(memory_entries[i].utf8_char, sizeof(char), 5, file);
            fwrite(&memory_entries[i].created_at, sizeof(long), 1, file);
            fwrite(memory_entries[i].embedding, sizeof(float), EMBEDDING_SIZE, file);
            fwrite(memory_entries[i].binary_signature, sizeof(uint8_t), BINARY_SIGNATURE_SIZE, file);
            fwrite(&memory_entries[i].access_count, sizeof(int), 1, file);
            fwrite(&memory_entries[i].importance, sizeof(float), 1, file);
        }
        
        fclose(file);
    }
}

void load_memory_from_bin() {
    FILE* file = fopen("memory.bin", "rb");
    if (file) {
        if (fread(&memory_count, sizeof(int), 1, file) != 1) {
            memory_count = 0;
            fclose(file);
            return;
        }
        if (memory_count > MAX_MEMORY_ENTRIES) memory_count = MAX_MEMORY_ENTRIES;
        
        for (int i = 0; i < memory_count; i++) {
            // Загружаем базовые поля
            if (fread(memory_entries[i].utf8_char, sizeof(char), 5, file) != 5) break;
            if (fread(&memory_entries[i].created_at, sizeof(long), 1, file) != 1) break;
            if (fread(memory_entries[i].embedding, sizeof(float), EMBEDDING_SIZE, file) != EMBEDDING_SIZE) break;
            if (fread(memory_entries[i].binary_signature, sizeof(uint8_t), BINARY_SIGNATURE_SIZE, file) != BINARY_SIGNATURE_SIZE) break;
            if (fread(&memory_entries[i].access_count, sizeof(int), 1, file) != 1) break;
            if (fread(&memory_entries[i].importance, sizeof(float), 1, file) != 1) break;
            
            // Фрактальные паттерны не загружаем из бинарного файла
            memory_entries[i].fractal_pattern = NULL;
        }
        fclose(file);
    }
}

// Безопасная функция ремонта UTF-8 строки
void repair_utf8_string_safe(char* str, int max_len) {
    if (!str || max_len <= 0) return;
    
    int input_len = strlen(str);
    if (input_len >= max_len) input_len = max_len - 1;
    
    char repaired[3000];
    int repair_pos = 0;
    int i = 0;
    
    while (i < input_len && repair_pos < (int)(sizeof(repaired) - 2)) {
        unsigned char byte = (unsigned char)str[i];
        int expected_len = utf8_char_length_safe(byte);
        
        if (expected_len <= 0 || i + expected_len > input_len) {
            repaired[repair_pos++] = '?';
            i++;
            continue;
        }
        
        int valid = 1;
        for (int j = 1; j < expected_len; j++) {
            if ((str[i + j] & 0xC0) != 0x80) {
                valid = 0;
                break;
            }
        }
        
        if (valid && repair_pos + expected_len < sizeof(repaired)) {
            memcpy(&repaired[repair_pos], &str[i], expected_len);
            repair_pos += expected_len;
            i += expected_len;
        } else {
            repaired[repair_pos++] = '?';
            i++;
        }
    }
    
    repaired[repair_pos] = '\0';
    strncpy(str, repaired, max_len - 1);
    str[max_len - 1] = '\0';
}

// Улучшенная функция активации с фрактальным кэшем
float get_fractal_activation(const char* pattern, float dimension, float intensity) {
    if (!global_fractal_cache || !pattern) return intensity;
    
    FractalHashEntry* entry = hash_cache_lookup(global_fractal_cache, pattern, dimension, intensity);
    if (entry) {
        return entry->cached_activation * (1.0f + HASH_CACHE_HIT_BONUS);
    }
    
    return intensity;
}

// Многострочный ввод
int read_multiline_input(char* buffer, int max_size) {
    int total_chars = 0;
    buffer[0] = '\0';
    
    printf("user>> ");
    fflush(stdout);
    
    int line_count = 0;
    
    while (total_chars < max_size - 1) {
        char line[512];
        if (fgets(line, sizeof(line), stdin) == NULL) {
            break;
        }
        
        int line_len = strlen(line);
        
        if (line_len == 1 && line[0] == '\n') {
            if (line_count > 0) {
                break;
            } else {
                printf("...... ");
                fflush(stdout);
                continue;
            }
        }
        
        if (total_chars + line_len < max_size - 1) {
            strcat(buffer, line);
            total_chars += line_len;
            line_count++;
        } else {
            strncat(buffer, line, max_size - total_chars - 1);
            break;
        }
        
        if (total_chars < max_size - 10) {
            printf("...... ");
            fflush(stdout);
        }
    }
    
    if (total_chars > 0 && buffer[total_chars - 1] == '\n') {
        buffer[total_chars - 1] = '\0';
        total_chars--;
    }
    
    return total_chars;
}

// Структура для хранения кандидатов с оценкой сходства
typedef struct {
    int index;
    float similarity;
} CandidateEntry;

// НОВАЯ: Улучшенная рекурсивная генерация ответа с фрактальными графами
char* generate_enhanced_recursive_response(const char* base_seed, float activation, 
                                          int depth, int max_depth, 
                                          FractalActivation* act,
                                          FractalGraphPattern* input_pattern) {
    if (depth >= max_depth || activation < 0.1f) {
        char* result = (char*)malloc(MAX_RESPONSE_LENGTH + 1);
        if (result) {
            strncpy(result, base_seed, MAX_RESPONSE_LENGTH);
            result[MAX_RESPONSE_LENGTH] = '\0';
        }
        return result;
    }
    
    char new_seed[MAX_RESPONSE_LENGTH * 2] = "";
    strncat(new_seed, base_seed, MAX_RESPONSE_LENGTH);
    
    // ИСПОЛЬЗУЕМ ФРАКТАЛЬНЫЕ ГРАФЫ ДЛЯ ВЫБОРА НАИБОЛЕЕ РЕЛЕВАНТНЫХ СИМВОЛОВ
    int sample_size = 10 + (int)(activation * 30);
    if (sample_size > memory_count) sample_size = memory_count;
    
    // Собираем кандидатов с оценкой сходства
    CandidateEntry candidates[MAX_MEMORY_ENTRIES];
    int candidate_count = 0;
    
    for (int i = 0; i < memory_count && candidate_count < sample_size * 2; i++) {
        if (memory_entries[i].fractal_pattern) {
            float similarity = fractal_graph_similarity(
                input_pattern, 
                memory_entries[i].fractal_pattern,
                global_resonance
            );
            
            if (similarity > 0.3f) { // Порог сходства
                candidates[candidate_count].index = i;
                candidates[candidate_count].similarity = similarity;
                candidate_count++;
            }
        }
    }
    
    // Сортируем кандидатов по сходству (простейшая сортировка)
    for (int i = 0; i < candidate_count - 1; i++) {
        for (int j = i + 1; j < candidate_count; j++) {
            if (candidates[i].similarity < candidates[j].similarity) {
                CandidateEntry temp = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = temp;
            }
        }
    }
    
    // Берем лучшие кандидаты
    int selected_count = (candidate_count < sample_size) ? candidate_count : sample_size;
    for (int i = 0; i < selected_count; i++) {
        const char* ch = memory_entries[candidates[i].index].utf8_char;
        if (ch && strlen(new_seed) + strlen(ch) < sizeof(new_seed) - 1) {
            strcat(new_seed, ch);
        }
    }
    
    // Если кандидатов мало, добавляем случайные
    if (selected_count < sample_size) {
        int needed = sample_size - selected_count;
        for (int i = 0; i < needed; i++) {
            int random_index = rand() % memory_count;
            const char* ch = memory_entries[random_index].utf8_char;
            if (ch && strlen(new_seed) + strlen(ch) < sizeof(new_seed) - 1) {
                strcat(new_seed, ch);
            }
        }
    }
    
    float new_activation = activation * 0.7f;
    char* recursive_result = generate_enhanced_recursive_response(
        new_seed, new_activation, 
        depth + 1, max_depth, act, input_pattern
    );
    
    char* final_result = (char*)malloc(MAX_RESPONSE_LENGTH * 2);
    if (final_result) {
        snprintf(final_result, MAX_RESPONSE_LENGTH * 2, "%s|%s", 
                 base_seed, recursive_result ? recursive_result : "");
        
        if (strlen(final_result) > MAX_RESPONSE_LENGTH) {
            final_result[MAX_RESPONSE_LENGTH] = '\0';
        }
        
        free(recursive_result);
    }
    
    return final_result ? final_result : strdup(base_seed);
}

// Основная функция памяти с интеграцией ФРАКТАЛЬНЫХ ГРАФОВ
void IOmemory(const char* user_message) {
    float base_activation = get_fractal_activation(user_message, 1.5f, 0.85f);
    
    FractalActivation* act = create_fractal_activation(base_activation, 0.70f, 0.75f, 4, 0.3f);

    // НОВОЕ: Создаем фрактальный граф для пользовательского ввода
    FractalGraphPattern* user_pattern = create_fractal_graph_pattern(4, 8);
    if (user_pattern) {
        process_input_with_fractal_graph(user_message, user_pattern);
    }

    if (global_resonance) {
        apply_resonance_to_activation(act, global_resonance);
    }

    // ФРАКТАЛЬНОЕ ОБРАТНОЕ РАСПРОСТРАНЕНИЕ
    FractalBackprop* bp = create_fractal_backprop(act->fractalDepth);
    float learning_rate = 0.8f * 0.01f;
    
    fractal_gradient_descent(act, learning_rate);
    float total = get_total_activation(act);
    
    float target_activation = 0.7f;
    fractal_backward_pass(bp, act, target_activation, total);
    apply_fractal_gradients(act, bp);
    
    total = get_total_activation(act);

    if (global_resonance) {
        update_resonance_parameters(global_resonance, total);
    }

    int depth = (int)(act->fractalDepth * total);
    if (depth < 1) depth = 1;
    if (depth > 8) depth = 8;

    // Сбор данных из памяти С ИСПОЛЬЗОВАНИЕМ ФРАКТАЛЬНЫХ ГРАФОВ
    char combined[3000] = "";
    int sample_count = depth * 50;
    if (sample_count > memory_count) sample_count = memory_count;
    
    // НОВОЕ: Используем фрактальные графы для выбора наиболее релевантных символов
    if (user_pattern) {
        CandidateEntry ranked_memory[MAX_MEMORY_ENTRIES];
        int ranked_count = 0;
        
        for (int i = 0; i < memory_count && ranked_count < sample_count * 2; i++) {
            if (memory_entries[i].fractal_pattern) {
                float similarity = fractal_graph_similarity(
                    user_pattern, 
                    memory_entries[i].fractal_pattern,
                    global_resonance
                );
                
                if (similarity > 0.2f) { // Порог сходства
                    ranked_memory[ranked_count].index = i;
                    ranked_memory[ranked_count].similarity = similarity;
                    ranked_count++;
                }
            }
        }
        
        // Сортируем по сходству
        for (int i = 0; i < ranked_count - 1; i++) {
            for (int j = i + 1; j < ranked_count; j++) {
                if (ranked_memory[i].similarity < ranked_memory[j].similarity) {
                    CandidateEntry temp = ranked_memory[i];
                    ranked_memory[i] = ranked_memory[j];
                    ranked_memory[j] = temp;
                }
            }
        }
        
        // Берем лучшие matches
        int selected = (ranked_count < sample_count) ? ranked_count : sample_count;
        for (int i = 0; i < selected; i++) {
            const char* ch = memory_entries[ranked_memory[i].index].utf8_char;
            if (ch && strlen(combined) + strlen(ch) < sizeof(combined) - 1) {
                strcat(combined, ch);
            }
        }
        
        printf("NAIC>> Использовано фрактальное сходство: %d/%d записей (макс. сходство: %.3f)\n", 
               selected, ranked_count, ranked_count > 0 ? ranked_memory[0].similarity : 0.0f);
    } else {
        // Резервный вариант: случайная выборка
        for (int i = 0; i < sample_count; i++) {
            int random_index = rand() % memory_count;
            const char* ch = memory_entries[random_index].utf8_char;
            if (ch && strlen(combined) + strlen(ch) < sizeof(combined) - 1) {
                strcat(combined, ch);
            }
        }
    }

    int symbol_count = strlen(combined);
    int base_window = 5 + (int)(total * 30);
    int dynamic_factor = (int)(0.8f * 8);
    int window = base_window + dynamic_factor;
    if (window > MAX_RESPONSE_LENGTH) window = MAX_RESPONSE_LENGTH;
    if (window < 5) window = 5;
    if (window > symbol_count) window = symbol_count;

    char bot_response[MAX_RESPONSE_LENGTH + 1] = "";
    
    if (symbol_count > 0 && window > 0) {
        char* initial_seed = (char*)malloc(window + 1);
        if (initial_seed) {
            int start = rand() % (symbol_count - window + 1);
            int safe_start = start;
            while (safe_start > 0) {
                unsigned char c = combined[safe_start];
                if ((c & 0x80) == 0 || (c & 0xC0) == 0xC0) break;
                safe_start--;
            }
            strncpy(initial_seed, &combined[safe_start], window);
            initial_seed[window] = '\0';
            
            // НОВОЕ: Используем улучшенную рекурсивную генерацию с фрактальными графами
            char* recursive_response = generate_enhanced_recursive_response(
                initial_seed, total, 0, RECURSION_DEPTH, act, user_pattern
            );
            
            if (recursive_response) {
                strncpy(bot_response, recursive_response, MAX_RESPONSE_LENGTH);
                bot_response[MAX_RESPONSE_LENGTH] = '\0';
                free(recursive_response);
            }
            
            free(initial_seed);
        }
    }

    if (strlen(bot_response) == 0) {
        const char* fallbacks[] = {
            "Размышляю над этим...", "Интересный вопрос!", 
            "Позвольте подумать...", "Ммм, нужно обдумать...",
            "Понимаю вашу мысль...", "Это требует анализа..."
        };
        strcpy(bot_response, fallbacks[rand() % 6]);
    }

    if (strlen(bot_response) > MAX_RESPONSE_LENGTH) {
        bot_response[MAX_RESPONSE_LENGTH] = '\0';
    }

    repair_utf8_string_safe(bot_response, MAX_RESPONSE_LENGTH);

    // ОЦЕНКА КАЧЕСТВА И ОБУЧЕНИЕ НА ЛЕТУ
    float response_quality = 0.5f;
    int response_length = strlen(bot_response);
    
    if (response_length > 10 && response_length < MAX_RESPONSE_LENGTH - 10) {
        response_quality = 0.7f;
    }
    if (response_length > 30) {
        response_quality = 0.8f;
    }
    
    int unique_chars = 0;
    int char_counts[256] = {0};
    for (int i = 0; i < response_length; i++) {
        unsigned char c = bot_response[i];
        if (char_counts[c] == 0) unique_chars++;
        char_counts[c]++;
    }
    float diversity = (response_length > 0) ? (float)unique_chars / response_length : 0.0f;
    response_quality += diversity * 0.2f;
    
    CLAMP(response_quality);

    // ОБУЧЕНИЕ НА ЛЕТУ С ИНТЕГРАЦИЕЙ ИЗ KERNEL.C
    if (global_fractal_cache && global_resonance) {
        enhanced_online_learning(user_message, total, response_quality, 1.5f);
        
        fractal_backward_pass(bp, act, response_quality, total);
        apply_fractal_gradients(act, bp);
        
        // Оптимизация энергии кэша
        optimize_hash_energy(global_fractal_cache, 0.8f - fabsf(response_quality - total) * 0.3f);
    }

    if (total < 0.15f) {
        printf("NAIC>> ... (активация низкая: %.2f)\n", total);
    } else {
        printf("NAIC>> %s [активация: %.2f, качество: %.2f]\n", 
               bot_response, total, response_quality);
    }

    // Сохраняем в фрактальный кэш
    if (global_fractal_cache) {
        hash_cache_store(global_fractal_cache, user_message, 1.5f, total, total);
    }

    // Очистка
    destroy_fractal_backprop(bp);
    destroy_fractal_activation(act);
    if (user_pattern) {
        destroy_fractal_graph_pattern(user_pattern);
    }
}

// === ОСНОВНАЯ ФУНКЦИЯ ===

int main(void) {
    srand((unsigned int)time(NULL));
    
    // Инициализация AI системы с фрактальным кэшем
    global_fractal_cache = create_fractal_hash_cache(1000);
    global_resonance = create_neural_resonance(1.0f, 0.5f, 0.01f);
    
    load_memory_from_bin();
    
    printf("NAIC>> Система инициализирована с ФРАКТАЛЬНО-ГРАФОВЫМИ МЕТРИКАМИ.\n");
    printf("      Готов к многострочному вводу!\n");
    printf("      (вводите текст, пустая строка завершает ввод)\n");
    printf("      Доступные команды: /exit, /debug, /forget\n\n");

    while (1) {
        // Проверяем необходимость постепенного забывания
        gradual_memory_forgetting();

        char message[MAX_DATA_INPUT] = "";
        
        int chars_read = read_multiline_input(message, sizeof(message));
        
        if (chars_read > 0) {
            if (strcmp(message, "/exit") == 0) {
                printf("NAIC>> Пока! Активация: 0.0\n");
                break;
            }
            
            if (strcmp(message, "/debug") == 0) {
                printf("\n=== DEBUG INFORMATION ===\n");
                printf("Memory entries: %d/%d\n", memory_count, MAX_MEMORY_ENTRIES);
                
                if (global_fractal_cache) {
                    printf("Fractal cache: %d/%d entries\n", global_fractal_cache->size, global_fractal_cache->capacity);
                } else {
                    printf("Fractal cache: NULL\n");
                }
                
                if (global_resonance) {
                    printf("Global resonance: freq=%.3f, amp=%.3f, damp=%.3f\n", 
                           global_resonance->frequency, global_resonance->amplitude, global_resonance->damping);
                } else {
                    printf("Global resonance: NULL\n");
                }
                
                if (memory_count > 0) {
                    long oldest = memory_entries[0].created_at;
                    long newest = memory_entries[memory_count-1].created_at;
                    printf("Memory time range: %ld - %ld\n", oldest, newest);
                    
                    // Статистика важности
                    float avg_importance = 0.0f;
                    int patterns_with_fractal = 0;
                    for (int i = 0; i < memory_count; i++) {
                        avg_importance += memory_entries[i].importance;
                        if (memory_entries[i].fractal_pattern) patterns_with_fractal++;
                    }
                    avg_importance /= memory_count;
                    printf("Average memory importance: %.3f\n", avg_importance);
                    printf("Entries with fractal patterns: %d/%d\n", patterns_with_fractal, memory_count);
                }
                
                printf("========================\n\n");
                continue;
            }
            
            if (strcmp(message, "/forget") == 0) {
                printf("NAIC>> Принудительный запуск постепенного забывания...\n");
                gradual_memory_forgetting();
                printf("NAIC>> Забывание завершено\n");
                continue;
            }

            int is_trusted = 0;
            float intensity = 0.8f;
            if (strncmp(message, "!!", 2) == 0) {
                is_trusted = 1;
                intensity = 1.0f;
                memmove(message, message + 2, strlen(message + 2) + 1);
            }

            save_message_to_memory_enhanced(message);

            float enhanced_intensity = get_fractal_activation(message, 1.5f, intensity);
            
            // Сохраняем в фрактальный кэш
            if (global_fractal_cache) {
                hash_cache_store(global_fractal_cache, message, 1.5f, enhanced_intensity, enhanced_intensity);
            }

            IOmemory(message);
        } else if (chars_read == 0) {
            continue;
        } else {
            fprintf(stderr, "Error reading input.\n");
            break;
        }
    }

    // Сохраняем резонансное состояние
    if (global_resonance) {
        update_resonance_parameters(global_resonance, 0.5f);
    }
    
    // Очистка AI системы
    if (global_fractal_cache) {
        destroy_fractal_hash_cache(global_fractal_cache);
    }
    if (global_resonance) {
        destroy_neural_resonance(global_resonance);
    }
    
    // Очистка фрактальных паттернов в памяти
    for (int i = 0; i < memory_count; i++) {
        if (memory_entries[i].fractal_pattern) {
            destroy_fractal_graph_pattern(memory_entries[i].fractal_pattern);
        }
    }
    
    save_memory_to_bin();
    
    return 0;
}