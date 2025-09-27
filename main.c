#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "SQL/sqlite3.h"
#include "kernel.h"
#include <ctype.h>
#define MAX_DATA_INPUT 3000  // Увеличили для многострочного ввода
#define DB_NAME "AI MEMORY.db"
#define MAX_SPIKE_AGE_SECONDS 10000
#define MAX_RESPONSE_LENGTH 200  // Увеличили максимальную длину ответа
#define RECURSION_DEPTH 3        // Глубина рекурсивной генерации

// Глобальные структуры для улучшения интеллекта
static FractalHashCache* global_cache = NULL;
static NeuralResonance* global_resonance = NULL;

// Вспомогательные функции UTF-8
static int utf8_char_length(unsigned char byte) {
    if ((byte & 0x80) == 0) return 1;
    if ((byte & 0xE0) == 0xC0) return 2;
    if ((byte & 0xF0) == 0xE0) return 3;
    if ((byte & 0xF8) == 0xF0) return 4;
    return 1;
}

// === РАБОТА С БАЗОЙ ДАННЫХ ===
void save_message_to_db(sqlite3 *db, const char *message);
void IOmemory(sqlite3 *db, const char* user_message);
void save_fractal_spike_to_db(sqlite3 *db, FractalSpike *spike);
void cleanup_old_data(sqlite3 *db);

// === АДАПТИВНАЯ ЛОГИКА ===
float get_fractal_rarity(sqlite3 *db, float dim);
int get_active_spike_count(sqlite3 *db);
void maybe_spawn_new_neuron(sqlite3 *db, FractalSpike *trigger_spike);
float calculate_semantic_weight(const char* text, float base_weight);

// === УЛУЧШЕННЫЕ ФУНКЦИИ ИЗ KERNEL.H ===
void initialize_ai_system();
void cleanup_ai_system();
float get_enhanced_activation(FractalHashCache* cache, const char* pattern, 
                             float dimension, float intensity);
void optimize_system_resonance(NeuralResonance* resonance, float performance);
void load_resonance_state(sqlite3* db);
void save_resonance_state(sqlite3* db);

// === НОВЫЕ ФУНКЦИИ ДЛЯ МНОГОСТРОЧНОГО ВВОДА И РЕКУРСИВНОЙ ГЕНЕРАЦИИ ===
int read_multiline_input(char* buffer, int max_size);
char* generate_recursive_response(sqlite3 *db, const char* base_seed, float activation, 
                                 int depth, int max_depth, FractalActivation* act);

// ===================================================================

int main(void)
{
    srand((unsigned int)time(NULL));
    
    // Инициализация улучшенной AI системы
    initialize_ai_system();

    sqlite3 *db;
    int result = sqlite3_open(DB_NAME, &db);
    if (result != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        cleanup_ai_system();
        return 1;
    }

    // Создание таблиц
    const char *sql_create_memory = "CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY, ch TEXT, created_at INTEGER);";
    const char *sql_create_spikes = "CREATE TABLE IF NOT EXISTS spikes ("
                                    "id INTEGER PRIMARY KEY, "
                                    "timestamp INTEGER, "
                                    "intensity REAL, "
                                    "source TEXT, "
                                    "fractalDimension REAL, "
                                    "propagationPath TEXT);";
    const char *sql_create_resonance = "CREATE TABLE IF NOT EXISTS resonance_stats ("
                                      "id INTEGER PRIMARY KEY, "
                                      "frequency REAL, "
                                      "amplitude REAL, "
                                      "damping REAL, "
                                      "performance REAL, "
                                      "timestamp INTEGER);";
                                      
    char *err_msg = 0;

    result = sqlite3_exec(db, sql_create_memory, 0, 0, &err_msg);
    if (result != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
    }

    result = sqlite3_exec(db, sql_create_spikes, 0, 0, &err_msg);
    if (result != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
    }

    result = sqlite3_exec(db, sql_create_resonance, 0, 0, &err_msg);
    if (result != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
    }

    // Загрузка предыдущего состояния резонанса
    load_resonance_state(db);

    printf("NAIC>> Система инициализирована. Готов к многострочному вводу!\n");
    printf("      (вводите текст, пустая строка завершает ввод)\n\n");

    while (1) {
        cleanup_old_data(db);

        char message[MAX_DATA_INPUT] = "";
        
        // Многострочный ввод
        int chars_read = read_multiline_input(message, sizeof(message));
        
        if (chars_read > 0) {
            if (strcmp(message, "/exit") == 0) {
                printf("NAIC>> Пока! Активация: 0.0\n");
                break;
            }

            int is_trusted = 0;
            float intensity = 0.8f;
            if (strncmp(message, "!!", 2) == 0) {
                is_trusted = 1;
                intensity = 1.0f;
                memmove(message, message + 2, strlen(message + 2) + 1);
            }

            save_message_to_db(db, message);

            long timestamp = time(NULL);
            char* path[] = {"user_input", "memory", "spike"};
            if (is_trusted) {
                path[0] = "trusted_input";
            }

            // Использование улучшенной активации через кэш
            float enhanced_intensity = get_enhanced_activation(global_cache, message, 1.5f, intensity);
            
            FractalSpike* user_spike = create_fractal_spike(timestamp, enhanced_intensity, message, 1.5f, path, 3);
            save_fractal_spike_to_db(db, user_spike);

            // Адаптивное рождение узлов с улучшенной логикой
            maybe_spawn_new_neuron(db, user_spike);

            destroy_fractal_spike(user_spike);

            IOmemory(db, message);
        } else if (chars_read == 0) {
            // Пустой ввод - продолжаем цикл
            continue;
        } else {
            fprintf(stderr, "Error reading input.\n");
            break;
        }
    }

    // Сохранение состояния резонанса перед выходом
    save_resonance_state(db);
    
    sqlite3_close(db);
    cleanup_ai_system();
    return 0;
}

// ===================================================================
// === РЕАЛИЗАЦИЯ УЛУЧШЕННЫХ ФУНКЦИЙ ===

void initialize_ai_system() {
    global_cache = create_fractal_hash_cache(1000); // Кэш на 1000 записей
    global_resonance = create_neural_resonance(1.0f, 0.5f, 0.01f);
}

void cleanup_ai_system() {
    if (global_cache) destroy_fractal_hash_cache(global_cache);
    if (global_resonance) destroy_neural_resonance(global_resonance);
}

float get_enhanced_activation(FractalHashCache* cache, const char* pattern, 
                             float dimension, float intensity) {
    if (!cache || !pattern) return intensity;
    
    // Поиск в кэше для адаптивного обучения
    FractalHashEntry* entry = hash_cache_lookup(cache, pattern, dimension, intensity);
    if (entry) {
        // Бонус за попадание в кэш
        return intensity * (1.0f + HASH_CACHE_HIT_BONUS);
    }
    
    return intensity;
}

void optimize_system_resonance(NeuralResonance* resonance, float performance) {
    if (!resonance) return;
    
    // Адаптация параметров резонанса на основе производительности
    update_resonance_parameters(resonance, performance);
    
    // Оптимизация энергии системы
    if (performance > 0.7f) {
        resonance->amplitude *= 1.05f; // Увеличиваем амплитуду при хорошей производительности
    } else if (performance < 0.3f) {
        resonance->damping *= 1.1f; // Увеличиваем демпфирование при плохой производительности
    }
    
    CLAMP(resonance->amplitude);
    CLAMP(resonance->damping);
}

void load_resonance_state(sqlite3* db) {
    const char* sql = "SELECT frequency, amplitude, damping FROM resonance_stats ORDER BY timestamp DESC LIMIT 1;";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            if (global_resonance) {
                global_resonance->frequency = (float)sqlite3_column_double(stmt, 0);
                global_resonance->amplitude = (float)sqlite3_column_double(stmt, 1);
                global_resonance->damping = (float)sqlite3_column_double(stmt, 2);
            }
        }
        sqlite3_finalize(stmt);
    }
}

void save_resonance_state(sqlite3* db) {
    if (!global_resonance) return;
    
    const char* sql = "INSERT INTO resonance_stats (frequency, amplitude, damping, performance, timestamp) VALUES (?, ?, ?, ?, ?);";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_double(stmt, 1, (double)global_resonance->frequency);
        sqlite3_bind_double(stmt, 2, (double)global_resonance->amplitude);
        sqlite3_bind_double(stmt, 3, (double)global_resonance->damping);
        sqlite3_bind_double(stmt, 4, 0.5); // Базовая производительность
        sqlite3_bind_int64(stmt, 5, time(NULL));
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
}

float calculate_semantic_weight(const char* text, float base_weight) {
    if (!text) return base_weight;
    
    // Простой анализ семантической значимости текста
    int length = strlen(text);
    float complexity = 0.0f;
    
    // Учет длины и разнообразия символов
    int unique_chars = 0;
    int char_counts[256] = {0};
    
    for (int i = 0; i < length; i++) {
        unsigned char c = text[i];
        if (char_counts[c] == 0) unique_chars++;
        char_counts[c]++;
    }
    
    if (length > 0) {
        complexity = (float)unique_chars / length;
        // Более сложные тексты получают больший вес
        return base_weight * (0.7f + complexity * 0.3f);
    }
    
    return base_weight;
}

// === МНОГОСТРОЧНЫЙ ВВОД ===
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
        
        // Пустая строка (только Enter) завершает ввод
        if (line_len == 1 && line[0] == '\n') {
            if (line_count > 0) {
                break; // Завершаем если это не первая строка
            } else {
                printf("...... "); // Для первой строки показываем продолжение
                fflush(stdout);
                continue;
            }
        }
        
        // Добавляем строку к буферу
        if (total_chars + line_len < max_size - 1) {
            strcat(buffer, line);
            total_chars += line_len;
            line_count++;
        } else {
            strncat(buffer, line, max_size - total_chars - 1);
            break;
        }
        
        // Показываем индикатор продолжения
        if (total_chars < max_size - 10) {
            printf("...... ");
            fflush(stdout);
        }
    }
    
    // Убираем завершающий перевод строки если есть
    if (total_chars > 0 && buffer[total_chars - 1] == '\n') {
        buffer[total_chars - 1] = '\0';
        total_chars--;
    }
    
    return total_chars;
}

void save_message_to_db(sqlite3 *db, const char *message)
{
    if (!message) return;
    const char *sql = "INSERT INTO memory (ch, created_at) VALUES (?, ?);";
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Prepare failed: %s\n", sqlite3_errmsg(db));
        return;
    }

    long now = time(NULL);
    for (size_t i = 0; message[i] != '\0'; ) {
        int len = utf8_char_length((unsigned char)message[i]);
        if (len <= 0 || len > 4) len = 1;
        char temp[5] = {0};
        strncpy(temp, &message[i], len);
        temp[len] = '\0';

        sqlite3_bind_text(stmt, 1, temp, len, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 2, now);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
        i += len;
    }
    sqlite3_finalize(stmt);
}

void save_fractal_spike_to_db(sqlite3 *db, FractalSpike *spike)
{
    const char *sql = "INSERT INTO spikes (timestamp, intensity, source, fractalDimension, propagationPath) VALUES (?, ?, ?, ?, ?);";
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Prepare failed: %s\n", sqlite3_errmsg(db));
        return;
    }

    sqlite3_bind_int64(stmt, 1, spike->timestamp);
    sqlite3_bind_double(stmt, 2, (double)spike->intensity);
    sqlite3_bind_text(stmt, 3, spike->source, -1, SQLITE_STATIC);
    sqlite3_bind_double(stmt, 4, (double)spike->fractalDimension);

    char path_str[1000] = "";
    for (int i = 0; i < spike->pathSize; i++) {
        if (i > 0) strcat(path_str, ",");
        strncat(path_str, spike->propagationPath[i], sizeof(path_str) - strlen(path_str) - 1);
    }
    sqlite3_bind_text(stmt, 5, path_str, -1, SQLITE_STATIC);

    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

void cleanup_old_data(sqlite3 *db)
{
    long cutoff = time(NULL) - MAX_SPIKE_AGE_SECONDS;
    char sql[200];
    snprintf(sql, sizeof(sql), "DELETE FROM memory WHERE created_at < %ld;", cutoff);
    sqlite3_exec(db, sql, 0, 0, 0);
    snprintf(sql, sizeof(sql), "DELETE FROM spikes WHERE timestamp < %ld;", cutoff);
    sqlite3_exec(db, sql, 0, 0, 0);
}

float get_fractal_rarity(sqlite3 *db, float dim) {
    char sql[256];
    sqlite3_stmt *stmt;
    snprintf(sql, sizeof(sql),
        "SELECT COUNT(*) FROM spikes WHERE ABS(fractalDimension - %.2f) < 0.2;", dim);
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        sqlite3_finalize(stmt);
        return 1.0f;
    }
    
    float count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = (float)sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return 1.0f / (1.0f + count * 0.1f);
}

int get_active_spike_count(sqlite3 *db) {
    const char *sql = "SELECT COUNT(*) FROM spikes WHERE timestamp > ?";
    sqlite3_stmt *stmt;
    long cutoff = time(NULL) - MAX_SPIKE_AGE_SECONDS / 2;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        sqlite3_finalize(stmt);
        return 1000;
    }
    
    sqlite3_bind_int64(stmt, 1, cutoff);
    int count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return count;
}

void maybe_spawn_new_neuron(sqlite3 *db, FractalSpike *trigger_spike) {
    int active_count = get_active_spike_count(db);
    if (active_count > 200) return;

    float rarity = get_fractal_rarity(db, trigger_spike->fractalDimension);
    float birth_prob = trigger_spike->intensity * rarity * 0.4f;

    if ((float)rand() / RAND_MAX < birth_prob) {
        char newborn_source[256];
        snprintf(newborn_source, sizeof(newborn_source),
                 "[auto:%ld:%.2f]", trigger_spike->timestamp, trigger_spike->fractalDimension);
        
        float new_dim = trigger_spike->fractalDimension + 
                       ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
        if (new_dim < 1.0f) new_dim = 1.0f;
        if (new_dim > 2.0f) new_dim = 2.0f;
        
        char* newborn_path[] = {"auto_neuron", "resonance", "birth"};
        FractalSpike* newborn = create_fractal_spike(
            time(NULL),
            0.5f + (float)rand() / RAND_MAX * 0.3f,
            newborn_source,
            new_dim,
            newborn_path,
            3
        );
        save_fractal_spike_to_db(db, newborn);
        printf("  [рождён новый узел: %.2f]\n", new_dim);
        destroy_fractal_spike(newborn);
    }
}

// === РЕКУРСИВНАЯ ГЕНЕРАЦИЯ ОТВЕТА ===
char* generate_recursive_response(sqlite3 *db, const char* base_seed, float activation, 
                                 int depth, int max_depth, FractalActivation* act) {
    if (depth >= max_depth || activation < 0.1f) {
        char* result = malloc(MAX_RESPONSE_LENGTH + 1);
        if (result) {
            strncpy(result, base_seed, MAX_RESPONSE_LENGTH);
            result[MAX_RESPONSE_LENGTH] = '\0';
        }
        return result;
    }
    
    // Генерация нового семени на основе предыдущего
    char new_seed[MAX_RESPONSE_LENGTH * 2] = "";
    strncat(new_seed, base_seed, MAX_RESPONSE_LENGTH);
    
    // Добавляем случайные данные из памяти
    char sql[256];
    int sample_size = 10 + (int)(activation * 30);
    snprintf(sql, sizeof(sql), 
             "SELECT ch FROM memory WHERE ch != '' ORDER BY RANDOM() LIMIT %d;", 
             sample_size);
    
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* ch = (const char*)sqlite3_column_text(stmt, 0);
            if (ch && strlen(new_seed) + strlen(ch) < sizeof(new_seed) - 1) {
                strcat(new_seed, ch);
            }
        }
        sqlite3_finalize(stmt);
    }
    
    // Рекурсивный вызов с уменьшенной активацией
    float new_activation = activation * 0.7f; // Уменьшаем активацию с каждой глубиной
    char* recursive_result = generate_recursive_response(db, new_seed, new_activation, 
                                                        depth + 1, max_depth, act);
    
    // Комбинируем результаты
    char* final_result = malloc(MAX_RESPONSE_LENGTH * 2);
    if (final_result) {
        snprintf(final_result, MAX_RESPONSE_LENGTH * 2, "%s|%s", 
                 base_seed, recursive_result ? recursive_result : "");
        
        // Обрезаем до максимальной длины
        if (strlen(final_result) > MAX_RESPONSE_LENGTH) {
            final_result[MAX_RESPONSE_LENGTH] = '\0';
        }
        
        free(recursive_result);
    }
    
    return final_result ? final_result : strdup(base_seed);
}

void IOmemory(sqlite3 *db, const char* user_message)
{
    // Использование улучшенной активации через кэш
    float base_activation = get_enhanced_activation(global_cache, user_message, 1.5f, 0.85f);
    
    FractalActivation* act = create_fractal_activation(base_activation, 0.70f, 0.75f, 4, 0.3f);

    // Применение резонанса к активации
    if (global_resonance) {
        apply_resonance_to_activation(act, global_resonance);
    }

    const char *sql = "SELECT intensity FROM spikes ORDER BY timestamp DESC LIMIT 1;";
    sqlite3_stmt *stmt;
    float user_intensity = 0.8f;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW) {
        user_intensity = (float)sqlite3_column_double(stmt, 0);
    }
    sqlite3_finalize(stmt);

    // === ФРАКТАЛЬНОЕ ОБРАТНОЕ РАСПРОСТРАНЕНИЕ ===
    FractalBackprop* bp = create_fractal_backprop(act->fractalDepth);
    float learning_rate = user_intensity * 0.01f;
    
    // Прямой проход
    fractal_gradient_descent(act, learning_rate);
    float total = get_total_activation(act);
    
    // Обратный проход (обучение на основе результата)
    float target_activation = 0.7f; // Целевой уровень активации
    fractal_backward_pass(bp, act, target_activation, total);
    apply_fractal_gradients(act, bp);
    
    // Обновляем общую активацию после обучения
    total = get_total_activation(act);
    // === КОНЕЦ ОБРАТНОГО РАСПРОСТРАНЕНИЯ ===

    // Оптимизация резонанса на основе производительности
    if (global_resonance) {
        optimize_system_resonance(global_resonance, total);
        
        // Онлайн-обучение на лету
        fractal_online_learning(global_cache, global_resonance, 
                              user_message, total, target_activation, 1.5f);
    }

    int depth = (int)(act->fractalDepth * total);
    if (depth < 1) depth = 1;
    if (depth > 8) depth = 8;

    // Базовый сбор данных из памяти
    char combined[3000] = "";
    char sql2[256];
    int sample_count = depth * 50;
    snprintf(sql2, sizeof(sql2), "SELECT ch FROM memory ORDER BY RANDOM() LIMIT %d;", sample_count);
    
    sqlite3_stmt *stmt2;
    rc = sqlite3_prepare_v2(db, sql2, -1, &stmt2, NULL);
    if (rc == SQLITE_OK) {
        while ((rc = sqlite3_step(stmt2)) == SQLITE_ROW) {
            const char* ch = (const char*)sqlite3_column_text(stmt2, 0);
            if (ch && strlen(combined) + strlen(ch) < sizeof(combined) - 1) {
                strcat(combined, ch);
            }
        }
        sqlite3_finalize(stmt2);
    }

    int symbol_count = strlen(combined);
    int base_window = 5 + (int)(total * 30);
    int dynamic_factor = (int)(user_intensity * 8);
    int window = base_window + dynamic_factor;
    if (window > MAX_RESPONSE_LENGTH) window = MAX_RESPONSE_LENGTH;
    if (window < 5) window = 5;
    if (window > symbol_count) window = symbol_count;

    char bot_response[MAX_RESPONSE_LENGTH + 1] = "";
    
    if (symbol_count > 0 && window > 0) {
        // Рекурсивная генерация ответа
        char* initial_seed = malloc(window + 1);
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
            
            // Запускаем рекурсивную генерацию
            char* recursive_response = generate_recursive_response(db, initial_seed, total, 
                                                                  0, RECURSION_DEPTH, act);
            
            if (recursive_response) {
                strncpy(bot_response, recursive_response, MAX_RESPONSE_LENGTH);
                bot_response[MAX_RESPONSE_LENGTH] = '\0';
                free(recursive_response);
            }
            
            free(initial_seed);
        }
    }

    // Фолбэки если генерация не удалась
    if (strlen(bot_response) == 0) {
        const char* fallbacks[] = {
            "Размышляю над этим...", "Интересный вопрос!", 
            "Позвольте подумать...", "Ммм, нужно обдумать...",
            "Понимаю вашу мысль...", "Это требует анализа..."
        };
        strcpy(bot_response, fallbacks[rand() % 6]);
    }

    // Дополнительная проверка длины
    if (strlen(bot_response) > MAX_RESPONSE_LENGTH) {
        bot_response[MAX_RESPONSE_LENGTH] = '\0';
    }

    // === ОЦЕНКА КАЧЕСТВА И ДОПОЛНИТЕЛЬНОЕ ОБУЧЕНИЕ ===
    float response_quality = 0.5f;
    int response_length = strlen(bot_response);
    
    // Простая оценка качества ответа
    if (response_length > 10 && response_length < MAX_RESPONSE_LENGTH - 10) {
        response_quality = 0.7f;
    }
    if (response_length > 30) {
        response_quality = 0.8f;
    }
    
    // Учет уникальности символов
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

    // Дополнительное обучение на основе качества ответа
    if (global_resonance) {
        fractal_online_learning(global_cache, global_resonance, 
                              user_message, total, response_quality, 1.5f);
        
        // Второй проход обратного распространения с учетом качества
        fractal_backward_pass(bp, act, response_quality, total);
        apply_fractal_gradients(act, bp);
    }

    if (total < 0.15f) {
        printf("NAIC>> ... (активация низкая: %.2f)\n", total);
    } else {
        printf("NAIC>> %s [активация: %.2f, качество: %.2f]\n", 
               bot_response, total, response_quality);
    }

    long timestamp = time(NULL);
    char* path[] = {"output", "fractal_response", "recursive_generation"};
    FractalSpike* bot_spike = create_fractal_spike(timestamp, total, bot_response, 1.6f, path, 3);
    save_fractal_spike_to_db(db, bot_spike);
    
    // Сохранение в кэш для будущего использования
    if (global_cache) {
        hash_cache_store(global_cache, user_message, 1.5f, total, total);
        
        // Сохраняем также информацию о качестве
        char quality_key[256];
        snprintf(quality_key, sizeof(quality_key), "quality:%s", user_message);
        hash_cache_store(global_cache, quality_key, 1.5f, response_quality, response_quality);
    }
    
    destroy_fractal_spike(bot_spike);
    destroy_fractal_backprop(bp);
    destroy_fractal_activation(act);
}

// Вспомогательная функция для оценки качества (добавить в main.c перед IOmemory)
float evaluate_response_quality_simple(const char* response, float activation) {
    if (!response || strlen(response) == 0) return 0.1f;
    
    int len = strlen(response);
    float quality = 0.3f; // Базовая оценка
    
    // Оценка длины
    if (len > 5 && len < 50) quality += 0.2f;
    else if (len >= 50 && len < 100) quality += 0.3f;
    else if (len >= 100) quality += 0.1f; // Слишком длинные могут быть плохи
    
    // Оценка разнообразия
    int unique = 0;
    int counts[256] = {0};
    for (int i = 0; i < len; i++) {
        unsigned char c = response[i];
        if (counts[c] == 0) unique++;
        counts[c]++;
    }
    quality += ((float)unique / len) * 0.3f;
    
    // Учет активации
    quality *= (0.4f + activation * 0.6f);
    
    CLAMP(quality);
    return quality;
}

// Функция оценки качества ответа для обучения с подкреплением
float evaluate_response_quality(const char* user_message, const char* bot_response, 
                               float activation_level) {
    if (!bot_response || strlen(bot_response) == 0) return 0.1f;
    
    float quality = 0.5f; // Базовая оценка
    
    // Оценка длины ответа
    int response_len = strlen(bot_response);
    if (response_len > 5 && response_len < 100) {
        quality += 0.2f;
    }
    
    // Оценка разнообразия символов
    int unique_chars = 0;
    int char_counts[256] = {0};
    for (int i = 0; i < response_len; i++) {
        unsigned char c = bot_response[i];
        if (char_counts[c] == 0) unique_chars++;
        char_counts[c]++;
    }
    
    float diversity = (float)unique_chars / response_len;
    quality += diversity * 0.3f;
    
    // Учет уровня активации
    quality *= (0.3f + activation_level * 0.7f);
    
    // ДОБАВИМ ИСПОЛЬЗОВАНИЕ user_message чтобы убрать warning
    if (user_message && strlen(user_message) > 0) {
        // Простая проверка релевантности: если ответ содержит слова из запроса
        int matches = 0;
        char user_lower[256], bot_lower[256];
        
        // Копируем и переводим в нижний регистр для сравнения
        strncpy(user_lower, user_message, sizeof(user_lower)-1);
        strncpy(bot_lower, bot_response, sizeof(bot_lower)-1);
        user_lower[sizeof(user_lower)-1] = '\0';
        bot_lower[sizeof(bot_lower)-1] = '\0';
        
        // Приводим к нижнему регистру
        for (int i = 0; user_lower[i]; i++) user_lower[i] = tolower(user_lower[i]);
        for (int i = 0; bot_lower[i]; i++) bot_lower[i] = tolower(bot_lower[i]);
        
        // Проверяем совпадения ключевых слов
        char* token = strtok(user_lower, " ,.!?;:");
        while (token && strlen(token) > 2) { // Слова длиннее 2 символов
            if (strstr(bot_lower, token)) {
                matches++;
            }
            token = strtok(NULL, " ,.!?;:");
        }
        
        if (matches > 0) {
            quality += matches * 0.1f; // Бонус за релевантность
        }
    }
    
    CLAMP(quality);
    return quality;
}