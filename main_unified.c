#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <time.h>
#include <locale.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <assert.h>
#include <signal.h>
#include <execinfo.h>
#include "core.h"

#define MAX_INPUT_LINES 100
#define MAX_LINE_LENGTH 256

// ===== Глобальные режимы =====
static int debug_mode = 0;      // 0 - обычный, 1 - отладка
static int line_counter = 1;    // Счетчик строк для обычного режима

// ===== Система оценки разнообразия =====
static uint8_t last_response_hash = 0;
static uint8_t diversity_score = 128; // 0-255, выше = больше разнообразия
static uint8_t repetition_penalty = 0;

// ===== Обработчик сигналов (только в debug режиме) =====
void signal_handler(int sig) {
    if (!debug_mode) return;
    
    void *array[20];
    size_t size;
    
    printf("\n!!! SEGMENTATION FAULT DETECTED !!!\n");
    printf("Signal: %d\n", sig);
    
    size = backtrace(array, 20);
    printf("Stack trace:\n");
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    
    printf("\n[ЭКСТРЕННОЕ СОХРАНЕНИЕ...]\n");
    save_state_to_file("crash_dump.bin");
    
    exit(1);
}

// ===== Функции режима отладки =====
void debug_printf(const char* format, ...) {
    if (!debug_mode) return;
    
    va_list args;
    va_start(args, format);
    
    printf("[DEBUG] ");
    vprintf(format, args);
    
    va_end(args);
    fflush(stdout);
}

void show_debug_info(void) {
    if (!debug_mode) return;
    
    printf("\n=== DEBUG INFO ===\n");
    printf("tnsrs array start: %p\n", (void*)tnsrs);
    printf("lnks array start: %p\n", (void*)lnks);
    printf("memo array start: %p\n", (void*)memo);
    printf("working_mem array start: %p\n", (void*)working_mem);
    printf("System time: %u\n", (uint32_t)time(NULL));
    
    if (tnsr_count > 0) {
        printf("\nFirst tensor (%p):\n", (void*)&tnsrs[0]);
        printf("  data pointer: %p\n", (void*)tnsrs[0].data);
        printf("  rows: %u, cols: %u\n", tnsrs[0].rows, tnsrs[0].cols);
        printf("  act: %u, res: %u, ent: %u\n", tnsrs[0].act, tnsrs[0].res, tnsrs[0].ent);
    }
    
    printf("Debug mode: %s\n", debug_mode ? "ON" : "OFF");
    printf("Diversity score: %u, Repetition penalty: %u\n", diversity_score, repetition_penalty);
    
    // Новая информация о самоорганизации
    printf("Кластеры: %u, Эпизоды: %u, Концепции: %u\n", 
           cluster_count, episode_count, concept_count);
    printf("Глобальный хеш контекста: %08X\n", global_context_hash);
    printf("Включена самоорганизация: %s\n", goals.self_organization_enabled ? "ДА" : "НЕТ");
}

// ===== Система разнообразия =====
uint8_t calculate_response_hash(const char* response) {
    if (!response || !*response) return 0;
    
    uint8_t hash = 0;
    for (int i = 0; response[i] && i < 20; i++) {
        hash = (hash * 31 + response[i]) % 256;
    }
    return hash;
}

void update_diversity_score(uint8_t current_hash, uint8_t thought_strength) {
    if (current_hash == last_response_hash) {
        // То же самое что и в прошлый раз - плохо
        repetition_penalty = repetition_penalty < 200 ? repetition_penalty + 10 : 200;
        diversity_score = diversity_score > 20 ? diversity_score - 15 : 20;
    } else {
        // Новый ответ - хорошо
        if (repetition_penalty > 0) {
            repetition_penalty = repetition_penalty > 10 ? repetition_penalty - 10 : 0;
        }
        diversity_score = diversity_score < 235 ? diversity_score + 20 : 255;
    }
    
    last_response_hash = current_hash;
    
    if (debug_mode) {
        printf("[DIVERSITY] score=%u, penalty=%u, hash=%u\n", 
               diversity_score, repetition_penalty, current_hash);
    }
}

uint32_t apply_diversity_adjustment(uint32_t original_score) {
    float multiplier = (float)diversity_score / 128.0f; // 0.5-2.0
    float penalty_reduction = (float)repetition_penalty / 255.0f; // 0.0-0.8
    
    if (repetition_penalty > 100) {
        multiplier *= (1.0f - penalty_reduction * 0.8f);
    }
    
    return (uint32_t)(original_score * multiplier);
}

void get_diverse_response(BitTensor* active_thought, uint8_t* strength_output, char* text_output, size_t max_len) {
    if (!active_thought || !active_thought->data) {
        *strength_output = 0;
        text_output[0] = '\0';
        return;
    }
    
    uint32_t total_bits = active_thought->rows * active_thought->cols;
    uint32_t total_bytes = (total_bits + 7) / 8;
    
    // Базовые параметры
    uint32_t base_strength = (active_thought->act * active_thought->efficiency * active_thought->res) / 65025;
    base_strength = base_strength > 100 ? 100 : base_strength;
    
    // Ищем связанные мысли с хорошим резонансом
    BitTensor* best_alternative = NULL;
    uint32_t best_alt_score = 0;
    
    if (diversity_score < 100 && repetition_penalty > 50) {
        // Ищем альтернативные мысли
        for (uint16_t i = 0; i < lnk_count; i++) {
            if (lnks[i].src == active_thought && lnks[i].strength > 150) {
                BitTensor* alt = lnks[i].tgt;
                if (alt && alt != active_thought && alt->act > 50) {
                    uint32_t alt_score = alt->act * alt->efficiency * lnks[i].strength;
                    if (alt_score > best_alt_score) {
                        best_alt_score = alt_score;
                        best_alternative = alt;
                    }
                }
            }
        }
    }
    
    // Выбираем что выводить
    BitTensor* output_tensor = active_thought;
    if (best_alternative && best_alt_score > (base_strength * 10000)) {
        output_tensor = best_alternative;
        if (debug_mode) printf("[DIVERSE] Using alternative thought\n");
    }
    
    // Пересчитываем для выбранного тензора
    uint32_t final_strength = (output_tensor->act * output_tensor->efficiency * output_tensor->res) / 65025;
    final_strength = final_strength > 100 ? 100 : final_strength;
    
    // Применяем коррекцию разнообразия
    final_strength = apply_diversity_adjustment(final_strength);
    if (final_strength > 100) final_strength = 100;
    
    // Извлекаем текст
    int printed = 0;
    uint32_t bytes_to_check = total_bytes < 100 ? total_bytes : 100;
    
    for (uint32_t i = 0; i < bytes_to_check && printed < (int)max_len - 1; i++) {
        uint8_t c = output_tensor->data[i];
        
        // Меняем стратегию в зависимости от разнообразия
        if (diversity_score < 80) {
            // Больше разнообразия - показываем разные символы
            if (c >= 32 && c <= 126) {
                text_output[printed++] = c;
            } else if (c == 0) {
                // Пропускаем нули
                continue;
            } else if (printed < (int)max_len - 4) {
                // Для непечатаемых - hex
                sprintf(text_output + printed, "[%02X]", c);
                printed += 4;
            }
        } else {
            // Стандартный подход
            if (c >= 32 && c <= 126) {
                text_output[printed++] = c;
            }
        }
    }
    
    text_output[printed] = '\0';
    
    // Вычисляем хэш ответа для следующего сравнения
    uint8_t response_hash = calculate_response_hash(text_output);
    update_diversity_score(response_hash, final_strength);
    
    *strength_output = (uint8_t)final_strength;
}

// ===== Утилиты =====
int file_exists(const char *filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

void encode_utf8_to_binary(const char* utf8_str, uint8_t* binary_output, size_t* output_len, size_t max_len) {
    if (!utf8_str || !binary_output || !output_len || max_len == 0) return;
    const unsigned char* ptr = (const unsigned char*)utf8_str;
    size_t pos = 0;
    while (*ptr && pos < max_len) {
        unsigned char c = *ptr;
        uint8_t char_len = 0;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;
        else { ptr++; continue; }
        for (uint8_t i = 0; i < char_len && *ptr && pos < max_len; i++) {
            binary_output[pos++] = *ptr++;
        }
    }
    *output_len = pos;
}

// ===== Многострочный ввод для обычного режима =====
int read_input_multiline(char* buffer, size_t max_len, int show_prompt) {
    if (!buffer || max_len == 0) return 0;
    
    buffer[0] = '\0';
    size_t total_len = 0;
    char line[256];
    int line_number = 1;
    int empty_line_count = 0;
    
    if (show_prompt) {
        printf("(%d)you>> ", line_counter);
        fflush(stdout);
    }
    
    while (1) {
        if (!fgets(line, sizeof(line), stdin)) {
            if (total_len > 0) break;
            return 0;
        }
        
        size_t line_len = strlen(line);
        if (line_len > 0 && line[line_len - 1] == '\n') {
            line[--line_len] = '\0';
        }
        
        if (line_len == 0) {
            empty_line_count++;
            if (empty_line_count >= 1) break; // Один пустой строка для завершения в обычном режиме
            continue;
        }
        
        empty_line_count = 0;
        
        // Проверяем, не команда ли это
        if (line[0] == '/') {
            // Если это команда, и у нас уже есть ввод - завершаем текущий ввод
            if (total_len > 0) {
                // Сохраняем буфер и возвращаем команду для обработки отдельно
                // Для простоты просто завершаем ввод
                break;
            } else {
                // Это чистая команда, копируем её
                if (total_len + line_len + 1 < max_len) {
                    strcpy(buffer, line);
                    return 1;
                }
            }
        }
        
        if (total_len + line_len + 2 < max_len) {
            if (total_len > 0) {
                buffer[total_len++] = ' ';
            }
            strcpy(buffer + total_len, line);
            total_len += line_len;
            line_number++;
        } else {
            printf("[!] Превышен лимит ввода\n");
            break;
        }
        
        // В обычном режиме не показываем дополнительные промпты
    }
    
    return total_len > 0 ? 1 : 0;
}

// ===== Старый ввод для режима отладки =====
int read_input_debug(char* buffer, size_t max_len, const char* prompt) {
    if (!buffer || max_len == 0) return 0;
    buffer[0] = '\0';
    size_t total_len = 0;
    char line[256];
    int line_number = 0;
    int empty_line_count = 0;
    printf("%s (двойной Enter для отправки):\n", prompt);
    while (1) {
        if (line_number > 0) printf("%d> ", line_number + 1);
        else printf("> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) {
            if (total_len > 0) break;
            return 0;
        }
        size_t line_len = strlen(line);
        if (line_len > 0 && line[line_len - 1] == '\n') {
            line[--line_len] = '\0';
        }
        if (line_len == 0) {
            empty_line_count++;
            if (empty_line_count >= 2) break;
            if (total_len > 0 && total_len + 1 < max_len) {
                buffer[total_len++] = '\n';
                buffer[total_len] = '\0';
                line_number++;
            }
            continue;
        }
        empty_line_count = 0;
        if (total_len + line_len + 2 < max_len) {
            if (total_len > 0) buffer[total_len++] = '\n';
            strcpy(buffer + total_len, line);
            total_len += line_len;
            line_number++;
        } else {
            printf("[!] Превышен лимит\n");
            break;
        }
    }
    if (total_len > 0 && buffer[total_len - 1] == '\n') {
        buffer[--total_len] = '\0';
    }
    return total_len > 0 ? 1 : 0;
}

// ===== Основная логика =====
void generate_response(void) {
    debug_printf("Entering generate_response()\n");
    
    if (tnsr_count == 0 || working_mem_count == 0) {
        if (debug_mode) {
            printf("[Система]: Нет активных мыслей\n");
        } else {
            printf("nai>>0\n");
        }
        return;
    }
    
    BitTensor* active_thought = find_significant_tensor(SEARCH_RESONANT, NULL);
    
    if (!active_thought || active_thought->act < 30) {
        active_thought = find_significant_tensor(SEARCH_MOST_ACTIVE, NULL);
    }
    
    if (!active_thought || active_thought->act < 30) {
        active_thought = find_significant_tensor(SEARCH_EFFICIENT, NULL);
    }
    
    if (!active_thought || active_thought->act < 20) {
        if (debug_mode) {
            printf("[Система]: Мысли слишком слабые\n");
        } else {
            printf("nai>>0\n");
        }
        return;
    }
    
    if (!active_thought->data) {
        if (debug_mode) {
            printf("[ERROR] Active thought has no data!\n");
        }
        return;
    }
    
    // ===== РАЗНЫЙ ВЫВОД ДЛЯ РЕЖИМОВ =====
    if (debug_mode) {
        // DEBUG MODE: подробный вывод
        printf("\n[МЫШЛЕНИЕ] ");
        
        uint32_t total_bits = active_thought->rows * active_thought->cols;
        uint32_t total_bytes = (total_bits + 7) / 8;
        
        debug_printf("Tensor size: %ux%u (%u bytes)\n", 
                     active_thought->rows, active_thought->cols, total_bytes);
        
        if (total_bytes > 0) {
            int printed = 0;
            for (uint32_t i = 0; i < total_bytes && printed < 100; i++) {
                uint8_t c = active_thought->data[i];
                if (c >= 32 && c <= 126) {
                    putchar(c);
                    printed++;
                } else if (c == '\n' || c == '\t' || c == '\r') {
                    putchar(' ');
                    printed++;
                } else if (c == 0) {
                    continue;
                } else if (printed < 95) {
                    printf("[%02X]", c);
                    printed += 4;
                }
                
                if (printed >= 100) break;
            }
        }
        
        printf("\n\n[СТАТИСТИКА МЫСЛИ]\n");
        printf("  Активность: %u/255\n", active_thought->act);
        printf("  Эффективность: %u/255\n", active_thought->efficiency);
        printf("  Резонанс: %u/255\n", active_thought->res);
        printf("  Стабильность: %u/255\n", active_thought->stab);
        printf("  Энтропия: %u/255\n", active_thought->ent);
        printf("  Связей: %u\n", active_thought->conn);
        printf("  Размер: %ux%u (%u байт)\n", 
               active_thought->rows, active_thought->cols,
               (active_thought->rows * active_thought->cols + 7) / 8);
        
        if (active_thought->cluster_id != 0) {
            printf("  Кластер: %u, Концепция: %s\n", 
                   active_thought->cluster_id,
                   active_thought->is_concept ? "ДА" : "НЕТ");
        }
        
        printf("\n[СВЯЗАННЫЕ МЫСЛИ]\n");
        int shown_links = 0;
        for (uint16_t i = 0; i < lnk_count && shown_links < 5; i++) {
            if (lnks[i].src == active_thought && lnks[i].strength > 50) {
                BitTensor* linked = lnks[i].tgt;
                if (linked && linked->data) {
                    printf("  → ");
                    int max_bytes = (linked->rows * linked->cols + 7) / 8;
                    if (max_bytes > 20) max_bytes = 20;
                    for (int j = 0; j < max_bytes; j++) {
                        uint8_t c = linked->data[j];
                        if (c >= 32 && c <= 126) putchar(c);
                        else putchar('.');
                    }
                    printf(" (str=%u, type=%d)\n", lnks[i].strength, lnks[i].semantic_type);
                    shown_links++;
                }
            }
        }
        if (shown_links == 0) {
            printf("  Нет сильных связей\n");
        }
    } else {
        // NORMAL MODE: минималистичный вывод с учетом разнообразия
        uint8_t final_strength;
        char response_text[256];
        
        get_diverse_response(active_thought, &final_strength, response_text, sizeof(response_text));
        
        printf("nai>>%u %s\n", final_strength, response_text);
    }
    
    save_tnsr(active_thought);
    active_thought->act = (active_thought->act * 6) / 10;
    active_thought->lu = (uint32_t)time(NULL);
    
    if (debug_mode) printf("\n");
    debug_printf("Exiting generate_response()\n");
}

// ===== Обработка ввода =====
void process_input(const char* input_text) {
    debug_printf("Entering process_input()\n");
    
    if (!input_text || !*input_text) {
        if (debug_mode) printf("[!] Пустой ввод\n");
        return;
    }
    
    size_t input_len = strlen(input_text);
    if (debug_mode) printf("\n[Обработка %zu символов...]\n", input_len);
    
    uint16_t rows = 128;
    uint16_t cols = 128;
    
    debug_printf("Creating tensor %ux%u...\n", rows, cols);
    BitTensor* input_tensor = create_tnsr(rows, cols);
    if (!input_tensor) {
        if (debug_mode) printf("[ERROR] Не создан тензор\n");
        return;
    }
    
    debug_printf("Tensor created at %p\n", (void*)input_tensor);
    
    size_t total_bytes = (rows * cols + 7) / 8;
    size_t copy_len = (input_len < total_bytes) ? input_len : total_bytes;
    
    debug_printf("total_bytes=%zu, copy_len=%zu\n", total_bytes, copy_len);
    
    if (copy_len > 0 && input_tensor->data) {
        uint8_t encoded_data[8192];
        size_t encoded_len = 0;
        encode_utf8_to_binary(input_text, encoded_data, &encoded_len, sizeof(encoded_data));
        
        size_t actual_copy = (encoded_len < copy_len) ? encoded_len : copy_len;
        
        debug_printf("Encoding complete: %zu bytes\n", encoded_len);
        debug_printf("Copying %zu bytes to tensor...\n", actual_copy);
        
        memcpy(input_tensor->data, encoded_data, actual_copy);
        if (debug_mode) {
            printf("[Создан тензор %ux%u (%lu байт), скопировано %zu байт]\n", 
                   rows, cols, (unsigned long)total_bytes, actual_copy);
        }
    }
    
    input_tensor->act = 180 + (rand() % 50);
    input_tensor->res = 150 + (rand() % 80);
    input_tensor->ent = calc_bit_ent(input_tensor, input_tensor->cols);
    input_tensor->efficiency = calculate_efficiency(input_tensor);
    input_tensor->lu = (uint32_t)time(NULL);
    
    if (debug_mode) {
        printf("[Параметры: act=%u, res=%u, ent=%u, eff=%u]\n", 
               input_tensor->act, input_tensor->res, input_tensor->ent, input_tensor->efficiency);
    }
    
    if (tnsr_count > 1) {
        if (debug_mode) printf("[Создание связей...]\n");
        int links_created = 0;
        for (uint16_t i = 0; i < tnsr_count-1 && links_created < 5; i++) {
            if (&tnsrs[i] != input_tensor) {
                debug_printf("Checking similarity with tensor %u...\n", i);
                uint8_t sim = calc_bit_sim(input_tensor, &tnsrs[i]);
                debug_printf("Similarity with tensor %u: %u\n", i, sim);
                if (sim > 30) {
                    debug_printf("Creating link...\n");
                    create_link(input_tensor, &tnsrs[i]);
                    links_created++;
                    if (debug_mode) printf("[Связь с тензором %u: sim=%u]\n", i, sim);
                }
            }
        }
        if (debug_mode) printf("[Создано %d связей]\n", links_created);
    }
    
    debug_printf("Saving tensor to memory...\n");
    save_tnsr(input_tensor);
    
    debug_printf("Adding to working memory...\n");
    add_to_working_memory(input_tensor);
    
    if (debug_mode) printf("[Поиск соответствий...]\n");
    
    debug_printf("Calling find_efficient_match()...\n");
    BitTensor* match = find_efficient_match(input_tensor);
    if (match) {
        debug_printf("Match found at %p\n", (void*)match);
        if (debug_mode) {
            printf("[Найдено соответствие! act=%u, eff=%u, res=%u]\n", 
                   match->act, match->efficiency, match->res);
        }
        debug_printf("Calling fast_contextual_activation()...\n");
        fast_contextual_activation(match);
    } else {
        debug_printf("No match found\n");
    }
    
    debug_printf("Updating thought stream...\n");
    update_thought_stream();
    
    debug_printf("Generating response...\n");
    generate_response();
    
    // Автосохранение только в debug или каждые 10 запросов
    static int request_count = 0;
    request_count++;
    
    if (debug_mode || (request_count % 10 == 0)) {
        if (debug_mode) printf("\n[Автосохранение...]\n");
        if (save_state_to_file("memory.bin") == 0) {
            if (debug_mode) {
                FILE *f = fopen("memory.bin", "rb");
                if (f) {
                    fseek(f, 0, SEEK_END);
                    long size = ftell(f);
                    fclose(f);
                    printf("[Сохранено: %ld байт]\n", size);
                    printf("[Система: тензоры=%u, связи=%u, память=%u]\n", 
                           tnsr_count, lnk_count, memo_size);
                    printf("[Самоорганизация: кластеры=%u, эпизоды=%u, концепции=%u]\n",
                           cluster_count, episode_count, concept_count);
                }
            }
        }
    }
    
    debug_printf("Exiting process_input()\n");
}

// ===== Команды =====
void execute_command(const char* cmd) {
    if (!cmd || !*cmd) return;
    
    if (strcmp(cmd, "/exit") == 0) {
        if (debug_mode) printf("[Сохранение перед выходом...]\n");
        save_state_to_file("memory.bin");
        printf("[Выход]\n");
        exit(0);
    }
    else if (strcmp(cmd, "/debug") == 0) {
        debug_mode = !debug_mode;
        if (debug_mode) {
            printf("[Режим отладки ВКЛЮЧЕН]\n");
            signal(SIGSEGV, signal_handler);
            signal(SIGABRT, signal_handler);
        } else {
            printf("[Режим отладки ВЫКЛЮЧЕН]\n");
            signal(SIGSEGV, SIG_DFL);
            signal(SIGABRT, SIG_DFL);
            line_counter = 1; // Сброс счетчика при переходе в обычный режим
        }
    }
    else if (strcmp(cmd, "/think") == 0) {
        if (debug_mode) printf("[Принудительное мышление...]\n");
        update_thought_stream();
        generate_response();
    }
    else if (strcmp(cmd, "/stats") == 0) {
        printf("\n=== СТАТИСТИКА СИСТЕМЫ ===\n");
        printf("Тензоры: %u\n", tnsr_count);
        printf("Связи: %u\n", lnk_count);
        printf("Память (memo): %u\n", memo_size);
        printf("Рабочая память: %u\n", working_mem_count);
        printf("Резонанс системы: %u\n", sys_res);
        printf("Цель эффективности: %u\n", goals.target_efficiency);
        
        uint16_t high_act = 0, high_eff = 0, high_res = 0;
        for (uint16_t i = 0; i < tnsr_count; i++) {
            if (tnsrs[i].act > 100) high_act++;
            if (tnsrs[i].efficiency > 200) high_eff++;
            if (tnsrs[i].res > 150) high_res++;
        }
        printf("Активные (>100): %u\n", high_act);
        printf("Эффективные (>200): %u\n", high_eff);
        printf("Резонансные (>150): %u\n", high_res);
        printf("Разнообразие: %u/255, Штраф: %u/255\n", diversity_score, repetition_penalty);
        
        // Новая статистика самоорганизации
        printf("\n=== САМООРГАНИЗАЦИЯ ===\n");
        printf("Кластеры: %u\n", cluster_count);
        printf("Эпизоды: %u\n", episode_count);
        printf("Концепции: %u\n", concept_count);
        printf("Глобальный хеш контекста: %08X\n", global_context_hash);
        printf("Включена самоорганизация: %s\n", goals.self_organization_enabled ? "ДА" : "НЕТ");
        
        if (cluster_count > 0) {
            printf("\nТоп-5 кластеров по активности:\n");
            for (uint16_t i = 0; i < cluster_count && i < 5; i++) {
                printf("  [%u] id=%u, размер=%u, стабильность=%u, активность=%u\n",
                       i, clusters[i].cluster_id, clusters[i].size,
                       clusters[i].stability, clusters[i].activation_level);
            }
        }
    }
    else if (strcmp(cmd, "/links") == 0) {
        printf("\n=== СВЯЗИ (%u всего) ===\n", lnk_count);
        for (uint16_t i = 0; i < lnk_count && i < 20; i++) {
            const char* type_str = "";
            switch(lnks[i].semantic_type) {
                case 0: type_str = "обыч"; break;
                case 1: type_str = "внут"; break;
                case 2: type_str = "межк"; break;
                case 3: type_str = "конц"; break;
                default: type_str = "неизв"; break;
            }
            printf("[%u] str=%3u, res=%3u, type=%s, use=%3u, age=%us\n", 
                   i, lnks[i].strength, lnks[i].res, type_str,
                   lnks[i].use_count, 
                   (uint32_t)time(NULL) - lnks[i].last_act);
        }
    }
    else if (strcmp(cmd, "/clusters") == 0) {
        printf("\n=== КЛАСТЕРЫ (%u всего) ===\n", cluster_count);
        for (uint16_t i = 0; i < cluster_count && i < 20; i++) {
            const char* category_str = "";
            switch(clusters[i].category) {
                case 0: category_str = "неопр"; break;
                case 1: category_str = "дейст"; break;
                case 2: category_str = "состо"; break;
                case 3: category_str = "конц."; break;
                default: category_str = "неизв"; break;
            }
            printf("[%u] id=%u, размер=%u, стабильность=%u, активность=%u\n",
                   i, clusters[i].cluster_id, clusters[i].size,
                   clusters[i].stability, clusters[i].activation_level);
            printf("     Категория: %s, связей: %u, создан: %us назад\n",
                   category_str, clusters[i].link_count,
                   (uint32_t)time(NULL) - clusters[i].creation_time);
        }
    }
    else if (strcmp(cmd, "/concepts") == 0) {
        printf("\n=== КОНЦЕПЦИИ (%u всего) ===\n", concept_count);
        for (uint8_t i = 0; i < concept_count && i < 20; i++) {
            if (concepts[i].concept_tensor) {
                uint16_t tensor_idx = concepts[i].concept_tensor - tnsrs;
                printf("[%u] тензор=%u, абстракция=%u, когерентность=%u\n",
                       i, tensor_idx, concepts[i].abstraction_level,
                       concepts[i].coherence);
                printf("     членов=%u, последнее использование: %us назад\n",
                       concepts[i].member_count,
                       (uint32_t)time(NULL) - concepts[i].last_used);
            }
        }
    }
    else if (strcmp(cmd, "/episodes") == 0) {
        printf("\n=== ЭПИЗОДЫ (%u всего) ===\n", episode_count);
        for (uint16_t i = 0; i < episode_count && i < 10; i++) {
            printf("[%u] длина=%u, важность=%u, успешность=%u\n",
                   i, episodes[i].length, episodes[i].importance, 
                   episodes[i].success_score);
            printf("     вспоминаний=%u, последнее: %us назад\n",
                   episodes[i].recall_count,
                   (uint32_t)time(NULL) - episodes[i].last_recall);
        }
    }
    else if (strcmp(cmd, "/clean") == 0) {
        if (debug_mode) printf("[Агрессивная чистка памяти...]\n");
        uint32_t before_t = tnsr_count, before_l = lnk_count;
        uint32_t before_c = cluster_count, before_e = episode_count;
        aggressive_memory_cleanup();
        build_link_index();
        if (debug_mode) {
            printf("[Удалено: тензоры=%u, связи=%u]\n", 
                   before_t - tnsr_count, before_l - lnk_count);
            printf("[Осталось: тензоры=%u, связи=%u, кластеры=%u, эпизоды=%u]\n", 
                   tnsr_count, lnk_count, cluster_count, episode_count);
        }
    }
    else if (strcmp(cmd, "/save") == 0) {
        save_state_to_file("memory.bin");
        if (debug_mode) {
            printf("[Сохранено: тензоры=%u, связи=%u, кластеры=%u, эпизоды=%u]\n",
                   tnsr_count, lnk_count, cluster_count, episode_count);
        }
    }
    else if (strcmp(cmd, "/reset") == 0) {
        if (debug_mode) printf("[Сброс системы...]\n");
        for (uint16_t i = 0; i < tnsr_count; i++) { 
            if (tnsrs[i].data) free(tnsrs[i].data); 
        }
        tnsr_count = 0; lnk_count = 0; memo_size = 0;
        working_mem_count = 0;
        cluster_count = 0; episode_count = 0; concept_count = 0;
        remove("memory.bin");
        if (debug_mode) printf("[Система сброшена]\n");
    }
    else if (strcmp(cmd, "/resetrep") == 0) {
        repetition_penalty = 0;
        diversity_score = 128;
        last_response_hash = 0;
        if (debug_mode) printf("[Сброс системы повторений]\n");
    }
    else if (strcmp(cmd, "/organize") == 0) {
        if (debug_mode) printf("[Принудительная самоорганизация памяти...]\n");
        self_organize_memory_clusters();
        memory_consolidation();
        build_link_index();
        if (debug_mode) {
            printf("[Самоорганизация завершена: кластеры=%u, эпизоды=%u, концепции=%u]\n",
                   cluster_count, episode_count, concept_count);
        }
    }
    else if (strcmp(cmd, "/toggleorg") == 0) {
        goals.self_organization_enabled = !goals.self_organization_enabled;
        printf("[Самоорганизация %s]\n", 
               goals.self_organization_enabled ? "ВКЛЮЧЕНА" : "ВЫКЛЮЧЕНА");
    }
    else if (strcmp(cmd, "/info") == 0) {
        show_debug_info();
    }
    else if (strcmp(cmd, "/help") == 0) {
        if (debug_mode) {
            printf("\n=== ДОСТУПНЫЕ КОМАНДЫ ===\n");
            printf("  Основные:\n");
            printf("    /debug     - переключить режим отладки\n");
            printf("    /think     - принудительное мышление\n");
            printf("    /stats     - статистика системы\n");
            printf("    /links     - список связей\n");
            printf("    /clean     - очистка памяти\n");
            printf("    /save      - сохранить состояние\n");
            printf("    /reset     - полный сброс (осторожно!)\n");
            printf("    /resetrep  - сброс системы повторений\n");
            printf("    /info      - отладочная информация\n");
            printf("    /exit      - выход с сохранением\n");
            printf("  Самоорганизация:\n");
            printf("    /clusters  - показать кластеры памяти\n");
            printf("    /concepts  - показать концепции\n");
            printf("    /episodes  - показать эпизоды\n");
            printf("    /organize  - принудительная самоорганизация\n");
            printf("    /toggleorg - вкл/выкл самоорганизацию\n");
            printf("  Справка:\n");
            printf("    /help      - эта справка\n");
        } else {
            printf("Команды: /debug /think /stats /links /clean /save /reset\n");
            printf("         /resetrep /clusters /concepts /episodes /organize\n");
            printf("         /toggleorg /exit /help\n");
        }
    }
    else {
        printf("[Неизвестная команда '%s']\n", cmd);
        printf("Используйте /help для списка команд\n");
    }
}

// ===== Главная функция =====
int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "en_US.UTF-8");
    srand((uint32_t)time(NULL));
    
    // Проверяем аргументы командной строки
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-debug") == 0 || strcmp(argv[i], "--debug") == 0) {
            debug_mode = 1;
            printf("[Запуск в режиме отладки]\n");
        }
        else if (strcmp(argv[i], "-normal") == 0 || strcmp(argv[i], "--normal") == 0) {
            debug_mode = 0;
            printf("[Запуск в обычном режиме]\n");
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Использование: %s [опции]\n", argv[0]);
            printf("Опции:\n");
            printf("  -debug, --debug    запуск в режиме отладки\n");
            printf("  -normal, --normal  запуск в обычном режиме\n");
            printf("  -h, --help         показать эту справку\n");
            return 0;
        }
    }
    
    // Настройка обработчиков сигналов если в debug режиме
    if (debug_mode) {
        signal(SIGSEGV, signal_handler);
        signal(SIGABRT, signal_handler);
    }
    
    memset(&sstate, 0, sizeof(BitSystemState));
    memset(working_mem, 0, sizeof(working_mem));
    sstate.coh = 128;
    sstate.energy = 128;
    sstate.hist_idx = 0;
    sstate.consolidation_timer = (uint32_t)time(NULL);
    sstate.self_org_timer = (uint32_t)time(NULL);
    
    if (debug_mode) {
        printf("[DEBUG] Initializing system...\n");
        printf("[Проверка файла memory.bin...]\n");
    }
    
    if (file_exists("memory.bin")) {
        if (debug_mode) printf("[Файл найден, загрузка...]\n");
        if (load_state_from_file("memory.bin") < 0) {
            if (debug_mode) printf("[Ошибка загрузки, новый запуск]\n");
        } else {
            if (debug_mode) {
                printf("[Загружено успешно!]\n");
                printf("[Тензоры: %u, Связи: %u, Память: %u]\n", 
                       tnsr_count, lnk_count, memo_size);
                if (cluster_count > 0) {
                    printf("[Самоорганизация: кластеры=%u, эпизоды=%u, концепции=%u]\n",
                           cluster_count, episode_count, concept_count);
                }
            }
            build_link_index();
        }
    } else {
        if (debug_mode) printf("[Файл не найден, новый запуск]\n");
    }
    
    if (debug_mode) {
        printf("\n=== AGI СИСТЕМА v5.0 (УНИВЕРСАЛЬНАЯ) ===\n");
        printf("Режим отладки: ВКЛЮЧЕН\n");
        printf("Система самоорганизации памяти: %s\n",
               goals.self_organization_enabled ? "ВКЛЮЧЕНА" : "ВЫКЛЮЧЕНА");
        printf("Команда /debug для переключения режимов\n\n");
    } else {
        printf("\n=== NAI AGI v5.0 ===\n");
        printf("Режим: минималистичный\n");
        printf("Формат: (номер)you>> [многострочный ввод, пустая строка для завершения]\n");
        printf("        nai>> ответ\n");
        printf("Команда /debug для подробного режима\n\n");
        line_counter = 1;
    }
    
    char input_buffer[8192];
    
    while (1) {
        if (debug_mode) {
            // Режим отладки: старый стиль ввода (двойной Enter для завершения)
            if (!read_input_debug(input_buffer, sizeof(input_buffer), "Введите текст или команду")) {
                break;
            }
        } else {
            // Обычный режим: многострочный ввод (одна пустая строка для завершения)
            if (!read_input_multiline(input_buffer, sizeof(input_buffer), 1)) {
                break;
            }
            
            // Увеличиваем счетчик строк только если был некомандный ввод
            if (input_buffer[0] != '/' || input_buffer[1] == '\0') {
                line_counter++;
            }
        }
        
        // Обработка команды или обычного ввода
        if (input_buffer[0] == '/' && input_buffer[1] != '\0') {
            execute_command(input_buffer);
        } else if (strlen(input_buffer) > 0) {
            process_input(input_buffer);
        }
        
        // Небольшая пауза для стабильности
        usleep(100000); // 0.1 секунда
    }
    
    if (debug_mode) {
        printf("\n=== ЗАВЕРШЕНИЕ РАБОТЫ ===\n");
        printf("Финальная статистика:\n");
        printf("  Тензоры: %u\n", tnsr_count);
        printf("  Связи: %u\n", lnk_count);
        printf("  Память: %u записей\n", memo_size);
        printf("  Рабочая память: %u\n", working_mem_count);
        printf("  Разнообразие: %u, Штраф: %u\n", diversity_score, repetition_penalty);
        printf("  Самоорганизация:\n");
        printf("    Кластеры: %u\n", cluster_count);
        printf("    Эпизоды: %u\n", episode_count);
        printf("    Концепции: %u\n", concept_count);
    }
    
    // Сохранение перед выходом
    save_state_to_file("memory.bin");
    
    for (uint16_t i = 0; i < tnsr_count; i++) { 
        if (tnsrs[i].data) free(tnsrs[i].data); 
    }
    
    return 0;
}