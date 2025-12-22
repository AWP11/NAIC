#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <locale.h>
#include <wchar.h>
#include "core.h" // Теперь включает обновлённые прототипы

// ===== UTF-8 ЭНКОДЕР/ДЕКОДЕР =====

// Кодирует UTF-8 строку в бинарный формат для тензора
void encode_utf8_to_binary(const char* utf8_str, uint8_t* binary_output, size_t* output_len, size_t max_len) {
    if (!utf8_str || !binary_output || !output_len || max_len == 0) {
        if (output_len) *output_len = 0;
        return;
    }
    
    const unsigned char* ptr = (const unsigned char*)utf8_str;
    size_t pos = 0;
    
    while (*ptr && pos < max_len) {
        unsigned char c = *ptr;
        uint8_t char_len = 0;
        
        // Определяем длину UTF-8 символа
        if ((c & 0x80) == 0) {
            // 1-байтовый символ (ASCII)
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        } else {
            // Некорректный UTF-8, пропускаем
            ptr++;
            continue;
        }
        
        // Копируем все байты символа
        for (uint8_t i = 0; i < char_len && *ptr && pos < max_len; i++) {
            binary_output[pos++] = *ptr++;
        }
    }
    
    *output_len = pos;
}

// Декодирует бинарные данные из тензора обратно в UTF-8
void decode_binary_to_utf8(const uint8_t* binary_data, size_t data_len, char* output, size_t max_output_len) {
    if (!binary_data || !output || max_output_len == 0) {
        if (output) output[0] = '\0';
        return;
    }
    
    size_t out_pos = 0;
    size_t in_pos = 0;
    
    while (in_pos < data_len && out_pos < max_output_len - 1) {
        unsigned char c = binary_data[in_pos];
        
        // Проверяем, является ли это началом корректного UTF-8 символа
        uint8_t char_len = 0;
        if ((c & 0x80) == 0) {
            char_len = 1;  // ASCII
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        } else {
            // Некорректный UTF-8, пропускаем этот байт
            in_pos++;
            continue;
        }
        
        // Проверяем, что у нас достаточно байт для полного символа
        if (in_pos + char_len > data_len) {
            // Недостаточно данных, выходим
            break;
        }
        
        // Проверяем, что continuation bytes корректны
        int valid = 1;
        for (uint8_t i = 1; i < char_len; i++) {
            if ((binary_data[in_pos + i] & 0xC0) != 0x80) {
                valid = 0;
                break;
            }
        }
        
        if (!valid) {
            // Некорректный UTF-8 символ, пропускаем первый байт
            in_pos++;
            continue;
        }
        
        // Копируем символ в вывод
        for (uint8_t i = 0; i < char_len && out_pos < max_output_len - 1; i++) {
            output[out_pos++] = binary_data[in_pos + i];
        }
        
        in_pos += char_len;
    }
    
    output[out_pos] = '\0';
}

// Вспомогательная: декодирует тензор в UTF-8 строку
void decode_tensor_to_utf8(BitTensor* t, char* output, size_t max_len) {
    if (!t || !t->data || !output || max_len == 0) {
        if (output) output[0] = '\0';
        return;
    }
    
    // Вычисляем размер данных тензора в байтах
    uint32_t total_bits = t->rows * t->cols;
    uint32_t total_bytes = (total_bits + 7) / 8;
    
    // Декодируем бинарные данные
    decode_binary_to_utf8(t->data, total_bytes, output, max_len);
}

// ===== ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТА =====

// Новая функция: генерация ответа из внутренних мыслей
void generate_response_from_thoughts(void) {
    if (tnsr_count == 0 || working_mem_count == 0) {
        printf("[Система]: Нет активных мыслей для ответа.\n");
        return;
    }
    
    // Получаем наиболее активный тензор как "текущую мысль" через новую функцию
    BitTensor* active_thought = find_significant_tensor(SEARCH_MOST_ACTIVE, NULL);
    if (!active_thought || active_thought->act < 50) {
        printf("[Система]: Мысли слишком слабые для ответа.\n");
        return;
    }
    
    // Получаем связанные тензоры (ассоциации)
    BitTensor* associations[MAX_LINKS];
    uint16_t assoc_count = 0;
    
    for (uint16_t i = 0; i < lnk_count; i++) {
        if (lnks[i].src == active_thought && lnks[i].strength > 40 && 
            lnks[i].tgt->act > 30 && !lnks[i].tgt->dropout) {
            if (assoc_count < MAX_LINKS) {
                associations[assoc_count++] = lnks[i].tgt;
            }
        }
    }
    
// Если связей мало, ищем и используем похожие тензоры для обучения
if (assoc_count < 3) {
    for (uint16_t i = 0; i < tnsr_count && assoc_count < 10; i++) {
        BitTensor* candidate = &tnsrs[i];
        
        if (candidate == active_thought || candidate->dropout) continue;
        
        // Более мягкая проверка сходства
        uint8_t similarity = calc_bit_sim(active_thought, candidate);
        
        if (similarity > 40 && similarity < 90) {  // Похожи, но не слишком
            uint8_t activity_score = candidate->act * (100 - candidate->efficiency) / 100;
            
            if (activity_score > 20) {
                associations[assoc_count++] = candidate;
                
                // 🔥 КЛЮЧЕВОЕ УЛУЧШЕНИЕ: ИЗУЧАЕМ СХОДСТВО
                // Создаем временный буфер для обучения
                uint8_t learning_buffer[200];
                uint8_t learning_len = 0;
                
                // Смешиваем характеристики похожего тензора с активным
                for (uint8_t j = 0; j < 50 && learning_len < 195; j++) {
                    uint8_t mix_byte = (active_thought->data[j] ^ candidate->data[j]) | 
                                      (active_thought->data[j] & candidate->data[j]);
                    learning_buffer[learning_len++] = mix_byte;
                }
                
                // Добавляем мета-информацию
                learning_buffer[learning_len++] = similarity;
                learning_buffer[learning_len++] = candidate->efficiency;
                learning_buffer[learning_len++] = (active_thought->act + candidate->act) / 2;
                
                // 🔥 ВЫЗОВ ФУНКЦИИ ОБУЧЕНИЯ
                learn_by_binary_update(active_thought, learning_buffer, learning_len);
                
                // Увеличиваем резонанс
                active_thought->res = (active_thought->res + similarity / 4 > RES_MAX) ? 
                                      RES_MAX : active_thought->res + similarity / 4;
            }
        }
    }
}
    
    // Декодируем активную мысль
    char thought_buffer[MAX_OUTPUT];
    decode_tensor_to_utf8(active_thought, thought_buffer, sizeof(thought_buffer));
    
    // Формируем ответ на основе мыслей
    printf("[Мышление]: ");
    
    // Выводим основную мысль
    size_t thought_len = strlen(thought_buffer);
    if (thought_len > 0) {
        // Выводим до первого нулевого символа или конца строки
        size_t print_len = thought_len < 100 ? thought_len : 100;
        for (size_t i = 0; i < print_len && thought_buffer[i] != '\0'; i++) {
            // Пропускаем только управляющие символы
            if (thought_buffer[i] >= 32 || thought_buffer[i] == '\n' || thought_buffer[i] == '\t') {
                putchar(thought_buffer[i]);
            } else {
                // Заменяем непечатаемые символы на '?'
                putchar('?');
            }
        }
    }
    
    // Добавляем ассоциации (случайный выбор)
    if (assoc_count > 0) {
        printf(" | Связи: ");
        uint8_t printed_assocs = 0;
        for (uint8_t i = 0; i < assoc_count && printed_assocs < 3; i++) {
            // Случайно выбираем ассоциации для разнообразия
            if (rand() % 100 < 40) {
                char assoc_buf[100];
                decode_tensor_to_utf8(associations[i], assoc_buf, sizeof(assoc_buf));
                size_t assoc_len = strlen(assoc_buf);
                if (assoc_len > 0 && assoc_len < 30) {
                    if (printed_assocs > 0) printf(", ");
                    
                    // Выводим только читаемые символы
                    for (size_t j = 0; j < assoc_len && j < 20; j++) {
                        if (assoc_buf[j] >= 32 || assoc_buf[j] == '\n' || assoc_buf[j] == '\t') {
                            putchar(assoc_buf[j]);
                        } else {
                            putchar('?');
                        }
                    }
                    printed_assocs++;
                }
            }
        }
    }
    
    // Добавляем рефлексию
    if (active_thought->stab > 150 && active_thought->res > 100) {
        printf(" [рефлексия]");
    }
    
    // Статистика мысли
    printf("\n[Стат: act=%u, eff=%u, res=%u, связей=%u]\n", 
           active_thought->act, active_thought->efficiency, 
           active_thought->res, assoc_count);
    
    // Обновляем систему после генерации мысли
    active_thought->act = (active_thought->act * 9) / 10; // Снижаем активность после "озвучивания"
    active_thought->lu = (uint32_t)time(NULL);
    
    // Сохраняем мысль в память
    save_tnsr(active_thought);
}

// ===== УНИВЕРСАЛЬНАЯ ФУНКЦИЯ МНОГОСТРОЧНОГО ВВОДА =====

// Читает многострочный ввод до пустой строки (как в Vim)
// Enter = новая строка, Двойной Enter = отправка
int read_vim_style_input(char* buffer, size_t max_len, const char* prompt) {
    if (!buffer || max_len == 0) return 0;
    
    buffer[0] = '\0';
    size_t total_len = 0;
    char line[256];
    int line_number = 0;
    int empty_line_count = 0;
    
    printf("%s (двойной Enter для отправки):\n", prompt);
    
    while (1) {
        // Показываем номер строки (если это не первая строка)
        if (line_number > 0) {
            printf("%d> ", line_number + 1);
        } else {
            printf("> ");
        }
        
        fflush(stdout);
        
        if (!fgets(line, sizeof(line), stdin)) {
            if (total_len > 0) break;  // EOF, но есть данные
            return 0;  // EOF без данных
        }
        
        size_t line_len = strlen(line);
        // Убираем символ новой строки
        if (line_len > 0 && line[line_len - 1] == '\n') {
            line[--line_len] = '\0';
        }
        
        // Если пустая строка
        if (line_len == 0) {
            empty_line_count++;
            
            // Если это вторая пустая строка подряд - отправляем
            if (empty_line_count >= 2) {
                break;
            }
            
            // Первая пустая строка - добавляем \n
            if (total_len > 0 && total_len + 1 < max_len) {
                buffer[total_len++] = '\n';
                buffer[total_len] = '\0';
                line_number++;
            }
            continue;
        }
        
        // Сбрасываем счетчик пустых строк
        empty_line_count = 0;
        
        // Проверяем, достаточно ли места
        if (total_len + line_len + 2 < max_len) {
            if (total_len > 0) {
                buffer[total_len++] = '\n';  // Добавляем разделитель строк
            }
            strcpy(buffer + total_len, line);
            total_len += line_len;
            line_number++;
        } else {
            printf("[!] Достигнут предел длины ввода\n");
            break;
        }
    }
    
    // Убираем лишний \n в конце, если он есть
    if (total_len > 0 && buffer[total_len - 1] == '\n') {
        buffer[--total_len] = '\0';
    }
    
    return total_len > 0 ? 1 : 0;
}

// ===== ОБРАБОТКА ВВОДА ПОЛЬЗОВАТЕЛЯ =====

void process_user_input(const char* input_text) {
    if (!input_text || !*input_text) {
        printf("[!] Пустой ввод\n");
        return;
    }
    
    printf("\n[Обработка %zu символов...]\n", strlen(input_text));
    
    // Кодируем UTF-8 в бинарный формат для обработки
    uint8_t encoded_data[MAX_INPUT * 4]; // UTF-8 может быть до 4 байт на символ
    size_t encoded_len = 0;
    
    encode_utf8_to_binary(input_text, encoded_data, &encoded_len, sizeof(encoded_data));
    
    // Передаем закодированные данные в ядро
    proc_bit_input_raw(encoded_data, (uint16_t)encoded_len);
    
    // Генерируем ответ
    update_thought_stream();
    generate_response_from_thoughts();
}

// ===== ГЛАВНАЯ ФУНКЦИЯ =====

int main(void) {
    // Устанавливаем локаль для поддержки UTF-8
    setlocale(LC_ALL, "en_US.UTF-8");
    
    srand((uint32_t)time(NULL));
    memset(&sstate, 0, sizeof(BitSystemState));
    memset(working_mem, 0, sizeof(working_mem));
    sstate.coh = 128;
    sstate.energy = 128;

    // === ЗАГРУЗКА СОСТОЯНИЯ ===
    if (load_state_from_file("memory.bin") < 0) {
        printf("[WARN] Не удалось загрузить состояние — запуск с чистого листа.\n");
    } else {
        printf("[LOAD] Состояние восстановлено.\n");
    }

    printf("=== Низкоуровневая AGI v2.1 ===\n");
    printf("Мыслящие тензоры, резонансные петли, XOR/AND/NOT обучение\n");
    printf("UTF-8 кодирование для обработки и отображения\n");
    printf("Vim-стиль ввода (двойной Enter для отправки)\n");
    printf("Цель эффективности: %u\n", goals.target_efficiency);
    printf("Дропаут: %s\n", goals.dropout_enabled ? "ON" : "OFF");
    printf("\n");
    printf("Использование:\n");
    printf("  • Вводите текст, нажимайте Enter для новой строки\n");
    printf("  • Дважды нажмите Enter для отправки сообщения\n");
    printf("  • Команды начинаются с /\n");
    printf("\n");
    printf("Команды:\n");
    printf("  /raw      - байтовый ввод (старый режим)\n");
    printf("  /think    - принудительная генерация мысли\n");
    printf("  /stats    - статистика системы\n");
    printf("  /links    - показать связи\n");
    printf("  /echo     - последняя активная мысль\n");
    printf("  /help     - справка\n");
    printf("  /exit     - выход с сохранением\n");
    printf("\n");

    char input_buffer[MAX_INPUT];
    uint8_t raw_buffer[MAX_INPUT];
    uint8_t encoded_buffer[MAX_INPUT * 4];
    size_t encoded_len;
    char line[256];
    uint32_t last_response_time = 0;

    while (1) {
        // Всегда используем многострочный ввод (Vim-стиль)
        if (read_vim_style_input(input_buffer, sizeof(input_buffer), "Введите текст")) {
            // Проверяем, не команда ли это
            if (input_buffer[0] == '/' && input_buffer[1] != '\0') {
                // Обрабатываем команду
                if (strcmp(input_buffer, "/exit") == 0 || strcmp(input_buffer, "/quit") == 0) {
                    // === СОХРАНЕНИЕ ПЕРЕД ВЫХОДОМ ===
                    if (save_state_to_file("memory.bin") < 0) {
                        printf("[ERROR] Не удалось сохранить состояние!\n");
                    } else {
                        printf("[SAVE] Состояние сохранено в memory.bin\n");
                    }
                    break;
                }
                
                else if (strcmp(input_buffer, "/think") == 0) {
                    // Команда для принудительной генерации мысли
                    update_thought_stream();
                    generate_response_from_thoughts();
                }
                
                else if (strcmp(input_buffer, "/raw") == 0) {
                    // Байтовый ввод (старая функция)
                    printf("Длина в байтах: ");
                    fflush(stdout);
                    if (!fgets(line, sizeof(line), stdin)) break;
                    
                    long n = strtol(line, NULL, 10);
                    if (n <= 0 || n > MAX_INPUT) {
                        printf("Неверная длина (1..%d)\n", MAX_INPUT);
                        continue;
                    }
                    
                    printf("Ожидаем %ld байт:\n", n);
                    fflush(stdout);
                    size_t input_len = fread(raw_buffer, 1, (size_t)n, stdin);
                    if (input_len == 0) {
                        printf("[!] Нет данных\n");
                        continue;
                    }
                    
                    printf("[OK] Принято %zu байт\n", input_len);
                    proc_bit_input_raw(raw_buffer, (uint16_t)input_len);
                    
                    // Генерируем ответ после обработки
                    update_thought_stream();
                    generate_response_from_thoughts();
                }
                
                else if (strcmp(input_buffer, "/goal") == 0) {
                    printf("Цель эффективности: %u\n", goals.target_efficiency);
                    printf("Прирост эффективности: %u\n", goals.efficiency_gain);
                    printf("Режим экономии: %s\n", goals.energy_saving_mode ? "ON" : "OFF");
                    printf("Общая стоимость: %u\n", goals.total_compute_cost);
                }
                
                else if (strcmp(input_buffer, "/dropout") == 0) {
                    goals.dropout_enabled = !goals.dropout_enabled;
                    printf("Дропаут: %s\n", goals.dropout_enabled ? "ON" : "OFF");
                }
                
                else if (strcmp(input_buffer, "/workmem") == 0) {
                    printf("Рабочая память (%u записей):\n", working_mem_count);
                    for (uint8_t i = 0; i < working_mem_count; i++) {
                        if (working_mem[i].tensor) {
                            char buf[100];
                            decode_tensor_to_utf8(working_mem[i].tensor, buf, sizeof(buf));
                            // Очищаем непечатаемые символы
                            for (size_t j = 0; buf[j] != '\0'; j++) {
                                if (buf[j] < 32 && buf[j] != '\n' && buf[j] != '\t') {
                                    buf[j] = '?';
                                }
                            }
                            printf("  [%u] prio:%u acc:%u: %.30s\n", 
                                   i, working_mem[i].priority, 
                                   working_mem[i].access_count, buf);
                        }
                    }
                }
                
                else if (strcmp(input_buffer, "/stats") == 0) {
                    printf("Тензоры: %u\n", tnsr_count);
                    printf("Связи: %u\n", lnk_count);
                    printf("Записи памяти: %u\n", memo_size);
                    printf("Тензор-Тензоры: %u\n", tt_count);
                    printf("Резонанс системы: %u\n", sys_res);
                    uint16_t active = 0;
                    uint16_t dropout = 0;
                    uint32_t total_eff = 0;
                    for (uint16_t i = 0; i < tnsr_count; i++) {
                        if (tnsrs[i].act > 50) active++;
                        if (tnsrs[i].dropout) dropout++;
                        total_eff += tnsrs[i].efficiency;
                    }
                    printf("Активные тензоры: %u\n", active);
                    printf("Тензоры в дропауте: %u\n", dropout);
                    if (tnsr_count > 0) {
                        printf("Средняя эффективность: %u\n", (uint32_t)total_eff / tnsr_count);
                    }
                }
                
                else if (strcmp(input_buffer, "/links") == 0) {
                    printf("Связи (%u всего):\n", lnk_count);
                    for (uint16_t i = 0; i < lnk_count; i++) {
                        char src_buf[50], tgt_buf[50];
                        decode_tensor_to_utf8(lnks[i].src, src_buf, sizeof(src_buf));
                        decode_tensor_to_utf8(lnks[i].tgt, tgt_buf, sizeof(tgt_buf));
                        
                        // Очищаем непечатаемые символы
                        for (size_t j = 0; src_buf[j] != '\0'; j++) {
                            if (src_buf[j] < 32 && src_buf[j] != '\n' && src_buf[j] != '\t') {
                                src_buf[j] = '?';
                            }
                        }
                        for (size_t j = 0; tgt_buf[j] != '\0'; j++) {
                            if (tgt_buf[j] < 32 && tgt_buf[j] != '\n' && tgt_buf[j] != '\t') {
                                tgt_buf[j] = '?';
                            }
                        }
                        
                        printf("  [%u] str:%u use:%u succ:%u: %.20s -> %.20s\n", 
                               i, lnks[i].strength, lnks[i].use_count, 
                               lnks[i].success_count, src_buf, tgt_buf);
                    }
                }
                
                else if (strcmp(input_buffer, "/clear") == 0) {
                    for (uint16_t i = 0; i < tnsr_count; i++) { 
                        if (tnsrs[i].data) free(tnsrs[i].data); 
                    }
                    for (uint16_t i = 0; i < tt_count; i++) { 
                        if (t_tnsrs[i].data) free(t_tnsrs[i].data); 
                        if (t_tnsrs[i].tensor_indices) free(t_tnsrs[i].tensor_indices);
                    }
                    tnsr_count = 0; 
                    tt_count = 0; 
                    lnk_count = 0; 
                    memo_size = 0;
                    working_mem_count = 0;
                    sys_res = RES_HALF;
                    goals.target_efficiency = 180;
                    printf("Система очищена.\n");
                }
                
                else if (strcmp(input_buffer, "/echo") == 0) {
                    // Эхо-тест: декодирует последний активный тензор
                    // Используем новую функцию для поиска
                    BitTensor* last_active = find_significant_tensor(SEARCH_MOST_ACTIVE, NULL);
                    if (last_active) {
                        char buf[MAX_OUTPUT];
                        decode_tensor_to_utf8(last_active, buf, sizeof(buf));
                        
                        // Очищаем непечатаемые символы
                        for (size_t j = 0; buf[j] != '\0'; j++) {
                            if (buf[j] < 32 && buf[j] != '\n' && buf[j] != '\t') {
                                buf[j] = '?';
                            }
                        }
                        
                        printf("Последняя активная мысль: %s\n", buf);
                        printf("Act: %u, Res: %u, Eff: %u\n", 
                               last_active->act, last_active->res, last_active->efficiency);
                    } else {
                        printf("Нет активных мыслей\n");
                    }
                }
                
                else if (strcmp(input_buffer, "/test") == 0) {
                    // Тест UTF-8 кодирования/декодирования
                    printf("Тест UTF-8 кодирования:\n");
                    const char* test_str = "Привет мир! Hello 世界! 😊";
                    printf("Оригинал: %s\n", test_str);
                    
                    // Кодируем
                    encode_utf8_to_binary(test_str, encoded_buffer, &encoded_len, sizeof(encoded_buffer));
                    printf("Закодировано: %zu байт\n", encoded_len);
                    
                    // Декодируем обратно
                    char decoded[MAX_OUTPUT];
                    decode_binary_to_utf8(encoded_buffer, encoded_len, decoded, sizeof(decoded));
                    printf("Декодировано: %s\n", decoded);
                }
                
                else if (strcmp(input_buffer, "/help") == 0) {
                    printf("справка:\n");
                    printf("  /help     - эта справка\n");
                    printf("  /raw      - байтовый ввод\n");
                    printf("  /think    - генерация мысли\n");
                    printf("  /stats    - статистика\n");
                    printf("  /workmem  - рабочая память\n");
                    printf("  /links    - список связей\n");
                    printf("  /echo     - последняя мысль\n");
                    printf("  /clear    - очистка системы\n");
                    printf("  /test     - тест UTF-8\n");
                    printf("  /exit     - выход\n");
                }
                
                else {
                    printf("Неизвестная команда. Используйте /help для списка команд\n");
                }
            } else {
                // Обычный текстовый ввод
                process_user_input(input_buffer);
            }
        }

        // === АВТОМАТИЧЕСКОЕ ОБНОВЛЕНИЕ МЫШЛЕНИЯ ===
        uint32_t current_time = (uint32_t)time(NULL);
        if (current_time - last_response_time > 45) {
            // Автоматическая генерация мысли каждые 45 секунд простоя
            update_thought_stream();
            if (rand() % 100 < 15) { // 15% шанс на спонтанную мысль
                printf("\n[Спонтанная мысль]: ");
                generate_response_from_thoughts();
            }
            last_response_time = current_time;
        }
    }

    printf("\nВыход. Финальная цель эффективности: %u\n", goals.target_efficiency);

    // Очистка при завершении
    for (uint16_t i = 0; i < tnsr_count; i++) { 
        if (tnsrs[i].data) free(tnsrs[i].data); 
    }
    for (uint16_t i = 0; i < tt_count; i++) { 
        if (t_tnsrs[i].tensor_indices) free(t_tnsrs[i].tensor_indices);
    }

    return 0;
}