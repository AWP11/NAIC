#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "core.h"
#define M_PI 3.1415
// ============================================
// ПРОТОТИПЫ ФУНКЦИЙ
// ============================================

void generate_noise_from_tensor(BitTensor* t, uint8_t* pixels, int width, int height);
void generate_interference_pattern(BitTensor* t, uint8_t* pixels, int width, int height);
void generate_fractal_pattern(BitTensor* t, uint8_t* pixels, int width, int height);
void generate_link_visualization(uint8_t* pixels, int width, int height);
void draw_line(uint8_t* pixels, int width, int height, 
               int x1, int y1, int x2, int y2, int thickness, uint8_t value);
void draw_circle(uint8_t* pixels, int width, int height, 
                 int cx, int cy, int radius, uint8_t value);
int save_pgm(const char* filename, uint8_t* pixels, int width, int height);
int save_ppm(const char* filename, uint8_t* pixels_r, uint8_t* pixels_g, 
             uint8_t* pixels_b, int width, int height);
void generate_rgb_from_tensors(BitTensor* r_t, BitTensor* g_t, BitTensor* b_t,
                               uint8_t* r_pixels, uint8_t* g_pixels, uint8_t* b_pixels,
                               int width, int height);
void create_animation_series(const char* basename, int frames, int width, int height);
BitTensor* find_tensor_by_features(uint8_t min_act, uint8_t min_res, uint8_t max_ent);
BitTensor* create_tensor_from_description(const char* desc);
static uint16_t tensor_to_index(BitTensor* t);

// ============================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ СЕРИАЛИЗАЦИИ
// ============================================

static uint16_t tensor_to_index(BitTensor* t) {
    if (!t) return 0xFFFF;
    intptr_t diff = t - tnsrs;
    return (diff >= 0 && diff < MAX_TENSORS) ? (uint16_t)diff : 0xFFFF;
}

// ============================================
// ИЗОБРАЗИТЕЛЬНЫЕ ГЕНЕРАТОРЫ
// ============================================

// 1. Генератор шума на основе тензора
void generate_noise_from_tensor(BitTensor* t, uint8_t* pixels, int width, int height) {
    if (!t || !t->data || width * height == 0) return;
    
    uint32_t total_bits = t->rows * t->cols;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            uint32_t bit_idx = idx % total_bits;
            uint8_t bit = BIT_GET(t->data[bit_idx / 8], bit_idx % 8);
            
            // Преобразуем бит в значение 0-255
            uint8_t value = bit ? 255 : 0;
            
            // Добавляем влияние активности и резонанса
            value = (value * t->act) / 255;
            
            // Добавляем шум на основе энтропии
            if (t->ent > 100) {
                value ^= (rand() % 64);
            }
            
            pixels[idx] = value;
        }
    }
}

// 2. Генератор градиента с интерференцией
void generate_interference_pattern(BitTensor* t, uint8_t* pixels, int width, int height) {
    if (!t || width * height == 0) return;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            
            // Базовый градиент
            float dx = (float)x / width;
            float dy = (float)y / height;
            float gradient = (dx + dy) * 0.5f;
            
            // Интерференция от тензора
            float interference = 0.0f;
            if (t->data) {
                uint32_t bit_idx = (x + y * width) % (t->rows * t->cols);
                uint8_t bit = BIT_GET(t->data[bit_idx / 8], bit_idx % 8);
                interference = bit ? 0.3f : -0.3f;
            }
            
            // Модуляция активностью и резонансом
            float mod = (t->act / 255.0f) * (t->res / 255.0f);
            float value_f = (gradient + interference * mod) * 255.0f;
            
            // Ограничение
            if (value_f < 0) value_f = 0;
            if (value_f > 255) value_f = 255;
            
            pixels[idx] = (uint8_t)value_f;
        }
    }
}

// 3. Генератор фрактальных паттернов
void generate_fractal_pattern(BitTensor* t, uint8_t* pixels, int width, int height) {
    if (width * height == 0) return;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            
            // Фрактальный шум
            float nx = (float)x / width * 10.0f;
            float ny = (float)y / height * 10.0f;
            float value_f = 0.0f;
            float freq = 1.0f;
            float amp = 1.0f;
            
            for (int i = 0; i < 4; i++) {
                value_f += sin(nx * freq) * cos(ny * freq) * amp;
                freq *= 2.0f;
                amp *= 0.5f;
            }
            
            // Нормализация
            value_f = (value_f + 1.0f) * 0.5f * 255.0f;
            
            // Влияние тензора
            if (t && t->data) {
                uint32_t bit_idx = (x ^ y) % (t->rows * t->cols);
                if (BIT_GET(t->data[bit_idx / 8], bit_idx % 8)) {
                    value_f = 255 - value_f; // Инверсия
                }
            }
            
            pixels[idx] = (uint8_t)value_f;
        }
    }
}

// 4. Вспомогательная функция: рисование линии
void draw_line(uint8_t* pixels, int width, int height, 
               int x1, int y1, int x2, int y2, int thickness, uint8_t value) {
    // Алгоритм Брезенхэма
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;
    
    while (1) {
        // Рисуем пиксель с толщиной
        for (int ty = -thickness/2; ty <= thickness/2; ty++) {
            for (int tx = -thickness/2; tx <= thickness/2; tx++) {
                int px = x1 + tx;
                int py = y1 + ty;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    pixels[py * width + px] = value;
                }
            }
        }
        
        if (x1 == x2 && y1 == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 < dx) { err += dx; y1 += sy; }
    }
}

// 5. Вспомогательная функция: рисование круга
void draw_circle(uint8_t* pixels, int width, int height, 
                 int cx, int cy, int radius, uint8_t value) {
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            if (x*x + y*y <= radius*radius) {
                int px = cx + x;
                int py = cy + y;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    pixels[py * width + px] = value;
                }
            }
        }
    }
}

// 6. Генератор связей между тензорами (визуализация графа)
void generate_link_visualization(uint8_t* pixels, int width, int height) {
    if (lnk_count == 0 || width * height == 0) return;
    
    // Очистка
    memset(pixels, 0, width * height);
    
    // Позиции тензоров на изображении
    float tensor_x[MAX_TENSORS];
    float tensor_y[MAX_TENSORS];
    
    // Распределяем тензоры по кругу
    for (uint16_t i = 0; i < tnsr_count && i < MAX_TENSORS; i++) {
        float angle = (2 * M_PI * i) / tnsr_count;
        float radius = 0.4f;
        tensor_x[i] = 0.5f + cos(angle) * radius;
        tensor_y[i] = 0.5f + sin(angle) * radius;
    }
    
    // Рисуем связи
    for (uint16_t i = 0; i < lnk_count; i++) {
        BitLink* link = &lnks[i];
        uint16_t src_idx = tensor_to_index(link->src);
        uint16_t tgt_idx = tensor_to_index(link->tgt);
        
        if (src_idx >= tnsr_count || tgt_idx >= tnsr_count) continue;
        
        int x1 = (int)(tensor_x[src_idx] * width);
        int y1 = (int)(tensor_y[src_idx] * height);
        int x2 = (int)(tensor_x[tgt_idx] * width);
        int y2 = (int)(tensor_y[tgt_idx] * height);
        
        // Толщина линии зависит от силы связи
        int thickness = link->strength / 64;
        if (thickness < 1) thickness = 1;
        
        // Цвет зависит от успешности
        uint8_t value = (link->success_count * 255) / (link->use_count + 1);
        
        // Рисуем линию
        draw_line(pixels, width, height, x1, y1, x2, y2, thickness, value);
    }
    
    // Рисуем узлы (тензоры)
    for (uint16_t i = 0; i < tnsr_count && i < MAX_TENSORS; i++) {
        int x = (int)(tensor_x[i] * width);
        int y = (int)(tensor_y[i] * height);
        int radius = tnsrs[i].act / 32;
        if (radius < 2) radius = 2;
        if (radius > 20) radius = 20;
        
        draw_circle(pixels, width, height, x, y, radius, 255);
    }
}

// 7. Сохранение в PGM (Portable Graymap)
int save_pgm(const char* filename, uint8_t* pixels, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;
    
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    fwrite(pixels, 1, width * height, f);
    fclose(f);
    
    return 0;
}

// 8. Сохранение в PPM (Portable Pixmap) - RGB
int save_ppm(const char* filename, uint8_t* pixels_r, uint8_t* pixels_g, 
             uint8_t* pixels_b, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;
    
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    
    for (int i = 0; i < width * height; i++) {
        fputc(pixels_r[i], f);
        fputc(pixels_g[i], f);
        fputc(pixels_b[i], f);
    }
    
    fclose(f);
    return 0;
}

// 9. Генератор цветного изображения из трёх тензоров (RGB)
void generate_rgb_from_tensors(BitTensor* r_t, BitTensor* g_t, BitTensor* b_t,
                               uint8_t* r_pixels, uint8_t* g_pixels, uint8_t* b_pixels,
                               int width, int height) {
    // Генерируем каждый канал
    if (r_t) generate_noise_from_tensor(r_t, r_pixels, width, height);
    else memset(r_pixels, 128, width * height);
    
    if (g_t) generate_interference_pattern(g_t, g_pixels, width, height);
    else memset(g_pixels, 128, width * height);
    
    if (b_t) generate_fractal_pattern(b_t, b_pixels, width, height);
    else memset(b_pixels, 128, width * height);
}

// 10. Создание гиф-анимации (серия PGM)
void create_animation_series(const char* basename, int frames, int width, int height) {
    printf("Создание анимации (%d кадров)...\n", frames);
    
    uint8_t* pixels = malloc(width * height);
    if (!pixels) return;
    
    for (int frame = 0; frame < frames; frame++) {
        // Выбираем случайный тензор для этого кадра
        BitTensor* t = NULL;
        if (tnsr_count > 0) {
            t = &tnsrs[rand() % tnsr_count];
        }
        
        // Генерируем паттерн
        generate_fractal_pattern(t, pixels, width, height);
        
        // Сохраняем кадр
        char filename[256];
        snprintf(filename, sizeof(filename), "%s_%04d.pgm", basename, frame);
        save_pgm(filename, pixels, width, height);
        
        // Обновляем мышление между кадрами
        update_thought_stream();
        
        if (frame % 10 == 0) {
            printf("  Кадр %d/%d\n", frame + 1, frames);
        }
    }
    
    free(pixels);
    printf("Анимация сохранена в %s_*.pgm\n", basename);
}

// ============================================
// УТИЛИТЫ ДЛЯ РАБОТЫ С ТЕНЗОРАМИ
// ============================================

// Поиск тензора по характеристикам
BitTensor* find_tensor_by_features(uint8_t min_act, uint8_t min_res, uint8_t max_ent) {
    BitTensor* best = NULL;
    uint32_t best_score = 0;
    
    for (uint16_t i = 0; i < tnsr_count; i++) {
        BitTensor* t = &tnsrs[i];
        if (t->act < min_act || t->res < min_res || t->ent > max_ent) continue;
        
        uint32_t score = (uint32_t)t->act * t->res * (255 - t->ent);
        if (score > best_score) {
            best_score = score;
            best = t;
        }
    }
    
    return best;
}

// Создание нового тензора из описания
BitTensor* create_tensor_from_description(const char* desc) {
    // Создаём тензор
    BitTensor* t = create_tnsr(8, strlen(desc) * 3);
    if (!t) return NULL;
    
    // Кодируем описание
    encode_tnsr(t, (const uint8_t*)desc, strlen(desc));
    
    // Настраиваем параметры на основе описания
    t->act = 180;
    t->res = 200;
    t->ent = calc_bit_ent(t);
    t->efficiency = calculate_efficiency(t);
    
    // Добавляем в рабочую память
    add_to_working_memory(t);
    
    return t;
}

// ============================================
// ГЛАВНАЯ ПРОГРАММА - ГЕНЕРАТОР ИЗОБРАЖЕНИЙ
// ============================================

int main(void) {
    srand((uint32_t)time(NULL));
    memset(&sstate, 0, sizeof(BitSystemState));
    memset(working_mem, 0, sizeof(working_mem));
    sstate.coh = 128;
    sstate.energy = 128;

    // Загрузка состояния
    if (load_state_from_file("art_memory.bin") < 0) {
        printf("[WARN] Не удалось загрузить состояние — начинаем с нуля.\n");
        // Создаём начальные тензоры для искусства
        create_tensor_from_description("cosmic noise");
        create_tensor_from_description("organic patterns");
        create_tensor_from_description("digital dreams");
        create_tensor_from_description("fractal universe");
    } else {
        printf("[LOAD] Художественное состояние восстановлено.\n");
    }

    printf("=== Tensor Art Generator v1.0 ===\n");
    printf("Использует BitTensor AGI для генерации изображений\n");
    printf("Тензоров: %u, Связей: %u\n", tnsr_count, lnk_count);
    printf("\nДоступные команды:\n");
    printf("  /noise [W] [H] [name]    — Шумовой паттерн\n");
    printf("  /fractal [W] [H] [name]  — Фрактальный паттерн\n");
    printf("  /interf [W] [H] [name]   — Интерференция\n");
    printf("  /links [W] [H] [name]    — Визуализация связей\n");
    printf("  /rgb [W] [H] [name]      — Цветное RGB изображение\n");
    printf("  /anim [W] [H] [frames]   — Анимация\n");
    printf("  /batch [N] [W] [H]       — Пакетная генерация\n");
    printf("  /learn [текст]           — Обучить на тексте\n");
    printf("  /tensors                 — Список тензоров\n");
    printf("  /stats                   — Статистика\n");
    printf("  /clear                   — Очистить\n");
    printf("  /exit                    — Выход с сохранением\n");
    printf("\n");

    char line[256];

    while (1) {
        printf("\nart> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;

        size_t line_len = strlen(line);
        if (line_len > 0 && line[line_len - 1] == '\n') {
            line[--line_len] = '\0';
        }

        // ===== КОМАНДЫ ГЕНЕРАЦИИ =====
        if (strncmp(line, "/noise", 6) == 0) {
            int width = 512, height = 512;
            char name[256] = "noise";
            
            if (line_len > 6) {
                sscanf(line + 6, "%d %d %255s", &width, &height, name);
            }
            
            if (width <= 0 || height <= 0) {
                printf("Неверные размеры\n");
                continue;
            }
            
            // Находим или создаём тензор
            BitTensor* t = get_most_active_tensor();
            if (!t && tnsr_count > 0) t = &tnsrs[0];
            
            // Генерируем изображение
            uint8_t* pixels = malloc(width * height);
            generate_noise_from_tensor(t, pixels, width, height);
            
            // Сохраняем
            char filename[256];
            snprintf(filename, sizeof(filename), "%s.pgm", name);
            save_pgm(filename, pixels, width, height);
            free(pixels);
            
            printf("Сохранено: %s (%dx%d)\n", filename, width, height);
        }
        else if (strncmp(line, "/fractal", 8) == 0) {
            int width = 512, height = 512;
            char name[256] = "fractal";
            
            if (line_len > 8) {
                sscanf(line + 8, "%d %d %255s", &width, &height, name);
            }
            
            // Находим тензор с высокой энтропией
            BitTensor* t = find_tensor_by_features(100, 100, 200);
            
            uint8_t* pixels = malloc(width * height);
            generate_fractal_pattern(t, pixels, width, height);
            
            char filename[256];
            snprintf(filename, sizeof(filename), "%s.pgm", name);
            save_pgm(filename, pixels, width, height);
            free(pixels);
            
            printf("Сохранено: %s\n", filename);
        }
        else if (strncmp(line, "/interf", 7) == 0) {
            int width = 512, height = 512;
            char name[256] = "interference";
            
            if (line_len > 7) {
                sscanf(line + 7, "%d %d %255s", &width, &height, name);
            }
            
            BitTensor* t = get_resonant_tensor();
            
            uint8_t* pixels = malloc(width * height);
            generate_interference_pattern(t, pixels, width, height);
            
            char filename[256];
            snprintf(filename, sizeof(filename), "%s.pgm", name);
            save_pgm(filename, pixels, width, height);
            free(pixels);
            
            printf("Сохранено: %s\n", filename);
        }
        else if (strncmp(line, "/links", 6) == 0) {
            int width = 800, height = 600;
            char name[256] = "links";
            
            if (line_len > 6) {
                sscanf(line + 6, "%d %d %255s", &width, &height, name);
            }
            
            if (lnk_count == 0) {
                printf("Нет связей для визуализации\n");
                continue;
            }
            
            uint8_t* pixels = malloc(width * height);
            generate_link_visualization(pixels, width, height);
            
            char filename[256];
            snprintf(filename, sizeof(filename), "%s.pgm", name);
            save_pgm(filename, pixels, width, height);
            free(pixels);
            
            printf("Сохранено: %s (связей: %u)\n", filename, lnk_count);
        }
        else if (strncmp(line, "/rgb", 4) == 0) {
            int width = 512, height = 512;
            char name[256] = "rgb";
            
            if (line_len > 4) {
                sscanf(line + 4, "%d %d %255s", &width, &height, name);
            }
            
            // Находим три разных тензора для RGB
            BitTensor* r_t = find_tensor_by_features(150, 100, 150);
            BitTensor* g_t = find_tensor_by_features(100, 150, 150);
            BitTensor* b_t = find_tensor_by_features(100, 100, 200);
            
            uint8_t* r_pixels = malloc(width * height);
            uint8_t* g_pixels = malloc(width * height);
            uint8_t* b_pixels = malloc(width * height);
            
            generate_rgb_from_tensors(r_t, g_t, b_t, 
                                     r_pixels, g_pixels, b_pixels,
                                     width, height);
            
            char filename[256];
            snprintf(filename, sizeof(filename), "%s.ppm", name);
            save_ppm(filename, r_pixels, g_pixels, b_pixels, width, height);
            
            free(r_pixels);
            free(g_pixels);
            free(b_pixels);
            
            printf("Сохранено: %s (RGB)\n", filename);
        }
        else if (strncmp(line, "/anim", 5) == 0) {
            int width = 256, height = 256, frames = 30;
            
            if (line_len > 5) {
                sscanf(line + 5, "%d %d %d", &width, &height, &frames);
            }
            
            if (frames > 1000) frames = 1000;
            
            create_animation_series("animation", frames, width, height);
        }
        else if (strncmp(line, "/batch", 6) == 0) {
            int count = 10, width = 256, height = 256;
            
            if (line_len > 6) {
                sscanf(line + 6, "%d %d %d", &count, &width, &height);
            }
            
            if (count > 100) count = 100;
            
            printf("Генерация %d изображений...\n", count);
            for (int i = 0; i < count; i++) {
                BitTensor* t = NULL;
                if (tnsr_count > 0) {
                    t = &tnsrs[rand() % tnsr_count];
                }
                
                uint8_t* pixels = malloc(width * height);
                
                // Чередуем генераторы
                switch (i % 3) {
                    case 0: generate_noise_from_tensor(t, pixels, width, height); break;
                    case 1: generate_interference_pattern(t, pixels, width, height); break;
                    case 2: generate_fractal_pattern(t, pixels, width, height); break;
                }
                
                char filename[256];
                snprintf(filename, sizeof(filename), "batch_%04d.pgm", i);
                save_pgm(filename, pixels, width, height);
                
                free(pixels);
                
                // Обновляем мышление
                if (i % 5 == 0) {
                    update_thought_stream();
                }
                
                if (i % 10 == 0) {
                    printf("  %d/%d\n", i + 1, count);
                }
            }
            printf("Готово!\n");
        }
        else if (strncmp(line, "/learn ", 7) == 0) {
            const char* text = line + 7;
            proc_bit_input(text);
            printf("Обучаемся на: \"%s\"\n", text);
        }
        else if (strcmp(line, "/tensors") == 0) {
            printf("Тензоры (%u):\n", tnsr_count);
            for (uint16_t i = 0; i < tnsr_count; i++) {
                BitTensor* t = &tnsrs[i];
                char buf[100];
                decode_tnsr(t, buf, sizeof(buf));
                printf("  [%3u] Act:%3u Res:%3u Ent:%3u Eff:%3u: %.30s\n",
                       i, t->act, t->res, t->ent, t->efficiency, buf);
            }
        }
        else if (strcmp(line, "/stats") == 0) {
            printf("=== Статистика искусства ===\n");
            printf("Тензоры: %u\n", tnsr_count);
            printf("Связи: %u\n", lnk_count);
            printf("Память: %u\n", memo_size);
            printf("Резонанс системы: %u\n", sys_res);
            
            uint16_t active = 0;
            uint32_t total_ent = 0;
            for (uint16_t i = 0; i < tnsr_count; i++) {
                if (tnsrs[i].act > 50) active++;
                total_ent += tnsrs[i].ent;
            }
            printf("Активные тензоры: %u\n", active);
            printf("Средняя энтропия: %u\n", tnsr_count > 0 ? total_ent / tnsr_count : 0);
            printf("Цель эффективности: %u\n", goals.target_efficiency);
        }
        else if (strcmp(line, "/clear") == 0) {
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
            
            // Создаём новые художественные тензоры
            create_tensor_from_description("abstract art");
            create_tensor_from_description("generative design");
            create_tensor_from_description("neural patterns");
            
            printf("Система очищена, созданы новые художественные тензоры.\n");
        }
        else if (strcmp(line, "/exit") == 0) {
            if (save_state_to_file("art_memory.bin") < 0) {
                printf("[ERROR] Не удалось сохранить состояние!\n");
            } else {
                printf("[SAVE] Художественное состояние сохранено.\n");
            }
            break;
        }
        else if (line_len > 0 && line[0] == '/') {
            printf("Неизвестная команда. Введите /help для списка команд.\n");
        }
        else if (line_len > 0) {
            // Текст как художественное описание
            BitTensor* t = create_tensor_from_description(line);
            if (t) {
                printf("Создан художественный тензор из: \"%s\"\n", line);
                printf("Активность: %u, Резонанс: %u, Энтропия: %u\n",
                       t->act, t->res, t->ent);
            }
        }

        // ===== ФОНОВОЕ МЫШЛЕНИЕ =====
        update_thought_stream();
    }

    printf("\nTensor Art Generator завершён.\n");

    // Очистка
    for (uint16_t i = 0; i < tnsr_count; i++) { 
        if (tnsrs[i].data) free(tnsrs[i].data); 
    }
    for (uint16_t i = 0; i < tt_count; i++) { 
        if (t_tnsrs[i].data) free(t_tnsrs[i].data); 
        if (t_tnsrs[i].tensor_indices) free(t_tnsrs[i].tensor_indices);
    }

    return 0;
}