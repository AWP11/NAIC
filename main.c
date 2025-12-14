// main.c
// Битовая фрактально-резонансная AGI v2.0 — ДВА СЛОЯ: перцептивный + интегративный
// Всё есть биты. Никаких float. Только целые, сдвиги, XOR, popcount.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define MAX_INPUT       1500
#define MAX_OUTPUT      2000
#define MAX_PATTERN     256
#define MAX_MEM_ENTRIES 1024
#define MAX_TENSORS     300
#define MAX_LINKS       200
#define HISTORY_SIZE    64

// === БИНАРНАЯ МЕТАСИГМОЙДА ===
#define METASIGMOID_THRESHOLD 128 // Порог для бинарного переключения

// Бинарная сигмойда: (x >= 128) ? 255 : 0
uint8_t binary_metasigmoid(uint8_t x) {
    return (x >= METASIGMOID_THRESHOLD) ? 255 : 0;
}

// === БИТОВЫЕ КОНСТАНТЫ И МАКРОСЫ ===

#define BIT_SET(byte, bit)    ((byte) |= (1 << (bit)))
#define BIT_CLEAR(byte, bit)  ((byte) &= ~(1 << (bit)))
#define BIT_TOGGLE(byte, bit) ((byte) ^= (1 << (bit)))
#define BIT_GET(byte, bit)    (((byte) >> (bit)) & 1)

// Резонанс хранится в 8 битах: 0-255 (0.0-1.0 * 255)
#define RESONANCE_MAX   255
#define RESONANCE_HALF  128
#define ACTIVATION_MAX  255
#define DOC_MAX         255  // DOC: 0-255 (1.0-10.0 * 25.5)

// === СТРУКТУРЫ (БИТОВЫЕ ВЕРСИИ) ===

typedef struct {
    uint8_t* data;          // Бинарные данные
    uint16_t rows;          // Размерность X (биты)
    uint16_t cols;          // Размерность Y (байты)
    uint8_t resonance;      // Резонанс [0-255]
    uint8_t activation;     // Активация [0-255]
    uint8_t entropy;        // Энтропия [0-255]
    uint8_t stability;      // Стабильность [0-255]
    uint16_t connections;   // Количество связей
    uint32_t last_used;     // Timestamp
} BitTensor;

typedef struct {
    BitTensor* source;
    BitTensor* target;
    uint8_t strength;       // Сила связи [0-255]
    uint8_t resonance;      // Резонанс связи [0-255]
    uint16_t weight;        // Вес связи [0-65535]
    uint32_t created;
    uint32_t last_active;
} BitLink;

typedef struct {
    uint8_t data[MAX_PATTERN];
    uint8_t len;
    uint16_t count;
    uint8_t resonance;
    uint8_t activation;
    uint8_t entropy;
    uint32_t timestamp;
    uint32_t first_seen;
    uint8_t doc_score;      // Вклад в DOC [0-255]
} BitMemory;

typedef struct {
    uint8_t activation_hist[HISTORY_SIZE];
    uint8_t entropy_hist[HISTORY_SIZE];
    uint8_t resonance_hist[HISTORY_SIZE];
    uint8_t hist_index;
    uint8_t coherence;      // Когерентность системы [0-255]
    uint8_t energy;         // Энергия системы [0-255]
} BitSystemState;

// === ТИПЫ СЛОЁВ ===
typedef enum {
    LAYER_PERCEPTUAL = 0,
    LAYER_INTEGRATIVE = 1
} LayerType;

// === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ===
static BitMemory memory[MAX_MEM_ENTRIES];
static BitTensor tensors[MAX_TENSORS];
static BitLink links[MAX_LINKS];
static BitSystemState sys_state;

// СЛОИ:
static BitTensor* perceptual_layer[MAX_TENSORS];
static BitTensor* integrative_layer[MAX_TENSORS];
static uint16_t perceptual_count = 0;
static uint16_t integrative_count = 0;

static uint16_t tensor_count = 0;
static uint16_t link_count = 0;
static uint16_t memory_size = 0;

static uint8_t current_DOC = 25;   // 25 ≈ DOC 1.0 (25.5)
static uint32_t interaction_count = 0;
static uint8_t sys_resonance = 128; // 128 ≈ 0.5

// === ПРОТОТИПЫ ФУНКЦИЙ ===
BitTensor* create_bit_tensor(uint16_t rows, uint16_t cols);
BitTensor* create_bit_tensor_typed(uint16_t rows, uint16_t cols, LayerType type);
uint8_t calculate_bit_entropy(BitTensor* t);
uint8_t calculate_bit_similarity(BitTensor* a, BitTensor* b);
void set_bit_tensor(BitTensor* t, uint16_t row, uint16_t col, uint8_t value);
uint8_t get_bit_tensor(BitTensor* t, uint16_t row, uint16_t col);
BitLink* create_bit_link(BitTensor* src, BitTensor* tgt);
void update_bit_network(void);
void generate_bit_pattern(BitLink* link);
void update_bit_DOC(uint16_t active_links, uint32_t total_resonance);
void process_bit_input(const char* input);
BitTensor* get_most_active_bit_tensor(void);
void decode_bit_tensor(BitTensor* t, char* buffer, uint16_t buf_size);
void save_bit_tensor(BitTensor* t);

// === БИТОВАЯ МАТЕМАТИКА ===

// Быстрое вычисление энтропии (битовая версия)
uint8_t calculate_bit_entropy(BitTensor* t) {
    if (!t || !t->data) return 0;
    
    uint32_t total_bits = t->rows * t->cols;
    if (total_bits == 0) return 0;
    
    uint32_t ones = 0;
    
    // Подсчет единичных битов
    for (uint32_t i = 0; i < total_bits; i++) {
        uint32_t byte_idx = i / 8;
        uint8_t bit_idx = i % 8;
        if (BIT_GET(t->data[byte_idx], bit_idx)) ones++;
    }
    
    // Вычисление вероятности P(1)
    uint32_t p1_fixed = (ones << 8) / total_bits;
    uint32_t p0_fixed = 256 - p1_fixed;
    
    // Таблица приближенного log2
    static const uint8_t log2_table[256] = {
        0,0,1,1,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
        6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
        7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
        7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
        7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
        7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7
    };
    
    uint8_t log2_p0 = p0_fixed ? log2_table[p0_fixed] : 0;
    uint8_t log2_p1 = p1_fixed ? log2_table[p1_fixed] : 0;
    
    uint32_t h_fixed = (p0_fixed * log2_p0 + p1_fixed * log2_p1) >> 8;
    
    return (uint8_t)(h_fixed * 32);
}

// Быстрое вычисление сходства тензоров
uint8_t calculate_bit_similarity(BitTensor* a, BitTensor* b) {
    if (!a || !b || !a->data || !b->data) return 0;
    
    uint32_t total_bits_a = a->rows * a->cols;
    uint32_t total_bits_b = b->rows * b->cols;
    uint32_t min_bits = total_bits_a < total_bits_b ? total_bits_a : total_bits_b;
    
    if (min_bits == 0) return 0;
    
    uint32_t matches = 0;
    
    // Быстрое сравнение по байтам
    for (uint32_t i = 0; i < min_bits / 8; i++) {
        uint8_t byte_a = a->data[i];
        uint8_t byte_b = b->data[i];
        
        uint8_t xor_result = byte_a ^ byte_b;
        matches += 8 - __builtin_popcount(xor_result);
    }
    
    // Остаточные биты
    uint32_t remaining_bits = min_bits % 8;
    if (remaining_bits > 0) {
        uint32_t last_byte = min_bits / 8;
        uint8_t mask = (1 << remaining_bits) - 1;
        uint8_t byte_a = a->data[last_byte] & mask;
        uint8_t byte_b = b->data[last_byte] & mask;
        uint8_t xor_result = byte_a ^ byte_b;
        matches += remaining_bits - __builtin_popcount(xor_result);
    }
    
    return (uint8_t)((matches * 255) / min_bits);
}

// Создание битового тензора (базовый)
BitTensor* create_bit_tensor(uint16_t rows, uint16_t cols) {
    if (tensor_count >= MAX_TENSORS || rows == 0 || cols == 0) 
        return NULL;
    
    BitTensor* t = &tensors[tensor_count++];
    uint32_t total_bytes = (rows * cols + 7) / 8;
    
    t->data = (uint8_t*)calloc(total_bytes, 1);
    t->rows = rows;
    t->cols = cols;
    t->resonance = RESONANCE_HALF;
    t->activation = ACTIVATION_MAX / 2;
    t->entropy = 0;
    t->stability = 128;
    t->connections = 0;
    t->last_used = (uint32_t)time(NULL);
    
    return t;
}

// Типизированный создатель — связывает с нужным слоем
BitTensor* create_bit_tensor_typed(uint16_t rows, uint16_t cols, LayerType type) {
    BitTensor* t = create_bit_tensor(rows, cols);
    if (!t) return NULL;
    
    if (type == LAYER_INTEGRATIVE) {
        t->resonance = RESONANCE_HALF + 32;   // 160 — базовый резонанс для слоя 2
        t->activation = ACTIVATION_MAX / 2 + 20;
        integrative_layer[integrative_count++] = t;
    } else {
        // По умолчанию — перцептивный
        perceptual_layer[perceptual_count++] = t;
    }
    return t;
}

// Установка бита в тензоре
void set_bit_tensor(BitTensor* t, uint16_t row, uint16_t col, uint8_t value) {
    if (!t || !t->data) return;
    
    uint32_t bit_index = row * t->cols + col;
    uint32_t byte_index = bit_index / 8;
    uint8_t bit_offset = bit_index % 8;
    
    if (value) {
        BIT_SET(t->data[byte_index], bit_offset);
    } else {
        BIT_CLEAR(t->data[byte_index], bit_offset);
    }
}

// Получение бита из тензора
uint8_t get_bit_tensor(BitTensor* t, uint16_t row, uint16_t col) {
    if (!t || !t->data) return 0;
    
    uint32_t bit_index = row * t->cols + col;
    uint32_t byte_index = bit_index / 8;
    uint8_t bit_offset = bit_index % 8;
    
    return BIT_GET(t->data[byte_index], bit_offset);
}

// Создание битовой связи
BitLink* create_bit_link(BitTensor* src, BitTensor* tgt) {
    if (link_count >= MAX_LINKS || !src || !tgt) return NULL;
    
    BitLink* link = &links[link_count++];
    link->source = src;
    link->target = tgt;
    
    link->strength = calculate_bit_similarity(src, tgt);
    link->resonance = (src->resonance + tgt->resonance) / 2;
    link->weight = (uint16_t)link->strength * link->resonance;
    
    link->created = (uint32_t)time(NULL);
    link->last_active = link->created;
    
    src->connections++;
    tgt->connections++;
    
    return link;
}

// Обновление резонансной сети (с межслойным резонансом)
void update_bit_network(void) {
    // --- МЕЖСЛОЙНЫЙ РЕЗОНАНС: перцептивный → интегративный ---
    for (uint16_t i = 0; i < integrative_count; i++) {
        BitTensor* comp = integrative_layer[i];
        if (comp->activation < 60) continue;

        uint16_t boost = 0;
        for (uint16_t j = 0; j < perceptual_count; j++) {
            BitTensor* p = perceptual_layer[j];
            if (p->activation > 100) {
                uint8_t sim = calculate_bit_similarity(comp, p);
                if (sim > 40) {
                    boost += sim;
                }
            }
        }
        // Применяем	boost с бинарным мета-фильтром
        if (boost > 50) {
            comp->activation = (comp->activation * 240 + (boost >> 2) * 16) >> 8;
            if (comp->activation > 255) comp->activation = 255;
        }
    }

    // --- ОСНОВНОЙ РЕЗОНАНС (внутри всей сети) ---
    uint32_t total_resonance = 0;
    uint16_t active_links = 0;
    
    for (uint16_t i = 0; i < link_count; i++) {
        BitLink* link = &links[i];
        
        if (link->resonance > 25) {
            uint8_t src_act = link->source->activation;
            uint8_t tgt_act = link->target->activation;
            
            uint8_t interaction = (src_act & tgt_act) + ((src_act ^ tgt_act) >> 1);
            
            link->resonance = (link->resonance * 230 + interaction * 25) >> 8;
            
            uint16_t transfer = (link->weight * interaction) >> 8;
            if (link->target->activation + transfer > 255) {
                link->target->activation = 255;
            } else {
                link->target->activation += transfer;
            }
            
            link->last_active = (uint32_t)time(NULL);
            total_resonance += link->resonance;
            active_links++;
            
            if (link->resonance > 200 && (rand() & 0xFF) < 20) {
                generate_bit_pattern(link);
            }
        }
    }
    
    if (active_links > 0) {
        uint8_t avg_resonance = (uint8_t)(total_resonance / active_links);
        sys_resonance = (sys_resonance * 230 + avg_resonance * 25) >> 8;
    }
    
    update_bit_DOC(active_links, total_resonance);
}

// Эмерджентная генерация паттерна (битовая)
void generate_bit_pattern(BitLink* link) {
    if (!link || !link->source || !link->target) return;
    
    BitTensor* emergent = create_bit_tensor_typed(link->source->rows, link->source->cols, LAYER_PERCEPTUAL);
    if (!emergent) return;
    
    uint32_t total_bits = link->source->rows * link->source->cols;
    uint32_t total_bytes = (total_bits + 7) / 8;
    
    for (uint32_t i = 0; i < total_bytes; i++) {
        uint8_t src_byte = link->source->data[i];
        uint8_t tgt_byte = i < (link->target->rows * link->target->cols + 7)/8 ? link->target->data[i] : 0;
        
        uint8_t base = src_byte ^ tgt_byte;
        
        if (link->resonance > 180) {
            uint8_t random_mask = rand() & 0xFF;
            base ^= random_mask & ((link->resonance - 180) >> 1);
        }
        
        if (link->resonance > 150) {
            base = (base & 0xF0) >> 4 | (base & 0x0F) << 4;
            base = (base & 0xCC) >> 2 | (base & 0x33) << 2;
            base = (base & 0xAA) >> 1 | (base & 0x55) << 1;
        }
        
        emergent->data[i] = base;
    }
    
    emergent->resonance = link->resonance;
    emergent->activation = (link->source->activation + link->target->activation) >> 1;
    emergent->entropy = calculate_bit_entropy(emergent);
    emergent->stability = 128;
    emergent->last_used = (uint32_t)time(NULL);
    
    create_bit_link(link->source, emergent);
    create_bit_link(link->target, emergent);
    
    printf("[Генерация] Создан паттерн, резонанс=%u\n", emergent->resonance);
}

// Обновление DOC (битовая версия)
void update_bit_DOC(uint16_t active_links, uint32_t total_resonance) {
    uint8_t avg_link_res = active_links ? (uint8_t)(total_resonance / active_links) : 0;
    uint8_t net_density = (uint8_t)((active_links * 255) / MAX_LINKS);
    
    uint16_t new_DOC = 25;
    
    new_DOC += (avg_link_res * 3) >> 8;
    new_DOC += (net_density * 2) >> 8;
    new_DOC += (sys_resonance * 2) >> 8;
    new_DOC += ((255 - sys_state.coherence) * 1) >> 8;
    
    if (new_DOC > DOC_MAX) new_DOC = DOC_MAX;
    if (new_DOC < 6) new_DOC = 6;
    
    uint8_t max_change = current_DOC >> 3;
    if (new_DOC > current_DOC + max_change) {
        current_DOC += max_change;
    } else if (new_DOC < current_DOC - max_change) {
        current_DOC -= max_change;
    } else {
        current_DOC = (uint8_t)new_DOC;
    }
    
    sys_state.resonance_hist[sys_state.hist_index] = sys_resonance;
    sys_state.hist_index = (sys_state.hist_index + 1) % HISTORY_SIZE;
}

// === ОСНОВНАЯ ОБРАБОТКА ===

void process_bit_input(const char* input) {
    if (!input || !*input) return;
    
    printf("[Вход] %s\n", input);
    
    uint8_t input_len = (uint8_t)strlen(input);
    if (input_len == 0 || input_len > MAX_INPUT) return;

    // --- СЛОЙ 1: Перцептивный ---
    BitTensor* input_tensor = create_bit_tensor_typed(8, input_len, LAYER_PERCEPTUAL);
    if (!input_tensor) return;
    
    for (uint16_t i = 0; i < input_len; i++) {
        uint8_t byte = (uint8_t)input[i];
        for (uint8_t bit = 0; bit < 8; bit++) {
            set_bit_tensor(input_tensor, bit, i, (byte >> bit) & 1);
        }
    }
    
    input_tensor->resonance = 150;
    input_tensor->activation = 180;
    input_tensor->entropy = calculate_bit_entropy(input_tensor);
    
    // Поиск в перцептивном слое
    BitTensor* best_perceptual = NULL;
    for (uint16_t i = 0; i < perceptual_count; i++) {
        if (perceptual_layer[i] == input_tensor) continue;
        uint8_t sim = calculate_bit_similarity(input_tensor, perceptual_layer[i]);
        if (sim > 150 && perceptual_layer[i]->activation > 80) {
            best_perceptual = perceptual_layer[i];
            break;
        }
    }

    if (best_perceptual) {
        create_bit_link(input_tensor, best_perceptual);
        printf("[Слой 1] Резонанс с перцептивным паттерном\n");
    }

    // --- СЛОЙ 2: Интегративный — генерация композитных паттернов ---
    // Условие: ≥2 активных в слое 1 И текущий ДОС достаточно высок
    if (perceptual_count >= 2 && current_DOC >= 15 && (rand() & 0x3F) < current_DOC) {
        // Выбираем 2 самых активных перцептивных тензора
        BitTensor* top1 = NULL, *top2 = NULL;
        uint8_t max1 = 0, max2 = 0;

        for (uint16_t i = 0; i < perceptual_count; i++) {
            uint8_t a = perceptual_layer[i]->activation;
            if (a > max1) {
                max2 = max1; top2 = top1;
                max1 = a; top1 = perceptual_layer[i];
            } else if (a > max2 && perceptual_layer[i] != top1) {
                max2 = a; top2 = perceptual_layer[i];
            }
        }

        if (top1 && top2 && max1 > 100 && max2 > 70) {
            uint16_t rows = top1->rows;
            uint16_t cols = (top1->cols + top2->cols + 1) >> 1;

            BitTensor* comp = create_bit_tensor_typed(rows, cols, LAYER_INTEGRATIVE);
            if (comp) {
                for (uint16_t r = 0; r < rows; r++) {
                    for (uint16_t c = 0; c < cols; c++) {
                        uint8_t b1 = get_bit_tensor(top1, r, c % top1->cols);
                        uint8_t b2 = get_bit_tensor(top2, r, c % top2->cols);
                        uint8_t comp_bit = b1 ^ b2; // XOR — нелинейное смешение
                        if ((rand() & 0x0F) < (current_DOC >> 2)) comp_bit ^= 1; // адаптивный шум
                        set_bit_tensor(comp, r, c, comp_bit);
                    }
                }

                comp->resonance = (top1->resonance + top2->resonance + current_DOC) / 2;
                comp->activation = (top1->activation + top2->activation) * 64 / 255;
                comp->entropy = calculate_bit_entropy(comp);

                create_bit_link(top1, comp);
                create_bit_link(top2, comp);

                printf("[Слой 2] Композит: %ux%u (рез=%u, акт=%u)\n", 
                       rows, cols, comp->resonance, comp->activation);
            }
        }
    }

    update_bit_network();
    
    // === ОТВЕТ: приоритет — слой 2, затем слой 1 ===
    BitTensor* response = NULL;

    // Сначала: слой 2 (интегративный)
    for (uint16_t i = 0; i < integrative_count; i++) {
        BitTensor* t = integrative_layer[i];
        // Пропускаем через бинарную метасигмойду
        if (binary_metasigmoid(t->activation) && t->activation > 120) {
            if (!response || (t->activation * t->resonance) > (response->activation * response->resonance)) {
                response = t;
            }
        }
    }

    // Затем: fallback на любой активный
    if (!response) {
        response = get_most_active_bit_tensor();
    }

    if (response) {
        char output[MAX_OUTPUT];
        decode_bit_tensor(response, output, sizeof(output));
        printf("\nNAIC[ДОС=%u]: %s\n", current_DOC, output);
        save_bit_tensor(response);
    } else {
        printf("\nNAIC: (анализ… ДОС=%u)\n", current_DOC);
    }
}

// Получение наиболее активного тензора (без учёта слоёв — fallback)
BitTensor* get_most_active_bit_tensor(void) {
    BitTensor* most_active = NULL;
    uint16_t max_activity = 0;
    
    for (uint16_t i = 0; i < tensor_count; i++) {
        BitTensor* t = &tensors[i];
        if (!t->data || binary_metasigmoid(t->activation) == 0) continue;
        
        uint16_t activity = (uint16_t)t->activation * t->resonance;
        if (activity > max_activity) {
            max_activity = activity;
            most_active = t;
        }
    }
    return most_active;
}

// Декодирование тензора в текст
void decode_bit_tensor(BitTensor* t, char* buffer, uint16_t buf_size) {
    if (!t || !buffer || buf_size == 0) return;
    
    uint16_t max_bytes = t->cols < buf_size - 1 ? t->cols : buf_size - 1;
    
    for (uint16_t i = 0; i < max_bytes; i++) {
        uint8_t byte = 0;
        for (uint8_t bit = 0; bit < 8; bit++) {
            byte |= (get_bit_tensor(t, bit, i) << bit);
        }
        
        if (byte >= 32 && byte <= 126) {
            buffer[i] = (char)byte;
        } else {
            buffer[i] = '.';
        }
    }
    
    buffer[max_bytes] = '\0';
}

// Сохранение тензора в память
void save_bit_tensor(BitTensor* t) {
    if (!t || !t->data || memory_size >= MAX_MEM_ENTRIES) return;
    
    uint16_t total_bits = t->rows * t->cols;
    uint8_t total_bytes = (uint8_t)((total_bits + 7) / 8);
    
    if (total_bytes > MAX_PATTERN) return;
    
    for (uint16_t i = 0; i < memory_size; i++) {
        if (memory[i].len == total_bytes && 
            memcmp(memory[i].data, t->data, total_bytes) == 0) {
            memory[i].count++;
            memory[i].resonance = (memory[i].resonance + t->resonance) >> 1;
            memory[i].activation = (memory[i].activation + t->activation) >> 1;
            memory[i].entropy = (memory[i].entropy + t->entropy) >> 1;
            memory[i].timestamp = (uint32_t)time(NULL);
            memory[i].doc_score = current_DOC;
            return;
        }
    }
    
    BitMemory* entry = &memory[memory_size++];
    memcpy(entry->data, t->data, total_bytes);
    entry->len = total_bytes;
    entry->count = 1;
    entry->resonance = t->resonance;
    entry->activation = t->activation;
    entry->entropy = t->entropy;
    entry->timestamp = (uint32_t)time(NULL);
    entry->first_seen = entry->timestamp;
    entry->doc_score = current_DOC;
}

// === MAIN ===

int main(void) {
    srand((uint32_t)time(NULL));
    
    memset(&sys_state, 0, sizeof(BitSystemState));
    sys_state.coherence = 128;
    sys_state.energy = 128;
    current_DOC = 10;
    
    printf("Битовая фрактально-резонансная AGI v2.0\n");
    printf("Два слоя: 1) Перцептивный, 2) Интегративный\n");
    printf("Всё — биты. Никаких float. Метасигмойда: порог %d\n", METASIGMOID_THRESHOLD);
    printf("ДОС: %u (%.1f)\n\n", current_DOC, current_DOC / 25.5f);
    
    char input[MAX_INPUT];
    while (1) {
        printf("\n> ");
        fflush(stdout);
        
        if (!fgets(input, sizeof(input), stdin)) break;
        
        size_t len = strlen(input);
        if (len > 0 && input[len-1] == '\n') input[--len] = '\0';
        if (len == 0) continue;
        
        if (strcmp(input, "/exit") == 0) break;
        if (strcmp(input, "/doc") == 0) {
            printf("ДОС: %u (%.1f)\n", current_DOC, current_DOC / 25.5f);
            printf("Резонанс: %u\n", sys_resonance);
            printf("Тензоров: %u (P:%u, I:%u)\n", tensor_count, perceptual_count, integrative_count);
            printf("Связей: %u\n", link_count);
            continue;
        }
        if (strcmp(input, "/mem") == 0) {
            printf("Память: %u записей\n", memory_size);
            uint32_t total_uses = 0;
            for (uint16_t i = 0; i < memory_size; i++) {
                total_uses += memory[i].count;
            }
            printf("Использований: %u\n", total_uses);
            continue;
        }
        if (strcmp(input, "/layer") == 0) {
            printf("Слой 1 (перцептивный): %u тензоров\n", perceptual_count);
            printf("Слой 2 (интегративный): %u тензоров\n", integrative_count);
            uint16_t active2 = 0;
            for (uint16_t i = 0; i < integrative_count; i++) {
                if (binary_metasigmoid(integrative_layer[i]->activation)) active2++;
            }
            printf("→ Активно в слое 2: %u\n", active2);
            continue;
        }
        if (strcmp(input, "/clear") == 0) {
            for (uint16_t i = 0; i < tensor_count; i++) {
                if (tensors[i].data) free(tensors[i].data);
            }
            tensor_count = perceptual_count = integrative_count = link_count = memory_size = 0;
            current_DOC = 25;
            printf("Сеть очищена\n");
            continue;
        }
        
        process_bit_input(input);
    }
    
    printf("\nЗавершение. Финальный ДОС: %u\n", current_DOC);
    
    for (uint16_t i = 0; i < tensor_count; i++) {
        if (tensors[i].data) free(tensors[i].data);
    }
    
    return 0;
}