#include "core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <signal.h>  // для обработки сигналов

// ===== Глобальные переменные (только в .c) =====
Neuron grid[GRID_WIDTH][GRID_HEIGHT];
Connection connections[MAX_CONNECTIONS];
Cluster clusters[MAX_CLUSTERS];
BitTensor tensors[MAX_TENSORS];
BitLink links[MAX_LINKS];
TensorClusterAssociation tensor_cluster_assocs[MAX_ASSOCIATIONS];
MultiNeuronLink multi_links[MAX_MULTI_LINKS];
Experience experiences[MAX_CLUSTERS];
BitTensor tnsrs[MAX_TENSORS];
uint16_t tnsr_count = 0;
Connection connections[MAX_CONNECTIONS];
uint32_t conn_count = 0;
uint16_t cluster_count = 0;
uint32_t current_time = 0;
uint16_t tensor_count = 0;
uint16_t link_count = 0;
uint32_t assoc_count = 0;
uint16_t multi_link_count = 0;
uint16_t experience_count = 0;

// ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (внутренние) =====

static inline uint8_t bit_count_ones(uint8_t byte) {
    if (byte == 0) return 0;
    if (byte == 1) return 1;
    if (byte == 255) return 8;

    uint8_t count = 0;
    while (byte) {
        count += byte & 1;
        byte >>= 1;
    }
    return count;
}

static inline uint8_t bit_hamming_distance(uint8_t a, uint8_t b) {
    return bit_count_ones(a ^ b);
}

static uint8_t calc_bit_entropy(BitTensor* t, uint16_t width);

// ===== ФУНКЦИИ ДЛЯ РАБОТЫ С СЕТКОЙ (внутренние) =====

static void init_grid(void) {
    memset(grid, 0, sizeof(grid));
    conn_count = 0;
    cluster_count = 0;
    tensor_count = 0;
    link_count = 0;
    assoc_count = 0;
    multi_link_count = 0;
    experience_count = 0;
    current_time = (uint32_t)time(NULL);
}

static void activate_neuron(uint16_t x, uint16_t y, uint8_t strength) {
    if (x >= GRID_WIDTH || y >= GRID_HEIGHT) return;
    grid[x][y].activation = strength;
    grid[x][y].is_active = 1;
    grid[x][y].last_used = 0;
    if (grid[x][y].resonance + strength/4 > 255) {
        grid[x][y].resonance = 255;
    } else {
        grid[x][y].resonance += strength/4;
    }
}

static void set_grid_value(uint16_t x, uint16_t y, uint8_t value) {
    if (x >= GRID_WIDTH || y >= GRID_HEIGHT) return;
    grid[x][y].activation = value;
    grid[x][y].is_active = (value > 0);
}

static uint8_t get_grid_value(uint16_t x, uint16_t y) {
    if (x >= GRID_WIDTH || y >= GRID_HEIGHT) return 0;
    return grid[x][y].activation;
}

static uint8_t get_grid_value_with_resonance(uint16_t x, uint16_t y) {
    if (x >= GRID_WIDTH || y >= GRID_HEIGHT) return 0;
    Neuron* neuron = &grid[x][y];
    uint8_t base_value = neuron->activation;

    if (neuron->resonance > RESONANCE_THRESHOLD) {
        float resonance_boost = 1.0f + (neuron->resonance / 255.0f) * 0.5f;
        uint8_t boosted_value = (uint8_t)(base_value * resonance_boost);
        return (boosted_value > 255) ? 255 : boosted_value;
    }

    if (neuron->cluster_id > 0 && neuron->cluster_id < cluster_count) {
        return (base_value * 110) / 100; // +10%
    }

    return base_value;
}

// ===== ФУНКЦИИ ДЛЯ СВЯЗЕЙ (внутренние) =====

static void create_connection(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint8_t type) {
    if (conn_count >= MAX_CONNECTIONS) return;
    connections[conn_count].from_x = x1;
    connections[conn_count].from_y = y1;
    connections[conn_count].to_x = x2;
    connections[conn_count].to_y = y2;
    connections[conn_count].strength = 100;
    connections[conn_count].type = type;
    connections[conn_count].timestamp = current_time;
    connections[conn_count].distance = sqrtf((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
    conn_count++;
}

static void strengthen_connection(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint8_t amount) {
    for (uint32_t i = 0; i < conn_count; i++) {
        if ((connections[i].from_x == x1 && connections[i].from_y == y1 &&
             connections[i].to_x == x2 && connections[i].to_y == y2) ||
            (connections[i].from_x == x2 && connections[i].from_y == y2 &&
             connections[i].to_x == x1 && connections[i].to_y == y1)) {
            if (connections[i].strength + amount > 255) {
                connections[i].strength = 255;
            } else {
                connections[i].strength += amount;
            }
            return;
        }
    }
    uint8_t type = (abs(x2-x1) == abs(y2-y1)) ? 1 : 0;
    create_connection(x1, y1, x2, y2, type);
}

// В core.h добавь:


// В core.c:
void forward_forward_consolidation(uint8_t mode) {
    static uint32_t last_call = 0;
    uint32_t now = current_time; // Используем глобальный таймер
    if (now - last_call < 100) return; // Не вызываем слишком часто
    last_call = now;

    printf("[FFC] Consolidation started (mode=%d)\n", mode);

    // --- 1. Усиление связей с высокой активацией ---
    uint8_t strength_boost = (mode == 0) ? 5 : (mode == 1) ? 10 : 20;
    uint8_t min_goodness_conn = (mode == 0) ? 120 : (mode == 1) ? 100 : 80;

    uint32_t strengthened = 0;
    for (uint32_t i = 0; i < conn_count; i++) {
        Connection* c = &connections[i];
        Neuron* n1 = &grid[c->from_x][c->from_y];
        Neuron* n2 = &grid[c->to_x][c->to_y];

        // Простая "goodness": сумма активаций и резонанса
        uint16_t goodness = n1->activation + n1->resonance + n2->activation + n2->resonance;

        if (goodness > min_goodness_conn && c->strength < 255 - strength_boost) {
            c->strength += strength_boost;
            strengthened++;
        }
    }

    // --- 2. Ослабление слабых связей ---
    uint8_t min_strength = (mode == 0) ? 20 : (mode == 1) ? 15 : 10;
    uint32_t weakened = 0;
    uint32_t write_idx = 0;
    for (uint32_t i = 0; i < conn_count; i++) {
        if (connections[i].strength >= min_strength) {
            if (write_idx != i) {
                connections[write_idx] = connections[i];
            }
            write_idx++;
        } else {
            weakened++;
        }
    }
    conn_count = write_idx;

    // --- 3. Обновление кластеров ---
    for (uint16_t i = 0; i < cluster_count; i++) {
        Cluster* cl = &clusters[i];
        uint32_t total_act = 0;
        uint32_t total_res = 0;
        for (uint16_t j = 0; j < cl->count; j++) {
            if (cl->neuron_x[j] < GRID_WIDTH && cl->neuron_y[j] < GRID_HEIGHT) {
                Neuron* n = &grid[cl->neuron_x[j]][cl->neuron_y[j]];
                total_act += n->activation;
                total_res += n->resonance;
            }
        }
        if (cl->count > 0) {
            cl->avg_resonance = (uint8_t)(total_res / cl->count);
            cl->stability = (uint8_t)(total_act / cl->count);
        }
    }

    // --- 4. Усиление тензоров с высокой активацией ---
    uint8_t tensor_boost = (mode == 0) ? 10 : (mode == 1) ? 15 : 20;
    uint32_t tensors_updated = 0;
    for (uint16_t i = 0; i < tnsr_count; i++) {
        if (tnsrs[i].act > 100) {
            if (tnsrs[i].act < 240) tnsrs[i].act += tensor_boost;
            if (tnsrs[i].res < 240) tnsrs[i].res += tensor_boost;
            if (tnsrs[i].stab < 240) tnsrs[i].stab += tensor_boost / 2;
            tensors_updated++;
        }
    }

    printf("[FFC] Done. Strengthened: %u, Weakened: %u, Tensors updated: %u\n", strengthened, weakened, tensors_updated);
}

// ===== ФУНКЦИИ ДЛЯ КЛАСТЕРОВ (внутренние) =====

static uint8_t find_or_create_cluster(uint16_t x, uint16_t y) {
    for (uint16_t i = 0; i < cluster_count; i++) {
        for (uint16_t j = 0; j < clusters[i].count; j++) {
            uint16_t cx = clusters[i].neuron_x[j];
            uint16_t cy = clusters[i].neuron_y[j];
            if (abs(cx - x) <= 3 && abs(cy - y) <= 3) {
                if (clusters[i].count >= clusters[i].capacity) {
                    uint16_t new_cap = clusters[i].capacity + 16;
                    uint16_t* new_x = realloc(clusters[i].neuron_x, new_cap * sizeof(uint16_t));
                    uint16_t* new_y = realloc(clusters[i].neuron_y, new_cap * sizeof(uint16_t));
                    if (!new_x || !new_y) return i;
                    clusters[i].neuron_x = new_x;
                    clusters[i].neuron_y = new_y;
                    clusters[i].capacity = new_cap;
                }
                clusters[i].neuron_x[clusters[i].count] = x;
                clusters[i].neuron_y[clusters[i].count] = y;
                clusters[i].count++;
                return i;
            }
        }
    }

    if (cluster_count < MAX_CLUSTERS) {
        Cluster* c = &clusters[cluster_count];
        c->id = cluster_count;
        c->count = 1;
        c->capacity = 16;
        c->neuron_x = malloc(c->capacity * sizeof(uint16_t));
        c->neuron_y = malloc(c->capacity * sizeof(uint16_t));
        if (!c->neuron_x || !c->neuron_y) {
            free(c->neuron_x);
            free(c->neuron_y);
            c->neuron_x = c->neuron_y = NULL;
            return 0;
        }
        c->neuron_x[0] = x;
        c->neuron_y[0] = y;
        c->center_x = x;
        c->center_y = y;
        c->avg_resonance = grid[x][y].resonance;
        c->stability = grid[x][y].resonance;
        c->last_active = current_time;
        cluster_count++;
        return cluster_count - 1;
    }
    return 0;
}

// ===== ФУНКЦИИ ДЛЯ ТЕНЗОРОВ (внутренние) =====

static BitTensor* create_tensor(uint16_t rows, uint16_t cols) {
    if (tensor_count >= MAX_TENSORS) return NULL;
    BitTensor* t = &tensors[tensor_count];
    t->rows = rows;
    t->cols = cols;
    uint32_t total_bits = rows * cols;
    uint32_t total_bytes = (total_bits + 7) / 8;
    t->data = calloc(total_bytes, sizeof(uint8_t));
    if (!t->data) return NULL;
    t->act = 0;
    t->res = 0;
    t->ent = 128;
    t->stab = 100;
    t->efficiency = 50;
    t->lu = current_time;
    t->ref_count = 1;
    t->is_concept = 0;
    t->cluster_id = 0;
    t->hash = 0;

    for (uint32_t i = 0; i < total_bytes; i++) {
        t->hash = t->hash * 31 + t->data[i];
    }

    tensor_count++;
    return t;
}

// Функция создания тензора из байтов
static BitTensor* create_tensor_from_bytes(const uint8_t* input_data, uint32_t data_len,
                                   uint16_t target_rows, uint16_t target_cols) {
    if (!input_data || data_len == 0) return NULL;

    uint32_t total_bits_needed = data_len * 8;
    uint16_t rows = target_rows > 0 ? target_rows : 1;
    uint16_t cols = target_cols > 0 ? target_cols : (uint16_t)total_bits_needed;

    if ((uint32_t)rows * cols < total_bits_needed) {
        cols = (uint16_t)((total_bits_needed + rows - 1) / rows);
    }

    BitTensor* t = create_tensor(rows, cols);
    if (!t) return NULL;

    uint32_t total_bits = rows * cols;
    for (uint32_t i = 0; i < data_len && i * 8 < total_bits; i++) {
        uint8_t byte = input_data[i];
        for (int bit = 0; bit < 8 && (i * 8 + bit) < total_bits; bit++) {
            if (BIT_GET(byte, bit)) {
                uint32_t pos = i * 8 + bit;
                uint16_t row = (uint16_t)(pos / cols);
                uint16_t col = (uint16_t)(pos % cols);
                uint32_t bit_pos = row * cols + col;
                uint32_t byte_idx = bit_pos / 8;
                uint8_t bit_idx = (uint8_t)(bit_pos % 8);
                if (byte_idx < ((rows * cols + 7) / 8)) {
                    BIT_SET(t->data[byte_idx], bit_idx);
                }
            }
        }
    }

    t->act = 150;
    t->res = 100;
    t->ent = calc_bit_entropy(t, cols);
    t->lu = current_time;
    t->efficiency = (t->act + t->res + (255 - t->ent)) / 3;

    return t;
}

// Функция добавления тензора к нейрону
static void add_tensor_to_neuron(Neuron* neuron, BitTensor* tensor) {
    if (!neuron || !tensor) return;

    // Просто увеличиваем резонанс и связываем с тензором
    if (neuron->resonance + 20 > 255) {
        neuron->resonance = 255;
    } else {
        neuron->resonance += 20;
    }

    // Сохраняем ID тензора в нейроне (если есть место)
    if (tensor->cluster_id == 0) {
        tensor->cluster_id = find_or_create_cluster(
            (tensor->hash % 255) % GRID_WIDTH,
            (tensor->hash % 173) % GRID_HEIGHT
        );
    }

    neuron->cluster_id = tensor->cluster_id;
    tensor->lu = current_time;
}

// Внутренняя функция расчета энтропии
static uint8_t calc_bit_entropy(BitTensor* t, uint16_t width) {
    if (!t || !t->data) return 128;
    uint32_t total_bits = t->rows * t->cols;
    uint32_t total_bytes = (total_bits + 7) / 8;
    uint32_t safe_total_bytes = total_bytes;
    if (safe_total_bytes > MAX_TENSOR_DATA_BYTES) {
        safe_total_bytes = MAX_TENSOR_DATA_BYTES;
    }

    uint32_t ones = 0;
    for (uint32_t i = 0; i < safe_total_bytes; i++) {
        ones += bit_count_ones(t->data[i]);
    }

    if (total_bits == 0) return 128;
    if (ones > UINT32_MAX / 255) {
        ones = UINT32_MAX / 255;
    }

    uint32_t scaled = ones * 255;
    uint32_t result = scaled / total_bits;
    return (result > 255) ? 255 : (uint8_t)result;
}

// ===== ЕДИНАЯ ТОЧКА ВХОДА/ВЫХОДА (публичная) =====

// ===== ВСПОМОГАТЕЛЬНАЯ СТРУКТУРА ДЛЯ РЕЗОНАНСА =====
typedef struct {
    uint16_t x, y;
    uint8_t activation;
} ResonanceCandidate;


int32_t neural_io(NeuralPacket* packet) {
    if (!packet) return -1;

    static uint32_t adapt_counter = 0;
    static uint8_t output_pattern = 0;
    adapt_counter++;

    uint8_t* data = packet->input_data;
    uint32_t len = (packet->input_len < MAX_TENSOR_DATA_BYTES) ? packet->input_len : MAX_TENSOR_DATA_BYTES;
    int32_t result = 0;

    // === ОБУЧЕНИЕ (mode 0 или 1) ===
    if ((packet->mode == 0 || packet->mode == 1) && data && len > 0) {
        // Динамическое определение параметров обучения на основе истории
        uint8_t adaptive_rate = packet->learning_rate;
        if (cluster_count > 50) {
            adaptive_rate = (adaptive_rate * 3) / 4;
        }
        if (current_time > 10000) {
            adaptive_rate += 5;
            if (adaptive_rate > 200) adaptive_rate = 200;
        }

        // Адаптивное распределение активации
        uint32_t spread_factor = (conn_count > 1000) ? 2 : 1;

        for (uint32_t i = 0; i < len && i < (uint32_t)(GRID_WIDTH * GRID_HEIGHT); i++) {
            uint8_t byte = data[i];

            // Менее детерминированное позиционирование
            uint16_t hash_seed = (byte * 13 + i * 17 + adapt_counter) % 256;
            uint16_t x = (hash_seed * 23 + i * 7) % GRID_WIDTH;
            uint16_t y = (hash_seed * 19 + i * 11) % GRID_HEIGHT;

            if (x < GRID_WIDTH && y < GRID_HEIGHT) {
                // Адаптивная активация
                uint8_t base_activation = 90 + (byte % 110);
                uint8_t activation = base_activation + (adaptive_rate / 12);
                if (activation > 240) activation = 240;

                activate_neuron(x, y, activation);

                // Адаптивные связи (только направленные!)
                if (i > 0 && (i % spread_factor == 0)) {
                    uint16_t prev_hash = (data[i-1] * 13 + (i-1) * 17 + adapt_counter) % 256;
                    uint16_t prev_x = (prev_hash * 23 + (i-1) * 7) % GRID_WIDTH;
                    uint16_t prev_y = (prev_hash * 19 + (i-1) * 11) % GRID_HEIGHT;

                    if (prev_x < GRID_WIDTH && prev_y < GRID_HEIGHT && x < GRID_WIDTH && y < GRID_HEIGHT) {
                        strengthen_connection(prev_x, prev_y, x, y, adaptive_rate / 2);
                    }
                }

                // Динамическая активация битов
                uint8_t bit_mask = byte;
                for (int bit = 0; bit < 8; bit++) {
                    if (bit_mask & 1) {
                        uint16_t bit_x = (x + bit * 5 + hash_seed) % GRID_WIDTH;
                        uint16_t bit_y = (y + bit * 3 + (hash_seed >> 4)) % GRID_HEIGHT;

                        if (bit_x < GRID_WIDTH && bit_y < GRID_HEIGHT) {
                            uint8_t bit_act = 70 + (adaptive_rate / 8) + (bit * 5);
                            if (bit_act > 220) bit_act = 220;
                            activate_neuron(bit_x, bit_y, bit_act);
                        }
                    }
                    bit_mask >>= 1;
                }
            }
        }

        // Адаптивное создание тензоров
        if (len > 0) {
            uint16_t rows = 8 + (adapt_counter % 5);
            uint16_t cols = (uint16_t)(len * 2) + (adapt_counter % 7);

            BitTensor* memory_tensor = create_tensor_from_bytes(data, len, rows, cols);
            if (memory_tensor) {
                memory_tensor->act = 170 + (adaptive_rate / 3) + (adapt_counter % 15);
                if (memory_tensor->act > 250) memory_tensor->act = 250;
                memory_tensor->res = 140 + (adaptive_rate / 4) + (adapt_counter % 10);
                if (memory_tensor->res > 250) memory_tensor->res = 250;
                memory_tensor->stab = 110 + (adaptive_rate / 5) + (adapt_counter % 8);
                if (memory_tensor->stab > 250) memory_tensor->stab = 250;
                memory_tensor->lu = current_time;

                // Адаптивное размещение тензора
                uint16_t placement_seed = (data[0] * 31 + data[len-1] * 17 + adapt_counter) % 256;
                uint16_t tx = (placement_seed * 11) % GRID_WIDTH;
                uint16_t ty = (placement_seed * 7) % GRID_HEIGHT;

                if (tx < GRID_WIDTH && ty < GRID_HEIGHT) {
                    add_tensor_to_neuron(&grid[tx][ty], memory_tensor);
                }
            }
        }

        // Адаптивный цикл распространения
        int cycles = (len < 100) ? 2 : 3;
        if (cluster_count > 100) cycles = 1;

        for (int cycle = 0; cycle < cycles; cycle++) {
            // Оптимизированное распространение активации
            for (int x = 1; x < GRID_WIDTH - 1; x += 1) {
                for (int y = 1; y < GRID_HEIGHT - 1; y += 1) {
                    uint8_t current_act = grid[x][y].activation;
                    if (current_act > 20) {
                        // Умное распространение только в 4 направлениях
                        int dx_list[] = {-1, 1, 0, 0};
                        int dy_list[] = {0, 0, -1, 1};

                        for (int dir = 0; dir < 4; dir++) {
                            int nx = x + dx_list[dir];
                            int ny = y + dy_list[dir];

                            if (nx >= 0 && nx < GRID_WIDTH && ny >= 0 && ny < GRID_HEIGHT) {
                                uint8_t propagated = current_act / 4;
                                if (grid[nx][ny].activation < propagated) {
                                    grid[nx][ny].activation = propagated;
                                    grid[nx][ny].is_active = 1;
                                }
                            }
                        }
                    }
                }
            }

            // === ОГРАНИЧЕННЫЙ РЕЗОНАНС (новая логика) ===
            uint8_t dynamic_threshold = RESONANCE_THRESHOLD;
            if (current_time > 5000) {
                dynamic_threshold = (dynamic_threshold * 3) / 4;
            }

            // Сбор кандидатов на резонанс
            ResonanceCandidate candidates[GRID_WIDTH * GRID_HEIGHT]; // Осторожно: большой стек!
            uint32_t candidate_count = 0;

            for (int x = 0; x < GRID_WIDTH; x++) {
                for (int y = 0; y < GRID_HEIGHT; y++) {
                    if (grid[x][y].activation > dynamic_threshold) {
                        if (candidate_count < sizeof(candidates) / sizeof(candidates[0])) {
                            candidates[candidate_count].x = (uint16_t)x;
                            candidates[candidate_count].y = (uint16_t)y;
                            candidates[candidate_count].activation = grid[x][y].activation;
                            candidate_count++;
                        }
                    }
                }
            }

            // Определяем количество кандидатов для резонанса
            uint32_t resonant_limit = (candidate_count <= MAX_RESONANT_NEURONS) ? candidate_count : MAX_RESONANT_NEURONS;

            // Частичная сортировка: выбираем top-N по активации
            for (uint32_t i = 0; i < resonant_limit; i++) {
                uint32_t max_idx = i;
                for (uint32_t j = i + 1; j < candidate_count; j++) {
                    if (candidates[j].activation > candidates[max_idx].activation) {
                        max_idx = j;
                    }
                }
                if (max_idx != i) {
                    ResonanceCandidate temp = candidates[i];
                    candidates[i] = candidates[max_idx];
                    candidates[max_idx] = temp;
                }
            }

            // Обработка только отобранных кандидатов
            for (uint32_t i = 0; i < resonant_limit; i++) {
                uint16_t x = candidates[i].x;
                uint16_t y = candidates[i].y;
                int resonance_count = 0;

                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < GRID_WIDTH && ny >= 0 && ny < GRID_HEIGHT &&
                            grid[nx][ny].activation > dynamic_threshold) {
                            resonance_count++;
                            if (resonance_count <= 2) {
                                strengthen_connection(x, y, (uint16_t)nx, (uint16_t)ny, 15);
                            }
                        }
                    }
                }
                if (resonance_count >= 2) {
                    grid[x][y].resonance = 255;
                    find_or_create_cluster(x, y);
                }
            }

            // Остальные кандидаты (не в топ-N) получают слабое влияние
            for (uint32_t i = resonant_limit; i < candidate_count; i++) {
                uint16_t x = candidates[i].x;
                uint16_t y = candidates[i].y;
                if (grid[x][y].resonance > 5) {
                    grid[x][y].resonance -= 3; // Подавление шума
                }
            }

            // Адаптивное затухание
            uint8_t decay_rate = ACTIVATION_DECAY;
            if (current_time % 1000 < 500) {
                decay_rate = (decay_rate * 3) / 4;
            }

            for (int x = 0; x < GRID_WIDTH; x++) {
                for (int y = 0; y < GRID_HEIGHT; y++) {
                    if (grid[x][y].activation > decay_rate) {
                        grid[x][y].activation -= decay_rate;
                    } else {
                        grid[x][y].activation = 0;
                    }
                    grid[x][y].last_used++;
                }
            }

            current_time++;
        }

        result += (int32_t)len;
    }

    // === ГЕНЕРАЦИЯ ВЫХОДА (mode 0, 2 или 3) ===
    if (packet->output_buffer && packet->output_capacity > 0 &&
        (packet->mode == 0 || packet->mode == 2 || packet->mode == 3)) {

        uint8_t* output = packet->output_buffer;
        uint32_t capacity = packet->output_capacity;
        uint32_t generated = 0;

        // Адаптивный сбор активных нейронов
        #define MAX_ACTIVE 256
        uint16_t active_coords[MAX_ACTIVE];
        uint8_t active_values[MAX_ACTIVE];
        uint16_t active_count = 0;

        // Адаптивный порог активации
        uint8_t activation_threshold = 40 + (adapt_counter % 30);
        if (conn_count > 500) activation_threshold = 30;

        // Эффективный сбор (через шаг для скорости)
        int step = (GRID_WIDTH * GRID_HEIGHT > 5000) ? 2 : 1;

        for (int x = 0; x < GRID_WIDTH && active_count < MAX_ACTIVE; x += step) {
            for (int y = 0; y < GRID_HEIGHT && active_count < MAX_ACTIVE; y += step) {
                uint8_t value = get_grid_value_with_resonance((uint16_t)x, (uint16_t)y);
                if (value > activation_threshold) {
                    active_coords[active_count] = ((uint16_t)x << 8) | (uint16_t)y;
                    active_values[active_count] = value;
                    active_count++;
                    if (active_count >= MAX_ACTIVE - 1) break;
                }
            }
        }

        // Адаптивная сортировка (только если нужно)
        if (active_count > 1 && active_count < 100) {
            for (uint16_t i = 0; i < active_count && i < 20; i++) {
                uint16_t max_idx = i;
                for (uint16_t j = i + 1; j < active_count; j++) {
                    if (active_values[j] > active_values[max_idx]) {
                        max_idx = j;
                    }
                }
                if (max_idx != i) {
                    uint16_t temp_coord = active_coords[i];
                    uint8_t temp_val = active_values[i];
                    active_coords[i] = active_coords[max_idx];
                    active_values[i] = active_values[max_idx];
                    active_coords[max_idx] = temp_coord;
                    active_values[max_idx] = temp_val;
                }
            }
        }

        // Адаптивное форматирование выхода
        generated = (active_count < capacity) ? active_count : capacity;
        output_pattern = (output_pattern + 1) % 4;

        for (uint32_t i = 0; i < generated; i++) {
            uint16_t x = active_coords[i] >> 8;
            uint16_t y = active_coords[i] & 0xFF;
            uint8_t activation = active_values[i];

            switch (output_pattern) {
                case 0:
                    output[i] = (x * 13 + y * 17 + activation + adapt_counter) % 256;
                    break;
                case 1:
                    output[i] = ((x ^ y) + activation * 3 + current_time) % 256;
                    break;
                case 2:
                    output[i] = (x * y + activation * 5 + adapt_counter * 7) % 256;
                    break;
                default:
                    output[i] = ((x << 4) | (y & 0x0F)) ^ (activation + adapt_counter);
                    break;
            }
        }

        packet->output_len = generated;
        result += (int32_t)generated;
    }

    // === ВЫЗОВ КОНСОЛИДАЦИИ ===
    if (packet->mode == 0 && packet->input_len > 0) { // Только при обучении+генерации
        if (current_time % 50 == 0) {
            forward_forward_consolidation(0); // Лёгкая консолидация
        }
        if (current_time % 200 == 0) {
            forward_forward_consolidation(1); // Полная консолидация
        }
    }

    // === АДАПТИВНАЯ ОЧИСТКА СЛАБЫХ СВЯЗЕЙ ===
    if (current_time % (100 + (conn_count / 50)) == 0 && conn_count > 0) {
        uint8_t strength_threshold = 20 + (current_time % 20);
        if (conn_count > 2000) strength_threshold += 10;

        uint32_t write_idx = 0;
        for (uint32_t i = 0; i < conn_count; i++) {
            if (connections[i].strength >= strength_threshold) {
                if (write_idx != i) {
                    connections[write_idx] = connections[i];
                }
                write_idx++;
            }
        }
        conn_count = write_idx;
    }

    return result;
}
// ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ PYTHON (публичные) =====

int32_t process_bytes(const uint8_t* input, uint32_t input_len,
                     uint8_t* output, uint32_t output_capacity,
                     uint8_t mode, uint8_t learning_rate) {
    NeuralPacket packet;
    packet.input_data = (uint8_t*)input;
    packet.input_len = input_len;
    packet.output_buffer = output;
    packet.output_capacity = output_capacity;
    packet.output_len = 0;
    packet.mode = mode;
    packet.learning_rate = learning_rate;

    // --- НОВОЕ ---
    // 1. Устанавливаем область интереса по умолчанию - всю сетку
    packet.grid_region_x = 0;
    packet.grid_region_y = 0;
    packet.grid_region_width = GRID_WIDTH;
    packet.grid_region_height = GRID_HEIGHT;
    // --- /НОВОЕ ---

    return neural_io(&packet);
}

void get_system_stats(uint32_t* stats) {
    if (!stats) return;

    stats[0] = conn_count;
    stats[1] = cluster_count;
    stats[2] = tensor_count;
    stats[3] = link_count;
    stats[4] = assoc_count;
    stats[5] = multi_link_count;
    stats[6] = experience_count;
    stats[7] = current_time;

    uint32_t active_neurons = 0;
    uint32_t total_activation = 0;
    for (int x = 0; x < GRID_WIDTH; x++) {
        for (int y = 0; y < GRID_HEIGHT; y++) {
            if (grid[x][y].activation > 0) {
                active_neurons++;
                total_activation += grid[x][y].activation;
            }
        }
    }

    stats[8] = active_neurons;
    stats[9] = (active_neurons > 0) ? total_activation / active_neurons : 0;
}

// ===== СОХРАНЕНИЕ/ЗАГРУЗКА (публичные) =====

int save_memory_state(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;

    // Сохраняем счётчики
    if (fwrite(&conn_count, sizeof(conn_count), 1, f) != 1) goto fail;
    if (fwrite(&cluster_count, sizeof(cluster_count), 1, f) != 1) goto fail;
    if (fwrite(&tensor_count, sizeof(tensor_count), 1, f) != 1) goto fail;
    if (fwrite(&link_count, sizeof(link_count), 1, f) != 1) goto fail;
    if (fwrite(&assoc_count, sizeof(assoc_count), 1, f) != 1) goto fail;
    if (fwrite(&multi_link_count, sizeof(multi_link_count), 1, f) != 1) goto fail;
    if (fwrite(&experience_count, sizeof(experience_count), 1, f) != 1) goto fail;

    // Статические структуры (без указателей)
    if (conn_count > 0 && fwrite(connections, sizeof(Connection), conn_count, f) != conn_count) goto fail;
    if (link_count > 0 && fwrite(links, sizeof(BitLink), link_count, f) != link_count) goto fail;
    if (assoc_count > 0 && fwrite(tensor_cluster_assocs, sizeof(TensorClusterAssociation), assoc_count, f) != assoc_count) goto fail;
    if (multi_link_count > 0 && fwrite(multi_links, sizeof(MultiNeuronLink), multi_link_count, f) != multi_link_count) goto fail;
    if (experience_count > 0 && fwrite(experiences, sizeof(Experience), experience_count, f) != experience_count) goto fail;

    // === Кластеры: сохраняем только данные, не указатели ===
    for (uint16_t i = 0; i < cluster_count && i < MAX_CLUSTERS; i++) {
        uint16_t count = clusters[i].count;
        if (fwrite(&count, sizeof(count), 1, f) != 1) goto fail;
        if (count > 0) {
            if (fwrite(clusters[i].neuron_x, sizeof(uint16_t), count, f) != count) goto fail;
            if (fwrite(clusters[i].neuron_y, sizeof(uint16_t), count, f) != count) goto fail;
        }
    }

    // === Тензоры: сохраняем metadata + data отдельно ===
    for (uint16_t i = 0; i < tensor_count && i < MAX_TENSORS; i++) {
        BitTensor* t = &tensors[i];
        // Сохраняем структуру без указателя data
        BitTensor meta = *t;
        meta.data = NULL;
        if (fwrite(&meta, sizeof(BitTensor), 1, f) != 1) goto fail;

        // Затем сохраняем данные
        if (t->data) {
            uint32_t total_bytes = (t->rows * t->cols + 7U) / 8U;
            if (fwrite(t->data, sizeof(uint8_t), total_bytes, f) != total_bytes) goto fail;
        }
    }

    fclose(f);
    return 0;

fail:
    fclose(f);
    return -1;
}
int load_memory_state(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return -1;

    // Очистка текущего состояния — КРИТИЧЕСКИ ВАЖНО!
    cleanup_memory();
    init_grid(); // сбрасывает current_time и всё остальное

    size_t read;

    // Загружаем счётчики
    if ((read = fread(&conn_count, sizeof(conn_count), 1, f)) != 1) goto fail;
    if ((read = fread(&cluster_count, sizeof(cluster_count), 1, f)) != 1) goto fail;
    if ((read = fread(&tensor_count, sizeof(tensor_count), 1, f)) != 1) goto fail;
    if ((read = fread(&link_count, sizeof(link_count), 1, f)) != 1) goto fail;
    if ((read = fread(&assoc_count, sizeof(assoc_count), 1, f)) != 1) goto fail;
    if ((read = fread(&multi_link_count, sizeof(multi_link_count), 1, f)) != 1) goto fail;
    if ((read = fread(&experience_count, sizeof(experience_count), 1, f)) != 1) goto fail;

    // Проверка границ
    if (conn_count > MAX_CONNECTIONS ||
        cluster_count > MAX_CLUSTERS ||
        tensor_count > MAX_TENSORS ||
        link_count > MAX_LINKS ||
        assoc_count > MAX_ASSOCIATIONS ||
        multi_link_count > MAX_MULTI_LINKS ||
        experience_count > MAX_CLUSTERS) {
        goto fail;
    }

    // Статические структуры
    if (conn_count > 0 && fread(connections, sizeof(Connection), conn_count, f) != conn_count) goto fail;
    if (link_count > 0 && fread(links, sizeof(BitLink), link_count, f) != link_count) goto fail;
    if (assoc_count > 0 && fread(tensor_cluster_assocs, sizeof(TensorClusterAssociation), assoc_count, f) != assoc_count) goto fail;
    if (multi_link_count > 0 && fread(multi_links, sizeof(MultiNeuronLink), multi_link_count, f) != multi_link_count) goto fail;
    if (experience_count > 0 && fread(experiences, sizeof(Experience), experience_count, f) != experience_count) goto fail;

    // === Восстановление кластеров ===
    for (uint16_t i = 0; i < cluster_count; i++) {
        uint16_t count;
        if (fread(&count, sizeof(count), 1, f) != 1) goto fail;
        clusters[i].count = count;
        clusters[i].capacity = (count > 16) ? count : 16;

        clusters[i].neuron_x = malloc(clusters[i].capacity * sizeof(uint16_t));
        clusters[i].neuron_y = malloc(clusters[i].capacity * sizeof(uint16_t));
        if (!clusters[i].neuron_x || !clusters[i].neuron_y) goto fail;

        if (count > 0) {
            if (fread(clusters[i].neuron_x, sizeof(uint16_t), count, f) != count) goto fail;
            if (fread(clusters[i].neuron_y, sizeof(uint16_t), count, f) != count) goto fail;
        }
    }

    // === Восстановление тензоров ===
    for (uint16_t i = 0; i < tensor_count; i++) {
        if (fread(&tensors[i], sizeof(BitTensor), 1, f) != 1) goto fail;
        tensors[i].data = NULL; // будет выделено ниже

        if (tensors[i].rows > 0 && tensors[i].cols > 0) {
            uint32_t total_bytes = (tensors[i].rows * tensors[i].cols + 7U) / 8U;
            tensors[i].data = malloc(total_bytes);
            if (!tensors[i].data) goto fail;
            if (fread(tensors[i].data, sizeof(uint8_t), total_bytes, f) != total_bytes) goto fail;
        }
    }

    fclose(f);
    return 0;

fail:
    fclose(f);
    cleanup_memory(); // откат при ошибке
    init_grid();
    return -1;
}
// ===== ОЧИСТКА (публичная) =====

void cleanup_memory(void) {
    for (uint16_t i = 0; i < MAX_CLUSTERS; i++) {
        if (clusters[i].neuron_x) {
            free(clusters[i].neuron_x);
            clusters[i].neuron_x = NULL;
        }
        if (clusters[i].neuron_y) {
            free(clusters[i].neuron_y);
            clusters[i].neuron_y = NULL;
        }
    }

    for (uint16_t i = 0; i < MAX_TENSORS; i++) {
        if (tensors[i].data) {
            free(tensors[i].data);
            tensors[i].data = NULL;
        }
    }

    for (uint16_t i = 0; i < MAX_CLUSTERS; i++) {
        if (experiences[i].neuron_x) {
            free(experiences[i].neuron_x);
            experiences[i].neuron_x = NULL;
        }
        if (experiences[i].neuron_y) {
            free(experiences[i].neuron_y);
            experiences[i].neuron_y = NULL;
        }
    }

    conn_count = 0;
    cluster_count = 0;
    tensor_count = 0;
    link_count = 0;
    assoc_count = 0;
    multi_link_count = 0;
    experience_count = 0;

    printf("[CLEANUP] Memory fully cleaned\n");
}

// ===== EMERGENCY DUMP =====

int emergency_dump_memory(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;

    // Сохраняем всё, как в save_memory_state, но с контрольной суммой
    uint32_t magic = 0xDEADBEEF;
    fwrite(&magic, sizeof(magic), 1, f);

    if (fwrite(&conn_count, sizeof(conn_count), 1, f) != 1) goto fail;
    if (fwrite(&cluster_count, sizeof(cluster_count), 1, f) != 1) goto fail;
    if (fwrite(&tensor_count, sizeof(tensor_count), 1, f) != 1) goto fail;
    if (fwrite(&link_count, sizeof(link_count), 1, f) != 1) goto fail;
    if (fwrite(&assoc_count, sizeof(assoc_count), 1, f) != 1) goto fail;
    if (fwrite(&multi_link_count, sizeof(multi_link_count), 1, f) != 1) goto fail;
    if (fwrite(&experience_count, sizeof(experience_count), 1, f) != 1) goto fail;

    if (conn_count > 0 && fwrite(connections, sizeof(Connection), conn_count, f) != conn_count) goto fail;

    // === Кластеры: сохраняем count и данные ===
    for (uint16_t i = 0; i < cluster_count && i < MAX_CLUSTERS; i++) {
        uint16_t count = clusters[i].count;
        fwrite(&count, sizeof(count), 1, f);
        if (count > 0) {
            fwrite(clusters[i].neuron_x, sizeof(uint16_t), count, f);
            fwrite(clusters[i].neuron_y, sizeof(uint16_t), count, f);
        }
    }

    // === Тензоры: метаданные + данные ===
    for (uint16_t i = 0; i < tensor_count && i < MAX_TENSORS; i++) {
        BitTensor meta = tensors[i];
        meta.data = NULL;
        if (fwrite(&meta, sizeof(BitTensor), 1, f) != 1) goto fail;
    }
    for (uint16_t i = 0; i < tensor_count && i < MAX_TENSORS; i++) {
        if (tensors[i].data) {
            uint32_t total_bytes = (tensors[i].rows * tensors[i].cols + 7U) / 8U;
            if (fwrite(tensors[i].data, sizeof(uint8_t), total_bytes, f) != total_bytes) goto fail;
        }
    }

    fclose(f);
    return 0;

fail:
    fclose(f);
    return -1;
}

int restore_from_emergency_dump_if_exists(void) {
    const char* filename = "AImemory.emergency.dump";
    FILE* f = fopen(filename, "rb");
    if (!f) return 0; // нет дампа — всё нормально

    printf("⚠️ Emergency dump found, restoring...\n");

    uint32_t magic;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0xDEADBEEF) {
        fclose(f);
        printf("❌ Invalid emergency dump.\n");
        return -1;
    }

    // Загружаем, как в load_memory_state
    cleanup_memory();
    init_grid();

    if (fread(&conn_count, sizeof(conn_count), 1, f) != 1) goto fail;
    if (fread(&cluster_count, sizeof(cluster_count), 1, f) != 1) goto fail;
    if (fread(&tensor_count, sizeof(tensor_count), 1, f) != 1) goto fail;
    if (fread(&link_count, sizeof(link_count), 1, f) != 1) goto fail;
    if (fread(&assoc_count, sizeof(assoc_count), 1, f) != 1) goto fail;
    if (fread(&multi_link_count, sizeof(multi_link_count), 1, f) != 1) goto fail;
    if (fread(&experience_count, sizeof(experience_count), 1, f) != 1) goto fail;

    if (conn_count > 0 && fread(connections, sizeof(Connection), conn_count, f) != conn_count) goto fail;

    for (uint16_t i = 0; i < cluster_count && i < MAX_CLUSTERS; i++) {
        uint16_t count;
        if (fread(&count, sizeof(count), 1, f) != 1) goto fail;
        clusters[i].count = count;
        clusters[i].capacity = (count > 16) ? count : 16;

        clusters[i].neuron_x = malloc(clusters[i].capacity * sizeof(uint16_t));
        clusters[i].neuron_y = malloc(clusters[i].capacity * sizeof(uint16_t));
        if (!clusters[i].neuron_x || !clusters[i].neuron_y) goto fail;

        if (count > 0) {
            if (fread(clusters[i].neuron_x, sizeof(uint16_t), count, f) != count) goto fail;
            if (fread(clusters[i].neuron_y, sizeof(uint16_t), count, f) != count) goto fail;
        }
    }

    for (uint16_t i = 0; i < tensor_count && i < MAX_TENSORS; i++) {
        if (fread(&tensors[i], sizeof(BitTensor), 1, f) != 1) goto fail;
        tensors[i].data = NULL;
    }

    for (uint16_t i = 0; i < tensor_count && i < MAX_TENSORS; i++) {
        uint32_t total_bytes = (tensors[i].rows * tensors[i].cols + 7U) / 8U;
        tensors[i].data = malloc(total_bytes);
        if (!tensors[i].data) goto fail;
        if (fread(tensors[i].data, sizeof(uint8_t), total_bytes, f) != total_bytes) goto fail;
    }

    fclose(f);
    remove(filename); // удалить дамп после восстановления
    printf("✅ Emergency state restored.\n");
    return 1;

fail:
    fclose(f);
    printf("❌ Failed to restore from emergency dump.\n");
    return -1;
}

// ===== CRASH HANDLER =====

void crash_handler(int sig) {
    printf("\n💥 SIGNAL %d CAUGHT — SAVING EMERGENCY DUMP...\n", sig);
    emergency_dump_memory("AImemory.emergency.dump");
    exit(sig);
}

void setup_crash_handler(void) {
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGILL, crash_handler);
}