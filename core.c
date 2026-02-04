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

int32_t neural_io(NeuralPacket* packet) {
    // === БАЗОВЫЕ ПРОВЕРКИ ===
    if (!packet) {
        return -1;
    }

    uint8_t* data = packet->input_data;
    uint32_t len = (packet->input_len < MAX_TENSOR_DATA_BYTES) ? packet->input_len : MAX_TENSOR_DATA_BYTES;
    int32_t result = 0;

    // === ОБУЧЕНИЕ (mode 0 или 1) ===
    if ((packet->mode == 0 || packet->mode == 1) && data && len > 0) {
        for (uint32_t i = 0; i < len && i < (uint32_t)(GRID_WIDTH * GRID_HEIGHT); i++) {
            uint8_t byte = data[i];
            uint16_t x = (i * 17 + byte * 13) % GRID_WIDTH;
            uint16_t y = (i * 13 + byte * 17) % GRID_HEIGHT;

            // Безопасная активация
            if (x < GRID_WIDTH && y < GRID_HEIGHT) {
                uint8_t activation = 100 + (byte % 100) + (packet->learning_rate / 10);
                if (activation > 255) activation = 255;
                activate_neuron(x, y, activation);

                // Связь с предыдущим нейроном
                if (i > 0) {
                    uint16_t prev_x = ((i-1) * 17 + data[i-1] * 13) % GRID_WIDTH;
                    uint16_t prev_y = ((i-1) * 13 + data[i-1] * 17) % GRID_HEIGHT;
                    if (prev_x < GRID_WIDTH && prev_y < GRID_HEIGHT && x < GRID_WIDTH && y < GRID_HEIGHT) {
                        strengthen_connection(prev_x, prev_y, x, y, packet->learning_rate);
                    }
                }

                // Активация битовых позиций
                for (int bit = 0; bit < 8; bit++) {
                    if (byte & (1 << bit)) {
                        uint16_t bit_x = (x + bit * 3) % GRID_WIDTH;
                        uint16_t bit_y = (y + bit * 5) % GRID_HEIGHT;
                        if (bit_x < GRID_WIDTH && bit_y < GRID_HEIGHT) {
                            uint8_t bit_activation = 80 + (packet->learning_rate / 5);
                            if (bit_activation > 255) bit_activation = 255;
                            activate_neuron(bit_x, bit_y, bit_activation);
                        }
                    }
                }
            }
        }

        // Создание тензора памяти
        if (len > 0) {
            BitTensor* memory_tensor = create_tensor_from_bytes(data, len, 8, (uint16_t)(len * 2));
            if (memory_tensor) {
                memory_tensor->act = 180 + (packet->learning_rate / 3);
                if (memory_tensor->act > 255) memory_tensor->act = 255;
                memory_tensor->res = 150 + (packet->learning_rate / 4);
                if (memory_tensor->res > 255) memory_tensor->res = 255;
                memory_tensor->stab = 120 + (packet->learning_rate / 5);
                if (memory_tensor->stab > 255) memory_tensor->stab = 255;
                memory_tensor->lu = current_time;

                uint16_t tx = (data[0] * 17) % GRID_WIDTH;
                uint16_t ty = (data[0] * 13) % GRID_HEIGHT;
                if (tx < GRID_WIDTH && ty < GRID_HEIGHT) {
                    add_tensor_to_neuron(&grid[tx][ty], memory_tensor);
                }
            }
        }

        // === ЦИКЛ РАСПРОСТРАНЕНИЯ И РЕЗОНАНСА (2 итерации) ===
        for (int cycle = 0; cycle < 2; cycle++) {
            // Распространение активации
            for (int x = 1; x < GRID_WIDTH - 1; x++) {
                for (int y = 1; y < GRID_HEIGHT - 1; y++) {
                    uint8_t current_act = grid[x][y].activation;
                    if (current_act > 0) {
                        for (int dx = -1; dx <= 1; dx++) {
                            for (int dy = -1; dy <= 1; dy++) {
                                if (dx == 0 && dy == 0) continue;
                                int nx = x + dx;
                                int ny = y + dy;
                                if (nx >= 0 && nx < GRID_WIDTH && ny >= 0 && ny < GRID_HEIGHT) {
                                    uint8_t propagated = current_act / 3;
                                    if (grid[nx][ny].activation < propagated) {
                                        grid[nx][ny].activation = propagated;
                                        grid[nx][ny].is_active = 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Формирование резонанса и кластеров
            for (int x = 0; x < GRID_WIDTH; x++) {
                for (int y = 0; y < GRID_HEIGHT; y++) {
                    if (grid[x][y].activation > RESONANCE_THRESHOLD) {
                        int resonance_count = 0;
                        for (int dx = -2; dx <= 2; dx++) {
                            for (int dy = -2; dy <= 2; dy++) {
                                if (dx == 0 && dy == 0) continue;
                                int nx = x + dx;
                                int ny = y + dy;
                                if (nx >= 0 && nx < GRID_WIDTH && ny >= 0 && ny < GRID_HEIGHT &&
                                    grid[nx][ny].activation > RESONANCE_THRESHOLD) {
                                    resonance_count++;
                                    strengthen_connection((uint16_t)x, (uint16_t)y, (uint16_t)nx, (uint16_t)ny, 15);
                                }
                            }
                        }
                        if (resonance_count >= 3) {
                            grid[x][y].resonance = 255;
                            find_or_create_cluster((uint16_t)x, (uint16_t)y);
                        }
                    }
                }
            }

            // Затухание активации
            for (int x = 0; x < GRID_WIDTH; x++) {
                for (int y = 0; y < GRID_HEIGHT; y++) {
                    if (grid[x][y].activation > ACTIVATION_DECAY) {
                        grid[x][y].activation -= ACTIVATION_DECAY;
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

        // Собираем активные нейроны (лимит 256 для безопасности стека)
        #define MAX_ACTIVE 256
        uint16_t active_coords[MAX_ACTIVE];
        uint8_t active_values[MAX_ACTIVE];
        uint16_t active_count = 0;

        for (int x = 0; x < GRID_WIDTH && active_count < MAX_ACTIVE; x++) {
            for (int y = 0; y < GRID_HEIGHT && active_count < MAX_ACTIVE; y++) {
                uint8_t value = get_grid_value_with_resonance((uint16_t)x, (uint16_t)y);
                if (value > 50) {
                    active_coords[active_count] = ((uint16_t)x << 8) | (uint16_t)y;
                    active_values[active_count] = value;
                    active_count++;
                }
            }
        }

        // Сортировка по убыванию активации (пузырьковая для простоты и безопасности)
        for (uint16_t i = 0; i < active_count; i++) {
            for (uint16_t j = i + 1; j < active_count; j++) {
                if (active_values[j] > active_values[i]) {
                    uint16_t temp_coord = active_coords[i];
                    uint8_t temp_val = active_values[i];
                    active_coords[i] = active_coords[j];
                    active_values[i] = active_values[j];
                    active_coords[j] = temp_coord;
                    active_values[j] = temp_val;
                }
            }
        }

        // Формирование выходного байтового потока
        generated = (active_count < capacity) ? active_count : capacity;
        for (uint32_t i = 0; i < generated; i++) {
            uint16_t x = active_coords[i] >> 8;
            uint16_t y = active_coords[i] & 0xFF;
            uint8_t activation = active_values[i];
            output[i] = (uint8_t)((x * 17 + y * 13 + activation * 7 + (current_time % 256)) % 256);
        }

        packet->output_len = generated;
        result += (int32_t)generated;
    }

    // === ОЧИСТКА СЛАБЫХ СВЯЗЕЙ (каждые 100 тактов) ===
    if (current_time % 100 == 0 && conn_count > 0) {
        uint32_t write_idx = 0;
        for (uint32_t i = 0; i < conn_count; i++) {
            if (connections[i].strength >= 20) {
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