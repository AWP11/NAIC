#ifndef CORE_H
#define CORE_H

#include <stdint.h>
#include <math.h>

// ===== КОНСТАНТЫ =====
#define GRID_WIDTH 200
#define GRID_HEIGHT 200
#define MAX_CONNECTIONS 10000
#define MAX_CLUSTERS 1000
#define MAX_TENSORS 5000
#define MAX_LINKS 5000
#define MAX_ASSOCIATIONS 5000
#define MAX_MULTI_LINKS 2000
#define MAX_RESONANT_NEURONS 60 
#define MAX_TENSOR_DATA_BYTES 1024
#define RESONANCE_THRESHOLD 128
#define ACTIVATION_DECAY 5

// ===== МАКРОСЫ =====
#define BIT_SET(byte, bit) ((byte) |= (1 << (bit)))
#define BIT_CLEAR(byte, bit) ((byte) &= ~(1 << (bit)))
#define BIT_GET(byte, bit) (((byte) >> (bit)) & 1)

// ===== СТРУКТУРЫ =====

typedef struct {
    uint16_t rows;
    uint16_t cols;
    uint8_t* data;
    uint8_t act;      // активность
    uint8_t res;      // резонанс
    uint8_t ent;      // энтропия
    uint8_t stab;     // стабильность
    uint8_t efficiency; // эффективность
    uint32_t lu;      // время последнего использования
    uint16_t ref_count; // счетчик ссылок
    uint8_t is_concept; // является ли концепцией
    uint16_t cluster_id; // ID кластера
    uint32_t hash;    // хэш для быстрого сравнения
} BitTensor;

typedef struct {
    uint16_t from_x, from_y;
    uint16_t to_x, to_y;
    uint8_t strength;
    uint8_t type;  // 0 - локальная, 1 - диагональная
    uint32_t timestamp;
    float distance;
} Connection;

typedef struct {
    uint16_t id;
    uint16_t count;
    uint16_t capacity;
    uint16_t* neuron_x;
    uint16_t* neuron_y;
    uint16_t center_x, center_y;
    uint8_t avg_resonance;
    uint8_t stability;
    uint32_t last_active;
} Cluster;

typedef struct {
    uint16_t neuron_x;
    uint16_t neuron_y;
    uint16_t tensor_id;
    uint8_t strength;
} TensorClusterAssociation;

typedef struct {
    uint16_t from_x, from_y;
    uint16_t to_x, to_y;
    uint8_t strength;
    uint8_t count;
} MultiNeuronLink;

typedef struct {
    uint16_t from_x, from_y;
    uint16_t to_x, to_y;
    uint8_t strength;
    uint32_t timestamp;
} BitLink;

typedef struct {
    uint16_t count;
    uint16_t capacity;
    uint16_t* neuron_x;
    uint16_t* neuron_y;
    uint8_t type;  // 0 - положительный опыт, 1 - отрицательный
    uint32_t timestamp;
} Experience;

typedef struct {
    uint8_t activation;
    uint8_t resonance;
    uint8_t cluster_id;
    uint8_t is_active;
    uint16_t last_used;
} Neuron;

typedef struct {
    uint8_t* input_data;
    uint32_t input_len;
    uint8_t* output_buffer;
    uint32_t output_capacity;
    uint32_t output_len;
    uint8_t mode;  // 0 - обучение+генерация, 1 - только обучение, 2 - только генерация
    uint8_t learning_rate;

    // --- НОВОЕ ---
    uint16_t grid_region_x;      // Координата X верхнего левого угла региона
    uint16_t grid_region_y;      // Координата Y верхнего левого угла региона
    uint16_t grid_region_width;  // Ширина региона
    uint16_t grid_region_height; // Высота региона
    // --- /НОВОЕ ---
} NeuralPacket;

// ===== ФУНКЦИИ (публичные) =====

// Основная точка входа
int32_t neural_io(NeuralPacket* packet);

// Интерфейс для Python
int32_t process_bytes(const uint8_t* input, uint32_t input_len,
                     uint8_t* output, uint32_t output_capacity,
                     uint8_t mode, uint8_t learning_rate);

// Получение статистики системы
void get_system_stats(uint32_t* stats);

// --- НОВОЕ ---
void forward_forward_consolidation(uint8_t mode);
// --- /НОВОЕ ---

// Сохранение/загрузка состояния
int save_memory_state(const char* filename);
int load_memory_state(const char* filename);

// Очистка памяти
void cleanup_memory(void);

// ===== EMERGENCY MEMORY FUNCTIONS =====

// Сохранение полного дампа в случае краха
int emergency_dump_memory(const char* filename);

// Восстановление из дампа при запуске (если файл существует)
int restore_from_emergency_dump_if_exists(void);

// Настройка обработчика сигналов для аварийного сохранения
void setup_crash_handler(void);

#endif // CORE_H