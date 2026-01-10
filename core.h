// core.h - Заголовочный файл для универсального модуля работы с бинарными тензорами с самоорганизацией памяти
#ifndef CORE_H
#define CORE_H

#include <stdint.h>
#include <time.h>

// ===== Настройки =====
#define MAX_INPUT 4096
#define MAX_OUTPUT 8192
#define MAX_PATTERN 1024
#define MAX_MEM_ENTRIES 1024
#define MAX_TENSORS 1024
#define MAX_LINKS 4096
#define HISTORY_SIZE 2048
#define LOW_ACT_THRESHOLD 45
#define MEM_REDUCE_INTERVAL 30
#define WORKING_MEM_SIZE 512
#define DROPOUT_RATE 1
#define LINK_STRENGTH_INC 20
#define LINK_STRENGTH_DEC 2
#define LINK_MIN_STRENGTH 20
#define LINK_MAX_STRENGTH 240
#define ENCODER_QUALITY 5
#define RES_MAX 255
#define RES_HALF 128
#define ACT_MAX 255
#define SIM_FUZZINESS_PERCENT 2

// === НАСТРОЙКИ САМООРГАНИЗАЦИИ ПАМЯТИ ===
#define MAX_CLUSTERS 128
#define MAX_EPISODES 512
#define CLUSTER_THRESHOLD 150      // Порог схожести для кластеризации (0-255)
#define MERGE_THRESHOLD 180        // Порог для слияния тензоров
#define CONCEPT_CREATION_THRESHOLD 3  // Минимальный размер кластера для концепции
#define EPISODE_MIN_LENGTH 3       // Минимальная длина эпизода
#define CONSOLIDATION_INTERVAL 30  // Интервал консолидации (сек)
#define SELF_ORG_INTERVAL 5        // Интервал самоорганизации (сек)

// ===== Битовые макросы =====
#define BIT_SET(byte, bit) ((byte) |= (1U << (bit)))
#define BIT_CLEAR(byte, bit) ((byte) &= ~(1U << (bit)))
#define BIT_TOGGLE(byte, bit) ((byte) ^= (1U << (bit)))
#define BIT_GET(byte, bit) (((byte) >> (bit)) & 1U)
#define BIT_NOT(byte) (~(byte))
#define BIT_AND(a, b) ((a) & (b))
#define BIT_XOR(a, b) ((a) ^ (b))
#define BIT_OR(a, b) ((a) | (b))

// ===== Структуры =====
typedef struct BitTensor BitTensor;
typedef struct BitLink BitLink;
typedef struct BitMemory BitMemory;
typedef struct WorkingMemoryEntry WorkingMemoryEntry;
typedef struct BitSystemState BitSystemState;
typedef struct SystemGoals SystemGoals;

// === НОВЫЕ СТРУКТУРЫ ДЛЯ САМООРГАНИЗАЦИИ ===
typedef struct MemoryCluster MemoryCluster;
typedef struct EpisodeMemory EpisodeMemory;
typedef struct MemoryConcept MemoryConcept;

struct BitTensor {
    uint8_t* data;            // Битовые данные тензора
    uint16_t rows;            // Количество строк
    uint16_t cols;            // Количество столбцов
    uint8_t res;              // Резонанс (0-255)
    uint8_t act;              // Активность (0-255)
    uint8_t ent;              // Энтропия (0-255)
    uint8_t stab;             // Стабильность (0-255)
    uint16_t conn;            // Количество связей
    uint32_t lu;              // Последнее обновление (timestamp)
    uint8_t mem_red;          // Флаг сжатия памяти
    uint8_t efficiency;       // Эффективность (0-255)
    uint32_t compute_cost;    // Вычислительная стоимость
    uint8_t goal_active;      // Флаг активной цели
    uint8_t dropout;          // Флаг dropout
    uint8_t cluster_id;       // ID кластера (0 = не в кластере)
    uint8_t is_concept;       // Флаг концепции
};

struct BitLink {
    BitTensor* src;           // Исходный тензор
    BitTensor* tgt;           // Целевой тензор
    uint8_t strength;         // Сила связи (0-255)
    uint8_t res;              // Резонанс связи
    uint16_t weight;          // Вес связи
    uint32_t ts;              // Время создания
    uint32_t last_act;        // Время последней активации
    uint16_t use_count;       // Количество использований
    uint16_t success_count;   // Количество успешных использований
    uint8_t semantic_type;    // Тип связи: 0=обычная, 1=внутрикластерная, 2=межкластерная, 3=концептуальная
};

struct BitMemory {
    uint8_t data[MAX_PATTERN]; // Данные паттерна
    uint8_t len;               // Длина в байтах
    uint16_t count;            // Количество повторений
    uint8_t res;               // Средний резонанс
    uint8_t act;               // Средняя активность
    uint8_t ent;               // Средняя энтропия
    uint32_t ts;               // Время последнего обновления
    uint8_t cluster_id;        // ID кластера (для организации)
};

struct WorkingMemoryEntry {
    BitTensor* tensor;         // Указатель на тензор
    uint32_t timestamp;        // Время добавления
    uint8_t priority;          // Приоритет (0-255)
    uint8_t access_count;      // Количество обращений
    uint8_t episode_marker;    // Маркер для эпизодной памяти
};

struct BitSystemState {
    uint8_t act_hist[HISTORY_SIZE]; // История активности
    uint8_t ent_hist[HISTORY_SIZE]; // История энтропии
    uint8_t res_hist[HISTORY_SIZE]; // История резонанса
    uint8_t hist_idx;                // Текущий индекс истории
    uint8_t coh;                     // Когерентность системы
    uint8_t energy;                  // Энергия системы
    uint32_t consolidation_timer;    // Таймер консолидации
    uint32_t self_org_timer;         // Таймер самоорганизации
};

struct SystemGoals {
    uint8_t target_efficiency;       // Целевая эффективность
    uint8_t energy_saving_mode;      // Режим энергосбережения
    uint32_t total_compute_cost;     // Общая вычислительная стоимость
    uint32_t efficiency_gain;        // Прирост эффективности
    uint8_t dropout_enabled;         // Включен ли dropout
    uint8_t self_organization_enabled; // Включена ли самоорганизация
    uint8_t memory_consolidation_mode; // Режим консолидации памяти
};

// === СТРУКТУРЫ САМООРГАНИЗАЦИИ ===
struct MemoryCluster {
    uint16_t tensor_indices[64];    // Индексы тензоров в кластере
    uint8_t cluster_id;             // ID кластера (1-255)
    uint8_t centroid[256];          // Центроид кластера (упрощенный вектор признаков)
    uint16_t size;                  // Размер кластера
    uint8_t stability;              // Стабильность кластера (0-255)
    uint32_t last_access;           // Последний доступ к кластеру
    uint8_t category;               // Категория: 0=неопределено, 1=действие, 2=состояние, 3=концепция
    uint8_t activation_level;       // Уровень активации кластера
    uint32_t creation_time;         // Время создания
    uint16_t link_count;            // Количество внешних связей
};

struct EpisodeMemory {
    uint16_t sequence[256];         // Последовательность тензоров/действий
    uint8_t context_hash[16];       // Хеш контекста эпизода
    uint32_t start_time;            // Время начала
    uint32_t end_time;              // Время окончания
    uint8_t success_score;          // Оценка успешности (0-255)
    uint8_t importance;             // Важность эпизода (0-255)
    uint8_t length;                 // Длина последовательности
    uint8_t reward_context;         // Контекст награды
    uint32_t last_recall;           // Время последнего вспоминания
    uint16_t recall_count;          // Количество вспоминаний
};

struct MemoryConcept {
    BitTensor* concept_tensor;      // Тензор-концепция
    uint16_t member_indices[32];    // Индексы членов концепции
    uint8_t member_count;           // Количество членов
    uint8_t abstraction_level;      // Уровень абстракции (1-3)
    uint8_t coherence;              // Связность концепции
    uint32_t last_used;             // Время последнего использования
};

// ===== Контекстная индексация связей =====
typedef struct {
    uint16_t tensor_idx;            // Индекс тензора
    uint16_t link_indices[32];      // Индексы связей
    uint8_t link_count;             // Количество связей
    uint8_t cluster_links[16];      // Связи внутри кластера
    uint8_t cluster_link_count;     // Количество внутрикластерных связей
} TensorLinks;

// ===== Типы для поиска =====
typedef enum {
    SEARCH_MOST_ACTIVE,           // Поиск самого активного
    SEARCH_RESONANT,              // Поиск самого резонансного
    SEARCH_EFFICIENT,             // Поиск самого эффективного
    SEARCH_CUSTOM_SCORE,          // Поиск по пользовательской функции
    SEARCH_BY_CLUSTER,            // Поиск по кластеру
    SEARCH_CONCEPTUAL             // Поиск концепций
} SearchStrategy;

typedef uint32_t (*ScoreFunction)(BitTensor* t, void* context);

// ===== Глобальные переменные (экспортируемые) =====
extern BitMemory memo[MAX_MEM_ENTRIES];
extern BitTensor tnsrs[MAX_TENSORS];
extern BitLink lnks[MAX_LINKS];
extern WorkingMemoryEntry working_mem[WORKING_MEM_SIZE];
extern TensorLinks tensor_links[MAX_TENSORS];
extern BitSystemState sstate;
extern SystemGoals goals;

// === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ САМООРГАНИЗАЦИИ ===
extern MemoryCluster clusters[MAX_CLUSTERS];
extern EpisodeMemory episodes[MAX_EPISODES];
extern MemoryConcept concepts[64];

extern uint16_t memo_size;
extern uint16_t tnsr_count;
extern uint16_t lnk_count;
extern uint8_t sys_res;
extern uint32_t interaction_count;
extern uint32_t last_mem_check_ts;
extern uint8_t working_mem_count;

extern uint16_t cluster_count;
extern uint16_t episode_count;
extern uint8_t concept_count;
extern uint32_t global_context_hash;
extern uint32_t last_cluster_reorg;
extern uint8_t next_cluster_id;

// ===== Прототипы функций =====

// === Основные функции работы с тензорами ===
BitTensor* create_tnsr(uint16_t rows, uint16_t cols);
void reduce_tnsr_mem(BitTensor* t);
uint8_t fast_log2(uint32_t x);
uint8_t calc_bit_ent(BitTensor* t, uint32_t unit_sz);
uint8_t calc_bit_sim(BitTensor* a, BitTensor* b);
uint8_t calc_res_match(BitTensor* a, BitTensor* b);
uint8_t calculate_efficiency(BitTensor* t);
void update_efficiency_goal(void);
BitTensor* find_efficient_match(BitTensor* input);
void optimize_tensor(BitTensor* t);
void update_bit_net_with_goals(void);
BitTensor* find_significant_tensor(SearchStrategy strategy, void* context);
void save_tnsr(BitTensor* t);
void add_to_working_memory(BitTensor* t);
BitTensor* get_from_working_memory(uint8_t min_priority);
void apply_dropout(void);

// === Функции работы со связями ===
BitLink* create_link(BitTensor* src, BitTensor* tgt);
void update_link_strength(BitLink* link, uint8_t was_successful);
void decay_unused_links(void);

// === Функции обучения ===
void learn_by_binary_update(BitTensor* target, const uint8_t* input_data, uint16_t input_len);
void prevent_overfitting_by_bit_shift(BitTensor* t);
void self_reflect_on_thought(void);

// === Функции сериализации ===
int save_state_to_file(const char* filename);
int load_state_from_file(const char* filename);
uint16_t tensor_to_index(BitTensor* t);
BitTensor* index_to_tensor(uint16_t idx);

// === Новые/обновленные функции ===
void build_link_index(void);
uint8_t check_context_fit(BitTensor* a, BitTensor* b, BitLink* link);
void fast_contextual_activation(BitTensor* context);
void update_thought_stream(void);
void aggressive_memory_cleanup(void);

// === ФУНКЦИИ САМООРГАНИЗАЦИИ ПАМЯТИ ===

// Кластеризация и организация памяти
void self_organize_memory_clusters(void);
void semantic_memory_binding(void);
void memory_consolidation(void);
void reorganize_clusters_by_activity(void);
void prune_weak_clusters(void);
void reassign_tensors_to_clusters(void);

// Работа с тензорами в контексте самоорганизации
void merge_tensors(BitTensor* a, BitTensor* b);
void transfer_links(BitTensor* from, BitTensor* to);
BitTensor* find_center_tensor_in_cluster(MemoryCluster* cluster);
BitLink* find_or_create_link(BitTensor* src, BitTensor* tgt);

// Концепции и абстракции
void create_concept_from_cluster(MemoryCluster* cluster);
void save_concept(BitTensor* concept, uint8_t cluster_id);

// Эпизодная память
void create_episode_from_working_memory(void);
void activate_relevant_episodes(void);
void forget_unimportant_episodes(void);
void reinforce_successful_episodes(void);
void extract_patterns_from_episodes(void);

// Вспомогательные функции самоорганизации
void update_cluster_stability(MemoryCluster* cluster);
uint8_t calculate_cluster_coherence(MemoryCluster* cluster);
void create_semantic_links_between_clusters(MemoryCluster* c1, MemoryCluster* c2);
uint32_t calculate_context_hash(void);
uint8_t is_similar_context(uint8_t* ctx1, uint8_t* ctx2);

// === Внешний API (для Python и других языков) ===
uint8_t sync_and_get_recommendation(uint8_t* vector_data, int vector_len, uint8_t last_action);
uint8_t get_global_resonance(void);
void learn_from_data(uint8_t* vector_data, int len, uint8_t action_idx, uint8_t reward);

// === Утилиты отладки и мониторинга ===
void print_tensor_info(BitTensor* t);
void print_cluster_info(MemoryCluster* cluster);
void print_system_status(void);
void reset_memory_system(void);

// === Константы для экспорта в Python ===
#ifdef __cplusplus
extern "C" {
#endif

// Экспорт констант для Python
#define CORE_MAX_TENSORS MAX_TENSORS
#define CORE_MAX_LINKS MAX_LINKS
#define CORE_MAX_CLUSTERS MAX_CLUSTERS
#define CORE_MAX_EPISODES MAX_EPISODES
#define CORE_WORKING_MEM_SIZE WORKING_MEM_SIZE
#define CORE_ACT_MAX ACT_MAX
#define CORE_RES_MAX RES_MAX

#ifdef __cplusplus
}
#endif

#endif // CORE_H