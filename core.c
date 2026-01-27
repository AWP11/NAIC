// core.c/h - универсальный модуль для работы с бинарными тензорами с самоорганизацией памяти
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

// ===== Настройки =====
#define MAX_PATTERN 1024
#define MAX_MEM_ENTRIES 1024
#define MAX_TENSORS 1024
#define MAX_LINKS 2048
#define HISTORY_SIZE 2048
#define LOW_ACT_THRESHOLD 45
#define MEM_REDUCE_INTERVAL 30
#define WORKING_MEM_SIZE 512
#define DROPOUT_RATE 1
#define LINK_STRENGTH_INC 20
#define LINK_STRENGTH_DEC 3
#define LINK_MIN_STRENGTH 15
#define LINK_MAX_STRENGTH 240
#define LEARN_SLICE_SIZE 64 // Размер среза данных для обновления за один шаг
#define ENCODER_QUALITY 5
#define RES_MAX 255
#define RES_HALF 128
#define ACT_MAX 255
#define SIM_FUZZINESS_PERCENT 2

// === НАСТРОЙКИ САМООРГАНИЗАЦИИ ПАМЯТИ ===
#define MAX_CLUSTERS 128
#define MAX_EPISODES 512
#define CLUSTER_THRESHOLD 150  // Порог схожести для кластеризации (0-255)
#define MERGE_THRESHOLD 180    // Порог для слияния тензоров
#define CONCEPT_CREATION_THRESHOLD 3  // Минимальный размер кластера для концепции
#define EPISODE_MIN_LENGTH 3   // Минимальная длина эпизода
#define CONSOLIDATION_INTERVAL 30  // Интервал консолидации (сек)
#define SELF_ORG_INTERVAL 5    // Интервал самоорганизации (сек)

// === НАСТРОЙКИ ПОТОКОВ МЫСЛЕЙ ===
#define MAX_THOUGHT_STREAMS 6        // Уменьшено: меньше, но качественнее
#define MAX_THOUGHT_CHAIN_LENGTH 12  // Уменьшено: короче, но глубже
#define THOUGHT_STREAM_LIFETIME 180  // 3 минуты: быстрее "забывает" слабые мысли

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

// === НОВАЯ СТРУКТУРА: Поток мыслей ===
typedef struct {
    BitTensor* thought_chain[MAX_THOUGHT_CHAIN_LENGTH];
    uint8_t chain_length;
    uint32_t timestamp;
    uint8_t coherence;
    uint8_t abstraction_level;
    uint16_t activation_counter;
    uint8_t is_active;
    uint8_t recursion_depth;  // НОВОЕ: глубина рекурсивного мышления
    BitTensor* meta_reflections[3];  // НОВОЕ: мета-тензоры рефлексии
} ThoughtStream;


// ===== Контекстная индексация связей =====
typedef struct {
    uint16_t tensor_idx;
    uint16_t link_indices[32];
    uint8_t link_count;
    uint8_t cluster_links[16];      // Связи внутри кластера
    uint8_t cluster_link_count;
} TensorLinks;

// ===== Типы для поиска =====
typedef enum {
    SEARCH_MOST_ACTIVE,
    SEARCH_RESONANT,
    SEARCH_EFFICIENT,
    SEARCH_CUSTOM_SCORE,
    SEARCH_BY_CLUSTER,           // Поиск по кластеру
    SEARCH_CONCEPTUAL            // Поиск концепций
} SearchStrategy;

typedef uint32_t (*ScoreFunction)(BitTensor* t, void* context);

// ===== Структуры данных =====
struct BitTensor {
    uint8_t* data;
    uint16_t rows;
    uint16_t cols;
    uint8_t res;
    uint8_t act;
    uint8_t ent;      // <-- Текущая энтропия
    uint8_t ent_last; // <-- Добавлено: Предыдущее значение энтропии
    uint8_t stab;
    uint16_t conn;
    uint32_t lu;
    uint8_t mem_red;
    uint8_t efficiency;
    uint16_t forward;
    uint32_t compute_cost;
    uint8_t goal_active;
    uint8_t dropout;
    uint8_t cluster_id;      // ID кластера, к которому принадлежит
    uint8_t is_concept;      // Флаг, что это концепция
};

struct BitLink {
    BitTensor* src;
    BitTensor* tgt;
    uint8_t strength;
    uint8_t res;
    uint16_t weight;
    uint32_t ts;
    uint32_t last_act;
    uint16_t use_count;
    uint16_t success_count;
    uint8_t semantic_type;   // Тип семантической связи: 0=обычная, 1=внутрикластерная, 2=межкластерная, 3=концептуальная
};

struct BitMemory {
    uint8_t data[MAX_PATTERN];
    uint8_t len;
    uint16_t count;
    uint8_t res;
    uint8_t act;
    uint8_t ent;
    uint32_t ts;
    uint8_t cluster_id;      // Для организации паттернов памяти
};

struct WorkingMemoryEntry {
    BitTensor* tensor;
    uint32_t timestamp;
    uint8_t priority;
    uint8_t access_count;
    uint8_t episode_marker;  // Отметка для эпизодной памяти
};

struct BitSystemState {
    uint8_t act_hist[HISTORY_SIZE];
    uint8_t ent_hist[HISTORY_SIZE];
    uint8_t res_hist[HISTORY_SIZE];
    uint8_t hist_idx;
    uint8_t coh;
    uint8_t energy;
    uint32_t consolidation_timer;
    uint32_t self_org_timer;
    uint32_t last_thought_cleanup;
};

struct SystemGoals {
    uint8_t target_efficiency;
    uint8_t energy_saving_mode;
    uint32_t total_compute_cost;
    uint32_t efficiency_gain;
    uint8_t dropout_enabled;
    uint8_t self_organization_enabled;  // Включена ли самоорганизация
    uint8_t memory_consolidation_mode;  // Режим консолидации: 0=авто, 1=агрессивный, 2=консервативный
    uint8_t thought_stream_enabled;     // Включены ли потоки мыслей
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
    uint32_t start_time;
    uint32_t end_time;
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
    uint8_t member_count;
    uint8_t abstraction_level;      // Уровень абстракции (1-3)
    uint8_t coherence;              // Связность концепции
    uint32_t last_used;
};

// ===== Глобальные состояния =====
BitMemory memo[MAX_MEM_ENTRIES] = {0};
BitTensor tnsrs[MAX_TENSORS] = {0};
BitLink lnks[MAX_LINKS] = {0};
WorkingMemoryEntry working_mem[WORKING_MEM_SIZE] = {0};
TensorLinks tensor_links[MAX_TENSORS] = {0};
BitSystemState sstate = {0}; // Все поля инициализируются нулями
SystemGoals goals = {150, 0, 0, 0, 1, 1, 0, 1};

// === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ САМООРГАНИЗАЦИИ ===
MemoryCluster clusters[MAX_CLUSTERS];
EpisodeMemory episodes[MAX_EPISODES];
MemoryConcept concepts[64];

// === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ПОТОКОВ МЫСЛЕЙ ===
ThoughtStream thought_streams[MAX_THOUGHT_STREAMS];
uint8_t active_thought_streams = 0;

uint16_t memo_size = 0;
uint16_t tnsr_count = 0;
uint16_t lnk_count = 0;
uint8_t sys_res = RES_HALF;
uint32_t interaction_count = 0;
uint32_t last_mem_check_ts = 0;
uint8_t working_mem_count = 0;
uint16_t cluster_count = 0;
uint16_t episode_count = 0;
uint8_t concept_count = 0;
uint32_t global_context_hash = 0;
uint32_t last_cluster_reorg = 0;
uint8_t next_cluster_id = 1;

// ===== Прототипы =====
// Основные функции
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
BitLink* create_link(BitTensor* src, BitTensor* tgt);
void update_link_strength(BitLink* link, uint8_t was_successful);
void decay_unused_links(void);
void learn_by_binary_update(BitTensor* target, const uint8_t* input_data, uint16_t input_len);
void prevent_overfitting_by_bit_shift(BitTensor* t);
void self_reflect_on_thought(void);
int save_state_to_file(const char* filename);
int load_state_from_file(const char* filename);
uint16_t tensor_to_index(BitTensor* t);
BitTensor* index_to_tensor(uint16_t idx);

// Новые/обновленные функции
void simple_memory_consolidation(void); 
void create_concept_from_tensors(BitTensor* t1, BitTensor* t2);
void build_link_index(void);
uint8_t check_context_fit(BitTensor* a, BitTensor* b, BitLink* link);
void fast_contextual_activation(BitTensor* context);
void update_thought_stream(void);
void aggressive_memory_cleanup(void);

// === ПРОТОТИПЫ ФУНКЦИЙ САМООРГАНИЗАЦИИ ===
void self_organize_memory_clusters(void);
void semantic_memory_binding(void);
void memory_consolidation(void);
void merge_tensors(BitTensor* a, BitTensor* b);
void create_concept_from_cluster(MemoryCluster* cluster);
void create_episode_from_working_memory(void);
void activate_relevant_episodes(void);
BitTensor* find_center_tensor_in_cluster(MemoryCluster* cluster);
BitLink* find_or_create_link(BitTensor* src, BitTensor* tgt);
void transfer_links(BitTensor* from, BitTensor* to);
void save_concept(BitTensor* concept, uint8_t cluster_id);
void reorganize_clusters_by_activity(void);
void prune_weak_clusters(void);
void update_cluster_stability(MemoryCluster* cluster);
uint8_t calculate_cluster_coherence(MemoryCluster* cluster);
void reassign_tensors_to_clusters(void);
void create_semantic_links_between_clusters(MemoryCluster* c1, MemoryCluster* c2);
void extract_patterns_from_episodes(void);
void forget_unimportant_episodes(void);
void reinforce_successful_episodes(void);
uint32_t calculate_context_hash(void);
uint8_t is_similar_context(uint8_t* ctx1, uint8_t* ctx2);

// === ПРОТОТИПЫ ФУНКЦИЙ ПОТОКОВ МЫСЛЕЙ ===
void form_thought_chain(void);
void evolve_thought_stream(ThoughtStream* stream);
void prune_old_thought_streams(void);
BitTensor* find_association(BitTensor* current);
uint8_t calculate_chain_coherence(ThoughtStream* stream);
void create_abstraction_from_chain(ThoughtStream* stream);
const char* get_current_thought(void);
void activate_thought_stream(void);
void check_working_memory_for_transfer(void);
void enhance_semantic_linking(void);
void force_transfer_to_longterm(BitTensor* tensor);
void activate_longterm_to_working(void);

// === ДРУГИЕ ФУНКЦИИ ===
void forward_forward_consolidation(uint8_t mode);

// ===== Вспомогательные функции =====
uint8_t fast_log2(uint32_t x) {
    if (x == 0) return 0;
    uint32_t orig = x;
    uint8_t log = 0;
    if (x & 0xFFFF0000) { x >>= 16; log += 16; }
    if (x & 0xFF00) { x >>= 8; log += 8; }
    if (x & 0xF0) { x >>= 4; log += 4; }
    if (x & 0xC) { x >>= 2; log += 2; }
    if (x & 0x2) { x >>= 1; log += 1; }
    uint32_t low_pow = 1UL << log;
    uint32_t diff = (1UL << (log + 1)) - low_pow;
    if(diff == 0) diff = 1;
    uint8_t fraction = ((orig - low_pow) * 255UL) / diff;
    fraction = (fraction > 255) ? 255 : fraction;
    return (log << 3) | (fraction >> 5);
}

/**
 * @brief Вычисляет энтропию битового тензора.
 *
 * @param t Указатель на структуру BitTensor. Должен быть валидным и указывать на инициализированную структуру.
 * @param unit_sz Множитель, влияющий на итоговое значение энтропии. Должен быть != 0.
 * @return uint8_t Вычисленное значение энтропии в диапазоне [0, 255].
 *                 Возвращает 0, если произошла ошибка (t == NULL, t->data == NULL, unit_sz == 0, total_bits == 0).
 */
uint8_t calc_bit_ent(BitTensor* t, uint32_t unit_sz) {
    // 1. Проверка на NULL указатели и нулевой размер единицы
    if (!t || !t->data || unit_sz == 0) {
        // Если указатель на тензор или его данные NULL, или unit_sz 0,
        // возвращаем значение по умолчанию (0).
        // Также обновляем ent_last, если t валиден.
        if (t) {
            t->ent_last = t->ent; // Сохраняем старое значение ent как ent_last
            t->ent = 0;           // Устанавливаем текущую ent в 0 из-за ошибки
        }
        return 0; // Возвращаем 0 при ошибке
    }

    // 2. Вычисление общего количества битов
    // Проверка на переполнение при умножении не включена, так как uint16_t * uint16_t максимум 2^32 - 2^17 + 1,
    // что всё равно помещается в uint32_t. Но результат может быть 0.
    uint32_t total_bits = (uint32_t)t->rows * t->cols;

    // 3. Проверка на нулевое количество битов
    if (total_bits == 0) {
         t->ent_last = t->ent; // Сохраняем старое значение перед обновлением
         t->ent = 0;           // Устанавливаем ent в 0
         return 0;             // Возвращаем 0 при ошибке
    }

    // 4. Подсчет единиц
    uint32_t ones = 0;
    for (uint32_t i = 0; i < total_bits; i++) {
        // Проверка границ (предполагается, что create_tnsr корректно выделяет память и total_bits <= allocated_bits)
        // Здесь делаем только безопасный доступ в рамках total_bits и выделенной памяти
        uint32_t byte_idx = i / 8;
        // uint8_t bit_idx = i % 8; // Не используется напрямую, так как BIT_GET сам делает &
        // if (byte_idx >= allocated_bytes_for_t) break; // Не делаем здесь, т.к. должно быть в create_tnsr
        if (BIT_GET(t->data[byte_idx], i % 8)) {
            ones++;
        }
    }

    // 5. Вычисление вероятностей (fixed-point arithmetic с 8 битами дробной части)
    // ones << 8 эквивалентно ones * 256
    // Деление на total_bits безопасно, так как мы проверили total_bits != 0
    uint32_t p1_fixed = (ones << 8) / total_bits; // p1_fixed в диапазоне [0, 256]
    // p0_fixed = 1 - p1_fixed (в fixed-point 8-bit fractional)
    uint32_t p0_fixed = 256 - p1_fixed; // p0_fixed в диапазоне [0, 256]
    uint8_t log_p0 = (p0_fixed > 0) ? fast_log2(p0_fixed) : 0;
    uint8_t log_p1 = (p1_fixed > 0) ? fast_log2(p1_fixed) : 0;

    // Более осторожное вычисление с промежуточными проверками (хотя overflow uint32_t не проверяется строго)
    uint32_t term_p0 = p0_fixed * log_p0; // Может быть до 256 * 2048 = 524288
    uint32_t term_p1 = p1_fixed * log_p1; // Может быть до 256 * 2048 = 524288
    uint32_t sum_terms = term_p0 + term_p1; // Может быть до 1048576
    uint32_t h_fixed_unscaled = sum_terms >> 8; // Может быть до 1048576 / 256 = 4096

    // 8. Применение веса unit_sz
    // Результат (h_fixed_unscaled * unit_sz) может легко превысить 255.
    // Например, h_fixed_unscaled = 4096, unit_sz = 1 -> 4096 > 255
    // Нужно нормализовать или ограничить результат.
    // Простое ограничение:
    uint32_t result_raw = (uint32_t)h_fixed_unscaled * unit_sz;
    uint8_t result = (result_raw > 255) ? 255 : (uint8_t)result_raw;

    // 9. Обновление полей тензора: сначала сохраняем текущее значение ent в ent_last
    t->ent_last = t->ent;
    t->ent = result; // Присваиваем новое вычисленное значение

    return result;
}
uint8_t calc_bit_sim(BitTensor* a, BitTensor* b) {
    if (!a || !b || !a->data || !b->data) return 0;

    uint32_t bits_a = a->rows * a->cols;
    uint32_t bits_b = b->rows * b->cols;

    if (bits_a == 0 || bits_b == 0) return 0;

    // Оптимизация: используем меньший из размеров для сравнения
    uint32_t bits_to_compare = (bits_a < bits_b) ? bits_a : bits_b;
    uint32_t bytes_to_compare = (bits_to_compare + 7) / 8;

    // Оптимизация: предварительная проверка через энтропию
    uint8_t entropy_diff = (a->ent > b->ent) ? (a->ent - b->ent) : (b->ent - a->ent);
    if (entropy_diff > 60) { // Большая разница в энтропии → низкая схожесть
        return 0;
    }

    // Оптимизация: использование локальных переменных для быстрого доступа
    uint8_t* data_a = a->data;
    uint8_t* data_b = b->data;
    uint32_t total_bytes_a = (bits_a + 7) / 8;
    uint32_t total_bytes_b = (bits_b + 7) / 8;

    float weighted_similarity_sum = 0.0f;
    float total_weight = 0.0f;
    float max_possible_similarity = 255.0f;
    float fuzziness_factor = ((float)SIM_FUZZINESS_PERCENT) / 100.0f;
    float fuzziness_threshold = fuzziness_factor * max_possible_similarity;

    // Оптимизация: развернутый цикл на 8 байт за итерацию
    uint32_t i = 0;
    for (; i + 8 <= bytes_to_compare; i += 8) {
        uint8_t bytes_a[8], bytes_b[8];
        
        // Копируем данные с проверкой границ
        for (uint8_t j = 0; j < 8; j++) {
            bytes_a[j] = (i + j < total_bytes_a) ? data_a[i + j] : 0;
            bytes_b[j] = (i + j < total_bytes_b) ? data_b[i + j] : 0;
        }

        // Быстрое сравнение 8 байт
        for (uint8_t byte_idx = 0; byte_idx < 8; byte_idx++) {
            uint8_t diff = bytes_a[byte_idx] ^ bytes_b[byte_idx];
            
            if (diff == 0) {
                // Все биты совпадают
                weighted_similarity_sum += max_possible_similarity * 8;
                total_weight += 8;
            } else if (diff == 0xFF) {
                // Все биты разные
                weighted_similarity_sum += fuzziness_threshold * 8;
                total_weight += 8;
            } else {
                // Частичное совпадение
                for (uint8_t bit = 0; bit < 8; bit++) {
                    uint8_t bit_a = (bytes_a[byte_idx] >> bit) & 1;
                    uint8_t bit_b = (bytes_b[byte_idx] >> bit) & 1;
                    
                    float actual_similarity = (bit_a == bit_b) ? 
                                             max_possible_similarity : fuzziness_threshold;
                    float bit_weight = ((float)(a->act + b->act) / 2.0f) * 
                                      ((float)(a->stab + b->stab) / 2.0f) / 255.0f;
                    
                    // Дополнительный вес для "важных" битов
                    if ((i * 8 + byte_idx * 8 + bit) < bits_to_compare) {
                        bit_weight *= 1.65f;
                    }
                    
                    weighted_similarity_sum += actual_similarity * bit_weight;
                    total_weight += bit_weight;
                }
            }
        }
    }

    // Обработка оставшихся байтов
    for (; i < bytes_to_compare; i++) {
        uint8_t byte_a = (i < total_bytes_a) ? data_a[i] : 0;
        uint8_t byte_b = (i < total_bytes_b) ? data_b[i] : 0;
        
        uint8_t diff = byte_a ^ byte_b;
        
        for (uint8_t bit = 0; bit < 8; bit++) {
            // Проверяем, не вышли ли за пределы bits_to_compare
            if ((i * 8 + bit) >= bits_to_compare) break;
            
            uint8_t bit_a = (byte_a >> bit) & 1;
            uint8_t bit_b = (byte_b >> bit) & 1;
            
            float actual_similarity = (bit_a == bit_b) ? 
                                     max_possible_similarity : fuzziness_threshold;
            float bit_weight = ((float)(a->act + b->act) / 2.0f) * 
                              ((float)(a->stab + b->stab) / 2.0f) / 255.0f;
            
            // Дополнительный вес для "важных" битов
            bit_weight *= 1.65f;
            
            weighted_similarity_sum += actual_similarity * bit_weight;
            total_weight += bit_weight;
        }
    }

    if (total_weight == 0.0f) return 0;

    float avg_weighted_similarity = weighted_similarity_sum / total_weight;
    
    // Оптимизация: используем аппроксимацию tanh через быструю математику
    float scaled_sim = avg_weighted_similarity / (max_possible_similarity / 5.0f);
    
    // Быстрая аппроксимация tanh(x) ≈ x / (1 + |x|) для x ∈ [-3, 3]
    float abs_scaled = (scaled_sim < 0) ? -scaled_sim : scaled_sim;
    if (abs_scaled > 3.0f) abs_scaled = 3.0f;
    float tanh_approx = scaled_sim / (1.0f + abs_scaled);
    
    uint8_t final_similarity = (uint8_t)(tanh_approx * 255.0f);

    // Дополнительная проверка энтропии для схожих данных
    if (final_similarity > 150) {
        // Вычисляем "медленную энтропию" только для сильно схожих данных
        static uint8_t slow_entropy_cache[256] = {0};
        static uint32_t cache_timestamp = 0;
        
        uint32_t now = (uint32_t)time(NULL);
        if (now - cache_timestamp > 300) { // Кэш обновляется каждые 5 минут
            memset(slow_entropy_cache, 0, sizeof(slow_entropy_cache));
            cache_timestamp = now;
        }
        
        uint8_t combined_hash = (uint8_t)((bits_a ^ bits_b) & 0xFF);
        if (slow_entropy_cache[combined_hash] == 0) {
            // "Медленная" энтропийная проверка для схожих данных
            uint32_t common_bits = 0;
            for (uint32_t j = 0; j < bits_to_compare; j += 16) { // Увеличили шаг для производительности
                uint32_t byte_idx = j / 8;
                uint8_t bit_idx = j % 8;
                if (byte_idx < total_bytes_a && byte_idx < total_bytes_b) {
                    uint8_t bit_a = (data_a[byte_idx] >> bit_idx) & 1;
                    uint8_t bit_b = (data_b[byte_idx] >> bit_idx) & 1;
                    if (bit_a == bit_b) common_bits++;
                }
            }
            
            float similarity_ratio = (float)common_bits / (bits_to_compare / 16.0f);
            slow_entropy_cache[combined_hash] = (uint8_t)(similarity_ratio * 255.0f);
            
            // Корректировка финальной схожести на основе "медленной" энтропии
            if (similarity_ratio > 0.8f) {
                final_similarity = (final_similarity * 11 + slow_entropy_cache[combined_hash] * 5) >> 4;
            } else if (similarity_ratio < 0.3f) {
                final_similarity = (final_similarity * 13) >> 4;
            }
        }
    }

    return (final_similarity > 255) ? 255 : final_similarity;
}

uint8_t calc_res_match(BitTensor* a, BitTensor* b) {
    if (!a || !b) return 0;
    
    // Быстрое вычисление различий с защитой от переполнения
    uint16_t diff_res = (a->res > b->res) ? (a->res - b->res) : (b->res - a->res);
    uint16_t diff_act = (a->act > b->act) ? (a->act - b->act) : (b->act - a->act);
    uint16_t diff_ent = (a->ent > b->ent) ? (a->ent - b->ent) : (b->ent - a->ent);
    
    // Сдвиги вместо деления для производительности
    uint16_t avg_diff = (diff_res + diff_act + diff_ent) / 3;
    uint8_t base_match = 255 - (avg_diff > 255 ? 255 : (uint8_t)avg_diff);
    
    // Стабильность как усилитель (сдвиг вместо деления)
    uint16_t diff_stab = (a->stab > b->stab) ? (a->stab - b->stab) : (b->stab - a->stab);
    uint8_t stab_boost = 255 - (diff_stab > 255 ? 255 : (uint8_t)diff_stab);
    
    // Быстрое усреднение (сдвиг вправо на 1 = деление на 2)
    uint8_t final_match = (base_match + stab_boost) >> 1;
    
    // Дополнительный буст для схожих тензоров
    if (diff_res < 10 && diff_act < 20 && diff_ent < 15) {
        final_match = (final_match * 3 + 255) >> 2;
    }
    
    return final_match > 255 ? 255 : final_match;
}

// ===== ОБНОВЛЕННАЯ ФУНКЦИЯ ЭФФЕКТИВНОСТИ =====
uint8_t calculate_efficiency(BitTensor* t) {
    // ===== ПРОВЕРКА ВХОДНЫХ ДАННЫХ =====
    if (!t || !t->data) return 0;
    
    // ===== 1. МЯГКАЯ БАЗОВАЯ ОЦЕНКА =====
    // Используем взвешенное среднее для большей стабильности
    uint16_t base_score;
    
    // Для новых тензоров даем более мягкую оценку
    uint32_t now = (uint32_t)time(NULL);
    uint32_t age = (t->lu > 0 && now > t->lu) ? (now - t->lu) : 3600;
    
    if (age < 60 && t->conn < 3) {
        // Молодой неопытный тензор - щадящая оценка
        base_score = 50 + (t->act * 2 + t->res) / 3;  // [50-255]
    } else {
        // Стандартная оценка
        base_score = (t->act + t->res) / 2;  // [0-255]
    }
    
    // ===== 2. КОРРЕКЦИЯ НА СТОИМОСТЬ (С ЗАЩИТОЙ ОТ НУЛЯ) =====
    uint32_t effective_cost = t->compute_cost;
    
    // Гарантируем минимальную стоимость
    if (effective_cost == 0) effective_cost = 1;
    
    // Автоматическое снижение стоимости для часто используемых тензоров
    if (t->lu > 0 && now > t->lu) {
        uint32_t time_since_use = now - t->lu;
        
        // Если использовался недавно и активно - снижаем "стоимость"
        if (time_since_use < 300 && t->act > 100) {  // 5 минут
            // Уменьшаем на 25%, но не ниже 1
            uint32_t reduced_cost = (effective_cost * 3) >> 2;
            effective_cost = (reduced_cost > 0) ? reduced_cost : 1;
        }
    }
    
    // ===== 3. СТАБИЛЬНОСТЬ как усилитель =====
    uint8_t stability_boost = t->stab >> 2;  // [0-63]
    
    // ===== 4. СВЯЗНОСТЬ как показатель полезности =====
    uint8_t connectivity_boost = (t->conn > 20) ? 20 : t->conn;  // Максимум +20
    
    // ===== 5. ИНТЕЛЛЕКТУАЛЬНЫЙ АНАЛИЗ ЭНТРОПИИ =====
    uint8_t entropy_effect = 0;
    
    // Анализируем контекст энтропии
    if (t->ent >= 80 && t->ent <= 180) {
        // Оптимальный диапазон - бонус
        uint8_t distance_from_130 = (t->ent > 130) ? (t->ent - 130) : (130 - t->ent);
        entropy_effect = 10 - (distance_from_130 / 10);  // [0-10] бонус
    } 
    else if (t->ent > 180 && t->ent <= 220) {
        // Повышенная энтропия - мягкий штраф или даже бонус для молодых
        if (age < 120 && t->act > 100) {  // Молодой и активный
            entropy_effect = (t->ent - 180) / 8;  // [0-5] штраф
        } else {
            entropy_effect = (t->ent - 180) / 4;  // [0-10] штраф
        }
        // Преобразуем в отрицательное значение
        entropy_effect = -entropy_effect;
    }
    else if (t->ent > 220) {
        // Очень высокая энтропия - штраф
        entropy_effect = -(10 + (t->ent - 220) / 3);  // [-10..-22] штраф
    }
    // Для t->ent < 80 оставляем entropy_effect = 0 (низкая энтропия - нормально)
    
    // ===== 6. КОНЦЕПЦИИ И КЛАСТЕРЫ =====
    uint8_t concept_bonus = t->is_concept ? 30 : 0;
    uint8_t cluster_bonus = (t->cluster_id != 0) ? 15 : 0;
    
    // ===== 7. ФИНАЛЬНЫЙ РАСЧЕТ (С ЗАЩИТОЙ ОТ ДЕЛЕНИЯ НА 0) =====
    uint32_t total_score = (uint32_t)base_score + stability_boost + connectivity_boost + 
                          concept_bonus + cluster_bonus;
    
    // Добавляем эффект энтропии (может быть отрицательным)
    if (entropy_effect > 0) {
        total_score += entropy_effect;
    } else {
        // entropy_effect отрицательный, вычитаем его модуль
        uint8_t penalty = -entropy_effect;
        total_score = (total_score > penalty) ? (total_score - penalty) : 0;
    }
    
    // ===== 8. КОРРЕКЦИЯ НА СТОИМОСТЬ (С ЗАЩИТОЙ) =====
    if (effective_cost > 0) {
        // cost_factor в диапазоне [0, 256]
        uint32_t cost_factor = (effective_cost > 100) ? 
                              ((uint32_t)100 * 256 / effective_cost) : 256;
        
        // Проверяем, чтобы не было деления на 0 при нормализации
        if (cost_factor > 0) {
            total_score = (total_score * cost_factor) >> 8;
        }
    }
    
    // ===== 9. ДИНАМИЧЕСКАЯ АДАПТАЦИЯ =====
    static uint8_t system_mood = 128;  // 0=пессимист, 255=оптимист
    static uint32_t last_mood_update = 0;
    
    if (now - last_mood_update > 60) {
        // Анализируем общую ситуацию в системе
        uint32_t total_activity = 0;
        uint16_t active_count = 0;
        
        for (uint16_t i = 0; i < tnsr_count && i < 100; i++) {
            if (tnsrs[i].act > 50 && !tnsrs[i].dropout) {
                total_activity += tnsrs[i].act;
                active_count++;
            }
        }
        
        if (active_count > 0) {
            uint8_t avg_activity = (uint8_t)(total_activity / active_count);
            
            // Настраиваем "настроение" системы
            if (avg_activity > 150) {
                system_mood = (system_mood * 7 + 200) >> 3;  // Становимся оптимистичнее
            } else if (avg_activity < 70) {
                system_mood = (system_mood * 7 + 50) >> 3;   // Становимся пессимистичнее
            } else {
                system_mood = (system_mood * 7 + 128) >> 3;  // Нейтральное
            }
        }
        
        last_mood_update = now;
    }
    
    // Применяем "настроение" системы
    total_score = (total_score * system_mood) >> 8;
    
    // ===== 10. НОРМАЛИЗАЦИЯ И ГАРАНТИИ =====
    uint8_t efficiency = (total_score > 255) ? 255 : (uint8_t)total_score;
    
    // Гарантируем минимальную эффективность для активных тензоров
    if (efficiency < 10) {
        if (t->act > 60) {
            efficiency = 10 + (t->act / 10);  // [10-25]
        } else {
            efficiency = 10;  // Абсолютный минимум
        }
    }
    
    // Защита от переобучения
    if (efficiency > 240 && t->ent < 30 && t->conn < 2) {
        // Слегка снижаем для тензоров с подозрением на переобучение
        efficiency = (efficiency * 9) >> 3;  // Уменьшаем на ~12.5%
    }
    
    // ===== 11. СОХРАНЕНИЕ РЕЗУЛЬТАТА =====
    t->efficiency = efficiency;
    
    return efficiency;
}

void update_efficiency_goal(void) {
    uint32_t total_efficiency = 0;
    uint16_t active_tensors = 0;
    
    // ===== 1. ОБНОВЛЕНИЕ ЭФФЕКТИВНОСТИ ВСЕХ АКТИВНЫХ ТЕНЗОРОВ =====
    for (uint16_t i = 0; i < tnsr_count; i++) {
        if (tnsrs[i].act > 50 && tnsrs[i].data) {
            tnsrs[i].efficiency = calculate_efficiency(&tnsrs[i]);
            total_efficiency += tnsrs[i].efficiency;
            active_tensors++;
        }
    }
    
    // ===== 2. РАСЧЕТ СРЕДНЕЙ ЭФФЕКТИВНОСТИ (С ЗАЩИТОЙ ОТ ДЕЛЕНИЯ НА 0) =====
    uint8_t avg_efficiency = 50;  // Значение по умолчанию
    
    if (active_tensors > 0) {
        // Безопасное деление
        avg_efficiency = (uint8_t)(total_efficiency / active_tensors);
        
        // Дополнительная проверка на валидность
        if (avg_efficiency == 0 && active_tensors > 0) {
            avg_efficiency = 50;  // Запасной вариант
        }
    }
    
    // ===== 3. ОБНОВЛЕНИЕ ЦЕЛЕВОЙ ЭФФЕКТИВНОСТИ =====
    if (active_tensors > 5) {  // Только если есть достаточная статистика
        int16_t diff = (int16_t)avg_efficiency - (int16_t)goals.target_efficiency;
        
        if (diff > 10) {
            // Эффективность значительно выросла - плавное увеличение цели
            goals.target_efficiency = (goals.target_efficiency * 9 + avg_efficiency) >> 3;
            goals.efficiency_gain++;
            
            // Лимитируем максимальное значение
            if (goals.target_efficiency > 200) goals.target_efficiency = 200;
            
        } else if (diff < -20) {
            // Эффективность значительно упала - плавное уменьшение
            goals.target_efficiency = (goals.target_efficiency * 7 + avg_efficiency) >> 3;
            goals.energy_saving_mode = 1;
            
            // Лимитируем минимальное значение
            if (goals.target_efficiency < 30) goals.target_efficiency = 30;
            
        } else {
            // Стабильная ситуация
            goals.energy_saving_mode = 0;
            
            // Микрокорректировка к средней эффективности
            goals.target_efficiency = (goals.target_efficiency * 15 + avg_efficiency) >> 4;
        }
    } else if (active_tensors > 0) {
        // Мало данных - осторожная корректировка
        goals.target_efficiency = (goals.target_efficiency * 3 + avg_efficiency) >> 2;
    }
    
    // ===== 4. АДАПТИВНЫЙ ПОРОГ АКТИВАЦИИ =====
    static uint8_t dynamic_threshold = 50;
    static uint32_t last_threshold_update = 0;
    uint32_t now = (uint32_t)time(NULL);
    
    if (now - last_threshold_update > 60 && active_tensors > 3) {
        // Рассчитываем динамический порог на основе медианы, а не среднего
        uint8_t efficiencies[100];
        uint8_t eff_count = 0;
        
        // Собираем эффективности активных тензоров
        for (uint16_t i = 0; i < tnsr_count && i < 100 && eff_count < 100; i++) {
            if (tnsrs[i].act > 50 && tnsrs[i].efficiency > 0) {
                efficiencies[eff_count++] = tnsrs[i].efficiency;
            }
        }
        
        if (eff_count > 3) {
            // Простая сортировка для нахождения медианы
            for (uint8_t i = 0; i < eff_count - 1; i++) {
                for (uint8_t j = i + 1; j < eff_count; j++) {
                    if (efficiencies[i] > efficiencies[j]) {
                        uint8_t temp = efficiencies[i];
                        efficiencies[i] = efficiencies[j];
                        efficiencies[j] = temp;
                    }
                }
            }
            
            // Берем медиану (средний элемент)
            uint8_t median = efficiencies[eff_count / 2];
            
            // Динамический порог = медиана - 20, но не ниже 30
            uint8_t new_threshold = (median > 50) ? (median - 20) : 30;
            
            // Плавная корректировка
            dynamic_threshold = (dynamic_threshold * 3 + new_threshold) >> 2;
            
            last_threshold_update = now;
        }
    }
    
    // ===== 5. ОБНОВЛЕНИЕ ГЛОБАЛЬНОГО ПОРОГА =====
    goals.target_efficiency = dynamic_threshold;
    
    // ===== 6. ЛОГИРОВАНИЕ (для отладки) =====
    static uint32_t last_log = 0;
    if (now - last_log > 300 && active_tensors > 0) {  // Каждые 5 минут
        printf("[EFF] Target: %u, Avg: %u, Active: %u, Threshold: %u\n",
               goals.target_efficiency, avg_efficiency, active_tensors, dynamic_threshold);
        last_log = now;
    }
}
// ===== ОСНОВНЫЕ ФУНКЦИИ =====
BitTensor* create_tensor(uint16_t rows, uint16_t cols) {
    // 1. ПРОВЕРКА ВХОДНЫХ ПАРАМЕТРОВ
    if (tnsr_count >= MAX_TENSORS) {
        printf("[ERROR] Достигнут максимальный лимит тензоров: %u\n", MAX_TENSORS);
        return NULL;
    }
    
    if (rows == 0 || cols == 0) {
        printf("[ERROR] Некорректные размеры тензора: %ux%u\n", rows, cols);
        return NULL;
    }
    
    // 2. БЕЗОПАСНОЕ ВЫЧИСЛЕНИЕ РАЗМЕРОВ
    uint32_t total_bits = (uint32_t)rows * cols;
    if (total_bits == 0 || total_bits > 0xFFFFF) { // Ограничение ~1 млн бит
        printf("[ERROR] Слишком большой тензор: %u бит\n", total_bits);
        return NULL;
    }
    
    uint32_t total_bytes = (total_bits + 7) / 8;
    
    // 3. ВЫДЕЛЕНИЕ ПАМЯТИ С ЗАЩИТОЙ ОТ ПЕРЕПОЛНЕНИЯ
    if (total_bytes > 1024 * 1024) { // Ограничение 1 МБ
        printf("[ERROR] Тензор слишком большой для памяти: %u байт\n", total_bytes);
        return NULL;
    }
    
    uint8_t* data = (uint8_t*)calloc(total_bytes, 1);
    if (!data) {
        printf("[ERROR] Не удалось выделить %u байт для тензора\n", total_bytes);
        return NULL;
    }
    
    // 4. ИНИЦИАЛИЗАЦИЯ СТРУКТУРЫ ТЕНЗОРА
    BitTensor* t = &tnsrs[tnsr_count];
    
    // Сначала инициализируем указатель на данные
    t->data = data;
    t->rows = rows;
    t->cols = cols;
    
    // Увеличиваем счетчик только после успешной инициализации
    tnsr_count++;
    
    // 5. ИНИЦИАЛИЗАЦИЯ ПАРАМЕТРОВ ТЕНЗОРА
    uint32_t now = (uint32_t)time(NULL);
    
    // Базовые параметры
    t->res = RES_HALF;        // 128
    t->act = 50;              // Начальная активность
    t->ent = 0;               // Начальная энтропия
    t->ent_last = 0;          // Предыдущая энтропия
    t->stab = 100;            // Начальная стабильность
    t->conn = 0;              // Нет связей
    t->lu = now;              // Время создания
    t->mem_red = 0;           // Не сжат
    t->compute_cost = total_bits; // Базовая стоимость
    
    // Целевая активность зависит от размера
    t->goal_active = (total_bits > 100) ? 1 : 0;
    
    // Dropout с вероятностью DROPOUT_RATE%
    t->dropout = (rand() % 100 < DROPOUT_RATE) ? 1 : 0;
    
    // Кластер и концепция
    t->cluster_id = 0;
    t->is_concept = 0;
    
    // 6. ВЫЧИСЛЕНИЕ ЭНТРОПИИ И ЭФФЕКТИВНОСТИ
    // Безопасное вычисление энтропии с проверкой
    uint8_t entropy = 0;
    if (total_bits > 0) {
        entropy = calc_bit_ent(t, cols);
    }
    t->ent = entropy;
    
    // Вычисление начальной эффективности
    t->efficiency = calculate_efficiency(t);
    
    // 7. СОЗДАНИЕ БАЗОВОЙ СВЯЗИ ДЛЯ САМОРЕФЛЕКСИИ
    // Создаем минимальную ссылку на себя для предотвращения изоляции
    if (tnsr_count > 1) {
        // Связываем с предыдущим тензором для создания начального контекста
        BitTensor* prev_tensor = &tnsrs[tnsr_count - 2];
        if (prev_tensor && prev_tensor->data) {
            BitLink* link = create_link(prev_tensor, t);
            if (link) {
                link->strength = 50; // Слабая начальная связь
                link->semantic_type = 0; // Обычная связь
            }
        }
    }
    
    // 8. ДОБАВЛЕНИЕ В РАБОЧУЮ ПАМЯТЬ
    add_to_working_memory(t);
    
    // 9. ЛОГИРОВАНИЕ (опционально)
    static uint32_t last_log = 0;
    uint32_t current_time = (uint32_t)time(NULL);
    if (current_time - last_log > 60) { // Логируем не чаще чем раз в минуту
        printf("[TENSOR] Создан тензор %u: %ux%u (%u бит), эфф=%u\n",
               tnsr_count - 1, rows, cols, total_bits, t->efficiency);
        last_log = current_time;
    }
    
    return t;
}

// Алиас для обратной совместимости
BitTensor* create_tnsr(uint16_t rows, uint16_t cols) {
    return create_tensor(rows, cols);
}

/**
 * Интеллектуальное сжатие памяти тензора
 */
void reduce_tnsr_mem(BitTensor* t) {
    if (!t || !t->data || t->mem_red) return;
    
    // Не сжимаем концепции и важные тензоры
    if (t->is_concept || t->act > 100 || t->conn > 10) {
        t->mem_red = 1; // Отмечаем как проверенный
        return;
    }
    
    uint32_t total_bits = t->rows * t->cols;
    
    // Определяем степень сжатия на основе энтропии
    uint8_t compression_level;
    if (t->ent < 50) {
        // Низкая энтропия - можно сильнее сжать
        compression_level = 4;
    } else if (t->ent < 150) {
        // Средняя энтропия - умеренное сжатие
        compression_level = 2;
    } else {
        // Высокая энтропия - минимальное сжатие или отказ
        if (total_bits < 256) {
            t->mem_red = 1;
            return;
        }
        compression_level = 1;
    }
    
    // Вычисляем новые размеры
    uint16_t new_rows = t->rows / compression_level;
    uint16_t new_cols = t->cols / compression_level;
    
    // Ограничения минимального размера
    if (new_rows < 4) new_rows = 4;
    if (new_cols < 4) new_cols = 4;
    if (new_rows > t->rows) new_rows = t->rows;
    if (new_cols > t->cols) new_cols = t->cols;
    
    // Если размер не изменился, просто отмечаем
    if (new_rows == t->rows && new_cols == t->cols) {
        t->mem_red = 1;
        return;
    }
    
    uint32_t new_bits = new_rows * new_cols;
    uint32_t new_bytes = (new_bits + 7) / 8;
    
    uint8_t* new_data = (uint8_t*)calloc(new_bytes, 1);
    if (!new_data) {
        t->mem_red = 1;
        return;
    }
    
    // Алгоритм пулинга: максимальное значение в области
    float row_ratio = (float)t->rows / new_rows;
    float col_ratio = (float)t->cols / new_cols;
    
    for (uint16_t i = 0; i < new_rows; i++) {
        for (uint16_t j = 0; j < new_cols; j++) {
            uint16_t src_start_row = (uint16_t)(i * row_ratio);
            uint16_t src_end_row = (uint16_t)((i + 1) * row_ratio);
            if (src_end_row > t->rows) src_end_row = t->rows;
            
            uint16_t src_start_col = (uint16_t)(j * col_ratio);
            uint16_t src_end_col = (uint16_t)((j + 1) * col_ratio);
            if (src_end_col > t->cols) src_end_col = t->cols;
            
            // Ищем максимальное значение в области
            uint8_t max_val = 0;
            for (uint16_t sr = src_start_row; sr < src_end_row; sr++) {
                for (uint16_t sc = src_start_col; sc < src_end_col; sc++) {
                    uint32_t src_idx = sr * t->cols + sc;
                    uint8_t val = BIT_GET(t->data[src_idx / 8], src_idx % 8);
                    if (val > max_val) max_val = val;
                }
            }
            
            // Устанавливаем бит в сжатом тензоре
            if (max_val) {
                uint32_t dst_idx = i * new_cols + j;
                BIT_SET(new_data[dst_idx / 8], dst_idx % 8);
            }
        }
    }
    
    // Сохраняем старые данные в память перед заменой
    save_tnsr(t);
    
    // Заменяем данные
    free(t->data);
    t->data = new_data;
    t->rows = new_rows;
    t->cols = new_cols;
    t->mem_red = 1;
    
    // Обновляем параметры
    t->ent = calc_bit_ent(t, t->cols);
    t->compute_cost = new_bits * 2;
    t->act = (uint8_t)(t->act * 0.9f); // Немного снижаем активность после сжатия
    t->efficiency = calculate_efficiency(t);
    
    printf("[MEM-RED] Тензор %lu сжат: %ux%u -> %ux%u, энтропия: %u->%u\n",
           (unsigned long)(t - tnsrs), t->rows * compression_level, t->cols * compression_level,
           t->rows, t->cols, t->ent_last, t->ent);
}



void add_to_working_memory(BitTensor* t) {
    if (!t || working_mem_count >= WORKING_MEM_SIZE) return;
    
    uint32_t now = (uint32_t)time(NULL);
    uint8_t existing_idx = 255;
    
    // === 1. AI-ОПТИМИЗАЦИЯ: ИНТЕЛЛЕКТУАЛЬНЫЙ ПОИСК С КЭШИРОВАНИЕМ ===
    // Используем локальный кэш для быстрого поиска (работает в 90% случаев)
    static uint16_t last_found_idx = 0;
    if (working_mem[last_found_idx].tensor == t && last_found_idx < working_mem_count) {
        existing_idx = last_found_idx;
    } else {
        // Стратегический поиск: начинаем с наиболее вероятных позиций
        uint8_t search_order[3] = {0, working_mem_count / 2, working_mem_count - 1};
        for (uint8_t attempt = 0; attempt < 3 && existing_idx == 255; attempt++) {
            uint8_t i = search_order[attempt];
            if (i < working_mem_count && working_mem[i].tensor == t) {
                existing_idx = i;
                last_found_idx = i;
                break;
            }
        }
        
        // Если не нашли в ключевых точках - полный поиск
        if (existing_idx == 255) {
            for (uint8_t i = 0; i < working_mem_count; i++) {
                if (working_mem[i].tensor == t) {
                    existing_idx = i;
                    last_found_idx = i;
                    break;
                }
            }
        }
    }
    
    if (existing_idx != 255) {
        WorkingMemoryEntry* entry = &working_mem[existing_idx];
        entry->access_count++;
        
        // === 2. AI-РАСЧЁТ: ИНТЕЛЛЕКТУАЛЬНОЕ РЕЗОНИРОВАНИЕ ===
        // Многофакторная оценка с нелинейными весами
        float resonance_factor = 0.0f;
        
        // 2.1. Базовое резонирование: системный резонанс × активация
        float base_resonance = (float)sys_res * t->act / 65025.0f; // 0.0-1.0
        
        // 2.2. Временное резонирование: чем свежее, тем сильнее
        uint32_t time_since_access = now - entry->timestamp;
        float time_resonance = (time_since_access < 10) ? 1.0f : 
                              (time_since_access < 60) ? 0.5f : 0.1f;
        
        // 2.3. Семантическое резонирование: связи и концепции
        float semantic_resonance = 0.0f;
        if (t->is_concept) {
            semantic_resonance = 0.8f + (t->conn > 5 ? 0.2f : 0.0f);
        } else if (t->cluster_id != 0) {
            semantic_resonance = 0.5f + (t->conn > 3 ? 0.2f : 0.0f);
        } else {
            semantic_resonance = 0.3f;
        }
        
        // 2.4. Эпизодное резонирование: если отмечен для эпизодной памяти
        float episode_resonance = entry->episode_marker ? 0.7f : 0.3f;
        
        // === 3. AI-ИНТЕГРАЦИЯ: НЕЙРОННАЯ СЕТЕВАЯ АГРЕГАЦИЯ ===
        // Взвешенная сумма с адаптивными коэффициентами
        resonance_factor = 
            base_resonance * 0.35f +      // Базовая активация
            time_resonance * 0.25f +      // Временная свежесть
            semantic_resonance * 0.20f +  // Семантическая значимость
            episode_resonance * 0.20f;    // Эпизодная релевантность
        
        // Нормализация и ограничение
        resonance_factor = resonance_factor > 1.0f ? 1.0f : resonance_factor;
        resonance_factor = resonance_factor < 0.0f ? 0.0f : resonance_factor;
        
        // === 4. AI-ОБНОВЛЕНИЕ: АДАПТИВНЫЙ АЛГОРИТМ ПРИОРИТЕТА ===
        uint8_t resonance_boost = (uint8_t)(resonance_factor * 100.0f); // 0-100
        
        // Интеллектуальная формула с защитой от переобучения
        uint16_t old_priority = entry->priority;
        
        // Динамический вес в зависимости от текущего приоритета
        uint8_t adaptive_weight;
        if (old_priority < 50) {
            adaptive_weight = 3;  // Низкий приоритет - быстрый рост
        } else if (old_priority < 150) {
            adaptive_weight = 5;  // Средний приоритет - умеренный рост
        } else {
            adaptive_weight = 7;  // Высокий приоритет - медленный рост
        }
        
        // Формула с плавным переходом
        uint16_t new_priority = 
            (old_priority * adaptive_weight + 100 + resonance_boost * 2) / 
            (adaptive_weight + 2);
        
        // Ограничение и применение
        entry->priority = (new_priority > 255) ? 255 : (uint8_t)new_priority;
        
        // === 5. AI-КОНТЕКСТ: ОБНОВЛЕНИЕ СВЯЗАННЫХ ПОЛЕЙ ===
        entry->timestamp = now;
        entry->episode_marker = 1;
        
        // Синхронизация с тензором
        t->lu = now;
        
        // Микроповышение активности при частом доступе
        if (entry->access_count % 5 == 0 && t->act < 240) {
            t->act += 3;
        }
        
        // === 6. AI-ОПТИМИЗАЦИЯ: АДАПТИВНОЕ УПРАВЛЕНИЕ ПАМЯТЬЮ ===
        // Если элемент стал очень приоритетным, проверяем возможность консолидации
        if (entry->priority > 200 && entry->access_count > 10) {
            // Автоматическая оптимизация связанного тензора
            if (t->efficiency < goals.target_efficiency) {
                optimize_tensor(t);
            }
            
            // Автоматический перенос в долговременную память при высокой стабильности
            if (t->stab > 180 && t->conn > 5 && !t->is_concept) {
                force_transfer_to_longterm(t);
            }
        }
        
    } else {
        // === 7. AI-ИНИЦИАЛИЗАЦИЯ: ИНТЕЛЛЕКТУАЛЬНОЕ ДОБАВЛЕНИЕ ===
        // Выбираем оптимальную позицию для вставки
        uint8_t insert_pos = working_mem_count;
        
        // Стратегия: если есть элементы с низким приоритетом, заменяем их
        if (working_mem_count == WORKING_MEM_SIZE) {
            uint8_t lowest_idx = 0;
            uint8_t lowest_priority = 255;
            
            for (uint8_t i = 0; i < WORKING_MEM_SIZE; i++) {
                if (working_mem[i].priority < lowest_priority) {
                    lowest_priority = working_mem[i].priority;
                    lowest_idx = i;
                    
                    // Прерываем если нашли очень низкий приоритет
                    if (lowest_priority < 30) break;
                }
            }
            
            // Заменяем только если новый тензор действительно лучше
            if (t->act > 50 && t->efficiency > 40) {
                insert_pos = lowest_idx;
                working_mem_count--; // Временное уменьшение для вставки
            } else {
                return; // Не добавляем слабые тензоры
            }
        }
        
        // Инициализация новой записи
        WorkingMemoryEntry* entry = &working_mem[insert_pos];
        entry->tensor = t;
        entry->timestamp = now;
        
        // Интеллектуальный начальный приоритет
        uint8_t initial_priority = 80; // Базовый
        
        // Бонусы за качество тензора
        if (t->is_concept) initial_priority += 40;
        if (t->cluster_id != 0) initial_priority += 20;
        if (t->act > 100) initial_priority += 15;
        if (t->efficiency > goals.target_efficiency) initial_priority += 25;
        
        entry->priority = (initial_priority > 200) ? 200 : initial_priority;
        entry->access_count = 1;
        entry->episode_marker = 1;
        
        if (insert_pos == working_mem_count) {
            working_mem_count++;
        }
        
        // AI-логирование для отладки
        static uint32_t last_log = 0;
        if (now - last_log > 60) {
            printf("[AI-WM] Добавлен тензор %u: приоритет=%u, акт=%u, эфф=%u\n",
                   (uint32_t)(t - tnsrs), entry->priority, t->act, t->efficiency);
            last_log = now;
        }
    }
    
    // === 8. AI-МОНИТОРИНГ: АНАЛИЗ СОСТОЯНИЯ РАБОЧЕЙ ПАМЯТИ ===
    static uint32_t last_analysis = 0;
    if (now - last_analysis > 30) {
        uint32_t total_priority = 0;
        uint8_t high_priority_count = 0;
        
        for (uint8_t i = 0; i < working_mem_count; i++) {
            total_priority += working_mem[i].priority;
            if (working_mem[i].priority > 150) high_priority_count++;
        }
        
        uint8_t avg_priority = working_mem_count ? (uint8_t)(total_priority / working_mem_count) : 0;
        
        // Адаптивная настройка системы на основе состояния рабочей памяти
        if (avg_priority > 180 && high_priority_count > 3) {
            goals.energy_saving_mode = 0; // Высокая активность
        } else if (avg_priority < 70 && high_priority_count < 2) {
            goals.energy_saving_mode = 1; // Низкая активность
            // Автоматическая активация для поддержания активности
            if ((rand() % 10) == 0) {
                activate_longterm_to_working();
            }
        }
        
        last_analysis = now;
    }
}

BitTensor* get_from_working_memory(uint8_t min_priority) {
    if (working_mem_count == 0) return NULL;
    uint8_t best_idx = 0;
    uint16_t best_score = 0;
    for (uint8_t i = 0; i < working_mem_count; i++) {
        if (working_mem[i].priority >= min_priority) {
            uint16_t score = working_mem[i].priority * working_mem[i].access_count;
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }
    }
    if (best_score > 0) {
        working_mem[best_idx].access_count++;
        working_mem[best_idx].episode_marker = 1;
        return working_mem[best_idx].tensor;
    }
    return NULL;
}

// ===== ФУНКЦИИ САМООРГАНИЗАЦИИ ПАМЯТИ =====

void self_organize_memory_clusters(void) {
    if (!goals.self_organization_enabled || tnsr_count < 2) return;

    uint32_t now = (uint32_t)time(NULL);

    // 1. Собираем активные тензоры для кластеризации
    BitTensor* active_tensors[MAX_TENSORS];
    uint16_t active_count = 0;

    for (uint16_t i = 0; i < tnsr_count; i++) {
        if (tnsrs[i].act > 30 && !tnsrs[i].dropout && tnsrs[i].data && !tnsrs[i].is_concept) {
            active_tensors[active_count++] = &tnsrs[i];
            if (active_count >= MAX_TENSORS - 1) break;
        }
    }

    if (active_count < 3) return;

    // 2. Вычисляем матрицу похожести
    uint8_t similarity_matrix[256][256] = {0};
    uint16_t limit = (active_count > 256) ? 256 : active_count;
    
    for (uint16_t i = 0; i < limit; i++) {
        for (uint16_t j = i + 1; j < limit; j++) {
            uint8_t sim = calc_bit_sim(active_tensors[i], active_tensors[j]);
            similarity_matrix[i][j] = sim;
            similarity_matrix[j][i] = sim;
        }
    }

    // 3. Алгоритм агломеративной кластеризации
    uint8_t cluster_assignments[256] = {0};
    uint8_t current_cluster_id = 1;

    for (uint16_t i = 0; i < limit; i++) {
        if (cluster_assignments[i] == 0) {
            cluster_assignments[i] = current_cluster_id;

            for (uint16_t j = i + 1; j < limit; j++) {
                if (cluster_assignments[j] == 0 &&
                    similarity_matrix[i][j] > CLUSTER_THRESHOLD) {
                    cluster_assignments[j] = current_cluster_id;
                }
            }

            current_cluster_id++;
            if (current_cluster_id >= 128) break;
        }
    }

    // 4. Обновляем или создаем кластеры
    for (uint8_t cid = 1; cid < current_cluster_id; cid++) {
        uint16_t cluster_size = 0;
        uint16_t tensor_indices[64];
        uint16_t cluster_act_sum = 0;

        for (uint16_t i = 0; i < limit; i++) {
            if (cluster_assignments[i] == cid) {
                uint16_t global_idx = active_tensors[i] - tnsrs;
                if (global_idx < MAX_TENSORS) {
                    tensor_indices[cluster_size] = global_idx;
                    cluster_act_sum += active_tensors[i]->act;
                    cluster_size++;
                }
                if (cluster_size >= 64) break;
            }
        }

        if (cluster_size >= 2) {
            MemoryCluster* cluster = NULL;
            uint8_t found_existing = 0;

            for (uint16_t ci = 0; ci < cluster_count; ci++) {
                uint8_t overlap = 0;
                for (uint16_t k = 0; k < cluster_size; k++) {
                    for (uint16_t l = 0; l < clusters[ci].size; l++) {
                        if (tensor_indices[k] == clusters[ci].tensor_indices[l]) {
                            overlap++;
                            break;
                        }
                    }
                }

                if (overlap >= cluster_size / 2 || overlap >= clusters[ci].size / 2) {
                    cluster = &clusters[ci];
                    found_existing = 1;
                    break;
                }
            }

            if (!found_existing && cluster_count < MAX_CLUSTERS) {
                cluster = &clusters[cluster_count++];
                cluster->cluster_id = next_cluster_id++;
                if (next_cluster_id == 0) next_cluster_id = 1;
                cluster->size = 0;
                cluster->stability = 100;
                cluster->last_access = now;
                cluster->creation_time = now;
                cluster->activation_level = (uint8_t)(cluster_act_sum / cluster_size);
                cluster->link_count = 0;
                cluster->category = 0;
            }

            if (cluster) {
                cluster->size = cluster_size;
                for (uint16_t k = 0; k < cluster_size; k++) {
                    cluster->tensor_indices[k] = tensor_indices[k];
                    tnsrs[tensor_indices[k]].cluster_id = cluster->cluster_id;
                }
                cluster->last_access = now;
                cluster->activation_level = (uint8_t)(cluster_act_sum / cluster_size);

                memset(cluster->centroid, 0, 256);
                for (uint16_t k = 0; k < cluster_size; k++) {
                    BitTensor* t = &tnsrs[tensor_indices[k]];
                    uint32_t bits = t->rows * t->cols;
                    uint32_t bytes = (bits + 7) / 8;
                    for (uint32_t b = 0; b < bytes && b < 256; b++) {
                        cluster->centroid[b] = (cluster->centroid[b] * (cluster_size - 1) + t->data[b]) / cluster_size;
                    }
                }

                update_cluster_stability(cluster);
            }
        }
    }

    prune_weak_clusters();
    last_cluster_reorg = now;
}

void update_cluster_stability(MemoryCluster* cluster) {
    if (!cluster) return;

    uint32_t now = (uint32_t)time(NULL);
    uint32_t age = now - cluster->creation_time;

    uint8_t age_factor = (age > 3600) ? 200 : (uint8_t)(age / 18);
    uint8_t activity_factor = cluster->activation_level;
    uint8_t coherence = calculate_cluster_coherence(cluster);

    cluster->stability = (age_factor * 3 + activity_factor * 5 + coherence * 2) / 10;

    if (cluster->stability > 255) cluster->stability = 255;
}

uint8_t calculate_cluster_coherence(MemoryCluster* cluster) {
    if (!cluster || cluster->size < 2) return 0;

    uint32_t total_similarity = 0;
    uint32_t comparisons = 0;

    for (uint16_t i = 0; i < cluster->size; i++) {
        for (uint16_t j = i + 1; j < cluster->size; j++) {
            BitTensor* t1 = &tnsrs[cluster->tensor_indices[i]];
            BitTensor* t2 = &tnsrs[cluster->tensor_indices[j]];

            uint8_t sim = calc_bit_sim(t1, t2);
            total_similarity += sim;
            comparisons++;
        }
    }

    if (comparisons == 0) return 0;
    return (uint8_t)(total_similarity / comparisons);
}

void prune_weak_clusters(void) {
    uint32_t now = (uint32_t)time(NULL);

    for (uint16_t ci = 0; ci < cluster_count; ci++) {
        MemoryCluster* cluster = &clusters[ci];

        if ((now - cluster->last_access > 7200 && cluster->stability < 50) ||
            cluster->stability < 20 ||
            (cluster->size < 2 && now - cluster->creation_time > 3600)) {

            for (uint16_t i = 0; i < cluster->size; i++) {
                uint16_t tensor_idx = cluster->tensor_indices[i];
                if (tensor_idx < tnsr_count) {
                    tnsrs[tensor_idx].cluster_id = 0;
                }
            }

            if (ci < cluster_count - 1) {
                clusters[ci] = clusters[cluster_count - 1];
            }
            cluster_count--;
            ci--;
        }
    }
}

BitTensor* find_center_tensor_in_cluster(MemoryCluster* cluster) {
    if (!cluster || cluster->size == 0) return NULL;

    BitTensor* best = NULL;
    uint32_t best_score = 0;

    for (uint16_t i = 0; i < cluster->size; i++) {
        uint16_t idx = cluster->tensor_indices[i];
        if (idx < tnsr_count) {
            BitTensor* t = &tnsrs[idx];
            uint32_t score = (uint32_t)t->act * t->res * (t->conn + 1);
            if (score > best_score) {
                best_score = score;
                best = t;
            }
        }
    }

    return best;
}

BitLink* find_or_create_link(BitTensor* src, BitTensor* tgt) {
    if (!src || !tgt) return NULL;

    for (uint16_t i = 0; i < lnk_count; i++) {
        if ((lnks[i].src == src && lnks[i].tgt == tgt) ||
            (lnks[i].src == tgt && lnks[i].tgt == src)) {
            return &lnks[i];
        }
    }

    return create_link(src, tgt);
}

void semantic_memory_binding(void) {
    if (cluster_count < 2) return;

    uint32_t now = (uint32_t)time(NULL);

    for (uint16_t i = 0; i < cluster_count; i++) {
        for (uint16_t j = i + 1; j < cluster_count; j++) {
            MemoryCluster* c1 = &clusters[i];
            MemoryCluster* c2 = &clusters[j];
            
            if (c1 == c2) continue;

            uint32_t time_diff = (c1->last_access > c2->last_access) ?
                                (c1->last_access - c2->last_access) :
                                (c2->last_access - c1->last_access);

            uint8_t common_links = 0;
            uint16_t max_check = (c1->size < c2->size) ? c1->size : c2->size;
            max_check = (max_check > 10) ? 10 : max_check;

            for (uint8_t k = 0; k < max_check; k++) {
                uint16_t idx1 = c1->tensor_indices[k];
                uint16_t idx2 = c2->tensor_indices[k];

                if (idx1 < tnsr_count && idx2 < tnsr_count) {
                    for (uint16_t li = 0; li < lnk_count; li++) {
                        if ((lnks[li].src == &tnsrs[idx1] && lnks[li].tgt == &tnsrs[idx2]) ||
                            (lnks[li].src == &tnsrs[idx2] && lnks[li].tgt == &tnsrs[idx1])) {
                            common_links++;
                            break;
                        }
                    }
                }
            }

            uint8_t semantic_score = 0;
            if (time_diff < 5) semantic_score += 80;
            else if (time_diff < 30) semantic_score += 40;

            semantic_score += common_links * 15;

            if (semantic_score > 70) {
                BitTensor* center1 = find_center_tensor_in_cluster(c1);
                BitTensor* center2 = find_center_tensor_in_cluster(c2);

                if (center1 && center2 && center1 != center2) {
                    BitLink* link = find_or_create_link(center1, center2);
                    if (link) {
                        link->strength = (link->strength * 7 + 180) >> 3;
                        link->res = (link->res * 7 + 200) >> 3;
                        link->semantic_type = 2;
                        link->last_act = now;

                        c1->link_count++;
                        c2->link_count++;
                    }
                }
            }
        }
    }
}

void create_concept_from_tensors(BitTensor* t1, BitTensor* t2) {
    if (!t1 || !t2 || t1 == t2) return;
    
    BitTensor* concept = create_tnsr(16, 16);
    if (!concept) return;
    
    uint32_t bits1 = t1->rows * t1->cols;
    uint32_t bits2 = t2->rows * t2->cols;
    uint32_t max_bits = (bits1 > bits2) ? bits1 : bits2;
    
    if (max_bits > 0) {
        uint32_t concept_bits = concept->rows * concept->cols;
        for (uint32_t i = 0; i < concept_bits && i < max_bits; i++) {
            uint8_t bit1 = 0, bit2 = 0;
            
            if (i < bits1) {
                bit1 = BIT_GET(t1->data[i / 8], i % 8);
            }
            
            if (i < bits2) {
                bit2 = BIT_GET(t2->data[i / 8], i % 8);
            }
            
            if (bit1 || bit2) {
                BIT_SET(concept->data[i / 8], i % 8);
            }
        }
    }
    
    concept->act = (t1->act + t2->act + 20) / 2;
    concept->res = (t1->res + t2->res + 30) / 2;
    concept->stab = 200;
    concept->ent = calc_bit_ent(concept, concept->cols);
    concept->efficiency = calculate_efficiency(concept);
    concept->is_concept = 1;
    concept->lu = (uint32_t)time(NULL);
    
    BitLink* link1 = create_link(concept, t1);
    BitLink* link2 = create_link(concept, t2);
    
    if (link1) {
        link1->strength = 180;
        link1->res = 200;
        link1->semantic_type = 3;
        link1->use_count = 2;
    }
    
    if (link2) {
        link2->strength = 180;
        link2->res = 200;
        link2->semantic_type = 3;
        link2->use_count = 2;
    }
    
    uint8_t direct_link_exists = 0;
    for (uint16_t i = 0; i < lnk_count; i++) {
        if ((lnks[i].src == t1 && lnks[i].tgt == t2) ||
            (lnks[i].src == t2 && lnks[i].tgt == t1)) {
            direct_link_exists = 1;
            break;
        }
    }
    
    if (!direct_link_exists) {
        BitLink* direct_link = create_link(t1, t2);
        if (direct_link) {
            direct_link->strength = 160;
            direct_link->semantic_type = 2;
        }
    }
    
    if (concept_count < 64) {
        MemoryConcept* new_concept = &concepts[concept_count++];
        new_concept->concept_tensor = concept;
        new_concept->member_count = 2;
        new_concept->member_indices[0] = t1 - tnsrs;
        new_concept->member_indices[1] = t2 - tnsrs;
        new_concept->abstraction_level = 1;
        new_concept->coherence = calc_bit_sim(t1, t2);
        new_concept->last_used = concept->lu;
        
        save_tnsr(concept);
    }
}

void merge_tensors(BitTensor* a, BitTensor* b) {
    // УСИЛЕННАЯ ПРОВЕРКА ВАЛИДНОСТИ
    if (!a || !b || a == b) {
        printf("[MERGE-ERROR] Нулевые или одинаковые тензоры\n");
        return;
    }
    
    // Проверка, что указатели находятся в пределах массива tnsrs
    if ((uintptr_t)a < (uintptr_t)tnsrs || 
        (uintptr_t)a >= (uintptr_t)(tnsrs + tnsr_count) ||
        (uintptr_t)b < (uintptr_t)tnsrs || 
        (uintptr_t)b >= (uintptr_t)(tnsrs + tnsr_count)) {
        printf("[MERGE-ERROR] Указатели вне массива тензоров\n");
        return;
    }
    
    if (!a->data || !b->data) {
        printf("[MERGE-ERROR] Нет данных в тензорах\n");
        return;
    }
    
    uint32_t bits_a = a->rows * a->cols;
    uint32_t bits_b = b->rows * b->cols;

    if (bits_a == 0 || bits_b == 0 || 
        bits_a > 65535 || bits_b > 65535) {
        printf("[MERGE-ERROR] Некорректный размер тензоров: %ux%u, %ux%u\n",
               a->rows, a->cols, b->rows, b->cols);
        return;
    }

    uint32_t new_bits = (bits_a + bits_b) >> 1;  // >> 1 вместо / 2
    if (new_bits < 8) new_bits = 8;

    uint16_t new_rows = 16;
    uint16_t new_cols = (uint16_t)((new_bits + new_rows - 1) / new_rows);

    BitTensor* merged = create_tnsr(new_rows, new_cols);
    if (!merged) {
        printf("[MERGE-ERROR] Не удалось создать новый тензор\n");
        return;
    }

    uint32_t bytes_a = (bits_a + 7) >> 3;  // >> 3 вместо / 8
    uint32_t bytes_b = (bits_b + 7) >> 3;
    uint32_t bytes_merged = (new_bits + 7) >> 3;

    for (uint32_t i = 0; i < bytes_merged; i++) {
        uint8_t byte_a = 0, byte_b = 0;

        if (i < bytes_a) byte_a = a->data[i];
        if (i < bytes_b) byte_b = b->data[i];

        uint8_t strong_bits = byte_a | byte_b;
        uint8_t weak_bits = byte_a & byte_b;
        merged->data[i] = strong_bits | (weak_bits >> 2);
    }

    // БЕЗОПАСНЫЕ ВЫЧИСЛЕНИЯ
    merged->act = (a->act + b->act) >> 1;  // Среднее арифметическое
    merged->res = (a->res + b->res) >> 1;
    
    // Вычисление стабильности: (a->stab * 3 + b->stab * 2) / 5
    // Эквивалент: (a->stab * 3 + b->stab * 2) * 51 >> 8 (51/256 ≈ 0.199 ≈ 1/5)
    merged->stab = ((uint16_t)a->stab * 3 + (uint16_t)b->stab * 2) * 51 >> 8;
    
    merged->ent = calc_bit_ent(merged, merged->cols);
    merged->efficiency = calculate_efficiency(merged);
    merged->cluster_id = (a->cluster_id == b->cluster_id) ? a->cluster_id : 0;

    transfer_links(a, merged);
    transfer_links(b, merged);

    a->act = 1;
    b->act = 1;
    a->dropout = 1;
    b->dropout = 1;

    printf("[MEM] Объединены тензоры [%u] и [%u] -> новый тензор [%u]\n",
           (uint32_t)(a - tnsrs), (uint32_t)(b - tnsrs), (uint32_t)(merged - tnsrs));
}

void transfer_links(BitTensor* from, BitTensor* to) {
    if (!from || !to || from == to) return;

    for (uint16_t i = 0; i < lnk_count; i++) {
        BitLink* link = &lnks[i];

        if (link->src == from) {
            link->src = to;
            to->conn++;
            if (from->conn > 0) from->conn--;
        }

        if (link->tgt == from) {
            link->tgt = to;
            to->conn++;
            if (from->conn > 0) from->conn--;
        }
    }
}

void create_concept_from_cluster(MemoryCluster* cluster) {
    if (!cluster || cluster->size < CONCEPT_CREATION_THRESHOLD) return;

    for (uint8_t ci = 0; ci < concept_count; ci++) {
        if (concepts[ci].concept_tensor &&
            concepts[ci].concept_tensor->cluster_id == cluster->cluster_id) {
            return;
        }
    }

    if (concept_count >= 64) {
        uint32_t oldest_time = UINT32_MAX;
        uint8_t oldest_idx = 0;

        for (uint8_t ci = 0; ci < concept_count; ci++) {
            if (concepts[ci].last_used < oldest_time) {
                oldest_time = concepts[ci].last_used;
                oldest_idx = ci;
            }
        }

        concept_count--;
        for (uint8_t ci = oldest_idx; ci < concept_count; ci++) {
            concepts[ci] = concepts[ci + 1];
        }
    }

    BitTensor* concept = create_tnsr(8, 8);
    if (!concept) return;

    for (uint32_t i = 0; i < 64; i++) {
        if (cluster->centroid[i % 256] > 128) {
            BIT_SET(concept->data[i / 8], i % 8);
        }
    }

    concept->act = cluster->stability / 2;
    concept->res = 200;
    concept->stab = cluster->stability;
    concept->ent = 60;
    concept->efficiency = 180;
    concept->is_concept = 1;
    concept->cluster_id = cluster->cluster_id;

    MemoryConcept* concept_entry = &concepts[concept_count++];
    concept_entry->concept_tensor = concept;
    concept_entry->member_count = (cluster->size > 32) ? 32 : cluster->size;
    concept_entry->abstraction_level = 1;
    concept_entry->coherence = calculate_cluster_coherence(cluster);
    concept_entry->last_used = (uint32_t)time(NULL);

    for (uint8_t i = 0; i < concept_entry->member_count; i++) {
        concept_entry->member_indices[i] = cluster->tensor_indices[i];
    }

    for (uint8_t i = 0; i < concept_entry->member_count; i++) {
        uint16_t idx = concept_entry->member_indices[i];
        if (idx < tnsr_count) {
            BitLink* link = create_link(concept, &tnsrs[idx]);
            if (link) {
                link->strength = 200;
                link->res = 220;
                link->semantic_type = 3;
            }
        }
    }

    save_concept(concept, cluster->cluster_id);

    printf("[CONCEPT] Создана концепция [%u] из кластера %u (размер: %u, стабильность: %u)\n",
           (uint32_t)(concept - tnsrs), cluster->cluster_id, cluster->size, cluster->stability);
}

void save_concept(BitTensor* concept, uint8_t cluster_id) {
    save_tnsr(concept);
}

void create_episode_from_working_memory(void) {
    if (working_mem_count < EPISODE_MIN_LENGTH) return;

    uint8_t new_episode = 0;
    for (uint8_t i = 0; i < working_mem_count; i++) {
        if (working_mem[i].episode_marker) {
            new_episode = 1;
            break;
        }
    }

    if (!new_episode) return;

    if (episode_count >= MAX_EPISODES) {
        uint8_t least_important_idx = 0;
        uint8_t min_importance = 255;

        for (uint16_t i = 0; i < episode_count; i++) {
            if (episodes[i].importance < min_importance) {
                min_importance = episodes[i].importance;
                least_important_idx = i;
            }
        }

        for (uint16_t i = least_important_idx; i < episode_count - 1; i++) {
            episodes[i] = episodes[i + 1];
        }
        episode_count--;
    }

    EpisodeMemory* episode = &episodes[episode_count++];
    memset(episode, 0, sizeof(EpisodeMemory));

    uint8_t seq_len = (working_mem_count < 256) ? working_mem_count : 256;
    episode->length = seq_len;

    for (uint8_t i = 0; i < seq_len; i++) {
        uint16_t tensor_idx = working_mem[i].tensor - tnsrs;
        episode->sequence[i] = tensor_idx;
        working_mem[i].episode_marker = 0;
    }

    uint32_t context_hash = calculate_context_hash();
    memcpy(episode->context_hash, &context_hash, sizeof(uint32_t));

    episode->start_time = working_mem[0].timestamp;
    episode->end_time = working_mem[seq_len - 1].timestamp;
    episode->success_score = 100;
    episode->importance = 100;
    episode->last_recall = 0;
    episode->recall_count = 0;

    printf("[EPISODE] Created episode %u (length: %u, context hash: %08X)\n",
        episode_count, seq_len, context_hash);

}

static inline uint32_t murmur_mix(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85EBCA6B;
    h ^= h >> 13;
    h *= 0xC2B2AE35;
    h ^= h >> 16;
    return h;
}

uint32_t calculate_context_hash(void) {
    uint32_t hash = 0x9747B28C;
    
    for (uint8_t i = 0; i < working_mem_count && i < 6; i++) {
        BitTensor* t = working_mem[i].tensor;
        if (t && t->data) {
            uint32_t total_bytes = (t->rows * t->cols + 7) / 8;
            uint32_t bytes_to_hash = total_bytes < 32 ? total_bytes : 32;
            uint32_t data_hash = 0;
            
            for (uint32_t j = 0; j < bytes_to_hash; j++) {
                data_hash = (data_hash << 3) ^ t->data[j] ^ (data_hash >> 29);
            }
            
            hash = murmur_mix(hash ^ data_hash ^ (i * 0xCC9E2D51));
        }
    }
    
    hash = murmur_mix(hash ^ (working_mem_count * 0x1B873593));
    
    return hash;
}

void activate_relevant_episodes(void) {
    if (episode_count == 0) return;

    uint32_t now = (uint32_t)time(NULL);
    uint32_t current_context = calculate_context_hash();

    for (uint16_t i = 0; i < episode_count; i++) {
        EpisodeMemory* episode = &episodes[i];
        uint32_t episode_context = *((uint32_t*)episode->context_hash);
        uint32_t context_diff = (current_context > episode_context) ?
                               (current_context - episode_context) :
                               (episode_context - current_context);

        if (context_diff < 0x100) {
            for (uint8_t j = 0; j < episode->length; j++) {
                uint16_t tensor_idx = episode->sequence[j];
                if (tensor_idx < tnsr_count) {
                    BitTensor* t = &tnsrs[tensor_idx];
                    t->act = (t->act * 7 + 120) >> 3;
                    t->lu = now;
                    add_to_working_memory(t);
                }
            }

            episode->last_recall = now;
            episode->recall_count++;

            if (episode->recall_count % 5 == 0) {
                episode->importance = (episode->importance * 9 + 110) >> 3;
                if (episode->importance > 200) episode->importance = 200;
            }

            printf("[EPISODE] Активирован эпизод %u (схожесть: %u%%)\n",
                   i, (uint32_t)(100 - (context_diff * 100 / 0x100)));

            if (i > 3) break;
        }
    }
}

// ===== ФУНКЦИИ ПОТОКОВ МЫСЛЕЙ =====

void form_thought_chain(void) {
    if (!goals.thought_stream_enabled || working_mem_count < 2) return;
    
    uint8_t stream_idx = active_thought_streams;
    uint8_t oldest_idx = 0;
    uint32_t now = (uint32_t)time(NULL);
    
    // 1. ПРОВЕРЯЕМ СУЩЕСТВУЮЩИЕ АКТИВНЫЕ ПОТОКИ
    // Если есть активные потоки с низкой связностью, переиспользуем их
    uint8_t found_reusable = 0;
    if (active_thought_streams > 0) {
        for (uint8_t i = 0; i < active_thought_streams; i++) {
            if (thought_streams[i].is_active && 
                thought_streams[i].coherence < 40 &&
                now - thought_streams[i].timestamp > 5) {
                // Переиспользуем слабый поток
                stream_idx = i;
                found_reusable = 1;
                break;
            }
        }
    }
    
    // 2. ЕСЛИ НЕТ ПЕРЕИСПОЛЬЗУЕМЫХ, ИЩЕМ САМЫЙ СТАРЫЙ
    if (!found_reusable && active_thought_streams >= MAX_THOUGHT_STREAMS) {
        uint32_t oldest_time = UINT32_MAX;
        uint8_t lowest_coherence = 255;
        
        for (uint8_t i = 0; i < active_thought_streams; i++) {
            // Ищем поток с наименьшей связностью и возрастом
            uint32_t stream_age = now - thought_streams[i].timestamp;
            uint8_t stream_coherence = thought_streams[i].coherence;
            
            // Весовая функция: возраст * (255 - связность)
            uint32_t score = stream_age * (255 - stream_coherence);
            uint32_t oldest_score = (oldest_time != UINT32_MAX) ? 
                                   oldest_time * (255 - lowest_coherence) : 0;
            
            if (score > oldest_score) {
                oldest_time = stream_age;
                lowest_coherence = stream_coherence;
                oldest_idx = i;
            }
        }
        
        stream_idx = oldest_idx;
        memset(&thought_streams[oldest_idx], 0, sizeof(ThoughtStream));
    } 
    else if (!found_reusable && active_thought_streams < MAX_THOUGHT_STREAMS) {
        // Создаем новый поток
        stream_idx = active_thought_streams;
        memset(&thought_streams[active_thought_streams], 0, sizeof(ThoughtStream));
    }
    
    ThoughtStream* stream = &thought_streams[stream_idx];
    
    // 3. ИНТЕЛЛЕКТУАЛЬНЫЙ ОТБОР ТЕНЗОРОВ ДЛЯ ЦЕПОЧКИ
    uint8_t chain_len = 0;
    
    // Сначала берем самые активные и свежие тензоры
    for (uint8_t attempt = 0; attempt < 2 && chain_len < MAX_THOUGHT_CHAIN_LENGTH; attempt++) {
        for (uint8_t i = 0; i < working_mem_count && chain_len < MAX_THOUGHT_CHAIN_LENGTH; i++) {
            BitTensor* tensor = working_mem[i].tensor;
            
            if (!tensor || tensor->dropout) continue;
            
            // Критерии отбора зависят от попытки
            uint8_t should_include = 0;
            
            if (attempt == 0) {
                // Первая попытка: самые активные и свежие
                should_include = (tensor->act > 70 && 
                                 now - working_mem[i].timestamp < 5);
            } else {
                // Вторая попытка: концепции и связанные тензоры
                should_include = (tensor->is_concept || tensor->cluster_id != 0) &&
                                (tensor->act > 50);
            }
            
            if (should_include) {
                // Проверяем, нет ли уже этого тензора в цепочке
                uint8_t already_in_chain = 0;
                for (uint8_t j = 0; j < chain_len; j++) {
                    if (stream->thought_chain[j] == tensor) {
                        already_in_chain = 1;
                        break;
                    }
                }
                
                if (!already_in_chain) {
                    stream->thought_chain[chain_len++] = tensor;
                }
            }
        }
    }
    
    // 4. ДОБАВЛЯЕМ СЕМАНТИЧЕСКИ СВЯЗАННЫЕ ТЕНЗОРЫ
    if (chain_len > 0 && chain_len < MAX_THOUGHT_CHAIN_LENGTH) {
        // Берем последний добавленный тензор и ищем его связи
        BitTensor* last_tensor = stream->thought_chain[chain_len - 1];
        
        // Ищем сильные связи
        for (uint16_t li = 0; li < lnk_count && chain_len < MAX_THOUGHT_CHAIN_LENGTH; li++) {
            if (lnks[li].strength > 80) {
                BitTensor* connected = NULL;
                
                if (lnks[li].src == last_tensor && lnks[li].tgt != last_tensor) {
                    connected = lnks[li].tgt;
                } else if (lnks[li].tgt == last_tensor && lnks[li].src != last_tensor) {
                    connected = lnks[li].src;
                }
                
                if (connected && !connected->dropout && connected->act > 40) {
                    // Проверяем, нет ли уже в цепочке
                    uint8_t already_in_chain = 0;
                    for (uint8_t j = 0; j < chain_len; j++) {
                        if (stream->thought_chain[j] == connected) {
                            already_in_chain = 1;
                            break;
                        }
                    }
                    
                    if (!already_in_chain) {
                        stream->thought_chain[chain_len++] = connected;
                        // Ограничиваем количество добавляемых связей
                        if (chain_len >= MAX_THOUGHT_CHAIN_LENGTH / 2) break;
                    }
                }
            }
        }
    }
    
    // 5. ИНИЦИАЛИЗИРУЕМ ПОТОК ЕСЛИ ЦЕПОЧКА ДОСТАТОЧНО ДЛИННАЯ
    if (chain_len >= 2) {
        stream->chain_length = chain_len;
        stream->timestamp = now;
        stream->coherence = calculate_chain_coherence(stream);
        
        // Интеллектуальное определение уровня абстракции
        uint8_t concept_count_in_chain = 0;
        uint8_t cluster_variety = 0;
        uint8_t cluster_ids[8] = {0};
        
        for (uint8_t i = 0; i < chain_len; i++) {
            BitTensor* t = stream->thought_chain[i];
            if (t->is_concept) concept_count_in_chain++;
            
            if (t->cluster_id != 0) {
                uint8_t cluster_found = 0;
                for (uint8_t j = 0; j < cluster_variety; j++) {
                    if (cluster_ids[j] == t->cluster_id) {
                        cluster_found = 1;
                        break;
                    }
                }
                if (!cluster_found && cluster_variety < 8) {
                    cluster_ids[cluster_variety++] = t->cluster_id;
                }
            }
        }
        
        // Уровень абстракции зависит от содержания цепочки
        if (concept_count_in_chain >= chain_len / 2) {
            stream->abstraction_level = 2 + (rand() % 2);  // 2-3
        } else if (cluster_variety > 1) {
            stream->abstraction_level = 1 + (rand() % 2);  // 1-2
        } else {
            stream->abstraction_level = rand() % 2;  // 0-1
        }
        
        // Инициализируем новые поля для рекурсивного мышления
        stream->activation_counter = 1;
        stream->is_active = 1;
        stream->recursion_depth = 0;
        memset(stream->meta_reflections, 0, sizeof(stream->meta_reflections));
        
        // Увеличиваем счетчик активных потоков если это новый поток
        if (!found_reusable && stream_idx == active_thought_streams && 
            active_thought_streams < MAX_THOUGHT_STREAMS) {
            active_thought_streams++;
        }
        
    } else {
        // Если цепочка слишком короткая, сбрасываем поток
        memset(stream, 0, sizeof(ThoughtStream));
    }
}

void evolve_thought_stream(ThoughtStream* stream) {
    if (!stream || !stream->is_active) return;
    
    uint32_t now = (uint32_t)time(NULL);
    
    if (now - stream->timestamp > THOUGHT_STREAM_LIFETIME) {
        stream->is_active = 0;
        return;
    }
    
    stream->activation_counter++;
        stream->coherence = calculate_chain_coherence(stream);
    
    // 1. РЕКУРСИВНОЕ МЫШЛЕНИЕ: анализ собственной цепочки
    // 1. РЕКУРСИВНОЕ МЫШЛЕНИЕ: анализ собственной цепочки
    if (stream->activation_counter % 8 == 0 && stream->chain_length >= 2) {
        // Создаем мета-тензор, представляющий саму цепочку мыслей
        // Используем достаточно большой тензор для хранения хэша
        BitTensor* meta_tensor = create_tnsr(8, 4);  // 32 бита = 4 байта
        if (meta_tensor) {
            // Кодируем хэш цепочки в тензор
            uint32_t chain_hash = 0;
            for (uint8_t i = 0; i < stream->chain_length && i < 4; i++) {
                if (stream->thought_chain[i]) {
                    chain_hash ^= (uint32_t)(stream->thought_chain[i] - tnsrs) << (i * 8);
                }
            }
            
            // Безопасное копирование с проверкой размера
            uint32_t total_bytes = (meta_tensor->rows * meta_tensor->cols + 7) / 8;
            if (total_bytes >= sizeof(uint32_t)) {
                memcpy(meta_tensor->data, &chain_hash, sizeof(uint32_t));
            } else {
                // Если тензор слишком мал, копируем только часть
                memcpy(meta_tensor->data, &chain_hash, total_bytes);
            }
            
            meta_tensor->act = stream->coherence;
            meta_tensor->is_concept = 1;
            meta_tensor->cluster_id = stream->abstraction_level;
            
            // Добавляем мета-тензор в конец цепочки (если есть место)
            if (stream->chain_length < MAX_THOUGHT_CHAIN_LENGTH - 1) {
                stream->thought_chain[stream->chain_length] = meta_tensor;
                stream->chain_length++;
                stream->coherence = (stream->coherence * 3 + calculate_chain_coherence(stream)) >> 2;
            }
        }
    }
       
    if (stream->activation_counter % 10 == 0 && stream->abstraction_level < 3) {
        stream->abstraction_level++;
        
        if (stream->abstraction_level == 2 && stream->chain_length >= 3) {
            create_abstraction_from_chain(stream);
        }
    }
    
    for (uint8_t i = 0; i < stream->chain_length; i++) {
        BitTensor* t = stream->thought_chain[i];
        if (t && !t->dropout) {
            t->act = (t->act * 7 + 100) >> 3;
            t->lu = now;
        }
    }
    
    // 2. РЕКУРСИВНОЕ МЫШЛЕНИЕ: обратные ассоциации
    if (stream->activation_counter % 7 == 0 && stream->chain_length > 1) {
        // Идем от конца к началу, ищем обратные связи
        for (int8_t i = stream->chain_length - 2; i >= 0 && i >= stream->chain_length - 4; i--) {
            BitTensor* current = stream->thought_chain[i];
            BitTensor* next = stream->thought_chain[i + 1];
            
            if (current && next) {
                // Создаем обратную связь (B → A если есть A → B)
                BitLink* reverse_link = NULL;
                for (uint16_t li = 0; li < lnk_count; li++) {
                    if (lnks[li].src == next && lnks[li].tgt == current) {
                        reverse_link = &lnks[li];
                        break;
                    }
                }
                
                if (!reverse_link) {
                    // Создаем обратную связь для рекурсивного мышления
                    create_link(next, current);
                }
            }
        }
    }
    
    if (stream->activation_counter % 5 == 0 && stream->chain_length > 0) {
        BitTensor* last = stream->thought_chain[stream->chain_length - 1];
        if (last) {
            BitTensor* association = find_association(last);
            if (association && stream->chain_length < MAX_THOUGHT_CHAIN_LENGTH) {
                stream->thought_chain[stream->chain_length] = association;
                stream->chain_length++;
                
                // 3. РЕКУРСИВНОЕ МЫШЛЕНИЕ: проверка циклов
                // Предотвращаем зацикливание (A→B→C→A)
                uint8_t is_cycle = 0;
                for (uint8_t i = 0; i < stream->chain_length - 1; i++) {
                    if (stream->thought_chain[i] == association) {
                        is_cycle = 1;
                        break;
                    }
                }
                
                if (is_cycle) {
                    // Цикл обнаружен - это хорошо для рекурсивного мышления!
                    // Усиливаем связность и создаем абстракцию из цикла
                    stream->coherence = (stream->coherence * 3 + 200) >> 2;
                    
                    if (stream->chain_length >= 4) {
                        // Создаем тензор, представляющий цикл
                        BitTensor* cycle_tensor = create_tnsr(6, 6);
                        if (cycle_tensor) {
                            uint32_t cycle_pattern = 0xC1C1C1C1;  // Паттерн цикла
                            memcpy(cycle_tensor->data, &cycle_pattern, sizeof(uint32_t));
                            cycle_tensor->act = 180;
                            cycle_tensor->is_concept = 1;
                            cycle_tensor->ent = 60;
                            
                            // Связываем с элементами цикла
                            uint8_t start_idx = stream->chain_length - 4;
                            for (uint8_t j = 0; j < 3 && start_idx + j < stream->chain_length; j++) {
                                create_link(cycle_tensor, stream->thought_chain[start_idx + j]);
                            }
                        }
                    }
                } else {
                    stream->coherence = calculate_chain_coherence(stream);
                }
            }
        }
    }
}

void create_recursive_abstraction(ThoughtStream* stream) {
    if (!stream || stream->chain_length < 3 || stream->recursion_depth >= 2) return;
    
    // Создаем абстракцию, которая представляет паттерн мышления
    // Используем достаточно большой тензор
    BitTensor* pattern_tensor = create_tnsr(8, 8);  // 64 бита = 8 байт
    if (!pattern_tensor) return;
    
    // Анализируем паттерн переходов в цепочке
    uint32_t transition_pattern = 0;
    for (uint8_t i = 0; i < stream->chain_length - 1 && i < 4; i++) {
        BitTensor* from = stream->thought_chain[i];
        BitTensor* to = stream->thought_chain[i + 1];
        
        if (from && to) {
            // Кодируем тип перехода (по силе связи, схожести и т.д.)
            uint8_t link_strength = 0;
            for (uint16_t li = 0; li < lnk_count; li++) {
                if ((lnks[li].src == from && lnks[li].tgt == to) ||
                    (lnks[li].src == to && lnks[li].tgt == from)) {
                    link_strength = lnks[li].strength;
                    break;
                }
            }
            
            transition_pattern |= (link_strength << (i * 8));
        }
    }
    
    // Безопасное копирование
    uint32_t total_bytes = (pattern_tensor->rows * pattern_tensor->cols + 7) / 8;
    if (total_bytes >= sizeof(uint32_t)) {
        memcpy(pattern_tensor->data, &transition_pattern, sizeof(uint32_t));
    } else {
        memcpy(pattern_tensor->data, &transition_pattern, total_bytes);
    }
    
    pattern_tensor->act = stream->coherence;
    pattern_tensor->is_concept = 1;
    pattern_tensor->ent = calc_bit_ent(pattern_tensor, pattern_tensor->cols);
    
    // Сохраняем как мета-рефлексию
    if (stream->recursion_depth < 3) {
        stream->meta_reflections[stream->recursion_depth] = pattern_tensor;
        stream->recursion_depth++;
        
        // Создаем связь между мета-тензором и оригинальной цепочкой
        for (uint8_t i = 0; i < 2 && i < stream->chain_length; i++) {
            create_link(pattern_tensor, stream->thought_chain[i]);
        }
    }
}

void prune_old_thought_streams(void) {
    uint32_t now = (uint32_t)time(NULL);
    
    for (uint8_t i = 0; i < active_thought_streams; i++) {
        ThoughtStream* stream = &thought_streams[i];
        
        if (!stream->is_active || 
            now - stream->timestamp > THOUGHT_STREAM_LIFETIME ||
            stream->coherence < 30) {
            
            if (i < active_thought_streams - 1) {
                thought_streams[i] = thought_streams[active_thought_streams - 1];
            }
            active_thought_streams--;
            i--;
        }
    }
}

BitTensor* find_association(BitTensor* current) {
    if (!current) return NULL;
    
    uint8_t best_score = 0;
    BitTensor* best_match = NULL;
    
    for (uint16_t i = 0; i < lnk_count && i < 50; i++) {
        BitLink* link = &lnks[i];
        
        if (link->src == current || link->tgt == current) {
            BitTensor* other = (link->src == current) ? link->tgt : link->src;
            
            if (other && other != current && !other->dropout && other->act > 40) {
                uint8_t score = link->strength;
                
                if (link->semantic_type > 0) score += 20;
                
                uint32_t now = (uint32_t)time(NULL);
                if (now - other->lu < 30) score += 10;
                
                if (score > best_score) {
                    best_score = score;
                    best_match = other;
                }
            }
        }
    }
    
    return best_match;
}

uint8_t calculate_chain_coherence(ThoughtStream* stream) {
    if (!stream || stream->chain_length < 2) return 0;
    
    uint32_t total_sim = 0;
    uint8_t comparisons = 0;
    
    for (uint8_t i = 0; i < stream->chain_length - 1; i++) {
        for (uint8_t j = i + 1; j < stream->chain_length && j < i + 3; j++) {
            BitTensor* t1 = stream->thought_chain[i];
            BitTensor* t2 = stream->thought_chain[j];
            
            if (t1 && t2 && t1 != t2) {
                uint8_t has_direct_link = 0;
                for (uint16_t li = 0; li < lnk_count; li++) {
                    if ((lnks[li].src == t1 && lnks[li].tgt == t2) ||
                        (lnks[li].src == t2 && lnks[li].tgt == t1)) {
                        has_direct_link = 1;
                        total_sim += lnks[li].strength;
                        break;
                    }
                }
                
                if (!has_direct_link) {
                    uint8_t sim = calc_bit_sim(t1, t2);
                    total_sim += sim / 2;
                }
                
                comparisons++;
            }
        }
    }
    
    if (comparisons == 0) return 0;
    return (uint8_t)(total_sim / comparisons);
}

void create_abstraction_from_chain(ThoughtStream* stream) {
    if (!stream || stream->chain_length < 3) return;
    
    BitTensor* abstraction = create_tnsr(8, 8);
    if (!abstraction) return;
    
    uint32_t total_bits = abstraction->rows * abstraction->cols;
    
    for (uint32_t i = 0; i < total_bits && i < 64; i++) {
        uint8_t combined_bit = 0;
        
        for (uint8_t j = 0; j < stream->chain_length && j < 3; j++) {
            BitTensor* t = stream->thought_chain[j];
            if (t && t->data) {
                uint32_t src_bits = t->rows * t->cols;
                if (i < src_bits) {
                    uint8_t bit_val = BIT_GET(t->data[i / 8], i % 8);
                    combined_bit |= bit_val;
                }
            }
        }
        
        if (combined_bit) {
            BIT_SET(abstraction->data[i / 8], i % 8);
        }
    }
    
    abstraction->act = 150;
    abstraction->res = 180;
    abstraction->stab = 170;
    abstraction->ent = calc_bit_ent(abstraction, abstraction->cols);
    abstraction->efficiency = calculate_efficiency(abstraction);
    abstraction->is_concept = 1;
    abstraction->lu = (uint32_t)time(NULL);
    
    for (uint8_t i = 0; i < stream->chain_length && i < 3; i++) {
        BitTensor* t = stream->thought_chain[i];
        if (t) {
            BitLink* link = create_link(abstraction, t);
            if (link) {
                link->semantic_type = 3;
                link->strength = 180;
                link->res = 200;
            }
        }
    }
    
    save_tnsr(abstraction);
}

// ===== ФУНКЦИИ ДЛЯ РАБОТЫ С СВЯЗЯМИ И СЕТЬЮ =====

BitLink* create_link(BitTensor* src, BitTensor* tgt) {
    if (!src || !tgt || lnk_count >= MAX_LINKS) return NULL;
    if (src == tgt && src->conn > 20) return NULL;

    uint16_t search_limit = (lnk_count > 1000) ? lnk_count / 4 : lnk_count;
    for (uint16_t i = 0; i < search_limit; i++) {
        if (lnks[i].src == src && lnks[i].tgt == tgt) {
            lnks[i].ts = (uint32_t)time(NULL);
            if (lnks[i].strength < LINK_MAX_STRENGTH) lnks[i].strength += 2;
            return &lnks[i];
        }
    }

    uint8_t sim_score = calc_bit_sim(src, tgt);
    uint32_t now = (uint32_t)time(NULL);
    uint32_t time_diff = (src->lu > tgt->lu) ? (src->lu - tgt->lu) : (tgt->lu - src->lu);
    
    // Нейросетевая модель: взвешенная активация
    float src_weight = (float)src->act / 255.0f;
    float tgt_weight = (float)tgt->act / 255.0f;
    float combined_activation = (src_weight + tgt_weight) / 2.0f;
    
    // Сила связи зависит от активации тензоров
    uint8_t base_strength = (uint8_t)(sim_score * combined_activation);
    
    // Контекстное усиление
    uint8_t context_boost = (time_diff < 2) ? 30 : 0;
    if (sim_score > 150) context_boost += 20;
    
    // Кластерное усиление
    uint8_t cluster_boost = 0;
    if (src->cluster_id != 0 && src->cluster_id == tgt->cluster_id) {
        cluster_boost = 40;
        // Усиление для стабильных кластеров
        for (uint16_t ci = 0; ci < cluster_count; ci++) {
            if (clusters[ci].cluster_id == src->cluster_id && clusters[ci].stability > 180) {
                cluster_boost += 50;
                break;
            }
        }
    }

    uint8_t final_strength = base_strength + context_boost + cluster_boost;
    if (final_strength > 255) final_strength = 255;
    
    // Минимальная сила для "значимых" связей
    if (final_strength < 20 && sim_score > 100) final_strength = 20;
    if (final_strength > 200) final_strength = 200;

    BitLink* link = &lnks[lnk_count++];
    link->src = src;
    link->tgt = tgt;
    link->strength = final_strength;
    
    // Нейросетевое взвешивание: сила влияет на передачу активации
    link->weight = (uint16_t)(final_strength * 256 / 255); // 0-256 scale
    
    link->res = calc_res_match(src, tgt);
    link->semantic_type = (src->cluster_id != 0 && src->cluster_id == tgt->cluster_id) ? 1 : 
                          (src->is_concept || tgt->is_concept) ? 3 : 
                          (context_boost > 30) ? 2 : 0;

    link->ts = now;
    link->last_act = now;
    link->use_count = 1;
    link->success_count = 0;

    // Передача активации между тензорами (нейросетевой эффект)
    uint8_t activation_transfer = (uint8_t)(combined_activation * final_strength / 10);
    src->act = (src->act > activation_transfer) ? src->act - activation_transfer : 0;
    tgt->act = (tgt->act > activation_transfer) ? tgt->act - activation_transfer : 0;

    // Обновление связей кластеров
    if (src->cluster_id != 0) {
        for (uint16_t ci = 0; ci < cluster_count; ci++) {
            if (clusters[ci].cluster_id == src->cluster_id) {
                clusters[ci].link_count++;
                break;
            }
        }
    }
    if (tgt->cluster_id != 0 && tgt->cluster_id != src->cluster_id) {
        for (uint16_t ci = 0; ci < cluster_count; ci++) {
            if (clusters[ci].cluster_id == tgt->cluster_id) {
                clusters[ci].link_count++;
                break;
            }
        }
    }

    return link;
}

void update_link_strength(BitLink* link, uint8_t was_successful) {
    if (!link) return;
    link->use_count++;
    link->ts = (uint32_t)time(NULL);

    // Нейросетевая модель обновления
    float activation_src = (float)link->src->act / 255.0f;
    float activation_tgt = (float)link->tgt->act / 255.0f;
    float combined_prev = (float)link->strength / 255.0f;
    
    if (was_successful) {
        link->success_count++;
        
        // Усиление на основе успешности и активации
        float success_factor = 1.0f + (float)link->success_count / (link->use_count + 1);
        float activation_factor = (activation_src + activation_tgt) / 2.0f;
        
        float new_strength = combined_prev * 0.8f + activation_factor * success_factor * 0.2f;
        link->strength = (uint8_t)(new_strength * 255.0f);
        
        if (link->strength > LINK_MAX_STRENGTH) link->strength = LINK_MAX_STRENGTH;
        
        // Небольшое усиление резонанса
        if (link->res < RES_MAX) link->res += 3;

        // Передача активации от успешной связи
        uint8_t reward = (uint8_t)(success_factor * 10);
        link->src->act = (link->src->act > ACT_MAX - reward) ? ACT_MAX : link->src->act + reward;
        link->tgt->act = (link->tgt->act > ACT_MAX - reward) ? ACT_MAX : link->tgt->act + reward;

    } else {
        // Ослабление на основе неуспешности
        float failure_penalty = (float)link->use_count / (link->success_count + 1);
        float new_strength = combined_prev * 0.9f - failure_penalty * 0.05f;
        
        if (new_strength < 0.0f) new_strength = 0.0f;
        link->strength = (uint8_t)(new_strength * 255.0f);
        
        if (link->strength < LINK_MIN_STRENGTH) link->strength = LINK_MIN_STRENGTH;

        // Уменьшение резонанса
        if (link->res > 10) link->res -= 2;

        // Уменьшение активации при неудаче
        if (link->src->act > 2) link->src->act -= 2;
        if (link->tgt->act > 2) link->tgt->act -= 2;
    }

    // Обновление веса на основе текущей силы связи
    link->weight = (uint16_t)(link->strength * 256 / 255);
    link->last_act = link->ts;
}
void decay_unused_links(void) {
    for (uint16_t i = 0; i < lnk_count && i < MAX_LINKS; i++) {
        BitLink* link = &lnks[i];
        
        // Проверяем, что ссылки на тензоры существуют и действительны
        if (link->src != NULL && link->tgt != NULL) {
            // Проверяем, что индексы тензоров в допустимом диапазоне
            uintptr_t src_offset = link->src - tnsrs;
            uintptr_t tgt_offset = link->tgt - tnsrs;
            
            if (src_offset < MAX_TENSORS && tgt_offset < MAX_TENSORS) {
                // Проверяем, что тензоры действительно инициализированы
                if (link->src->data != NULL && link->tgt->data != NULL) {
                    // Уменьшаем силу связи
                    if (link->strength > LINK_STRENGTH_DEC) {
                        link->strength -= LINK_STRENGTH_DEC;
                    } else {
                        link->strength = 0;
                    }
                    
                    // Удаляем слабые связи
                    if (link->strength < LINK_MIN_STRENGTH) {
                        // Сдвигаем все последующие элементы
                        for (uint16_t j = i; j < lnk_count - 1 && j < MAX_LINKS - 1; j++) {
                            lnks[j] = lnks[j + 1];
                        }
                        lnk_count--;
                        i--; // Проверяем текущий индекс снова после сдвига
                    }
                } else {
                    // Удаляем поврежденную ссылку
                    for (uint16_t j = i; j < lnk_count - 1 && j < MAX_LINKS - 1; j++) {
                        lnks[j] = lnks[j + 1];
                    }
                    lnk_count--;
                    i--;
                }
            } else {
                // Удаляем ссылки на тензоры за пределами допустимого диапазона
                for (uint16_t j = i; j < lnk_count - 1 && j < MAX_LINKS - 1; j++) {
                    lnks[j] = lnks[j + 1];
                }
                lnk_count--;
                i--;
            }
        } else {
            // Удаляем нулевые ссылки
            for (uint16_t j = i; j < lnk_count - 1 && j < MAX_LINKS - 1; j++) {
                lnks[j] = lnks[j + 1];
            }
            lnk_count--;
            i--;
        }
    }
}
void learn_by_binary_update(BitTensor* target, const uint8_t* input_data, uint16_t input_len) {
    if (!target || !input_data || input_len == 0) return;

    // Вычисляем общий размер данных тензора
    uint32_t total_data_size = target->rows * target->cols;

    // Если input_len больше, чем умещается в target->data, обрежем
    if (input_len > total_data_size) {
        input_len = total_data_size;
    }

    // --- Распределение (Spread) ---
    // Обновляем данные по срезам
    uint32_t bytes_processed = 0;
    while (bytes_processed < input_len) {
        // Определяем размер текущего среза
        uint16_t slice_size = (input_len - bytes_processed < LEARN_SLICE_SIZE) ?
                               (input_len - bytes_processed) : LEARN_SLICE_SIZE;

        // Проверяем границы
        if (bytes_processed + slice_size > total_data_size) {
            // Это защита, но должна быть невозможна при текущей логике
            slice_size = total_data_size - bytes_processed;
        }

        if (slice_size == 0) break; // Нечего обновлять

        // Выполняем обновление среза (например, XOR или AND/OR)
        // Это пример обновления. Можно использовать другие битовые операции или веса.
        for (uint16_t i = 0; i < slice_size; i++) {
            target->data[bytes_processed + i] ^= input_data[bytes_processed + i];
            // Пример с весом: target->data[bytes_processed + i] = (target->data[bytes_processed + i] & ~weight_mask) | (input_data[bytes_processed + i] & weight_mask);
        }

        // --- Потенциальная логика агрегации (Aggregation) ---
        // После обновления среза можно выполнить локальную агрегацию.
        // Например, обновить локальную энтропию, резонанс, эффективность для среза.
        // Или обновить связанные тензоры, если они есть.
        // Для простоты, обновим общую энтропию и активность тензора после каждого среза.
        // Но лучше это делать реже, например, раз в N срезов или в конце.
        if ((bytes_processed / LEARN_SLICE_SIZE) % 2 == 0) { // Пример: обновляем раз в 2 среза
            target->ent = calc_bit_ent(target, slice_size);
        }

        bytes_processed += slice_size;
    }

    // --- Агрегация (Aggregation) ---
    // Обновляем глобальные свойства тензора после обработки всех срезов
    if (bytes_processed > 0) {
        target->act = (target->act + bytes_processed) / 2; // Пример простого обновления активности
        target->lu = (uint32_t)time(NULL); // Обновляем время последнего обновления
        target->compute_cost += bytes_processed; // Обновляем счётчик вычислительных затрат
        // Обновляем эффективность на основе новых данных
        target->efficiency = calculate_efficiency(target);
        // Обновляем стабильность
        target->stab = (target->stab + 1) / 2; // Простой пример
    }
}

void prevent_overfitting_by_bit_shift(BitTensor* t) {
    if (!t || t->act < 200 || t->ent > 120) return;
    uint32_t total_bits = t->rows * t->cols;
    int m_count = (t->act - 200) / 10;
    if (m_count < 1) m_count = 1;
    for (int m = 0; m < m_count && m < 5; m++) {
        uint32_t bit_pos = rand() % total_bits;
        t->data[bit_pos / 8] ^= (1U << (bit_pos % 8));
    }
    t->ent = calc_bit_ent(t, t->cols);
    t->act = (uint8_t)((float)t->act * 0.8f);
    t->res = (uint8_t)((float)t->res * 0.95f);
    t->efficiency = calculate_efficiency(t);
}

void self_reflect_on_thought(void) {
    BitTensor* most_active = find_significant_tensor(SEARCH_MOST_ACTIVE, NULL);
    if (!most_active) return;

    BitLink* self_link = NULL;
    for (uint16_t i = 0; i < lnk_count; i++) {
        if (lnks[i].src == most_active && lnks[i].tgt == most_active) {
            self_link = &lnks[i]; break;
        }
    }
    if (!self_link) self_link = create_link(most_active, most_active);
    if (!self_link) return;

    update_link_strength(self_link, 1);
    uint32_t total_bits = most_active->rows * most_active->cols;
    for (uint32_t i = 0; i < (total_bits + 7) / 8; i++) {
        uint8_t old = most_active->data[i];
        uint8_t shifted = (old << 1) | (old >> 7);
        most_active->data[i] = BIT_XOR(old, shifted);
    }

    most_active->ent = calc_bit_ent(most_active, most_active->cols);
    most_active->act = (most_active->act > ACT_MAX * 0.9f) ? ACT_MAX :
                      (uint8_t)((float)most_active->act * 1.1f);
    most_active->efficiency = calculate_efficiency(most_active);

    if (most_active->is_concept) {
        for (uint8_t ci = 0; ci < concept_count; ci++) {
            if (concepts[ci].concept_tensor == most_active) {
                concepts[ci].last_used = (uint32_t)time(NULL);
                break;
            }
        }
    }
}

void apply_dropout(void) {
    if (!goals.dropout_enabled) return;

    for (uint16_t i = 0; i < tnsr_count; i++) {
        if (tnsrs[i].is_concept) {
            if (rand() % 1000 < (DROPOUT_RATE / 2)) {
                tnsrs[i].dropout = 1;
            }
            continue;
        }

        uint8_t cluster_active = 0;
        if (tnsrs[i].cluster_id != 0) {
            for (uint16_t ci = 0; ci < cluster_count; ci++) {
                if (clusters[ci].cluster_id == tnsrs[i].cluster_id) {
                    if (clusters[ci].activation_level > 100) {
                        cluster_active = 1;
                    }
                    break;
                }
            }
        }

        if (tnsrs[i].dropout) {
            if (tnsrs[i].act > 10) tnsrs[i].act = tnsrs[i].act >> 1;
            if (rand() % 100 < (cluster_active ? 3 : 1)) {
                tnsrs[i].dropout = 0;
                tnsrs[i].act = 50;
            }
        } else {
            uint8_t dropout_chance = cluster_active ? DROPOUT_RATE : DROPOUT_RATE * 2;
            if (rand() % 1000 < dropout_chance) {
                tnsrs[i].dropout = 1;
            }
        }
    }
}

BitTensor* find_efficient_match(BitTensor* input) {
    if (!input) return NULL;
    uint16_t best_score = 0;
    BitTensor* best_match = NULL;
    uint16_t best_link_idx = 0xFFFF;
    uint32_t now = (uint32_t)time(NULL);

    for (uint16_t i = 0; i < lnk_count; i++) {
        if (lnks[i].src == input && lnks[i].strength > 20) {
            BitTensor* t = lnks[i].tgt;

            if (t->is_concept && !input->is_concept) continue;

            uint8_t freshness = (now - t->lu > 300) ? 0 : 255 - (uint8_t)((now - t->lu)/2);
            uint8_t connectivity = (t->conn > 10) ? 255 : (t->conn * 25);
            uint8_t sim = calc_bit_sim(input, t);
            uint8_t res_match = calc_res_match(input, t);
            uint8_t eff_boost = (t->efficiency > goals.target_efficiency) ? 255 : t->efficiency;

            uint8_t cluster_bonus = 0;
            if (input->cluster_id != 0 && input->cluster_id == t->cluster_id) {
                cluster_bonus = 50;
            }

            uint32_t score = (uint32_t)(sim + cluster_bonus) * res_match * eff_boost * freshness * connectivity >> 16;
            score += lnks[i].strength * 2;

            if (score > best_score && score > 4000) {
                best_score = score;
                best_match = t;
                best_link_idx = i;
            }
        }
    }

    uint8_t link_created = 0;
    if (!best_match || best_score < 6000) {
        for (uint16_t i = 0; i < tnsr_count; i++) {
            BitTensor* t = &tnsrs[i];
            if (t == input || t->act < 20 || t->dropout) continue;
            if (t->is_concept && !input->is_concept) continue;

            uint8_t freshness = (now - t->lu > 300) ? 0 : 255 - (uint8_t)((now - t->lu)/2);
            uint8_t sim = calc_bit_sim(input, t);
            uint8_t res_match = calc_res_match(input, t);

            uint8_t cluster_bonus = 0;
            if (input->cluster_id != 0 && input->cluster_id == t->cluster_id) {
                cluster_bonus = 100;
            }

            uint32_t score = (uint32_t)(sim + cluster_bonus) * res_match * freshness * t->efficiency >> 8;

            if (score > best_score && score > 4000) {
                best_score = score;
                best_match = t;
            }
        }

        if (best_match) {
            BitLink* new_link = create_link(input, best_match);
            if (new_link) {
                best_link_idx = (uint16_t)(new_link - lnks);
                link_created = 1;
            }
        }
    }

    if (best_match) {
        best_match->lu = now;
        input->lu = now;
        add_to_working_memory(best_match);

        if (best_match->cluster_id != 0) {
            for (uint16_t ci = 0; ci < cluster_count; ci++) {
                if (clusters[ci].cluster_id == best_match->cluster_id) {
                    clusters[ci].last_access = now;
                    clusters[ci].activation_level = (clusters[ci].activation_level * 7 + 200) >> 3;
                    break;
                }
            }
        }

        if (best_link_idx < lnk_count) {
            BitLink* best_link = &lnks[best_link_idx];
            update_link_strength(best_link, 1);
            if (best_score > 8000 && !link_created) {
                best_link->weight = best_link->weight * 11 / 10;
            }
        }

        uint8_t act_increase = (best_score / 200 > 50) ? 50 : best_score / 200;
        best_match->act += act_increase;
        if (best_match->act > ACT_MAX) best_match->act = ACT_MAX;
    }

    return best_match;
}

void optimize_tensor(BitTensor* t) {
    if (!t || t->goal_active == 0) return;
    if (t->efficiency < goals.target_efficiency) {
        if (t->compute_cost > 100) t->compute_cost -= t->compute_cost / 10;
        if (t->stab < 200) t->stab += 5;
        t->goal_active = 2;
        t->efficiency = calculate_efficiency(t);
    }
}

void update_bit_net_with_goals(void) {
    uint32_t now = (uint32_t)time(NULL);

    if (now % 5 == 0) decay_unused_links();

    if (now - last_mem_check_ts >= MEM_REDUCE_INTERVAL) {
        last_mem_check_ts = now;
        for (uint16_t i = 0; i < tnsr_count; i++) {
            if (tnsrs[i].act < LOW_ACT_THRESHOLD && !tnsrs[i].mem_red) {
                reduce_tnsr_mem(&tnsrs[i]);
            }
        }
    }

    if (now % 10 == 0) {
        update_efficiency_goal();
        for (uint16_t i = 0; i < tnsr_count; i++) {
            if (tnsrs[i].efficiency < goals.target_efficiency - 20) {
                tnsrs[i].goal_active = 1;
            }
        }
        apply_dropout();

        for (uint16_t ci = 0; ci < cluster_count; ci++) {
            clusters[ci].activation_level = (clusters[ci].activation_level * 9) >> 3;
        }
    }

    for (uint16_t i = 0; i < tnsr_count; i++) {
        if (tnsrs[i].goal_active == 1) {
            optimize_tensor(&tnsrs[i]);
        }
    }

    uint32_t total_res_sum = 0;
    uint16_t active_links = 0;
    for (uint16_t i = 0; i < lnk_count; i++) {
        BitLink* lnk = &lnks[i];
        if (lnk->strength > 25 && !lnk->src->dropout && !lnk->tgt->dropout) {
            uint8_t src_act = lnk->src->act;
            uint8_t tgt_act = lnk->tgt->act;
            uint8_t interaction = (src_act & tgt_act) + ((src_act ^ tgt_act) >> 1);
            uint8_t was_successful = (interaction > 128) ? 1 : 0;
            update_link_strength(lnk, was_successful);

            uint16_t boost = (lnk->weight * interaction * lnk->strength) >> 16;
            lnk->src->act = (lnk->src->act > ACT_MAX - (boost >> 2)) ? ACT_MAX : lnk->src->act + (boost >> 2);
            lnk->tgt->act = (lnk->tgt->act > ACT_MAX - boost) ? ACT_MAX : lnk->tgt->act + boost;
            lnk->last_act = now;
            total_res_sum += lnk->res * lnk->strength;
            active_links++;
            add_to_working_memory(lnk->tgt);

            if (lnk->src->cluster_id != 0) {
                for (uint16_t ci = 0; ci < cluster_count; ci++) {
                    if (clusters[ci].cluster_id == lnk->src->cluster_id) {
                        clusters[ci].activation_level = (clusters[ci].activation_level * 7 + 150) >> 3;
                        break;
                    }
                }
            }
        }
    }

    if (now % 15 == 0) {
        for (uint16_t i = 0; i < tnsr_count; i++) {
            if (tnsrs[i].act > 100 && tnsrs[i].conn < 5 && !tnsrs[i].is_concept) {
                for (uint16_t j = 0; j < tnsr_count; j++) {
                    if (i != j && tnsrs[j].act > 50 && !tnsrs[j].is_concept) {
                        if (calc_bit_sim(&tnsrs[i], &tnsrs[j]) > 100) {
                            create_link(&tnsrs[i], &tnsrs[j]);
                            break;
                        }
                    }
                }
            }
        }
    }

    if (active_links > 0) {
        uint8_t avg_link_res = (uint8_t)(total_res_sum / active_links);
        sys_res = (sys_res * 230 + avg_link_res * 25) >> 8;
    }

    sstate.res_hist[sstate.hist_idx] = sys_res;
    sstate.hist_idx = (sstate.hist_idx + 1) % HISTORY_SIZE;

    if (goals.self_organization_enabled && now % SELF_ORG_INTERVAL == 0) {
        self_organize_memory_clusters();
    }
}

static uint32_t calculate_resonance_score(BitTensor* t, uint32_t now) {
    if (!t) return 0;

    uint32_t base_score = (uint32_t)t->act * t->res * t->efficiency;

    uint32_t cluster_bonus = 0;
    if (t->cluster_id != 0) {
        for (uint16_t ci = 0; ci < cluster_count; ci++) {
            if (clusters[ci].cluster_id == t->cluster_id) {
                cluster_bonus = clusters[ci].stability * clusters[ci].activation_level;
                break;
            }
        }
    }

    uint32_t link_score = 0, link_count = 0;
    for (uint16_t i = 0; i < lnk_count; i++) {
        BitLink* l = &lnks[i];
        if ((l->src == t || l->tgt == t) && l->strength > 40 && !l->src->dropout && !l->tgt->dropout) {
            link_score += l->strength * (l->semantic_type + 1);
            link_count++;
        }
    }

    uint32_t avg_link = link_count ? (link_score / link_count) : 0;
    uint32_t freshness = (now - t->lu > 300) ? 0 : 1000 - (now - t->lu) * 3;

    return base_score + cluster_bonus + (uint32_t)t->conn * avg_link * 10 +
           (uint32_t)t->stab * 10 + freshness;
}

BitTensor* find_significant_tensor(SearchStrategy strategy, void* context) {
    BitTensor* best = NULL;
    uint32_t best_score = 0;
    uint32_t now = (uint32_t)time(NULL);
    ScoreFunction custom_func = (strategy == SEARCH_CUSTOM_SCORE) ? (ScoreFunction)context : NULL;

    if (strategy == SEARCH_BY_CLUSTER && context != NULL) {
        uint8_t target_cluster_id = *(uint8_t*)context;
        for (uint16_t i = 0; i < tnsr_count; i++) {
            BitTensor* t = &tnsrs[i];
            if (t->cluster_id == target_cluster_id && !t->dropout) {
                uint32_t score = calculate_resonance_score(t, now);
                if (score > best_score) {
                    best_score = score;
                    best = t;
                }
            }
        }
        return best;
    }

    if (strategy == SEARCH_CONCEPTUAL) {
        for (uint8_t ci = 0; ci < concept_count; ci++) {
            BitTensor* t = concepts[ci].concept_tensor;
            if (t && !t->dropout) {
                uint32_t score = t->act * t->res * concepts[ci].coherence;
                if (score > best_score) {
                    best_score = score;
                    best = t;
                }
            }
        }
        if (best) return best;
        strategy = SEARCH_RESONANT;
    }

    for (uint16_t i = 0; i < tnsr_count; i++) {
        BitTensor* t = &tnsrs[i];
        if (!t->data || t->dropout) continue;

        uint32_t score = 0;
        switch (strategy) {
            case SEARCH_MOST_ACTIVE:
                score = t->act;
                break;
            case SEARCH_EFFICIENT:
                score = (uint32_t)t->efficiency * t->act;
                break;
            case SEARCH_RESONANT:
                score = calculate_resonance_score(t, now);
                break;
            case SEARCH_CUSTOM_SCORE:
                if (custom_func) score = custom_func(t, NULL);
                break;
            default:
                continue;
        }

        if (score > best_score) {
            best_score = score;
            best = t;
        }
    }

    for (uint8_t i = 0; i < working_mem_count; i++) {
        BitTensor* t = working_mem[i].tensor;
        if (!t || t->dropout) continue;

        uint32_t score = 0;
        uint8_t bonus = working_mem[i].priority * 5;

        switch (strategy) {
            case SEARCH_MOST_ACTIVE:
                score = t->act + bonus;
                break;
            case SEARCH_EFFICIENT:
                score = (uint32_t)t->efficiency * t->act + bonus;
                break;
            case SEARCH_RESONANT:
                score = calculate_resonance_score(t, now) + bonus;
                break;
            case SEARCH_CUSTOM_SCORE:
                if (custom_func) score = custom_func(t, NULL) + bonus;
                break;
            default:
                continue;
        }

        if (score > best_score) {
            best_score = score;
            best = t;
        }
    }

    if (strategy == SEARCH_RESONANT && best && best_score > 20000) {
        best->stab = (best->stab * 7 + 200) >> 3;

        if (best->cluster_id != 0) {
            for (uint16_t ci = 0; ci < cluster_count; ci++) {
                if (clusters[ci].cluster_id == best->cluster_id) {
                    clusters[ci].stability = (clusters[ci].stability * 9 + 220) >> 3;
                    break;
                }
            }
        }
    }

    return best;
}

// ===== FORWARD-FORWARD CONSOLIDATION =====

void forward_forward_consolidation(uint8_t mode) {
    uint32_t now = (uint32_t)time(NULL);
    printf("[F-F CONS] Начало (%s)...\n", 
           mode == 0 ? "быстрая" : mode == 1 ? "полная" : "агрессивная");
    
    // --- АЛГОРИТМ ФЮРЕРА ДЛЯ БЫСТРОГО ВЫЧИСЛЕНИЯ GOODNESS ---
    struct TensorGoodness {
        BitTensor* tensor;
        float goodness;
        uint8_t positive_examples;
        uint8_t negative_examples;
        uint16_t tensor_idx;
    } goodness_scores[MAX_TENSORS];
    
    uint16_t scored_tensors = 0;
    
    // Быстрый предварительный отбор активных тензоров
    uint16_t active_indices[MAX_TENSORS];
    uint16_t active_count = 0;
    
    for (uint16_t i = 0; i < tnsr_count && active_count < MAX_TENSORS; i++) {
        if (tnsrs[i].act > 30 && !tnsrs[i].dropout) {
            active_indices[active_count++] = i;
        }
    }
    
    // Ограничим количество для обработки
    uint16_t max_to_process = (mode == 0) ? 50 : (mode == 1) ? 100 : 200;
    if (active_count > max_to_process) {
        active_count = max_to_process;
    }
    
    // Быстрое вычисление goodness с кэшированием
    for (uint16_t a = 0; a < active_count; a++) {
        uint16_t i = active_indices[a];
        BitTensor* t = &tnsrs[i];
        
        float goodness = 0.0f;
        uint8_t positive = 0, negative = 0;
        
        // Быстрый проход по связям с ограничением
        uint16_t max_links_to_check = (mode == 0) ? 20 : 40;
        uint16_t links_checked = 0;
        
        // Используем индекс тензора для быстрого доступа к связям
        uint16_t tensor_idx = t - tnsrs;
        if (tensor_idx < MAX_TENSORS) {
            uint8_t link_count = tensor_links[tensor_idx].link_count;
            if (link_count > max_links_to_check) link_count = max_links_to_check;
            
            for (uint8_t li_idx = 0; li_idx < link_count; li_idx++) {
                uint16_t link_idx = tensor_links[tensor_idx].link_indices[li_idx];
                if (link_idx >= lnk_count) continue;
                
                BitLink* link = &lnks[link_idx];
                BitTensor* other = (link->src == t) ? link->tgt : link->src;
                
                if (other && other != t && !other->dropout) {
                    links_checked++;
                    
                    // Быстрое вычисление link_goodness
                    float link_goodness = link->strength / 255.0f;
                    link_goodness *= link->res / 255.0f;
                    
                    uint32_t inactive_time = now - link->last_act;
                    float freshness = (inactive_time > 255) ? 0 : (255 - inactive_time) / 255.0f;
                    link_goodness *= freshness;
                    
                    if (link_goodness > 0.5f) {
                        goodness += link_goodness;
                        positive++;
                    } else {
                        negative++;
                    }
                }
                
                if (links_checked >= max_links_to_check) break;
            }
        }
        
        if (positive > 0) {
            goodness /= positive;
            
            // Только если goodness значительный
            if (goodness > 0.2f || goodness < 0.8f) {
                goodness_scores[scored_tensors].tensor = t;
                goodness_scores[scored_tensors].tensor_idx = i;
                goodness_scores[scored_tensors].goodness = goodness;
                goodness_scores[scored_tensors].positive_examples = positive;
                goodness_scores[scored_tensors].negative_examples = negative;
                scored_tensors++;
                
                if (scored_tensors >= MAX_TENSORS - 10) break;
            }
        }
    }
    
    // --- ОПТИМИЗИРОВАННАЯ СОРТИРОВКА ПО GOODNESS ---
    // Используем алгоритм Фюрера для быстрой сортировки
    if (scored_tensors > 1) {
        // Быстрая сортировка с ограниченной глубиной рекурсии
        #define SWAP_GOODNESS(a, b) do { \
            struct TensorGoodness tmp = goodness_scores[a]; \
            goodness_scores[a] = goodness_scores[b]; \
            goodness_scores[b] = tmp; \
        } while(0)
        
        // Простой алгоритм - сортировка выбором для небольших массивов
        for (uint16_t i = 0; i < scored_tensors - 1; i++) {
            uint16_t max_idx = i;
            for (uint16_t j = i + 1; j < scored_tensors; j++) {
                if (goodness_scores[j].goodness > goodness_scores[max_idx].goodness) {
                    max_idx = j;
                }
            }
            if (max_idx != i) {
                SWAP_GOODNESS(i, max_idx);
            }
        }
    }
    
    uint16_t strengthened = 0, weakened = 0;
    
    // Обрабатываем только топ-N тензоров
    uint16_t top_n = (mode == 0) ? 10 : (mode == 1) ? 30 : scored_tensors;
    if (top_n > scored_tensors) top_n = scored_tensors;
    
    for (uint16_t i = 0; i < top_n; i++) {
        BitTensor* t = goodness_scores[i].tensor;
        float goodness = goodness_scores[i].goodness;
        
        if (goodness > 0.7f) {
            // Усиление
            if (t->act < 220) t->act = (t->act * 7 + 180) >> 3;
            if (t->stab < 230) t->stab = (t->stab * 7 + 200) >> 3;
            if (t->efficiency < 220) t->efficiency = (t->efficiency * 7 + 180) >> 3;
            
            // Рекурсивный forward с ограниченной глубиной (максимум 3 уровня)
            uint8_t recursion_depth = 0;
            BitTensor* current_tensor = t;
            BitTensor* visited[3] = {NULL, NULL, NULL}; // Максимум 3 уровня
            
            while (recursion_depth < 3 && current_tensor) {
                visited[recursion_depth] = current_tensor;
                
                // Проверяем, есть ли у тензора forward-ссылка
                if (current_tensor->forward > 0 && current_tensor->forward < tnsr_count) {
                    BitTensor* next_tensor = &tnsrs[current_tensor->forward];
                    
                    // Проверяем, не в цикле ли мы
                    uint8_t is_cycle = 0;
                    for (uint8_t cycle_check = 0; cycle_check < recursion_depth; cycle_check++) {
                        if (visited[cycle_check] == next_tensor) {
                            is_cycle = 1;
                            break;
                        }
                    }
                    
                    if (!is_cycle) {
                        // Усиливаем следующий тензор
                        if (next_tensor->act < 200) next_tensor->act = (next_tensor->act * 7 + 150) >> 3;
                        if (next_tensor->efficiency < 200) next_tensor->efficiency = (next_tensor->efficiency * 7 + 150) >> 3;
                        
                        // Усиливаем связь между текущим и следующим
                        BitLink* recursive_link = find_or_create_link(current_tensor, next_tensor);
                        if (recursive_link) {
                            uint8_t boost = (uint8_t)(goodness * 20);
                            if (recursive_link->strength < 255 - boost) {
                                recursive_link->strength += boost;
                                recursive_link->last_act = now;
                            }
                        }
                        
                        current_tensor = next_tensor;
                        recursion_depth++;
                    } else {
                        break; // Прерываем рекурсию при обнаружении цикла
                    }
                } else {
                    break; // Нет forward-ссылки
                }
            }
            
            // Усиливаем только сильные связи
            uint16_t tensor_idx = t - tnsrs;
            if (tensor_idx < MAX_TENSORS) {
                uint8_t links_to_strengthen = tensor_links[tensor_idx].link_count;
                if (links_to_strengthen > 10) links_to_strengthen = 10;
                
                for (uint8_t li_idx = 0; li_idx < links_to_strengthen; li_idx++) {
                    uint16_t link_idx = tensor_links[tensor_idx].link_indices[li_idx];
                    if (link_idx >= lnk_count) continue;
                    
                    BitLink* link = &lnks[link_idx];
                    if (link->strength > 100) {
                        uint8_t boost = (uint8_t)(goodness * 15);
                        if (link->strength < 255 - boost) {
                            link->strength += boost;
                            link->last_act = now;
                        }
                    }
                }
            }
            strengthened++;
            
        } else if (goodness < 0.3f && mode >= 1) {
            // Ослабление
            t->act = t->act >> 1;
            t->stab = (t->stab * 3) >> 2;
            
            // Ослабляем только слабые связи
            uint16_t tensor_idx = t - tnsrs;
            if (tensor_idx < MAX_TENSORS) {
                uint8_t links_to_weaken = tensor_links[tensor_idx].link_count;
                if (links_to_weaken > 5) links_to_weaken = 5;
                
                for (uint8_t li_idx = 0; li_idx < links_to_weaken; li_idx++) {
                    uint16_t link_idx = tensor_links[tensor_idx].link_indices[li_idx];
                    if (link_idx >= lnk_count) continue;
                    
                    BitLink* link = &lnks[link_idx];
                    if (link->strength < 50) {
                        link->strength = link->strength >> 1;
                        if (link->strength < 5) link->strength = 5;
                    }
                }
            }
            weakened++;
        }
    }
    
    uint8_t superclusters_created = 0;
    
    // --- ОПТИМИЗИРОВАННОЕ СОЗДАНИЕ СУПЕРКЛАСТЕРОВ ---
    if (mode >= 1 && cluster_count > 3 && cluster_count < MAX_CLUSTERS - 5) {
        uint16_t max_cluster_pairs = (mode == 1) ? 5 : 10;
        uint16_t pairs_checked = 0;
        
        for (uint16_t ci = 0; ci < cluster_count - 1 && pairs_checked < max_cluster_pairs; ci++) {
            for (uint16_t cj = ci + 1; cj < cluster_count && pairs_checked < max_cluster_pairs; cj++, pairs_checked++) {
                MemoryCluster* c1 = &clusters[ci];
                MemoryCluster* c2 = &clusters[cj];
                
                // Быстрая проверка: если кластеры уже связаны через концепцию
                uint8_t already_linked = 0;
                for (uint8_t cc = 0; cc < concept_count && !already_linked; cc++) {
                    if (concepts[cc].concept_tensor) {
                        if (concepts[cc].concept_tensor->cluster_id == c1->cluster_id ||
                            concepts[cc].concept_tensor->cluster_id == c2->cluster_id) {
                            // Проверим связи концепции
                            for (uint8_t m = 0; m < concepts[cc].member_count; m++) {
                                uint16_t member_idx = concepts[cc].member_indices[m];
                                if (member_idx < tnsr_count) {
                                    BitTensor* member = &tnsrs[member_idx];
                                    if (member->cluster_id == c1->cluster_id || 
                                        member->cluster_id == c2->cluster_id) {
                                        already_linked = 1;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                
                if (already_linked) continue;
                
                // Быстрая проверка связей между кластерами
                uint8_t inter_cluster_links = 0;
                uint8_t max_check = (c1->size < 3) ? c1->size : 3;
                uint8_t max_check2 = (c2->size < 3) ? c2->size : 3;
                
                for (uint8_t ti = 0; ti < max_check && inter_cluster_links < 2; ti++) {
                    for (uint8_t tj = 0; tj < max_check2 && inter_cluster_links < 2; tj++) {
                        BitTensor* t1 = &tnsrs[c1->tensor_indices[ti]];
                        BitTensor* t2 = &tnsrs[c2->tensor_indices[tj]];
                        
                        // Быстрая проверка существования связи
                        uint16_t t1_idx = t1 - tnsrs;
                        if (t1_idx < MAX_TENSORS) {
                            for (uint8_t li_idx = 0; li_idx < tensor_links[t1_idx].link_count; li_idx++) {
                                uint16_t link_idx = tensor_links[t1_idx].link_indices[li_idx];
                                if (link_idx < lnk_count) {
                                    BitLink* link = &lnks[link_idx];
                                    if ((link->src == t1 && link->tgt == t2) ||
                                        (link->src == t2 && link->tgt == t1)) {
                                        if (link->strength > 120) {
                                            inter_cluster_links++;
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                
                if (inter_cluster_links >= 2) {
                    BitTensor* super_tensor = create_tnsr(4, 4);
                    if (super_tensor) {
                        super_tensor->is_concept = 1;
                        super_tensor->act = 150;
                        super_tensor->res = 180;
                        super_tensor->stab = 170;
                        super_tensor->cluster_id = 0; // Пока без кластера
                        
                        BitTensor* center1 = find_center_tensor_in_cluster(c1);
                        BitTensor* center2 = find_center_tensor_in_cluster(c2);
                        
                        if (center1) {
                            BitLink* link1 = create_link(super_tensor, center1);
                            if (link1) {
                                link1->semantic_type = 3;
                                link1->strength = 180;
                                link1->last_act = now;
                            }
                        }
                        if (center2) {
                            BitLink* link2 = create_link(super_tensor, center2);
                            if (link2) {
                                link2->semantic_type = 3;
                                link2->strength = 180;
                                link2->last_act = now;
                            }
                        }
                        
                        // Создаем новый кластер для супертензора
                        if (cluster_count < MAX_CLUSTERS) {
                            MemoryCluster* super_cluster = &clusters[cluster_count];
                            super_cluster->cluster_id = next_cluster_id++;
                            if (next_cluster_id == 0) next_cluster_id = 1;
                            
                            super_cluster->size = 1;
                            super_cluster->tensor_indices[0] = super_tensor - tnsrs;
                            super_cluster->activation_level = 160;
                            super_cluster->stability = 150;
                            super_cluster->last_access = now;
                            super_cluster->creation_time = now;
                            super_cluster->link_count = 0;
                            super_cluster->category = 3; // Концептуальный
                            
                            // Инициализируем центроид
                            memset(super_cluster->centroid, 0, 256);
                            if (super_tensor->data) {
                                uint32_t bits = super_tensor->rows * super_tensor->cols;
                                uint32_t bytes = (bits + 7) / 8;
                                for (uint32_t b = 0; b < bytes && b < 256; b++) {
                                    super_cluster->centroid[b] = super_tensor->data[b];
                                }
                            }
                            
                            super_tensor->cluster_id = super_cluster->cluster_id;
                            cluster_count++;
                        }
                        
                        superclusters_created++;
                        
                        if (superclusters_created >= ((mode == 1) ? 2 : 3)) {
                            break;
                        }
                    }
                }
            }
            if (superclusters_created >= ((mode == 1) ? 2 : 3)) {
                break;
            }
        }
    }
    
    // --- ОПТИМИЗИРОВАННАЯ ОЧИСТКА ПРОТИВОРЕЧИВЫХ СВЯЗЕЙ ---
    if (mode == 2) {
        uint16_t pruned_links = 0;
        uint16_t max_to_check = (lnk_count > 200) ? 200 : lnk_count;
        
        // Создаем временный индекс для быстрого поиска обратных связей
        uint8_t* reverse_strength_cache = (uint8_t*)calloc(max_to_check, 1);
        if (reverse_strength_cache) {
            // Заполняем кэш обратных связей
            for (uint16_t li = 0; li < max_to_check; li++) {
                BitLink* link = &lnks[li];
                if (!link->src || !link->tgt) continue;
                
                // Ищем обратную связь
                uint16_t src_idx = link->src - tnsrs;
                if (src_idx < MAX_TENSORS) {
                    for (uint8_t li_idx = 0; li_idx < tensor_links[src_idx].link_count; li_idx++) {
                        uint16_t rev_link_idx = tensor_links[src_idx].link_indices[li_idx];
                        if (rev_link_idx < lnk_count && rev_link_idx != li) {
                            BitLink* rev_link = &lnks[rev_link_idx];
                            if (rev_link->src == link->tgt && rev_link->tgt == link->src) {
                                reverse_strength_cache[li] = rev_link->strength;
                                break;
                            }
                        }
                    }
                }
            }
            
            // Удаляем противоречивые связи
            for (int16_t li = max_to_check - 1; li >= 0 && pruned_links < 50; li--) {
                BitLink* link = &lnks[li];
                
                if (link->src && link->tgt) {
                    uint8_t reverse_strength = reverse_strength_cache[li];
                    
                    if (link->strength > 150 && reverse_strength < 30 && 
                        link->use_count < 3) {
                        
                        if (link->src->conn > 0) link->src->conn--;
                        if (link->tgt->conn > 0) link->tgt->conn--;
                        
                        // Обновляем кэш перед удалением
                        for (uint16_t lj = 0; lj < max_to_check; lj++) {
                            if (reverse_strength_cache[lj] == li) {
                                reverse_strength_cache[lj] = 0;
                            }
                        }
                        
                        lnks[li] = lnks[lnk_count - 1];
                        lnk_count--;
                        
                        // Обновляем индекс для переставленной связи
                        if (li < lnk_count && li < max_to_check) {
                            reverse_strength_cache[li] = reverse_strength_cache[lnk_count];
                        }
                        
                        pruned_links++;
                    }
                }
            }
            
            free(reverse_strength_cache);
        }
        
        printf("[F-F] Удалено противоречивых связей: %u\n", pruned_links);
    }
    
    // --- ОБНОВЛЕНИЕ КОНТЕКСТА И ВАЖНОСТИ ЭПИЗОДОВ ---
    if (working_mem_count > 2) {
        global_context_hash = calculate_context_hash();
        
        // Обновляем только топ эпизоды
        uint16_t max_episodes_to_update = (episode_count > 20) ? 20 : episode_count;
        for (uint16_t ei = 0; ei < max_episodes_to_update; ei++) {
            uint32_t episode_context = *((uint32_t*)episodes[ei].context_hash);
            uint32_t context_diff = (global_context_hash > episode_context) ?
                                   (global_context_hash - episode_context) :
                                   (episode_context - global_context_hash);
            
            if (context_diff < 0x1000) {
                episodes[ei].importance = (episodes[ei].importance * 9 + 110) >> 3;
                if (episodes[ei].importance > 200) episodes[ei].importance = 200;
                
                // Обновляем время последнего доступа
                episodes[ei].last_recall = now;
            }
        }
    }
    
    // --- ОБНОВЛЕНИЕ ЭФФЕКТИВНОСТИ ---
    uint16_t max_to_update = (mode == 0) ? 30 : (mode == 1) ? 60 : tnsr_count;
    if (max_to_update > tnsr_count) max_to_update = tnsr_count;
    
    for (uint16_t i = 0; i < max_to_update; i++) {
        if (tnsrs[i].act > 50) {
            tnsrs[i].efficiency = calculate_efficiency(&tnsrs[i]);
        }
    }
    
    printf("[F-F CONS] Done. Processed: %u tensors\n", scored_tensors);
    printf("[F-F CONS] Strengthened: %u, Weakened: %u, Superclusters created: %u\n", 
        strengthened, weakened, superclusters_created);
    printf("[F-F CONS] Current counters: tensors=%u, links=%u, clusters=%u\n",
       tnsr_count, lnk_count, cluster_count);
}
// ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====

void build_link_index(void) {
    memset(tensor_links, 0, sizeof(tensor_links));
    for (uint16_t i = 0; i < lnk_count; i++) {
        BitLink* link = &lnks[i];
        uint16_t src_idx = link->src - tnsrs;
        uint16_t tgt_idx = link->tgt - tnsrs;

        if (src_idx < MAX_TENSORS && tensor_links[src_idx].link_count < 32) {
            tensor_links[src_idx].link_indices[tensor_links[src_idx].link_count++] = i;

            if (link->src->cluster_id != 0 && link->src->cluster_id == link->tgt->cluster_id) {
                if (tensor_links[src_idx].cluster_link_count < 16) {
                    tensor_links[src_idx].cluster_links[tensor_links[src_idx].cluster_link_count++] = i;
                }
            }
        }
        if (tgt_idx < MAX_TENSORS && tensor_links[tgt_idx].link_count < 32) {
            tensor_links[tgt_idx].link_indices[tensor_links[tgt_idx].link_count++] = i;
        }
    }
}

uint8_t check_context_fit(BitTensor* a, BitTensor* b, BitLink* link) {
    if (!a || !b || !link) return 0;
    uint8_t base = link->res;

    for (uint16_t i = 0; i < lnk_count; i++) {
        if (lnks[i].src == b && lnks[i].tgt == a) {
            base = (base + lnks[i].strength + lnks[i].res) / 3;
            break;
        }
    }

    if (a->cluster_id != 0 && a->cluster_id == b->cluster_id) {
        base += 40;
    }

    uint32_t ts_diff = (a->lu > b->lu) ? a->lu - b->lu : b->lu - a->lu;
    if (ts_diff < 10) base += 30;
    else if (ts_diff < 60) base += 10;

    return (base > 255) ? 255 : base;
}

void fast_contextual_activation(BitTensor* context) {
    uint16_t ctx_idx = context - tnsrs;
    if (ctx_idx >= MAX_TENSORS) return;

    for (uint8_t li = 0; li < tensor_links[ctx_idx].link_count; li++) {
        uint16_t link_idx = tensor_links[ctx_idx].link_indices[li];
        BitLink* link = &lnks[link_idx];

        BitTensor* other = (link->src == context) ? link->tgt : link->src;

        if (link->strength > 100 &&
            link->res > 150 &&
            !context->dropout &&
            !other->dropout &&
            check_context_fit(context, other, link) > 140) {

            uint8_t activation = (context->act * link->strength * link->res) >> 16;
            other->act = (other->act > ACT_MAX - activation) ? ACT_MAX : other->act + activation;
            other->lu = (uint32_t)time(NULL);
            add_to_working_memory(other);
        }
    }

    if (context->cluster_id != 0) {
        for (uint16_t ci = 0; ci < cluster_count; ci++) {
            if (clusters[ci].cluster_id == context->cluster_id) {
                uint8_t to_activate = (clusters[ci].size > 5) ? 3 : clusters[ci].size / 2;
                for (uint8_t k = 0; k < to_activate; k++) {
                    uint16_t rand_idx = rand() % clusters[ci].size;
                    uint16_t tensor_idx = clusters[ci].tensor_indices[rand_idx];
                    if (tensor_idx < tnsr_count) {
                        BitTensor* t = &tnsrs[tensor_idx];
                        if (t != context && !t->dropout) {
                            t->act = (t->act * 7 + 120) >> 3;
                            t->lu = (uint32_t)time(NULL);
                        }
                    }
                }
                break;
            }
        }
    }
}

void update_thought_stream(void) {
    uint32_t now = (uint32_t)time(NULL);
    static uint8_t cycle_counter = 0;
    static uint32_t last_thought_cleanup = 0;
    static uint32_t last_energy_update = 0;
    
    cycle_counter = (cycle_counter + 1) & 0x7F;
    
    if (working_mem_count == 0 && tnsr_count < 5) {
        if (now - sstate.self_org_timer > 30) {
            sstate.self_org_timer = now;
            if (rand() % 10 == 0) {
                uint16_t idx = rand() % tnsr_count;
                if (idx < tnsr_count && !tnsrs[idx].dropout) {
                    tnsrs[idx].act = 50;
                    add_to_working_memory(&tnsrs[idx]);
                }
            }
        }
        return;
    }
    
    uint8_t cycle_mask = cycle_counter & 0x0F;
    
    for (uint8_t i = 0; i < working_mem_count && i < 8; i++) {
        if (working_mem[i].tensor && now - working_mem[i].timestamp < 2) {
            working_mem[i].tensor->act = (working_mem[i].tensor->act * 7 + 180) >> 3;
        }
    }
    
    if ((cycle_mask & 0x07) == 0) {
        build_link_index();
    }
    
    if (goals.self_organization_enabled) {
        if ((cycle_mask % 3) == 0) {
            uint16_t limit = (lnk_count > 80) ? 80 : lnk_count;
            for (uint16_t i = 0; i < limit; i++) {
                BitLink* link = &lnks[i];
                if (!link->src || !link->src->data || !link->tgt || !link->tgt->data) 
                    continue;
                    
                uint8_t pre_act = link->src->act;
                uint8_t post_act = link->tgt->act;
                
                if (pre_act > 40 && post_act > 40) {
                    uint8_t hebb = (pre_act * post_act) >> 9;
                    
                    if (pre_act > post_act * 2) {
                        if (link->weight > 20) link->weight -= 2;
                    } else if (post_act > pre_act * 2) {
                        link->weight = (link->weight * 15 + 240) >> 4;
                    }
                    
                    if (hebb > 15) {
                        link->strength = (link->strength * 7 + 200) >> 3;
                        if (link->strength > 250) link->strength = 250;
                        link->res = (link->res * 7 + 220) >> 3;
                    }
                    
                    if (now - link->last_act > 10 && hebb < 5) {
                        if (link->strength > LINK_MIN_STRENGTH + 10) {
                            link->strength = (link->strength * 13 + LINK_MIN_STRENGTH * 3) >> 4;
                        }
                    }
                    
                    link->last_act = now;
                }
            }
        }
        
        if ((cycle_mask % 7) == 0 && cluster_count > 2) {
            for (uint8_t ci = 0; ci < cluster_count && ci < 16; ci++) {
                MemoryCluster* c = &clusters[ci];
                if (c->size < 2 || c->activation_level < 60) continue;
                
                uint16_t total_act = 0;
                uint8_t active_cnt = 0;
                BitTensor* strongest = NULL;
                uint8_t max_act = 0;
                
                for (uint8_t ti = 0; ti < c->size && ti < 8; ti++) {
                    uint16_t idx = c->tensor_indices[ti];
                    if (idx < tnsr_count) {
                        BitTensor* t = &tnsrs[idx];
                        if (t->act > 30) {
                            total_act += t->act;
                            active_cnt++;
                            if (t->act > max_act) {
                                max_act = t->act;
                                strongest = t;
                            }
                        }
                    }
                }
                
                if (active_cnt > 2 && total_act > 400) {
                    uint8_t inhibition = (total_act / active_cnt > 150) ? 30 : 15;
                    
                    for (uint8_t ti = 0; ti < c->size && ti < 8; ti++) {
                        uint16_t idx = c->tensor_indices[ti];
                        if (idx < tnsr_count) {
                            BitTensor* t = &tnsrs[idx];
                            if (t != strongest && t->act > inhibition) {
                                t->act -= inhibition;
                            }
                        }
                    }
                    
                    if (strongest && strongest->act < ACT_MAX - 10) {
                        strongest->act += 5;
                    }
                }
            }
        }
        
        if (now - sstate.self_org_timer >= SELF_ORG_INTERVAL) {
            self_organize_memory_clusters();
            
            if (cluster_count > 4) {
                for (uint8_t ci = 0; ci < cluster_count - 1 && ci < 8; ci++) {
                    MemoryCluster* c1 = &clusters[ci];
                    MemoryCluster* c2 = &clusters[ci + 1];
                    
                    BitTensor* center1 = find_center_tensor_in_cluster(c1);
                    BitTensor* center2 = find_center_tensor_in_cluster(c2);
                    
                    if (center1 && center2 && center1 != center2) {
                        BitLink* link = find_or_create_link(center1, center2);
                        if (link) {
                            link->strength = 150;
                            link->res = 180;
                            link->semantic_type = 2;
                            link->weight = 128;
                        }
                    }
                }
            }
            
            semantic_memory_binding();
            sstate.self_org_timer = now;
        }
        
        if (now - sstate.consolidation_timer >= CONSOLIDATION_INTERVAL) {
            if (working_mem_count > 3) {
                for (uint8_t i = 0; i < working_mem_count && i < 6; i++) {
                    if (working_mem[i].tensor && working_mem[i].tensor->act > 60) {
                        BitTensor* t = working_mem[i].tensor;
                        for (uint8_t li = 0; li < tensor_links[t - tnsrs].link_count && li < 6; li++) {
                            uint16_t link_idx = tensor_links[t - tnsrs].link_indices[li];
                            BitLink* link = &lnks[link_idx];
                            BitTensor* other = (link->src == t) ? link->tgt : link->src;
                            
                            if (other && other != t && !other->dropout && (rand() & 0x3F) < 20) {
                                other->act = (other->act * 7 + 100) >> 3;
                                add_to_working_memory(other);
                            }
                        }
                    }
                }
            }
            
           forward_forward_consolidation(1);
            
            for (uint16_t i = 0; i < lnk_count && i < 40; i++) {
                if (lnks[i].use_count > 8 && lnks[i].success_count * 2 > lnks[i].use_count) {
                    if (lnks[i].strength < 200) {
                        lnks[i].strength = (lnks[i].strength * 15 + 210) >> 4;
                    }
                }
            }
            
            sstate.consolidation_timer = now;
        }
        
        if ((cycle_counter % 60) == 0) {
            uint32_t total_ex = 0;
            uint16_t excited = 0;
            
            uint16_t limit = (tnsr_count > 150) ? 150 : tnsr_count;
            for (uint16_t i = 0; i < limit; i++) {
                if (tnsrs[i].act > 30 && !tnsrs[i].dropout) {
                    total_ex += tnsrs[i].act;
                    excited++;
                }
            }
            
            if (excited > 0) {
                uint8_t avg_ex = total_ex / excited;
                
                if (avg_ex > 120 && excited > 15) {
                    uint8_t inhibition = (avg_ex - 100) >> 2;
                    
                    for (uint16_t i = 0; i < limit; i++) {
                        if (tnsrs[i].act > inhibition + 10 && !tnsrs[i].is_concept) {
                            tnsrs[i].act -= inhibition;
                        }
                    }
                    
                    for (uint8_t ci = 0; ci < concept_count; ci++) {
                        if (concepts[ci].concept_tensor && concepts[ci].concept_tensor->act > 20) {
                            concepts[ci].concept_tensor->act -= inhibition >> 1;
                        }
                    }
                }
                
                if (avg_ex < 40 && excited < 8) {
                    for (uint8_t i = 0; i < 4; i++) {
                        uint16_t rand_idx = rand() % tnsr_count;
                        if (rand_idx < tnsr_count && !tnsrs[rand_idx].dropout) {
                            tnsrs[rand_idx].act = (tnsrs[rand_idx].act * 3 + 80) >> 2;
                        }
                    }
                }
            }
        }
    }
    
    if ((cycle_mask & 0x03) == 0 && working_mem_count > 2) {
        form_thought_chain();
    }
    
    for (uint8_t ts = 0; ts < active_thought_streams && ts < MAX_THOUGHT_STREAMS; ts++) {
        evolve_thought_stream(&thought_streams[ts]);
    }
    
    if (now - last_thought_cleanup > 30) {
        prune_old_thought_streams();
        last_thought_cleanup = now;
    }
    
    if ((cycle_mask & 0x03) == 0) {
        update_bit_net_with_goals();
    }
    
    if ((cycle_mask & 0x07) == 0 && working_mem_count > 2) {
        static uint32_t last_context_hash = 0;
        uint32_t current_context = calculate_context_hash();
        uint32_t context_diff = (current_context > last_context_hash) ? 
                               (current_context - last_context_hash) : 
                               (last_context_hash - current_context);
        
        if (context_diff > 0x2000) {
            activate_relevant_episodes();
            last_context_hash = current_context;
        }
    }
    
    if ((cycle_mask & 0x01) == 0 && working_mem_count > 1) {
        for (uint8_t i = 0; i < working_mem_count - 1 && i < 3; i++) {
            BitTensor* t1 = working_mem[i].tensor;
            BitTensor* t2 = working_mem[i + 1].tensor;
            
            if (t1 && t2 && t1 != t2 && !t1->dropout && !t2->dropout) {
                uint8_t time_diff = (uint8_t)(working_mem[i + 1].timestamp - working_mem[i].timestamp);
                uint8_t temporal_sim = 255 - time_diff * 10;
                
                if (temporal_sim > 100) {
                    BitLink* link = find_or_create_link(t1, t2);
                    if (link) {
                        link->strength = (link->strength * 3 + 120) >> 2;
                        link->last_act = now;
                        link->semantic_type = 0;
                    }
                }
            }
        }
    }
    
    if ((cycle_counter & 0x3F) == 0 && (rand() & 0x7F) < 64) {
        uint16_t attempts = 0;
        BitTensor* weak_tensor = NULL;
        
        while (!weak_tensor && attempts < 30) {
            uint16_t idx = rand() % tnsr_count;
            if (idx < tnsr_count && tnsrs[idx].act < 40 && !tnsrs[idx].dropout) {
                weak_tensor = &tnsrs[idx];
            }
            attempts++;
        }
        
        if (weak_tensor) {
            weak_tensor->act = 80;
            add_to_working_memory(weak_tensor);
            
            for (uint8_t i = 0; i < 2; i++) {
                uint16_t target = rand() % tnsr_count;
                if (target < tnsr_count && &tnsrs[target] != weak_tensor) {
                    BitLink* link = create_link(weak_tensor, &tnsrs[target]);
                    if (link) {
                        link->strength = 60;
                        link->semantic_type = 0;
                    }
                }
            }
        }
    }
    
    if ((cycle_mask & 0x0F) == 0) {
        global_context_hash = calculate_context_hash();
    }
    
    if ((cycle_counter & 0x3F) == 0) {
        uint16_t limit = (tnsr_count > 100) ? 100 : tnsr_count;
        for (uint16_t i = 0; i < limit; i++) {
            if (tnsrs[i].act > 5 && now - tnsrs[i].lu > 3600) {
                tnsrs[i].act = (tnsrs[i].act * 15) >> 4;
                if (tnsrs[i].act < 5) tnsrs[i].act = 5;
            }
        }
        
        for (uint8_t ci = 0; ci < cluster_count && ci < 16; ci++) {
            if (now - clusters[ci].last_access > 1800) {
                clusters[ci].activation_level = (clusters[ci].activation_level * 7) >> 3;
                clusters[ci].stability = (clusters[ci].stability * 31) >> 5;
            }
        }
    }
    
    if (now - last_energy_update > 5) {
        uint16_t system_energy = 0;
        uint8_t limit = (working_mem_count > 8) ? 8 : working_mem_count;
        for (uint8_t i = 0; i < limit; i++) {
            if (working_mem[i].tensor) {
                system_energy += working_mem[i].tensor->act;
            }
        }
        
        if (system_energy > 1600) {
            goals.energy_saving_mode = 1;
        } else if (system_energy < 400) {
            goals.energy_saving_mode = 0;
        }
        last_energy_update = now;
    }
    
    if ((cycle_mask & 0x07) == 0) {
        uint32_t resonance_sum = 0;
        uint16_t resonance_cnt = 0;
        
        uint16_t limit = (lnk_count > 80) ? 80 : lnk_count;
        for (uint16_t i = 0; i < limit; i++) {
            if (lnks[i].strength > 50 && now - lnks[i].last_act < 30) {
                resonance_sum += (uint32_t)lnks[i].res * lnks[i].strength;
                resonance_cnt++;
            }
        }
        
        if (resonance_cnt > 0) {
            uint8_t avg_res = (uint8_t)(resonance_sum / resonance_cnt / 255);
            sys_res = (sys_res * 230 + avg_res * 25) >> 8;
        }
        
        sstate.res_hist[sstate.hist_idx] = sys_res;
        sstate.hist_idx = (sstate.hist_idx + 1) % HISTORY_SIZE;
    }
}

void aggressive_memory_cleanup(void) {
    uint32_t now = (uint32_t)time(NULL);
    uint16_t i;

    for (i = 0; i < lnk_count; i++) {
        BitLink* link = &lnks[i];
        uint32_t inactive_time = now - link->last_act;

        uint8_t should_remove = 0;

        if (link->semantic_type == 0) {
            should_remove = (inactive_time > 3600) ||
                           (link->strength < LINK_MIN_STRENGTH && link->use_count < 3) ||
                           (link->use_count > 0 && link->success_count * 10 < link->use_count);
        } else if (link->semantic_type == 1) {
            should_remove = (inactive_time > 7200) ||
                           (link->strength < LINK_MIN_STRENGTH / 2 && link->use_count < 5);
        } else {
            should_remove = (inactive_time > 10800) ||
                           (link->strength < 10 && link->use_count < 10);
        }

        if (should_remove) {
            if (link->src && link->src->conn > 0) link->src->conn--;
            if (link->tgt && link->tgt->conn > 0) link->tgt->conn--;

            lnks[i] = lnks[lnk_count - 1];
            lnk_count--;
            i--;
        }
    }

    for (i = 0; i < working_mem_count; i++) {
        if (working_mem[i].tensor &&
            (now - working_mem[i].timestamp > 300) &&
            working_mem[i].priority < 40) {
            working_mem[i] = working_mem[working_mem_count - 1];
            working_mem_count--;
            i--;
        }
    }

    for (int32_t t_idx = tnsr_count - 1; t_idx >= 0; t_idx--) {
        BitTensor* t = &tnsrs[t_idx];

        if (t->is_concept) continue;

        uint8_t in_working_mem = 0;
        for (uint8_t w = 0; w < working_mem_count; w++) {
            if (working_mem[w].tensor == t) {
                in_working_mem = 1;
                break;
            }
        }

        if (t->conn < 2 &&
            t->act < 10 &&
            !in_working_mem &&
            t->efficiency < 30 &&
            (now - t->lu > 600)) {

            if (t->cluster_id != 0) {
                for (uint16_t ci = 0; ci < cluster_count; ci++) {
                    if (clusters[ci].cluster_id == t->cluster_id) {
                        for (uint16_t k = 0; k < clusters[ci].size; k++) {
                            if (clusters[ci].tensor_indices[k] == (uint16_t)t_idx) {
                                for (uint16_t m = k; m < clusters[ci].size - 1; m++) {
                                    clusters[ci].tensor_indices[m] = clusters[ci].tensor_indices[m + 1];
                                }
                                clusters[ci].size--;
                                break;
                            }
                        }
                        break;
                    }
                }
            }

            for (int32_t l = lnk_count - 1; l >= 0; l--) {
                if (lnks[l].src == t || lnks[l].tgt == t) {
                    lnks[l] = lnks[lnk_count - 1];
                    lnk_count--;
                }
            }

            for (uint8_t w = 0; w < working_mem_count; w++) {
                if (working_mem[w].tensor == t) {
                    working_mem[w] = working_mem[working_mem_count - 1];
                    working_mem_count--;
                    break;
                }
            }

            if (t->data) {
                free(t->data);
                t->data = NULL;
            }

            if (t_idx < tnsr_count - 1) {
                tnsrs[t_idx] = tnsrs[tnsr_count - 1];

                BitTensor* moved_tensor = &tnsrs[t_idx];
                BitTensor* old_place = &tnsrs[tnsr_count - 1];

                if (old_place->cluster_id != 0) {
                    for (uint16_t ci = 0; ci < cluster_count; ci++) {
                        if (clusters[ci].cluster_id == old_place->cluster_id) {
                            for (uint16_t k = 0; k < clusters[ci].size; k++) {
                                if (clusters[ci].tensor_indices[k] == tnsr_count - 1) {
                                    clusters[ci].tensor_indices[k] = t_idx;
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }

                for (uint8_t ci = 0; ci < concept_count; ci++) {
                    for (uint8_t k = 0; k < concepts[ci].member_count; k++) {
                        if (concepts[ci].member_indices[k] == tnsr_count - 1) {
                            concepts[ci].member_indices[k] = t_idx;
                            break;
                        }
                    }
                }

                for (uint16_t ei = 0; ei < episode_count; ei++) {
                    for (uint8_t k = 0; k < episodes[ei].length; k++) {
                        if (episodes[ei].sequence[k] == tnsr_count - 1) {
                            episodes[ei].sequence[k] = t_idx;
                        }
                    }
                }

                for (uint16_t l = 0; l < lnk_count; l++) {
                    if (lnks[l].src == old_place) lnks[l].src = moved_tensor;
                    if (lnks[l].tgt == old_place) lnks[l].tgt = moved_tensor;
                }

                for (uint8_t w = 0; w < working_mem_count; w++) {
                    if (working_mem[w].tensor == old_place) working_mem[w].tensor = moved_tensor;
                }
            }

            tnsr_count--;
        }
    }

    for (i = 0; i < tnsr_count; i++) {
        BitTensor* t = &tnsrs[i];
        if (t->act < 30 && (now - t->lu > 1800) &&
            (t->rows * t->cols > 256) && !t->mem_red && !t->is_concept) {
            reduce_tnsr_mem(t);
        }
    }

    if (memo_size > 0) {
        for (uint16_t i = 0; i < memo_size; i++) {
            for (uint16_t j = i + 1; j < memo_size; j++) {
                float si = (float)memo[i].count * memo[i].act / ((now - memo[i].ts) + 1);
                float sj = (float)memo[j].count * memo[j].act / ((now - memo[j].ts) + 1);

                if (memo[i].cluster_id != 0) {
                    for (uint16_t ci = 0; ci < cluster_count; ci++) {
                        if (clusters[ci].cluster_id == memo[i].cluster_id &&
                            clusters[ci].activation_level > 150) {
                            si *= 1.5f;
                            break;
                        }
                    }
                }

                if (sj > si) {
                    BitMemory tmp = memo[i];
                    memo[i] = memo[j];
                    memo[j] = tmp;
                }
            }
        }

        uint16_t target_size = (MAX_MEM_ENTRIES / 2) > 50 ? (MAX_MEM_ENTRIES / 2) : 50;
        if (memo_size > target_size) {
            memo_size = target_size;
        }
    }

    if (episode_count > MAX_EPISODES / 2) {
        for (uint16_t ei = 0; ei < episode_count; ei++) {
            if (episodes[ei].importance < 30 &&
                now - episodes[ei].last_recall > 3600) {

                for (uint16_t ej = ei; ej < episode_count - 1; ej++) {
                    episodes[ej] = episodes[ej + 1];
                }
                episode_count--;
                ei--;
            }
        }
    }
}

// ===== СЕРИАЛИЗАЦИЯ =====

void save_tnsr(BitTensor* t) {
    if (!t || !t->data || memo_size >= MAX_MEM_ENTRIES) return;
    uint16_t total_bits = t->rows * t->cols;
    uint8_t total_bytes = (uint8_t)((total_bits + 7) / 8);
    if (total_bytes > MAX_PATTERN) return;

    if (t->is_concept) {
        for (uint16_t i = 0; i < memo_size; i++) {
            if (memo[i].len == total_bytes && !memcmp(memo[i].data, t->data, total_bytes)) {
                memo[i].count += 2;
                memo[i].res = (memo[i].res * 230 + t->res * 25) >> 8;
                memo[i].act = (memo[i].act * 230 + t->act * 25) >> 8;
                memo[i].ent = (memo[i].ent * 230 + t->ent * 25) >> 8;
                memo[i].ts = (uint32_t)time(NULL);
                memo[i].cluster_id = t->cluster_id;
                return;
            }
        }
    } else {
        for (uint16_t i = 0; i < memo_size; i++) {
            if (memo[i].len == total_bytes && !memcmp(memo[i].data, t->data, total_bytes)) {
                memo[i].count++;
                memo[i].res = (memo[i].res * 230 + t->res * 25) >> 8;
                memo[i].act = (memo[i].act * 230 + t->act * 25) >> 8;
                memo[i].ent = (memo[i].ent * 230 + t->ent * 25) >> 8;
                memo[i].ts = (uint32_t)time(NULL);
                memo[i].cluster_id = t->cluster_id;
                return;
            }
        }
    }

    BitMemory* entry = &memo[memo_size++];
    memcpy(entry->data, t->data, total_bytes);
    entry->len = total_bytes;
    entry->count = t->is_concept ? 2 : 1;
    entry->res = t->res;
    entry->act = t->act;
    entry->ent = t->ent;
    entry->ts = (uint32_t)time(NULL);
    entry->cluster_id = t->cluster_id;
}

int save_state_to_file(const char* filename) {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open(save)"); return -1; }

#define WRITE(ptr, size, count) do { \
    if (write(fd, (ptr), (size_t)(size)*(count)) != (ssize_t)((size_t)(size)*(count))) { \
        perror("write"); close(fd); return -1; } \
} while (0)

    WRITE(&tnsr_count, sizeof(uint16_t), 1);
    WRITE(&lnk_count, sizeof(uint16_t), 1);
    WRITE(&memo_size, sizeof(uint16_t), 1);
    WRITE(&working_mem_count, sizeof(uint8_t), 1);
    WRITE(&goals, sizeof(SystemGoals), 1);
    WRITE(&sys_res, sizeof(uint8_t), 1);
    WRITE(&interaction_count, sizeof(uint32_t), 1);
    WRITE(&last_mem_check_ts, sizeof(uint32_t), 1);
    WRITE(&sstate, sizeof(BitSystemState), 1);

    WRITE(&cluster_count, sizeof(uint16_t), 1);
    WRITE(&episode_count, sizeof(uint16_t), 1);
    WRITE(&concept_count, sizeof(uint8_t), 1);
    WRITE(&global_context_hash, sizeof(uint32_t), 1);
    WRITE(&next_cluster_id, sizeof(uint8_t), 1);

    for (uint16_t i = 0; i < tnsr_count; i++) {
        BitTensor* t = &tnsrs[i];
        WRITE(&t->rows, sizeof(uint16_t), 1);
        WRITE(&t->cols, sizeof(uint16_t), 1);
        WRITE(&t->res, sizeof(uint8_t), 1);
        WRITE(&t->act, sizeof(uint8_t), 1);
        WRITE(&t->ent, sizeof(uint8_t), 1);
        WRITE(&t->stab, sizeof(uint8_t), 1);
        WRITE(&t->conn, sizeof(uint16_t), 1);
        WRITE(&t->lu, sizeof(uint32_t), 1);
        WRITE(&t->mem_red, sizeof(uint8_t), 1);
        WRITE(&t->efficiency, sizeof(uint8_t), 1);
        WRITE(&t->compute_cost, sizeof(uint32_t), 1);
        WRITE(&t->goal_active, sizeof(uint8_t), 1);
        WRITE(&t->dropout, sizeof(uint8_t), 1);
        WRITE(&t->cluster_id, sizeof(uint8_t), 1);
        WRITE(&t->is_concept, sizeof(uint8_t), 1);

        uint32_t data_bytes = (t->rows * t->cols + 7) / 8;
        WRITE(&data_bytes, sizeof(uint32_t), 1);
        if (data_bytes > 0 && t->data) {
            WRITE(t->data, 1, data_bytes);
        }
    }

    for (uint16_t i = 0; i < lnk_count; i++) {
        uint16_t src_idx = tensor_to_index(lnks[i].src);
        uint16_t tgt_idx = tensor_to_index(lnks[i].tgt);
        WRITE(&src_idx, sizeof(uint16_t), 1);
        WRITE(&tgt_idx, sizeof(uint16_t), 1);
        WRITE(&lnks[i].strength, sizeof(uint8_t), 1);
        WRITE(&lnks[i].res, sizeof(uint8_t), 1);
        WRITE(&lnks[i].weight, sizeof(uint16_t), 1);
        WRITE(&lnks[i].ts, sizeof(uint32_t), 1);
        WRITE(&lnks[i].last_act, sizeof(uint32_t), 1);
        WRITE(&lnks[i].use_count, sizeof(uint16_t), 1);
        WRITE(&lnks[i].success_count, sizeof(uint16_t), 1);
        WRITE(&lnks[i].semantic_type, sizeof(uint8_t), 1);
    }

    WRITE(clusters, sizeof(MemoryCluster), cluster_count);
    WRITE(episodes, sizeof(EpisodeMemory), episode_count);

    for (uint8_t i = 0; i < concept_count; i++) {
        uint16_t concept_idx = tensor_to_index(concepts[i].concept_tensor);
        WRITE(&concept_idx, sizeof(uint16_t), 1);
        WRITE(&concepts[i].member_count, sizeof(uint8_t), 1);
        WRITE(concepts[i].member_indices, sizeof(uint16_t), concepts[i].member_count);
        WRITE(&concepts[i].abstraction_level, sizeof(uint8_t), 1);
        WRITE(&concepts[i].coherence, sizeof(uint8_t), 1);
        WRITE(&concepts[i].last_used, sizeof(uint32_t), 1);
    }

    WRITE(memo, sizeof(BitMemory), memo_size);

    for (uint8_t i = 0; i < working_mem_count; i++) {
        uint16_t t_idx = tensor_to_index(working_mem[i].tensor);
        WRITE(&t_idx, sizeof(uint16_t), 1);
        WRITE(&working_mem[i].timestamp, sizeof(uint32_t), 1);
        WRITE(&working_mem[i].priority, sizeof(uint8_t), 1);
        WRITE(&working_mem[i].access_count, sizeof(uint8_t), 1);
        WRITE(&working_mem[i].episode_marker, sizeof(uint8_t), 1);
    }

    close(fd);
    return 0;
}

int load_state_from_file(const char* filename) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        if (errno == ENOENT) return 0;
        perror("open(load)"); return -1;
    }

    for (uint16_t i = 0; i < tnsr_count; i++) free(tnsrs[i].data);
    memset(tnsrs, 0, sizeof(tnsrs));
    memset(lnks, 0, sizeof(lnks));
    memset(memo, 0, sizeof(memo));
    memset(working_mem, 0, sizeof(working_mem));
    memset(clusters, 0, sizeof(clusters));
    memset(episodes, 0, sizeof(episodes));
    memset(concepts, 0, sizeof(concepts));

    tnsr_count = lnk_count = memo_size = working_mem_count = 0;
    cluster_count = episode_count = concept_count = 0;
    sys_res = RES_HALF;
    global_context_hash = 0;
    next_cluster_id = 1;

#define READ(ptr, size, count) do { \
    if (read(fd, (ptr), (size_t)(size)*(count)) != (ssize_t)((size_t)(size)*(count))) { \
        perror("read"); close(fd); return -1; } \
} while (0)
    READ(&tnsr_count, sizeof(uint16_t), 1);
    READ(&lnk_count, sizeof(uint16_t), 1);
    READ(&memo_size, sizeof(uint16_t), 1);
    READ(&working_mem_count, sizeof(uint8_t), 1);

    if (tnsr_count > MAX_TENSORS || lnk_count > MAX_LINKS ||
        memo_size > MAX_MEM_ENTRIES || working_mem_count > WORKING_MEM_SIZE) {
        fprintf(stderr, "[ERR] Corrupted state\n");
        close(fd); return -1;
    }
    READ(&goals, sizeof(SystemGoals), 1);
    READ(&sys_res, sizeof(uint8_t), 1);
    READ(&interaction_count, sizeof(uint32_t), 1);
    READ(&last_mem_check_ts, sizeof(uint32_t), 1);
    READ(&sstate, sizeof(BitSystemState), 1);

    READ(&cluster_count, sizeof(uint16_t), 1);
    READ(&episode_count, sizeof(uint16_t), 1);
    READ(&concept_count, sizeof(uint8_t), 1);
    READ(&global_context_hash, sizeof(uint32_t), 1);
    READ(&next_cluster_id, sizeof(uint8_t), 1);

    if (cluster_count > MAX_CLUSTERS || episode_count > MAX_EPISODES || concept_count > 64) {
        fprintf(stderr, "[ERR] Corrupted self-organization state\n");
        close(fd); return -1;
    }
    
    for (uint16_t i = 0; i < tnsr_count; i++) {
        BitTensor* t = &tnsrs[i];
        READ(&t->rows, sizeof(uint16_t), 1);
        READ(&t->cols, sizeof(uint16_t), 1);
        READ(&t->res, sizeof(uint8_t), 1);
        READ(&t->act, sizeof(uint8_t), 1);
        READ(&t->ent, sizeof(uint8_t), 1);
        READ(&t->stab, sizeof(uint8_t), 1);
        READ(&t->conn, sizeof(uint16_t), 1);
        READ(&t->lu, sizeof(uint32_t), 1);
        READ(&t->mem_red, sizeof(uint8_t), 1);
        READ(&t->efficiency, sizeof(uint8_t), 1);
        READ(&t->compute_cost, sizeof(uint32_t), 1);
        READ(&t->goal_active, sizeof(uint8_t), 1);
        READ(&t->dropout, sizeof(uint8_t), 1);
        READ(&t->cluster_id, sizeof(uint8_t), 1);
        READ(&t->is_concept, sizeof(uint8_t), 1);

        uint32_t data_bytes;
        READ(&data_bytes, sizeof(uint32_t), 1);
        if (data_bytes > 0) {
            t->data = (uint8_t*)malloc(data_bytes);
            if (!t->data) { close(fd); return -1; }
            READ(t->data, 1, data_bytes);
        }
    }
    
    struct {
        uint16_t src_idx, tgt_idx;
        uint8_t strength, res, semantic_type;
        uint16_t weight;
        uint32_t ts, last_act;
        uint16_t use_count, success_count;
    } link_buf[MAX_LINKS];
    
    for (uint16_t i = 0; i < lnk_count; i++) {
        READ(&link_buf[i].src_idx, sizeof(uint16_t), 1);
        READ(&link_buf[i].tgt_idx, sizeof(uint16_t), 1);
        READ(&link_buf[i].strength, sizeof(uint8_t), 1);
        READ(&link_buf[i].res, sizeof(uint8_t), 1);
        READ(&link_buf[i].weight, sizeof(uint16_t), 1);
        READ(&link_buf[i].ts, sizeof(uint32_t), 1);
        READ(&link_buf[i].last_act, sizeof(uint32_t), 1);
        READ(&link_buf[i].use_count, sizeof(uint16_t), 1);
        READ(&link_buf[i].success_count, sizeof(uint16_t), 1);
        READ(&link_buf[i].semantic_type, sizeof(uint8_t), 1);
    }
    
    READ(clusters, sizeof(MemoryCluster), cluster_count);
    READ(episodes, sizeof(EpisodeMemory), episode_count);
    
    for (uint8_t i = 0; i < concept_count; i++) {
        uint16_t concept_idx;
        READ(&concept_idx, sizeof(uint16_t), 1);
        concepts[i].concept_tensor = index_to_tensor(concept_idx);
        READ(&concepts[i].member_count, sizeof(uint8_t), 1);
        READ(concepts[i].member_indices, sizeof(uint16_t), concepts[i].member_count);
        READ(&concepts[i].abstraction_level, sizeof(uint8_t), 1);
        READ(&concepts[i].coherence, sizeof(uint8_t), 1);
        READ(&concepts[i].last_used, sizeof(uint32_t), 1);
    }
    
    READ(memo, sizeof(BitMemory), memo_size);
    
    for (uint8_t i = 0; i < working_mem_count; i++) {
        uint16_t t_idx;
        READ(&t_idx, sizeof(uint16_t), 1);
        READ(&working_mem[i].timestamp, sizeof(uint32_t), 1);
        READ(&working_mem[i].priority, sizeof(uint8_t), 1);
        READ(&working_mem[i].access_count, sizeof(uint8_t), 1);
        READ(&working_mem[i].episode_marker, sizeof(uint8_t), 1);
        working_mem[i].tensor = index_to_tensor(t_idx);
    }
    
    for (uint16_t i = 0; i < lnk_count; i++) {
        BitLink* l = &lnks[i];
        l->src = index_to_tensor(link_buf[i].src_idx);
        l->tgt = index_to_tensor(link_buf[i].tgt_idx);
        l->strength = link_buf[i].strength;
        l->res = link_buf[i].res;
        l->weight = link_buf[i].weight;
        l->ts = link_buf[i].ts;
        l->last_act = link_buf[i].last_act;
        l->use_count = link_buf[i].use_count;
        l->success_count = link_buf[i].success_count;
        l->semantic_type = link_buf[i].semantic_type;
    }

    close(fd);
    build_link_index();
    printf("[LOAD] Загружено: %u тензоров, %u связей, %u кластеров, %u эпизодов, %u концепций\n",
           tnsr_count, lnk_count, cluster_count, episode_count, concept_count);
    return 0;
}

uint16_t tensor_to_index(BitTensor* t) {
    if (!t) return 0xFFFF;
    uintptr_t diff = t - tnsrs;
    return (diff < MAX_TENSORS) ? (uint16_t)diff : 0xFFFE;
}

BitTensor* index_to_tensor(uint16_t idx) {
    return (idx < MAX_TENSORS && idx < tnsr_count) ? &tnsrs[idx] : NULL;
}

// ===== ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ =====

void check_working_memory_for_transfer(void) {
    if (working_mem_count == 0) return;
    
    uint32_t now = (uint32_t)time(NULL);
    uint8_t transferred_count = 0;
    
    // Проверяем каждый элемент рабочей памяти
    for (int8_t i = working_mem_count - 1; i >= 0 && transferred_count < 3; i--) {
        WorkingMemoryEntry* entry = &working_mem[i];
        BitTensor* tensor = entry->tensor;
        
        if (!tensor || tensor->dropout) continue;
        
        // Критерии для переноса в долговременную память:
        // 1. Высокая активность
        // 2. Много обращений
        // 3. Длительное нахождение в рабочей памяти
        // 4. Хорошая эффективность
        
        uint8_t should_transfer = 0;
        uint32_t time_in_memory = now - entry->timestamp;
        
        if (tensor->act > 120 && 
            entry->access_count > 5 && 
            time_in_memory > 10 &&
            tensor->efficiency > goals.target_efficiency) {
            should_transfer = 1;
        }
        
        // Дополнительные критерии для концепций
        if (tensor->is_concept && tensor->act > 80 && entry->access_count > 3) {
            should_transfer = 1;
        }
        
        // Если тензор связан с кластером, проверяем активность кластера
        if (tensor->cluster_id != 0 && !should_transfer) {
            for (uint16_t ci = 0; ci < cluster_count; ci++) {
                if (clusters[ci].cluster_id == tensor->cluster_id) {
                    if (clusters[ci].activation_level > 150 && 
                        clusters[ci].stability > 100) {
                        should_transfer = 1;
                    }
                    break;
                }
            }
        }
        
        if (should_transfer) {
            // Усиливаем тензор и переносим
            tensor->stab = (tensor->stab * 7 + 200) >> 3;
            tensor->res = (tensor->res * 7 + 180) >> 3;
            
            // Если это еще не концепция, но заслуживает быть ей
            if (!tensor->is_concept && tensor->act > 140 && tensor->conn > 3) {
                tensor->is_concept = 1;
                
                // Создаем запись концепции если нужно
                if (concept_count < 64) {
                    MemoryConcept* concept = &concepts[concept_count++];
                    concept->concept_tensor = tensor;
                    concept->member_count = 1;
                    concept->member_indices[0] = tensor - tnsrs;
                    concept->abstraction_level = 1;
                    concept->coherence = tensor->stab;
                    concept->last_used = now;
                }
            }
            
            // Удаляем из рабочей памяти
            for (uint8_t j = i; j < working_mem_count - 1; j++) {
                working_mem[j] = working_mem[j + 1];
            }
            working_mem_count--;
            i--; // Корректируем индекс после удаления
            
            transferred_count++;
            
        }
    }
    
    // Если рабочая память почти полна, принудительно переносим старые элементы
    if (working_mem_count > WORKING_MEM_SIZE * 3 / 4) {
        uint8_t oldest_idx = 0;
        uint32_t oldest_time = working_mem[0].timestamp;
        
        for (uint8_t i = 1; i < working_mem_count; i++) {
            if (working_mem[i].timestamp < oldest_time) {
                oldest_time = working_mem[i].timestamp;
                oldest_idx = i;
            }
        }
        
        // Удаляем самый старый
        for (uint8_t i = oldest_idx; i < working_mem_count - 1; i++) {
            working_mem[i] = working_mem[i + 1];
        }
        working_mem_count--;
    }
}

void enhance_semantic_linking(void) {
    if (tnsr_count < 2 || lnk_count < 1) return;
    
    uint32_t now = (uint32_t)time(NULL);
    uint16_t enhanced_links = 0;
    uint8_t created_links = 0;
    
    // 1. Усиление существующих семантических связей
    for (uint16_t i = 0; i < lnk_count && enhanced_links < 20; i++) {
        BitLink* link = &lnks[i];
        
        if (!link->src || !link->tgt || 
            link->src->dropout || link->tgt->dropout) {
            continue;
        }
        
        // Усиливаем связи, которые:
        // - уже семантические
        // - имеют высокую прочность
        // - недавно использовались
        
        if (link->semantic_type > 0 && 
            link->strength > 100 &&
            now - link->last_act < 30) {
            
            // Вычисляем семантический вес
            uint8_t semantic_weight = 0;
            
            if (link->semantic_type == 1) {  // внутрикластерная
                semantic_weight = 40;
            } else if (link->semantic_type == 2) {  // межкластерная
                semantic_weight = 30;
            } else if (link->semantic_type == 3) {  // концептуальная
                semantic_weight = 50;
            }
            
            // Усиливаем связь
            if (link->strength < LINK_MAX_STRENGTH - semantic_weight) {
                link->strength += semantic_weight / 4;
                link->res = (link->res * 7 + 200) >> 3;
                enhanced_links++;
                
                // Также усиливаем связанные тензоры
                link->src->stab = (link->src->stab * 7 + 210) >> 3;
                link->tgt->stab = (link->tgt->stab * 7 + 210) >> 3;
            }
        }
    }
    
    // 2. Создание новых семантических связей между концепциями
    if (concept_count > 1 && created_links < 10) {
        for (uint8_t ci = 0; ci < concept_count - 1 && created_links < 5; ci++) {
            for (uint8_t cj = ci + 1; cj < concept_count && created_links < 5; cj++) {
                BitTensor* concept1 = concepts[ci].concept_tensor;
                BitTensor* concept2 = concepts[cj].concept_tensor;
                
                if (!concept1 || !concept2 || 
                    concept1->dropout || concept2->dropout) {
                    continue;
                }
                
                // Проверяем, нет ли уже связи
                uint8_t link_exists = 0;
                for (uint16_t li = 0; li < lnk_count; li++) {
                    if ((lnks[li].src == concept1 && lnks[li].tgt == concept2) ||
                        (lnks[li].src == concept2 && lnks[li].tgt == concept1)) {
                        link_exists = 1;
                        break;
                    }
                }
                
                if (!link_exists) {
                    // Проверяем семантическую близость
                    uint8_t coherence1 = concepts[ci].coherence;
                    uint8_t coherence2 = concepts[cj].coherence;
                    uint8_t avg_coherence = (coherence1 + coherence2) / 2;
                    
                    // Если обе концепции достаточно связные
                    if (avg_coherence > 100) {
                        BitLink* new_link = create_link(concept1, concept2);
                        if (new_link) {
                            new_link->semantic_type = 3;  // концептуальная связь
                            new_link->strength = 120 + (avg_coherence / 2);
                            new_link->res = 180;
                            created_links++;

                        }
                    }
                }
            }
        }
    }
    
    // 3. Улучшение связей внутри кластеров
    if (cluster_count > 0 && created_links < 15) {
        for (uint16_t ci = 0; ci < cluster_count && created_links < 10; ci++) {
            MemoryCluster* cluster = &clusters[ci];
            
            if (cluster->size < 2) continue;
            
            // Создаем связи между членами кластера
            uint8_t max_pairs = (cluster->size > 5) ? 5 : cluster->size;
            for (uint8_t ti = 0; ti < max_pairs - 1 && created_links < 10; ti++) {
                for (uint8_t tj = ti + 1; tj < max_pairs && created_links < 10; tj++) {
                    uint16_t idx1 = cluster->tensor_indices[ti];
                    uint16_t idx2 = cluster->tensor_indices[tj];
                    
                    if (idx1 >= tnsr_count || idx2 >= tnsr_count) continue;
                    
                    BitTensor* t1 = &tnsrs[idx1];
                    BitTensor* t2 = &tnsrs[idx2];
                    
                    if (t1->dropout || t2->dropout) continue;
                    
                    // Проверяем существование связи
                    uint8_t link_exists = 0;
                    for (uint16_t li = 0; li < lnk_count; li++) {
                        if ((lnks[li].src == t1 && lnks[li].tgt == t2) ||
                            (lnks[li].src == t2 && lnks[li].tgt == t1)) {
                            link_exists = 1;
                            break;
                        }
                    }
                    
                    if (!link_exists) {
                        // Создаем внутрикластерную связь
                        BitLink* new_link = create_link(t1, t2);
                        if (new_link) {
                            new_link->semantic_type = 1;  // внутрикластерная
                            new_link->strength = 100 + cluster->stability / 3;
                            new_link->res = 160;
                            created_links++;
                        }
                    }
                }
            }
        }
    }
    
    // 4. Создание связей на основе эпизодной памяти
    if (episode_count > 0 && created_links < 20) {
        for (uint16_t ei = 0; ei < episode_count && created_links < 10; ei++) {
            EpisodeMemory* episode = &episodes[ei];
            
            if (episode->importance < 50 || episode->length < 2) continue;
            
            // Создаем временные связи между последовательными элементами эпизода
            for (uint8_t pos = 0; pos < episode->length - 1 && created_links < 10; pos++) {
                uint16_t idx1 = episode->sequence[pos];
                uint16_t idx2 = episode->sequence[pos + 1];
                
                if (idx1 >= tnsr_count || idx2 >= tnsr_count) continue;
                
                BitTensor* t1 = &tnsrs[idx1];
                BitTensor* t2 = &tnsrs[idx2];
                
                if (t1->dropout || t2->dropout) continue;
                
                // Усиливаем существующую связь или создаем новую
                BitLink* link = NULL;
                for (uint16_t li = 0; li < lnk_count; li++) {
                    if ((lnks[li].src == t1 && lnks[li].tgt == t2) ||
                        (lnks[li].src == t2 && lnks[li].tgt == t1)) {
                        link = &lnks[li];
                        break;
                    }
                }
                
                if (link) {
                    // Усиливаем существующую связь
                    link->strength = (link->strength * 7 + 150) >> 3;
                    link->res = (link->res * 7 + 170) >> 3;
                    link->semantic_type = (link->semantic_type == 0) ? 2 : link->semantic_type;
                    enhanced_links++;
                } else {
                    // Создаем новую связь
                    link = create_link(t1, t2);
                    if (link) {
                        link->semantic_type = 2;  // межкластерная (временная)
                        link->strength = 80 + (episode->importance / 2);
                        link->res = 150;
                        created_links++;
                    }
                }
            }
        }
    }

    
    // Вызываем оригинальную semantic_memory_binding для завершения
    semantic_memory_binding();
}

void force_transfer_to_longterm(BitTensor* tensor) {
    if (!tensor) return;
    
    tensor->is_concept = 1;
    tensor->stab = (uint8_t)((tensor->stab * 7 + 220) >> 3);
    tensor->res = (uint8_t)((tensor->res * 7 + 200) >> 3);
    
    short temp_eff = (short)tensor->efficiency + 10;
    tensor->efficiency = (temp_eff > 255) ? 255 : (uint8_t)temp_eff;
    
    save_tnsr(tensor);
}

void activate_longterm_to_working(void) {
    if (concept_count > 0) {
        short idx = (short)(rand() % concept_count);
        BitTensor* concept = concepts[idx].concept_tensor;
        
        if (concept && !concept->dropout) {
            concept->act = 150;
            short new_conn = (short)concept->conn + 5;
            concept->conn = (new_conn > 65535) ? 65535 : (uint16_t)new_conn;
            
            add_to_working_memory(concept);
        }
    }
}

// ===== КОНЕЦ ФАЙЛА =====