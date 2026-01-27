#ifndef CORE_H
#define CORE_H

#include <stdint.h>
#include <time.h>

// ===== Константы =====
#define MAX_INPUT 4096
#define MAX_OUTPUT 8192
#define MAX_PATTERN 1024
#define MAX_MEM_ENTRIES 1024
#define MAX_TENSORS 2048
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
#define LEARN_SLICE_SIZE 64
#define ENCODER_QUALITY 5
#define RES_MAX 255
#define RES_HALF 128
#define ACT_MAX 255
#define SIM_FUZZINESS_PERCENT 2

// ===== Самоорганизация =====
#define MAX_CLUSTERS 128
#define MAX_EPISODES 512
#define CLUSTER_THRESHOLD 150
#define MERGE_THRESHOLD 180
#define CONCEPT_CREATION_THRESHOLD 3
#define EPISODE_MIN_LENGTH 3
#define CONSOLIDATION_INTERVAL 30
#define SELF_ORG_INTERVAL 5

// ===== Потоки мыслей =====
#define MAX_THOUGHT_STREAMS 6        // Уменьшено: меньше, но качественнее
#define MAX_THOUGHT_CHAIN_LENGTH 12  // Уменьшено: короче, но глубже
#define THOUGHT_STREAM_LIFETIME 180  // 3 минуты: быстрее "забывает" слабые мысли
#define MAX_META_REFLECTIONS 3       // Максимум мета-рефлексий

// ===== Битовые макросы =====
#define BIT_SET(byte, bit) ((byte) |= (1U << (bit)))
#define BIT_CLEAR(byte, bit) ((byte) &= ~(1U << (bit)))
#define BIT_TOGGLE(byte, bit) ((byte) ^= (1U << (bit)))
#define BIT_GET(byte, bit) (((byte) >> (bit)) & 1U)
#define BIT_NOT(byte) (~(byte))
#define BIT_AND(a, b) ((a) & (b))
#define BIT_XOR(a, b) ((a) ^ (b))
#define BIT_OR(a, b) ((a) | (b))

// ===== Отладка =====
#ifdef DEBUG_MODE
    #define DEBUG_PRINT(fmt, ...) \
        fprintf(stderr, "[DEBUG] %s:%d (%s): " fmt "\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)
    #define ERROR_EXIT(msg) \
        do { \
            fprintf(stderr, "[ERROR] %s:%d (%s): %s\n", __FILE__, __LINE__, __FUNCTION__, msg); \
            abort(); \
        } while(0)
#else
    #define DEBUG_PRINT(fmt, ...)
    #define ERROR_EXIT(msg) abort()
#endif

// ===== Структуры =====
typedef struct BitTensor BitTensor;
typedef struct BitLink BitLink;
typedef struct BitMemory BitMemory;
typedef struct WorkingMemoryEntry WorkingMemoryEntry;
typedef struct BitSystemState BitSystemState;
typedef struct SystemGoals SystemGoals;
typedef struct MemoryCluster MemoryCluster;
typedef struct EpisodeMemory EpisodeMemory;
typedef struct MemoryConcept MemoryConcept;
typedef struct TensorLinks TensorLinks;
typedef struct ThoughtStream ThoughtStream;

// ===== Типы для поиска =====
typedef enum {
    SEARCH_MOST_ACTIVE,
    SEARCH_RESONANT,
    SEARCH_EFFICIENT,
    SEARCH_CUSTOM_SCORE,
    SEARCH_BY_CLUSTER,
    SEARCH_CONCEPTUAL
} SearchStrategy;

typedef uint32_t (*ScoreFunction)(BitTensor* t, void* context);

// ===== Структуры данных =====
struct BitTensor {
    uint8_t* data;
    uint16_t rows;
    uint16_t cols;
    uint8_t res;
    uint8_t act;
    uint8_t ent;
    uint8_t ent_last;      // Предыдущее значение энтропии
    uint8_t stab;
    uint16_t conn;
    uint32_t lu;
    uint8_t mem_red;
    uint8_t efficiency;
    uint16_t forward;
    uint32_t compute_cost;
    uint8_t goal_active;
    uint8_t dropout;
    uint8_t cluster_id;
    uint8_t is_concept;
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
    uint8_t semantic_type;   // 0=обычная, 1=внутрикластерная, 2=межкластерная, 3=концептуальная
};

struct BitMemory {
    uint8_t data[MAX_PATTERN];
    uint8_t len;
    uint16_t count;
    uint8_t res;
    uint8_t act;
    uint8_t ent;
    uint32_t ts;
    uint8_t cluster_id;
};

struct WorkingMemoryEntry {
    BitTensor* tensor;
    uint32_t timestamp;
    uint8_t priority;
    uint8_t access_count;
    uint8_t episode_marker;
};

struct ThoughtStream {
    BitTensor* thought_chain[MAX_THOUGHT_CHAIN_LENGTH];
    uint8_t chain_length;
    uint32_t timestamp;
    uint8_t coherence;
    uint8_t abstraction_level;
    uint16_t activation_counter;
    uint8_t is_active;
    uint8_t recursion_depth;                         // Глубина рекурсивного мышления
    BitTensor* meta_reflections[MAX_META_REFLECTIONS]; // Мета-тензоры рефлексии
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
    uint8_t self_organization_enabled;
    uint8_t memory_consolidation_mode;
    uint8_t thought_stream_enabled;
};

struct MemoryCluster {
    uint16_t tensor_indices[64];
    uint8_t cluster_id;
    uint8_t centroid[256];
    uint16_t size;
    uint8_t stability;
    uint32_t last_access;
    uint8_t category;               // 0=неопределено, 1=действие, 2=состояние, 3=концепция
    uint8_t activation_level;
    uint32_t creation_time;
    uint16_t link_count;
};

struct EpisodeMemory {
    uint16_t sequence[256];
    uint8_t context_hash[16];
    uint32_t start_time;
    uint32_t end_time;
    uint8_t success_score;
    uint8_t importance;
    uint8_t length;
    uint8_t reward_context;
    uint32_t last_recall;
    uint16_t recall_count;
};

struct MemoryConcept {
    BitTensor* concept_tensor;
    uint16_t member_indices[32];
    uint8_t member_count;
    uint8_t abstraction_level;
    uint8_t coherence;
    uint32_t last_used;
};

struct TensorLinks {
    uint16_t tensor_idx;
    uint16_t link_indices[32];
    uint8_t link_count;
    uint8_t cluster_links[16];
    uint8_t cluster_link_count;
};

// ===== Глобальные переменные (extern) =====
extern BitMemory memo[MAX_MEM_ENTRIES];
extern BitTensor tnsrs[MAX_TENSORS];
extern BitLink lnks[MAX_LINKS];
extern WorkingMemoryEntry working_mem[WORKING_MEM_SIZE];
extern TensorLinks tensor_links[MAX_TENSORS];
extern ThoughtStream thought_streams[MAX_THOUGHT_STREAMS];
extern BitSystemState sstate;
extern SystemGoals goals;

// Самоорганизация
extern MemoryCluster clusters[MAX_CLUSTERS];
extern EpisodeMemory episodes[MAX_EPISODES];
extern MemoryConcept concepts[64];

// Счетчики
extern uint16_t memo_size;
extern uint16_t tnsr_count;
extern uint16_t lnk_count;
extern uint8_t sys_res;
extern uint32_t interaction_count;
extern uint32_t last_mem_check_ts;
extern uint8_t working_mem_count;
extern uint8_t active_thought_streams;

// Самоорганизация счетчики
extern uint16_t cluster_count;
extern uint16_t episode_count;
extern uint8_t concept_count;
extern uint32_t global_context_hash;
extern uint32_t last_cluster_reorg;
extern uint8_t next_cluster_id;

// ===== Основные функции =====
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

// ===== Функции самоорганизации =====
void self_organize_memory_clusters(void);
void semantic_memory_binding(void);
void memory_consolidation(void);
void create_concept_from_tensors(BitTensor* t1, BitTensor* t2);
void merge_tensors(BitTensor* a, BitTensor* b);
void create_concept_from_cluster(MemoryCluster* cluster);
void create_episode_from_working_memory(void);
void activate_relevant_episodes(void);
BitTensor* find_center_tensor_in_cluster(MemoryCluster* cluster);
BitLink* find_or_create_link(BitTensor* src, BitTensor* tgt);
void transfer_links(BitTensor* from, BitTensor* to);
void save_concept(BitTensor* concept, uint8_t cluster_id);
void prune_weak_clusters(void);
void update_cluster_stability(MemoryCluster* cluster);
uint8_t calculate_cluster_coherence(MemoryCluster* cluster);
void create_semantic_links_between_clusters(MemoryCluster* c1, MemoryCluster* c2);
uint32_t calculate_context_hash(void);
uint8_t is_similar_context(uint8_t* ctx1, uint8_t* ctx2);

// ===== Функции консолидации =====
void forward_forward_consolidation(uint8_t mode);
void simple_memory_consolidation(void);

// ===== Вспомогательные функции =====
void build_link_index(void);
uint8_t check_context_fit(BitTensor* a, BitTensor* b, BitLink* link);
void fast_contextual_activation(BitTensor* context);
void update_thought_stream(void);
void update_thought_stream_simple(void);  // Упрощенная версия
void aggressive_memory_cleanup(void);
void activate_longterm_to_working(void);
void force_transfer_to_longterm(BitTensor* tensor);

// ===== Функции управления памятью =====
void check_working_memory_for_transfer(void);  // Интеллектуальный перенос
void enhance_semantic_linking(void);           // Улучшенное семантическое связывание

// ===== Функции потоков мыслей =====
void form_thought_chain(void);
void evolve_thought_stream(ThoughtStream* stream);
void prune_old_thought_streams(void);
BitTensor* find_association(BitTensor* current);
uint8_t calculate_chain_coherence(ThoughtStream* stream);
void create_abstraction_from_chain(ThoughtStream* stream);
void create_recursive_abstraction(ThoughtStream* stream);  // Рекурсивное мышление
const char* get_current_thought(void);
void activate_thought_stream(void);

// ===== Функции для main_chat.c =====
void process_chat_input(const char* input);
void generate_response(void);
void extract_and_print_tensor_text(BitTensor* tensor);

// ===== Дополнительные функции =====
void reorganize_clusters_by_activity(void);
void extract_patterns_from_episodes(void);
void forget_unimportant_episodes(void);
void reinforce_successful_episodes(void);

// ===== RL-функции =====
uint8_t sync_and_get_recommendation(uint8_t* vector_data, int vector_len, uint8_t last_action);
uint8_t get_global_resonance(void);
void learn_from_data(uint8_t* vector_data, int len, uint8_t action_idx, uint8_t reward);

// ===== Утилиты =====
static inline uint32_t murmur_mix(uint32_t h);  // Вспомогательная функция для хеширования

#endif // CORE_H