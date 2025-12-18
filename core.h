#ifndef CORE_H
#define CORE_H

#include <stdint.h>
#include <time.h>

// ===== Настройки =====
#define MAX_INPUT 3000
#define MAX_OUTPUT 4000
#define MAX_PATTERN 384
#define MAX_MEM_ENTRIES 1536
#define MAX_TENSORS 1000
#define MAX_LINKS 500
#define MAX_TT_ENTRIES 100
#define HISTORY_SIZE 500
#define LOW_ACT_THRESHOLD 60
#define MEM_REDUCE_INTERVAL 15
#define WORKING_MEM_SIZE 50
#define DROPOUT_RATE 2
#define LINK_STRENGTH_INC 12
#define LINK_STRENGTH_DEC 6
#define LINK_MIN_STRENGTH 10
#define LINK_MAX_STRENGTH 220
#define ENCODER_QUALITY 3
#define RES_MAX 255
#define RES_HALF 96
#define ACT_MAX 255

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
typedef struct BitTensorTensor BitTensorTensor;
typedef struct BitMemory BitMemory;
typedef struct WorkingMemoryEntry WorkingMemoryEntry;
typedef struct BitSystemState BitSystemState;
typedef struct SystemGoals SystemGoals;

struct BitTensor {
    uint8_t* data;
    uint16_t rows;
    uint16_t cols;
    uint8_t res;
    uint8_t act;
    uint8_t ent;
    uint8_t stab;
    uint16_t conn;
    uint32_t lu;
    uint8_t mem_red;
    uint8_t efficiency;
    uint32_t compute_cost;
    uint8_t goal_active;
    uint8_t dropout;
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
};

struct BitTensorTensor {
    uint8_t* data;
    uint16_t num_tensors;
    uint8_t enc_type;
    uint8_t bits_per_idx;
    uint8_t res;
    uint8_t act;
    uint32_t lu;
    uint16_t* tensor_indices;
    uint8_t efficiency;
};

struct BitMemory {
    uint8_t data[MAX_PATTERN];
    uint8_t len;
    uint16_t count;
    uint8_t res;
    uint8_t act;
    uint8_t ent;
    uint32_t ts;
};

struct WorkingMemoryEntry {
    BitTensor* tensor;
    uint32_t timestamp;
    uint8_t priority;
    uint8_t access_count;
};

struct BitSystemState {
    uint8_t act_hist[HISTORY_SIZE];
    uint8_t ent_hist[HISTORY_SIZE];
    uint8_t res_hist[HISTORY_SIZE];
    uint8_t hist_idx;
    uint8_t coh;
    uint8_t energy;
};

struct SystemGoals {
    uint8_t target_efficiency;
    uint8_t energy_saving_mode;
    uint32_t total_compute_cost;
    uint32_t efficiency_gain;
    uint8_t dropout_enabled;
};

// ===== Внешние глобальные переменные =====
extern BitMemory memo[MAX_MEM_ENTRIES];
extern BitTensor tnsrs[MAX_TENSORS];
extern BitTensorTensor t_tnsrs[MAX_TT_ENTRIES];
extern BitLink lnks[MAX_LINKS];
extern WorkingMemoryEntry working_mem[WORKING_MEM_SIZE];
extern BitSystemState sstate;
extern SystemGoals goals;
extern uint16_t memo_size;
extern uint16_t tnsr_count;
extern uint16_t tt_count;
extern uint16_t lnk_count;
extern uint8_t sys_res;
extern uint32_t interaction_count;
extern uint32_t last_mem_check_ts;
extern uint8_t working_mem_count;

// ===== Прототипы функций =====
BitTensor* create_tnsr(uint16_t rows, uint16_t cols);
BitTensorTensor* create_tt(uint16_t num_tensors, uint8_t enc_type);
void reduce_tnsr_mem(BitTensor* t);
uint8_t fast_log2(uint32_t x);
uint8_t calc_bit_ent(BitTensor* t);
uint8_t calc_bit_sim(BitTensor* a, BitTensor* b);
uint8_t calc_res_match(BitTensor* a, BitTensor* b);
uint8_t calculate_efficiency(BitTensor* t);
void update_efficiency_goal(void);
BitTensor* find_efficient_match(BitTensor* input);
void optimize_tensor(BitTensor* t);
void update_bit_net_with_goals(void);
void proc_bit_input_raw(const uint8_t* binary, uint16_t input_len);
void proc_bit_input(const char* input);
BitTensor* get_resonant_tensor(void);
BitTensor* get_most_active_tensor(void);
void update_thought_stream(void);
void self_reflect_on_thought(void);

void decode_tnsr(BitTensor* t, char* buffer, uint16_t buf_size);
void encode_tnsr(BitTensor* t, const uint8_t* data, uint16_t data_len);
void save_tnsr(BitTensor* t);

void add_to_working_memory(BitTensor* t);
BitTensor* get_from_working_memory(uint8_t min_priority);
void apply_dropout(void);

BitLink* create_link(BitTensor* src, BitTensor* tgt);
void update_link_strength(BitLink* link, uint8_t was_successful);
void decay_unused_links(void);

void learn_by_binary_update(BitTensor* target, const uint8_t* input_data, uint16_t input_len);
void prevent_overfitting_by_bit_shift(BitTensor* t);

void set_tnsr_link(BitTensorTensor* tt, uint16_t row_idx, uint16_t col_idx, uint16_t tnsr_idx);
uint16_t get_tnsr_link(BitTensorTensor* tt, uint16_t row_idx, uint16_t col_idx);

int save_state_to_file(const char* filename);
int load_state_from_file(const char* filename);

#endif // CORE_H