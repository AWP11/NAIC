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

// ===== Глобальные состояния =====
BitMemory memo[MAX_MEM_ENTRIES];
BitTensor tnsrs[MAX_TENSORS];
BitTensorTensor t_tnsrs[MAX_TT_ENTRIES];
BitLink lnks[MAX_LINKS];
WorkingMemoryEntry working_mem[WORKING_MEM_SIZE];
BitSystemState sstate;
SystemGoals goals = {180, 0, 0, 0, 1};
uint16_t memo_size = 0;
uint16_t tnsr_count = 0;
uint16_t tt_count = 0;
uint16_t lnk_count = 0;
uint8_t sys_res = RES_HALF;
uint32_t interaction_count = 0;
uint32_t last_mem_check_ts = 0;
uint8_t working_mem_count = 0;

// ===== Прототипы =====
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
BitTensor* get_resonant_tensor(void);
void decode_tnsr(BitTensor* t, char* buffer, uint16_t buf_size);
void encode_tnsr(BitTensor* t, const uint8_t* data, uint16_t data_len);
void save_tnsr(BitTensor* t);
void add_to_working_memory(BitTensor* t);
BitTensor* get_from_working_memory(uint8_t min_priority);
void apply_dropout(void);
void set_tnsr_link(BitTensorTensor* tt, uint16_t row_idx, uint16_t col_idx, uint16_t tnsr_idx);
uint16_t get_tnsr_link(BitTensorTensor* tt, uint16_t row_idx, uint16_t col_idx);
BitLink* create_link(BitTensor* src, BitTensor* tgt);
void update_link_strength(BitLink* link, uint8_t was_successful);
void decay_unused_links(void);
void learn_by_binary_update(BitTensor* target, const uint8_t* input_data, uint16_t input_len);
void prevent_overfitting_by_bit_shift(BitTensor* t);
void update_thought_stream(void);
BitTensor* get_most_active_tensor(void);
void self_reflect_on_thought(void);
int save_state_to_file(const char* filename);
int load_state_from_file(const char* filename);

// ===== Вспомогательные для сериализации =====
static uint16_t tensor_to_index(BitTensor* t) {
    if (!t) return 0xFFFF;
    intptr_t diff = t - tnsrs;
    return (diff >= 0 && diff < MAX_TENSORS) ? (uint16_t)diff : 0xFFFF;
}

static BitTensor* index_to_tensor(uint16_t idx) {
    return (idx < MAX_TENSORS && tnsr_count > idx) ? &tnsrs[idx] : NULL;
}

// ===== Реализации =====

uint8_t fast_log2(uint32_t x) {
    if (x == 0) return 0;
    uint8_t log = 0;
    if (x & 0xFFFF0000) { x >>= 16; log += 16; }
    if (x & 0xFF00) { x >>= 8; log += 8; }
    if (x & 0xF0) { x >>= 4; log += 4; }
    if (x & 0xC) { x >>= 2; log += 2; }
    if (x & 0x2) { x >>= 1; log += 1; }
    uint32_t low_pow = 1 << log;
    uint32_t high_pow = low_pow << 1;
    uint8_t fraction = (x - low_pow) * 255 / (high_pow - low_pow);
    return (log << 3) | (fraction >> 5);
}

uint8_t calc_bit_ent(BitTensor* t) {
    if (!t || !t->data) return 0;
    uint32_t total_bits = t->rows * t->cols;
    if (total_bits == 0) return 0;
    uint32_t ones = 0;
    for (uint32_t i = 0; i < total_bits; i++) {
        if (BIT_GET(t->data[i / 8], i % 8)) ones++;
    }
    uint32_t p1_fixed = (ones << 8) / total_bits;
    uint32_t p0_fixed = 256 - p1_fixed;
    uint8_t log_p0 = p0_fixed ? fast_log2(p0_fixed) : 0;
    uint8_t log_p1 = p1_fixed ? fast_log2(p1_fixed) : 0;
    uint32_t h_fixed = (p0_fixed * log_p0 + p1_fixed * log_p1) >> 8;
    return (uint8_t)h_fixed;
}

uint8_t calc_bit_sim(BitTensor* a, BitTensor* b) {
    if (!a || !b || !a->data || !b->data) return 0;
    uint32_t bits_a = a->rows * a->cols;
    uint32_t bits_b = b->rows * b->cols;
    uint32_t min_bits = (bits_a < bits_b) ? bits_a : bits_b;
    if (min_bits == 0) return 0;
    uint32_t matches = 0;
    uint32_t bytes_to_compare = min_bits / 8;
    for (uint32_t i = 0; i < bytes_to_compare; i++) {
        uint8_t xor_res = a->data[i] ^ b->data[i];
        matches += 8 - __builtin_popcount(xor_res);
    }
    uint8_t rem_bits = min_bits % 8;
    if (rem_bits > 0) {
        uint32_t last_byte_idx = bytes_to_compare;
        uint8_t mask = (1 << rem_bits) - 1;
        uint8_t xor_res = (a->data[last_byte_idx] & mask) ^ (b->data[last_byte_idx] & mask);
        matches += rem_bits - __builtin_popcount(xor_res);
    }
    return (uint8_t)((matches * 255) / min_bits);
}

uint8_t calc_res_match(BitTensor* a, BitTensor* b) {
    if (!a || !b) return 0;
    uint8_t res_diff = (a->res > b->res) ? (a->res - b->res) : (b->res - a->res);
    uint8_t act_diff = (a->act > b->act) ? (a->act - b->act) : (b->act - a->act);
    uint8_t ent_diff = (a->ent > b->ent) ? (a->ent - b->ent) : (b->ent - a->ent);
    uint8_t diff_score = (res_diff + act_diff + ent_diff) / 3;
    return 255 - diff_score;
}

uint8_t calculate_efficiency(BitTensor* t) {
    if (!t) return 0;
    uint32_t benefit = (uint32_t)t->act * t->res;
    uint32_t cost = t->compute_cost + 1;
    uint8_t eff = (uint8_t)(benefit / cost);
    if (eff > 255) eff = 255;
    return eff;
}

void update_efficiency_goal(void) {
    uint32_t total_efficiency = 0;
    uint16_t active_tensors = 0;
    for (uint16_t i = 0; i < tnsr_count; i++) {
        if (tnsrs[i].act > 50) {
            tnsrs[i].efficiency = calculate_efficiency(&tnsrs[i]);
            total_efficiency += tnsrs[i].efficiency;
            active_tensors++;
        }
    }
    if (active_tensors > 0) {
        uint8_t avg_efficiency = (uint8_t)(total_efficiency / active_tensors);
        if (avg_efficiency > goals.target_efficiency) {
            goals.target_efficiency++;
            goals.efficiency_gain++;
        }
        if (avg_efficiency < goals.target_efficiency - 30) {
            goals.energy_saving_mode = 1;
        } else {
            goals.energy_saving_mode = 0;
        }
    }
}

BitTensor* create_tnsr(uint16_t rows, uint16_t cols) {
    if (tnsr_count >= MAX_TENSORS || rows == 0 || cols == 0) return NULL;
    BitTensor* t = &tnsrs[tnsr_count++];
    uint32_t total_bytes = (rows * cols + 7) / 8;
    t->data = (uint8_t*)calloc(total_bytes, 1);
    if (!t->data) { tnsr_count--; return NULL; }
    t->rows = rows;
    t->cols = cols;
    t->res = RES_HALF;
    t->act = ACT_MAX / 2;
    t->ent = 0;
    t->stab = 128;
    t->conn = 0;
    t->lu = (uint32_t)time(NULL);
    t->mem_red = 0;
    t->efficiency = 128;
    t->compute_cost = rows * cols * 2;
    t->goal_active = (rows * cols > 100) ? 1 : 0;
    t->dropout = (rand() % 100 < DROPOUT_RATE) ? 1 : 0;
    return t;
}

BitTensorTensor* create_tt(uint16_t num_tensors, uint8_t enc_type) {
    if (tt_count >= MAX_TT_ENTRIES || num_tensors == 0) return NULL;
    BitTensorTensor* tt = &t_tnsrs[tt_count++];
    tt->enc_type = enc_type;
    tt->res = RES_HALF;
    tt->act = ACT_MAX / 2;
    tt->lu = (uint32_t)time(NULL);
    tt->num_tensors = num_tensors;
    tt->efficiency = 128;
    if (enc_type == 1) {
        uint8_t bits_per_idx = 0;
        uint16_t max_idx_val = MAX_TENSORS - 1;
        while(max_idx_val > 0) { max_idx_val >>= 1; bits_per_idx++; }
        bits_per_idx = (bits_per_idx < 4) ? 4 : bits_per_idx;
        tt->bits_per_idx = bits_per_idx;
        uint32_t total_bits = num_tensors * bits_per_idx;
        uint32_t total_bytes = (total_bits + 7) / 8;
        tt->data = (uint8_t*)calloc(total_bytes, 1);
        if (!tt->data) { tt_count--; return NULL; }
        tt->tensor_indices = (uint16_t*)calloc(num_tensors, sizeof(uint16_t));
        if (!tt->tensor_indices) { free(tt->data); tt_count--; return NULL; }
    } else {
        uint32_t total_bytes = (num_tensors + 7) / 8;
        tt->data = (uint8_t*)calloc(total_bytes, 1);
        if (!tt->data) { tt_count--; return NULL; }
        tt->tensor_indices = NULL;
    }
    return tt;
}

void reduce_tnsr_mem(BitTensor* t) {
    if (!t || !t->data || t->mem_red) return;
    uint32_t current_bits = t->rows * t->cols;
    uint32_t new_bits = (current_bits >> 1);
    if (new_bits < 8) new_bits = 8;
    uint16_t new_cols = (t->cols > 1) ? t->cols >> 1 : 1;
    uint32_t new_size_bytes = (new_bits + 7) / 8;
    uint8_t* reduced_data = (uint8_t*)calloc(new_size_bytes, 1);
    if (!reduced_data) return;
    for (uint32_t i = 0; i < new_bits; i++) {
        uint32_t orig_bit_idx = i << 1;
        if (orig_bit_idx >= current_bits) break;
        uint8_t val = BIT_GET(t->data[orig_bit_idx / 8], orig_bit_idx % 8);
        if (val) BIT_SET(reduced_data[i / 8], i % 8);
    }
    free(t->data);
    t->data = reduced_data;
    t->cols = new_cols;
    t->act = (t->act > 1) ? t->act >> 1 : 1;
    t->mem_red = 1;
    t->ent = calc_bit_ent(t);
    t->compute_cost = t->rows * t->cols * 2;
    t->efficiency = calculate_efficiency(t);
}

void set_tnsr_link(BitTensorTensor* tt, uint16_t row_idx, uint16_t col_idx, uint16_t tnsr_idx) {
    if (!tt || tt->enc_type != 1 || !tt->tensor_indices) return;
    uint32_t element_idx = row_idx * tt->num_tensors + col_idx;
    if (element_idx < tt->num_tensors) {
        tt->tensor_indices[element_idx] = tnsr_idx;
    }
}

uint16_t get_tnsr_link(BitTensorTensor* tt, uint16_t row_idx, uint16_t col_idx) {
    if (!tt || tt->enc_type != 1 || !tt->tensor_indices) return MAX_TENSORS;
    uint32_t element_idx = row_idx * tt->num_tensors + col_idx;
    if (element_idx < tt->num_tensors) {
        return tt->tensor_indices[element_idx];
    }
    return MAX_TENSORS;
}

void optimize_tensor(BitTensor* t) {
    if (!t || t->goal_active == 0) return;
    if (t->efficiency < goals.target_efficiency) {
        if (t->compute_cost > 100) {
            t->compute_cost -= t->compute_cost / 10;
        }
        if (t->stab < 200) {
            t->stab += 5;
        }
        t->goal_active = 2;
        t->efficiency = calculate_efficiency(t);
    }
}

void apply_dropout(void) {
    if (!goals.dropout_enabled) return;
    for (uint16_t i = 0; i < tnsr_count; i++) {
        if (tnsrs[i].dropout) {
            if (tnsrs[i].act > 10) {
                tnsrs[i].act = tnsrs[i].act >> 1;
            }
            if (rand() % 100 < 10) {
                tnsrs[i].dropout = 0;
            }
        } else {
            if (rand() % 1000 < DROPOUT_RATE) {
                tnsrs[i].dropout = 1;
            }
        }
    }
}

void add_to_working_memory(BitTensor* t) {
    if (!t || working_mem_count >= WORKING_MEM_SIZE) return;
    uint8_t existing_idx = 255;
    for (uint8_t i = 0; i < working_mem_count; i++) {
        if (working_mem[i].tensor == t) {
            existing_idx = i;
            break;
        }
    }
    if (existing_idx != 255) {
        working_mem[existing_idx].access_count++;
        working_mem[existing_idx].priority = (working_mem[existing_idx].priority * 7 + 100) >> 3;
        working_mem[existing_idx].timestamp = (uint32_t)time(NULL);
    } else {
        working_mem[working_mem_count].tensor = t;
        working_mem[working_mem_count].timestamp = (uint32_t)time(NULL);
        working_mem[working_mem_count].priority = 100;
        working_mem[working_mem_count].access_count = 1;
        working_mem_count++;
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
        return working_mem[best_idx].tensor;
    }
    return NULL;
}

BitLink* create_link(BitTensor* src, BitTensor* tgt) {
    if (!src || !tgt || lnk_count >= MAX_LINKS) return NULL;
    for (uint16_t i = 0; i < lnk_count; i++) {
        if (lnks[i].src == src && lnks[i].tgt == tgt) {
            return &lnks[i];
        }
    }
    BitLink* link = &lnks[lnk_count++];
    link->src = src;
    link->tgt = tgt;
    link->strength = 128;
    link->res = calc_res_match(src, tgt);
    link->weight = (src->res + tgt->res) / 2;
    link->ts = (uint32_t)time(NULL);
    link->last_act = 0;
    link->use_count = 0;
    link->success_count = 0;
    src->conn++;
    tgt->conn++;
    return link;
}

void update_link_strength(BitLink* link, uint8_t was_successful) {
    if (!link) return;
    link->use_count++;
    if (was_successful) {
        link->success_count++;
        if (link->strength + LINK_STRENGTH_INC <= LINK_MAX_STRENGTH) {
            link->strength += LINK_STRENGTH_INC;
        } else {
            link->strength = LINK_MAX_STRENGTH;
        }
        uint8_t new_res = link->res + 5;
        if (new_res > RES_MAX) new_res = RES_MAX;
        link->res = new_res;
        link->weight = (link->weight * 9 + (link->src->act + link->tgt->act) * 127) >> 3;
    } else {
        if (link->strength > LINK_STRENGTH_DEC + LINK_MIN_STRENGTH) {
            link->strength -= LINK_STRENGTH_DEC;
        } else {
            link->strength = LINK_MIN_STRENGTH;
        }
        if (link->res > 10) link->res -= 2;
    }
    link->ts = (uint32_t)time(NULL);
}

void decay_unused_links(void) {
    uint32_t now = (uint32_t)time(NULL);
    for (uint16_t i = 0; i < lnk_count; i++) {
        BitLink* link = &lnks[i];
        if (now - link->last_act > 30) {
            if (link->strength > LINK_STRENGTH_DEC + LINK_MIN_STRENGTH) {
                link->strength -= LINK_STRENGTH_DEC;
            }
            if (link->strength < LINK_MIN_STRENGTH * 2 && link->use_count < 3) {
                if (link->src) link->src->conn--;
                if (link->tgt) link->tgt->conn--;
                lnks[i] = lnks[lnk_count - 1];
                lnk_count--;
                i--;
            }
        }
    }
}

void encode_tnsr(BitTensor* t, const uint8_t* data, uint16_t data_len) {
    if (!t || !data || data_len == 0) return;
    uint16_t max_cols = data_len < t->cols ? data_len : t->cols;
    for (uint16_t i = 0; i < max_cols; i++) {
        uint8_t byte = data[i];
        for (uint8_t bit = 0; bit < 8; bit++) {
            if (bit < t->rows) {
                uint32_t bit_idx = bit * t->cols + i;
                if (BIT_GET(byte, bit)) {
                    BIT_SET(t->data[bit_idx / 8], bit_idx % 8);
                } else {
                    BIT_CLEAR(t->data[bit_idx / 8], bit_idx % 8);
                }
            }
        }
    }
    for (uint16_t i = max_cols; i < t->cols; i++) {
        for (uint8_t bit = 0; bit < t->rows; bit++) {
            uint32_t bit_idx = bit * t->cols + i;
            BIT_CLEAR(t->data[bit_idx / 8], bit_idx % 8);
        }
    }
    t->ent = calc_bit_ent(t);
    t->efficiency = calculate_efficiency(t);
}

void decode_tnsr(BitTensor* t, char* buffer, uint16_t buf_size) {
    if (!t || !buffer || buf_size == 0) return;
    uint16_t max_cols = t->cols < buf_size - 1 ? t->cols : buf_size - 1;
    for (uint16_t i = 0; i < max_cols; i++) {
        uint8_t byte = 0;
        for (uint8_t bit = 0; bit < 8; bit++) {
            if (bit < t->rows) {
                uint32_t bit_idx = bit * t->cols + i;
                if (BIT_GET(t->data[bit_idx / 8], bit_idx % 8)) {
                    byte |= (1 << bit);
                }
            }
        }
        if (byte >= 32 && byte <= 126) {
            buffer[i] = (char)byte;
        } else if (byte == 0) {
            buffer[i] = ' ';
        } else {
            buffer[i] = '.';
        }
    }
    buffer[max_cols] = '\0';
}

void update_bit_net_with_goals(void) {
    uint32_t now = (uint32_t)time(NULL);
    if (now % 5 == 0) {
        decay_unused_links();
    }
    if (now - last_mem_check_ts >= MEM_REDUCE_INTERVAL) {
        last_mem_check_ts = now;
        for (uint16_t i = 0; i < tnsr_count; i++) {
            BitTensor* t = &tnsrs[i];
            if (t->act < LOW_ACT_THRESHOLD && !t->mem_red) {
                reduce_tnsr_mem(t);
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
            uint16_t activation_boost = (lnk->weight * interaction * lnk->strength) >> 16;
            lnk->src->act += activation_boost >> 2;
            if (lnk->src->act > ACT_MAX) lnk->src->act = ACT_MAX;
            lnk->tgt->act += activation_boost;
            if (lnk->tgt->act > ACT_MAX) lnk->tgt->act = ACT_MAX;
            lnk->last_act = now;
            total_res_sum += lnk->res * lnk->strength;
            active_links++;
            add_to_working_memory(lnk->tgt);
        }
    }
    if (now % 15 == 0) {
        for (uint16_t i = 0; i < tnsr_count; i++) {
            if (tnsrs[i].act > 100 && tnsrs[i].conn < 5) {
                for (uint16_t j = 0; j < tnsr_count; j++) {
                    if (i != j && tnsrs[j].act > 50) {
                        uint8_t sim = calc_bit_sim(&tnsrs[i], &tnsrs[j]);
                        if (sim > 100) {
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
}

BitTensor* find_efficient_match(BitTensor* input) {
    if (!input) return NULL;
    uint16_t best_score = 0;
    BitTensor* best_match = NULL;
    BitLink* best_link = NULL;
    uint32_t now = (uint32_t)time(NULL);
    for (uint16_t i = 0; i < lnk_count; i++) {
        if (lnks[i].src == input && lnks[i].strength > 20) {
            BitTensor* t = lnks[i].tgt;
            uint32_t time_since_act = now - t->lu;
            uint8_t freshness = (time_since_act > 300) ? 0 : 255 - (time_since_act / 2);
            uint8_t connectivity = (t->conn > 10) ? 255 : (t->conn * 25);
            uint8_t sim = calc_bit_sim(input, t);
            uint8_t res_match = calc_res_match(input, t);
            uint8_t efficiency_boost = (t->efficiency > goals.target_efficiency) ? 255 : t->efficiency;
            uint16_t link_bonus = lnks[i].strength * 2;
            if (lnks[i].use_count > 5) {
                link_bonus += (lnks[i].success_count * 100) / lnks[i].use_count;
            }
            uint32_t score = (uint32_t)sim * res_match;
            score = (score * efficiency_boost) >> 8;
            score = (score * freshness) >> 8;
            score = (score * connectivity) >> 8;
            score += link_bonus;
            uint32_t link_age = now - lnks[i].last_act;
            if (link_age > 60) {
                score = score * (100 - (link_age / 3)) / 100;
            }
            if (score > best_score && score > 3000) {
                best_score = (uint16_t)score;
                best_match = t;
                best_link = &lnks[i];
            }
        }
    }
    if (!best_match || best_score < 5000) {
        for (uint16_t i = 0; i < tnsr_count; i++) {
            BitTensor* t = &tnsrs[i];
            if (t == input || t->act < 20 || t->dropout) continue;
            uint32_t time_since_act = now - t->lu;
            uint8_t freshness = (time_since_act > 300) ? 0 : 255 - (time_since_act / 2);
            uint8_t connectivity = (t->conn > 10) ? 255 : (t->conn * 25);
            uint8_t sim = calc_bit_sim(input, t);
            uint8_t res_match = calc_res_match(input, t);
            uint8_t efficiency_boost = (t->efficiency > goals.target_efficiency) ? 255 : t->efficiency;
            uint16_t recency_bonus = (t->act > 100) ? t->act : 0;
            uint32_t score = (uint32_t)sim * res_match;
            score = (score * efficiency_boost) >> 8;
            score = (score * freshness) >> 8;
            score = (score * connectivity) >> 8;
            score += recency_bonus;
            if (goals.energy_saving_mode && t->compute_cost > 1000) {
                score = score * 3 / 4;
            }
            if (t->stab < 100) {
                score = score * t->stab / 100;
            }
            if (score > best_score && score > 3000) {
                best_score = (uint16_t)score;
                best_match = t;
            }
        }
        if (best_match && best_score > 4000) {
            best_link = create_link(input, best_match);
        }
    }
    if (best_match) {
        best_match->lu = now;
        input->lu = now;
        add_to_working_memory(best_match);
        if (best_link) {
            update_link_strength(best_link, 1);
            if (best_score > 6000) {
                best_link->weight += best_link->weight / 10;
                if (best_link->weight > 65500) best_link->weight = 65500;
            }
        }
        uint8_t activation_boost = best_score / 200;
        if (activation_boost > 50) activation_boost = 50;
        best_match->act += activation_boost;
        if (best_match->act > ACT_MAX) best_match->act = ACT_MAX;
    }
    return best_match;
}

void proc_bit_input(const char* input) {
    if (!input || !*input) return;
    size_t input_len = strlen(input);
    proc_bit_input_raw((const uint8_t*)input, (uint16_t)input_len);
}

void proc_bit_input_raw(const uint8_t* binary, uint16_t input_len) {
    if (!binary || input_len == 0) return;
    BitTensor* input_tnsr = create_tnsr(8, input_len * ENCODER_QUALITY);
    if (!input_tnsr) { printf("Error creating input tensor.\n"); return; }
    encode_tnsr(input_tnsr, binary, input_len);
    input_tnsr->res = 150;
    input_tnsr->act = 180;
    input_tnsr->efficiency = calculate_efficiency(input_tnsr);
    add_to_working_memory(input_tnsr);
    BitTensor* match_tnsr = find_efficient_match(input_tnsr);
    if (match_tnsr) {
        learn_by_binary_update(match_tnsr, binary, input_len);
    }
    update_bit_net_with_goals();
}

void learn_by_binary_update(BitTensor* target, const uint8_t* input_data, uint16_t input_len) {
    if (!target || !input_data || input_len == 0) return;
    uint32_t total_bits = target->rows * target->cols;
    uint32_t total_bytes = (total_bits + 7) / 8;
    if (total_bytes == 0) return;
    for (uint32_t i = 0; i < total_bytes; i++) {
        uint8_t input_byte = input_data[i % input_len];
        uint8_t old_byte = target->data[i];
        uint8_t new_byte = BIT_XOR(old_byte, input_byte);
        new_byte = BIT_AND(new_byte, input_byte | 0x7F);
        if (i % 2 == 0) {
            new_byte = (new_byte & 0xF0) | BIT_NOT(new_byte & 0x0F);
        }
        target->data[i] = new_byte;
    }
    target->ent = calc_bit_ent(target);
    target->act = (target->act + 20 > ACT_MAX) ? ACT_MAX : target->act + 20;
    target->res = (target->res + 5 > RES_MAX) ? RES_MAX : target->res + 5;
    target->efficiency = calculate_efficiency(target);
    target->lu = (uint32_t)time(NULL);
    if (target->act > 220) {
        prevent_overfitting_by_bit_shift(target);
    }
}

void prevent_overfitting_by_bit_shift(BitTensor* t) {
    if (!t || t->act < 200 || t->ent > 120) return;
    uint32_t total_bits = t->rows * t->cols;
    uint32_t total_bytes = (total_bits + 7) / 8;
    if (total_bytes == 0) return;
    uint8_t mutation_count = (t->act - 200) / 10;
    if (mutation_count == 0) mutation_count = 1;
    for (int m = 0; m < mutation_count; m++) {
        uint32_t bit_pos = rand() % total_bits;
        uint32_t byte_idx = bit_pos / 8;
        uint8_t bit_idx = bit_pos % 8;
        t->data[byte_idx] ^= (1 << bit_idx);
    }
    t->ent = calc_bit_ent(t);
    t->act = t->act * 0.8;
    t->res = t->res * 0.95;
    t->efficiency = calculate_efficiency(t);
}

void self_reflect_on_thought(void) {
    BitTensor* most_active = get_most_active_tensor();
    if (!most_active) return;
    BitLink* self_link = NULL;
    for (uint16_t i = 0; i < lnk_count; i++) {
        if (lnks[i].src == most_active && lnks[i].tgt == most_active) {
            self_link = &lnks[i];
            break;
        }
    }
    if (!self_link) {
        self_link = create_link(most_active, most_active);
    }
    if (self_link) {
        update_link_strength(self_link, 1);
        uint32_t total_bits = most_active->rows * most_active->cols;
        uint32_t total_bytes = (total_bits + 7) / 8;
        for (uint32_t i = 0; i < total_bytes; i++) {
            uint8_t old_byte = most_active->data[i];
            uint8_t shift_byte = (old_byte << 1) | (old_byte >> 7);
            most_active->data[i] = BIT_XOR(old_byte, shift_byte);
        }
        most_active->ent = calc_bit_ent(most_active);
        most_active->act = (most_active->act * 1.1 > ACT_MAX) ? ACT_MAX : (uint8_t)(most_active->act * 1.1);
        most_active->efficiency = calculate_efficiency(most_active);
    }
}

void update_thought_stream(void) {
    uint32_t now = (uint32_t)time(NULL);
    update_bit_net_with_goals();
    for (uint16_t i = 0; i < lnk_count; i++) {
        BitLink* lnk = &lnks[i];
        if (lnk->strength > 25 && !lnk->src->dropout && !lnk->tgt->dropout) {
            uint8_t src_act = lnk->src->act;
            uint8_t tgt_act = lnk->tgt->act;
            uint8_t interaction = (src_act & tgt_act) + ((src_act ^ tgt_act) >> 1);
            uint8_t was_successful = (interaction > 128) ? 1 : 0;
            update_link_strength(lnk, was_successful);
            uint16_t activation_boost = (lnk->weight * interaction * lnk->strength) >> 16;
            lnk->src->act += activation_boost >> 2;
            if (lnk->src->act > ACT_MAX) lnk->src->act = ACT_MAX;
            lnk->tgt->act += activation_boost;
            if (lnk->tgt->act > ACT_MAX) lnk->tgt->act = ACT_MAX;
            lnk->last_act = now;
        }
    }
    for (uint16_t i = 0; i < tnsr_count; i++) {
        prevent_overfitting_by_bit_shift(&tnsrs[i]);
    }
    self_reflect_on_thought();
}

BitTensor* get_most_active_tensor(void) {
    BitTensor* most_active = NULL;
    uint8_t max_act = 0;
    for (uint16_t i = 0; i < tnsr_count; i++) {
        if (tnsrs[i].act > max_act && !tnsrs[i].dropout) {
            max_act = tnsrs[i].act;
            most_active = &tnsrs[i];
        }
    }
    return most_active;
}

BitTensor* get_resonant_tensor(void) {
    if (tnsr_count < 2 || lnk_count == 0) return NULL;
    uint32_t now = (uint32_t)time(NULL);
    uint32_t best_loop_score = 0;
    BitTensor* best_candidate = NULL;
    for (uint16_t i = 0; i < lnk_count; i++) {
        BitLink* link_ab = &lnks[i];
        if (!link_ab->src || !link_ab->tgt || link_ab->strength < 40) continue;
        BitLink* link_ba = NULL;
        for (uint16_t j = 0; j < lnk_count; j++) {
            BitLink* candidate = &lnks[j];
            if (candidate->src == link_ab->tgt && candidate->tgt == link_ab->src) {
                link_ba = candidate;
                break;
            }
        }
        if (!link_ba || link_ba->strength < 40) continue;
        BitTensor* A = link_ab->src;
        BitTensor* B = link_ab->tgt;
        if (A->act < 30 || B->act < 30 || A->dropout || B->dropout) continue;
        if (now - A->lu > 120 || now - B->lu > 120) continue;
        uint32_t mutual_strength = (uint32_t)link_ab->strength * link_ba->strength;
        uint16_t avg_act = (A->act + B->act) / 2;
        uint16_t avg_res = (A->res + B->res) / 2;
        uint16_t avg_eff = (A->efficiency + B->efficiency) / 2;
        uint16_t coact_bonus = (A->act * B->act) >> 8;
        uint32_t score = mutual_strength;
        score = (score * avg_act) >> 8;
        score = (score * avg_res) >> 8;
        score = (score * avg_eff) >> 8;
        score += (coact_bonus << 4);
        uint32_t age_bonus = 255 - (now - link_ab->last_act) / 2;
        if (age_bonus > 255) age_bonus = 255;
        if (age_bonus < 0) age_bonus = 0;
        score = (score * age_bonus) >> 8;
        if (score > best_loop_score && score > 5000) {
            best_loop_score = score;
            best_candidate = (A->compute_cost <= B->compute_cost) ? A : B;
        }
    }
    if (!best_candidate) {
        for (uint16_t i = 0; i < lnk_count; i++) {
            BitLink* link = &lnks[i];
            if (!link->src || !link->tgt) continue;
            if (link->strength < 80 || link->tgt->act < 60 || link->tgt->dropout) continue;
            uint32_t score = (uint32_t)link->strength * link->tgt->act;
            if (score > best_loop_score) {
                best_loop_score = score;
                best_candidate = link->tgt;
            }
        }
    }
    if (!best_candidate) {
        for (uint16_t i = 0; i < tnsr_count; i++) {
            BitTensor* t = &tnsrs[i];
            if (!t->data || t->act < 20 || t->dropout || t->conn < 2) continue;
            uint16_t res_score = (uint16_t)t->res * t->act * t->conn;
            if (res_score > best_loop_score) {
                best_loop_score = res_score;
                best_candidate = t;
            }
        }
    }
    if (best_candidate && best_loop_score > 20000) {
        best_candidate->stab = (best_candidate->stab * 7 + 200) >> 3;
    }
    return best_candidate;
}

void save_tnsr(BitTensor* t) {
    if (!t || !t->data || memo_size >= MAX_MEM_ENTRIES) return;
    uint16_t total_bits = t->rows * t->cols;
    uint8_t total_bytes = (uint8_t)((total_bits + 7) / 8);
    if (total_bytes > MAX_PATTERN) return;
    for (uint16_t i = 0; i < memo_size; i++) {
        if (memo[i].len == total_bytes && memcmp(memo[i].data, t->data, total_bytes) == 0) {
            memo[i].count++;
            memo[i].res = (memo[i].res * 230 + t->res * 25) >> 8;
            memo[i].act = (memo[i].act * 230 + t->act * 25) >> 8;
            memo[i].ent = (memo[i].ent * 230 + t->ent * 25) >> 8;
            memo[i].ts = (uint32_t)time(NULL);
            return;
        }
    }
    BitMemory* entry = &memo[memo_size++];
    memcpy(entry->data, t->data, total_bytes);
    entry->len = total_bytes;
    entry->count = 1;
    entry->res = t->res;
    entry->act = t->act;
    entry->ent = t->ent;
    entry->ts = (uint32_t)time(NULL);
}

// ===== Сохранение/Загрузка =====

int save_state_to_file(const char* filename) {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open(save)"); return -1; }

#define WRITE(ptr, size, count) \
    if (write(fd, (ptr), (size_t)(size)*(count)) != (ssize_t)((size_t)(size)*(count))) { \
        perror("write"); close(fd); return -1; }

    WRITE(&tnsr_count, sizeof(uint16_t), 1);
    WRITE(&lnk_count, sizeof(uint16_t), 1);
    WRITE(&memo_size, sizeof(uint16_t), 1);
    WRITE(&tt_count, sizeof(uint16_t), 1);
    WRITE(&working_mem_count, sizeof(uint8_t), 1);
    WRITE(&goals, sizeof(SystemGoals), 1);
    WRITE(&sys_res, sizeof(uint8_t), 1);
    WRITE(&interaction_count, sizeof(uint32_t), 1);
    WRITE(&last_mem_check_ts, sizeof(uint32_t), 1);
    WRITE(&sstate, sizeof(BitSystemState), 1);

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
    }

    WRITE(memo, sizeof(BitMemory), memo_size);

    for (uint16_t i = 0; i < tt_count; i++) {
        BitTensorTensor* tt = &t_tnsrs[i];
        WRITE(&tt->enc_type, sizeof(uint8_t), 1);
        WRITE(&tt->num_tensors, sizeof(uint16_t), 1);
        WRITE(&tt->bits_per_idx, sizeof(uint8_t), 1);
        WRITE(&tt->res, sizeof(uint8_t), 1);
        WRITE(&tt->act, sizeof(uint8_t), 1);
        WRITE(&tt->lu, sizeof(uint32_t), 1);
        WRITE(&tt->efficiency, sizeof(uint8_t), 1);
        uint32_t data_bytes = (tt->enc_type == 1)
            ? ((uint32_t)tt->num_tensors * tt->bits_per_idx + 7) / 8
            : (tt->num_tensors + 7) / 8;
        WRITE(&data_bytes, sizeof(uint32_t), 1);
        if (data_bytes > 0 && tt->data) {
            WRITE(tt->data, 1, data_bytes);
        }
        if (tt->enc_type == 1 && tt->tensor_indices) {
            WRITE(tt->tensor_indices, sizeof(uint16_t), tt->num_tensors);
        }
    }

    for (uint8_t i = 0; i < working_mem_count; i++) {
        uint16_t t_idx = tensor_to_index(working_mem[i].tensor);
        WRITE(&t_idx, sizeof(uint16_t), 1);
        WRITE(&working_mem[i].timestamp, sizeof(uint32_t), 1);
        WRITE(&working_mem[i].priority, sizeof(uint8_t), 1);
        WRITE(&working_mem[i].access_count, sizeof(uint8_t), 1);
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
    for (uint16_t i = 0; i < tt_count; i++) {
        free(t_tnsrs[i].data);
        free(t_tnsrs[i].tensor_indices);
    }
    memset(tnsrs, 0, sizeof(tnsrs));
    memset(lnks, 0, sizeof(lnks));
    memset(memo, 0, sizeof(memo));
    memset(t_tnsrs, 0, sizeof(t_tnsrs));
    memset(working_mem, 0, sizeof(working_mem));
    tnsr_count = lnk_count = memo_size = tt_count = working_mem_count = 0;
    sys_res = RES_HALF;

#define READ(ptr, size, count) \
    if (read(fd, (ptr), (size_t)(size)*(count)) != (ssize_t)((size_t)(size)*(count))) { \
        perror("read"); close(fd); return -1; }

    READ(&tnsr_count, sizeof(uint16_t), 1);
    READ(&lnk_count, sizeof(uint16_t), 1);
    READ(&memo_size, sizeof(uint16_t), 1);
    READ(&tt_count, sizeof(uint16_t), 1);
    READ(&working_mem_count, sizeof(uint8_t), 1);

    if (tnsr_count > MAX_TENSORS || lnk_count > MAX_LINKS ||
        memo_size > MAX_MEM_ENTRIES || tt_count > MAX_TT_ENTRIES ||
        working_mem_count > WORKING_MEM_SIZE) {
        fprintf(stderr, "[ERR] Corrupted state: count overflow\n");
        close(fd); return -1;
    }

    READ(&goals, sizeof(SystemGoals), 1);
    READ(&sys_res, sizeof(uint8_t), 1);
    READ(&interaction_count, sizeof(uint32_t), 1);
    READ(&last_mem_check_ts, sizeof(uint32_t), 1);
    READ(&sstate, sizeof(BitSystemState), 1);

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

        uint32_t data_bytes;
        READ(&data_bytes, sizeof(uint32_t), 1);
        if (data_bytes > 0) {
            t->data = (uint8_t*)malloc(data_bytes);
            if (!t->data) { close(fd); return -1; }
            READ(t->data, 1, data_bytes);
        } else {
            t->data = NULL;
        }
    }

    struct {
        uint16_t src_idx, tgt_idx;
        uint8_t strength, res;
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
    }

    READ(memo, sizeof(BitMemory), memo_size);

    for (uint16_t i = 0; i < tt_count; i++) {
        BitTensorTensor* tt = &t_tnsrs[i];
        READ(&tt->enc_type, sizeof(uint8_t), 1);
        READ(&tt->num_tensors, sizeof(uint16_t), 1);
        READ(&tt->bits_per_idx, sizeof(uint8_t), 1);
        READ(&tt->res, sizeof(uint8_t), 1);
        READ(&tt->act, sizeof(uint8_t), 1);
        READ(&tt->lu, sizeof(uint32_t), 1);
        READ(&tt->efficiency, sizeof(uint8_t), 1);

        uint32_t data_bytes;
        READ(&data_bytes, sizeof(uint32_t), 1);
        if (data_bytes > 0) {
            tt->data = (uint8_t*)malloc(data_bytes);
            if (!tt->data) { close(fd); return -1; }
            READ(tt->data, 1, data_bytes);
        }

        if (tt->enc_type == 1) {
            tt->tensor_indices = (uint16_t*)malloc(tt->num_tensors * sizeof(uint16_t));
            if (!tt->tensor_indices) { close(fd); return -1; }
            READ(tt->tensor_indices, sizeof(uint16_t), tt->num_tensors);
        } else {
            tt->tensor_indices = NULL;
        }
    }

    for (uint8_t i = 0; i < working_mem_count; i++) {
        uint16_t t_idx;
        READ(&t_idx, sizeof(uint16_t), 1);
        READ(&working_mem[i].timestamp, sizeof(uint32_t), 1);
        READ(&working_mem[i].priority, sizeof(uint8_t), 1);
        READ(&working_mem[i].access_count, sizeof(uint8_t), 1);
        working_mem[i].tensor = index_to_tensor(t_idx);
    }

    for (uint16_t i = 0; i < lnk_count; i++) {
        BitLink* l = &lnks[i];
        l->src = index_to_tensor(link_buf[i].src_idx);
        l->tgt = index_to_tensor(link_buf[i].tgt_idx);
        l->strength     = link_buf[i].strength;
        l->res          = link_buf[i].res;
        l->weight       = link_buf[i].weight;
        l->ts           = link_buf[i].ts;
        l->last_act     = link_buf[i].last_act;
        l->use_count    = link_buf[i].use_count;
        l->success_count= link_buf[i].success_count;
    }

    close(fd);
    return 0;
}