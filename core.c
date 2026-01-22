// core.c/h - универсальный модуль для работы с бинарными тензорами с самоорганизацией памяти
/*
 * AGI_CORE - Core AGI Engine
 * Copyright (C) 2025 makushkin viktor 
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */
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
#define CLUSTER_THRESHOLD 150  // Порог схожести для кластеризации (0-255)
#define MERGE_THRESHOLD 180    // Порог для слияния тензоров
#define CONCEPT_CREATION_THRESHOLD 3  // Минимальный размер кластера для концепции
#define EPISODE_MIN_LENGTH 3   // Минимальная длина эпизода
#define CONSOLIDATION_INTERVAL 30  // Интервал консолидации (сек)
#define SELF_ORG_INTERVAL 5    // Интервал самоорганизации (сек)
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
};
struct SystemGoals {
    uint8_t target_efficiency;
    uint8_t energy_saving_mode;
    uint32_t total_compute_cost;
    uint32_t efficiency_gain;
    uint8_t dropout_enabled;
    uint8_t self_organization_enabled;  // Включена ли самоорганизация
    uint8_t memory_consolidation_mode;  // Режим консолидации: 0=авто, 1=агрессивный, 2=консервативный
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
// ===== Глобальные состояния =====
BitMemory memo[MAX_MEM_ENTRIES];
BitTensor tnsrs[MAX_TENSORS];
BitLink lnks[MAX_LINKS];
WorkingMemoryEntry working_mem[WORKING_MEM_SIZE];
TensorLinks tensor_links[MAX_TENSORS];
BitSystemState sstate;
SystemGoals goals = {180, 0, 0, 0, 1, 1, 0};  // self_organization_enabled=1 по умолчанию
// === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ САМООРГАНИЗАЦИИ ===
MemoryCluster clusters[MAX_CLUSTERS];
EpisodeMemory episodes[MAX_EPISODES];
MemoryConcept concepts[64];
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

// ===== Реализации вспомогательных функций =====
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

uint8_t calc_bit_ent(BitTensor* t, uint32_t unit_sz) {
    if (!t || !t->data || unit_sz == 0) return 0;
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
    return (uint8_t)(h_fixed * unit_sz);
}

uint8_t calc_bit_sim(BitTensor* a, BitTensor* b) {
    if (!a || !b || !a->data || !b->data) return 0;

    uint32_t bits_a = a->rows * a->cols;
    uint32_t bits_b = b->rows * b->cols;

    if (bits_a == 0 || bits_b == 0) return 0;

    uint32_t max_bits = (bits_a > bits_b) ? bits_a : bits_b;
    uint32_t min_bits = (bits_a < bits_b) ? bits_b : bits_b;
    uint32_t max_bytes = (max_bits + 7) / 8;

    uint8_t* buf_a = (uint8_t*)calloc(max_bytes, 1);
    uint8_t* buf_b = (uint8_t*)calloc(max_bytes, 1);

    if (!buf_a || !buf_b) {
        if (buf_a) free(buf_a);
        if (buf_b) free(buf_b);
        return 0;
    }

    uint32_t bytes_a = (bits_a + 7) / 8;
    uint32_t bytes_b = (bits_b + 7) / 8;
    uint32_t copy_bytes_a = (bytes_a < max_bytes) ? bytes_a : max_bytes;
    uint32_t copy_bytes_b = (bytes_b < max_bytes) ? bytes_b : max_bytes;

    memcpy(buf_a, a->data, copy_bytes_a);
    memcpy(buf_b, b->data, copy_bytes_b);

    if (bits_a < max_bits) {
        for (uint32_t i = bits_a; i < max_bits; i++) {
            uint32_t src_bit = i % bits_a;
            uint32_t src_byte = src_bit / 8;
            uint8_t src_bit_idx = src_bit % 8;
            uint8_t val = (buf_a[src_byte] >> src_bit_idx) & 1;
            uint32_t dst_byte = i / 8;
            uint8_t dst_bit_idx = i % 8;
            if (val) buf_a[dst_byte] |= (1U << dst_bit_idx);
        }
    }

    if (bits_b < max_bits) {
        for (uint32_t i = bits_b; i < max_bits; i++) {
            uint32_t src_bit = i % bits_b;
            uint32_t src_byte = src_bit / 8;
            uint8_t src_bit_idx = src_bit % 8;
            uint8_t val = (buf_b[src_byte] >> src_bit_idx) & 1;
            uint32_t dst_byte = i / 8;
            uint8_t dst_bit_idx = i % 8;
            if (val) buf_b[dst_byte] |= (1U << dst_bit_idx);
        }
    }

    float fuzziness_factor = ((float)SIM_FUZZINESS_PERCENT) / 100.0f;
    float max_possible_similarity = 255.0f;
    float fuzziness_threshold = fuzziness_factor * max_possible_similarity;

    float weighted_similarity_sum = 0.0f;
    float total_weight = 0.0f;

    for (uint32_t i = 0; i < min_bits; i++) {
        uint32_t byte_idx = i / 8;
        uint8_t bit_idx = i % 8;
        if (byte_idx >= max_bytes) break;

        uint8_t bit_a = (buf_a[byte_idx] >> bit_idx) & 1;
        uint8_t bit_b = (buf_b[byte_idx] >> bit_idx) & 1;

        float actual_similarity = (bit_a == bit_b) ? max_possible_similarity : fuzziness_threshold;
        float bit_weight = ((float)(a->act + b->act) / 2.0f) * ((float)(a->stab + b->stab) / 2.0f) / 255.0f;

        if (i < bits_a && i < bits_b) {
            bit_weight *= 1.5f;
        }

        weighted_similarity_sum += actual_similarity * bit_weight;
        total_weight += bit_weight;
    }

    free(buf_a);
    free(buf_b);

    if (total_weight == 0.0f) return 0;

    float avg_weighted_similarity = weighted_similarity_sum / total_weight;
    float scaled_sim = avg_weighted_similarity / (max_possible_similarity / 5.0f);
    float tanh_result = tanhf(scaled_sim);
    uint8_t final_similarity = (uint8_t)(tanh_result * 255.0f);

    return (final_similarity > 255) ? 255 : final_similarity;
}

uint8_t calc_res_match(BitTensor* a, BitTensor* b) {
    if (!a || !b) return 0;
    uint8_t res_diff = (a->res > b->res) ? (a->res - b->res) : (b->res - a->res);
    uint8_t act_diff = (a->act > b->act) ? (a->act - b->act) : (b->act - a->act);
    uint8_t ent_diff = (a->ent > b->ent) ? (a->ent - b->ent) : (b->ent - a->ent);
    uint8_t base_diff_score = (res_diff + act_diff + ent_diff) / 3;
    uint8_t base_similarity = 255 - base_diff_score;
    uint8_t stab_diff = (a->stab > b->stab) ? (a->stab - b->stab) : (b->stab - a->stab);
    uint8_t stab_bonus = 255 - stab_diff;
    uint8_t final_similarity = (base_similarity + stab_bonus) / 2;
    return final_similarity;
}

uint8_t calculate_efficiency(BitTensor* t) {
    if (!t) return 0;
    float base_benefit = (float)t->act * t->res * (255 - t->ent);
    float stability_bonus = sqrtf(t->stab) * 10.0f;
    float connectivity_bonus = 0.0f;
    if (t->conn < 50) {
        connectivity_bonus = sqrtf(t->conn) * 5.0f;
    }
    float total_benefit = base_benefit + stability_bonus + connectivity_bonus;
    float cost = (float)t->compute_cost + 1.0f;
    if (t->mem_red) cost *= 1.1f;
    float efficiency_score = total_benefit / cost;
    if (efficiency_score > 255.0f) return 255;
    return (uint8_t)efficiency_score;
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
    t->res = RES_HALF;      // Нейтрально
    t->act = 50;            // Низкая начальная активность
    t->ent = 0;
    t->stab = 100;          // Немного ниже среднего
    t->conn = 0;
    t->lu = (uint32_t)time(NULL);
    t->mem_red = 0;
    t->efficiency = 50;     // Низкая начальная эффективность
    t->compute_cost = rows * cols * 2;
    t->goal_active = (rows * cols > 100) ? 1 : 0;
    t->dropout = (rand() % 100 < DROPOUT_RATE) ? 1 : 0;
    t->cluster_id = 0;      // Не принадлежит кластеру
    t->is_concept = 0;      // Не концепция по умолчанию

    // Пересчитываем эффективность после установки базовых параметров
    t->efficiency = calculate_efficiency(t);

    return t;
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
    t->ent = calc_bit_ent(t, t->cols);
    t->compute_cost = t->rows * t->cols * 2;
    t->efficiency = calculate_efficiency(t);
    t->mem_red = 1;
}

void add_to_working_memory(BitTensor* t) {
    if (!t || working_mem_count >= WORKING_MEM_SIZE) return;
    uint8_t existing_idx = 255;
    for (uint8_t i = 0; i < working_mem_count; i++) {
        if (working_mem[i].tensor == t) {
            existing_idx = i; break;
        }
    }
    uint32_t now = (uint32_t)time(NULL);
    if (existing_idx != 255) {
        working_mem[existing_idx].access_count++;
        working_mem[existing_idx].priority = (working_mem[existing_idx].priority * 7 + 100) >> 3;
        working_mem[existing_idx].timestamp = now;
        working_mem[existing_idx].episode_marker = 1;  // Отмечаем для эпизода
    } else {
        working_mem[working_mem_count].tensor = t;
        working_mem[working_mem_count].timestamp = now;
        working_mem[working_mem_count].priority = 100;
        working_mem[working_mem_count].access_count = 1;
        working_mem[working_mem_count].episode_marker = 1;
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
        working_mem[best_idx].episode_marker = 1;
        return working_mem[best_idx].tensor;
    }
    return NULL;
}

// === ФУНКЦИИ САМООРГАНИЗАЦИИ ПАМЯТИ ===

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
    uint8_t similarity_matrix[256][256] = {0};  // Ограничиваем 256 тензорами для производительности

    uint16_t limit = (active_count > 256) ? 256 : active_count;
    for (uint16_t i = 0; i < limit; i++) {
        for (uint16_t j = i + 1; j < limit; j++) {
            uint8_t sim = calc_bit_sim(active_tensors[i], active_tensors[j]);
            similarity_matrix[i][j] = sim;
            similarity_matrix[j][i] = sim;
        }
    }

    // 3. Алгоритм агломеративной кластеризации (упрощенный)
    uint8_t cluster_assignments[256] = {0};
    uint8_t current_cluster_id = 1;

    for (uint16_t i = 0; i < limit; i++) {
        if (cluster_assignments[i] == 0) {
            // Начинаем новый кластер с этого тензора
            cluster_assignments[i] = current_cluster_id;

            // Добавляем похожие тензоры в этот кластер
            for (uint16_t j = i + 1; j < limit; j++) {
                if (cluster_assignments[j] == 0 &&
                    similarity_matrix[i][j] > CLUSTER_THRESHOLD) {
                    cluster_assignments[j] = current_cluster_id;
                }
            }

            current_cluster_id++;
            if (current_cluster_id >= 128) break;  // Максимум кластеров
        }
    }

    // 4. Обновляем или создаем кластеры
    for (uint8_t cid = 1; cid < current_cluster_id; cid++) {
        uint16_t cluster_size = 0;
        uint16_t tensor_indices[64];
        uint16_t cluster_act_sum = 0;

        // Собираем тензоры этого кластера
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

        if (cluster_size >= 2) {  // Минимум 2 тензора для кластера
            // Ищем существующий кластер с пересечением
            MemoryCluster* cluster = NULL;
            uint8_t found_existing = 0;

            for (uint16_t ci = 0; ci < cluster_count; ci++) {
                // Проверяем overlap с существующим кластером
                uint8_t overlap = 0;
                for (uint16_t k = 0; k < cluster_size; k++) {
                    for (uint16_t l = 0; l < clusters[ci].size; l++) {
                        if (tensor_indices[k] == clusters[ci].tensor_indices[l]) {
                            overlap++;
                            break;
                        }
                    }
                }

                // Если значительное пересечение (более 50%), обновляем существующий
                if (overlap >= cluster_size / 2 || overlap >= clusters[ci].size / 2) {
                    cluster = &clusters[ci];
                    found_existing = 1;
                    break;
                }
            }

            // Создаем новый кластер если не нашли похожего
            if (!found_existing && cluster_count < MAX_CLUSTERS) {
                cluster = &clusters[cluster_count++];
                cluster->cluster_id = next_cluster_id++;
                if (next_cluster_id == 0) next_cluster_id = 1;  // Зацикливаем
                cluster->size = 0;
                cluster->stability = 100;
                cluster->last_access = now;
                cluster->creation_time = now;
                cluster->activation_level = (uint8_t)(cluster_act_sum / cluster_size);
                cluster->link_count = 0;
                cluster->category = 0;  // Пока неопределено
            }

            if (cluster) {
                // Обновляем кластер
                cluster->size = cluster_size;
                for (uint16_t k = 0; k < cluster_size; k++) {
                    cluster->tensor_indices[k] = tensor_indices[k];
                    tnsrs[tensor_indices[k]].cluster_id = cluster->cluster_id;
                }
                cluster->last_access = now;
                cluster->activation_level = (uint8_t)(cluster_act_sum / cluster_size);

                // Обновляем центроид (усредненный вектор)
                memset(cluster->centroid, 0, 256);
                for (uint16_t k = 0; k < cluster_size; k++) {
                    BitTensor* t = &tnsrs[tensor_indices[k]];
                    uint32_t bits = t->rows * t->cols;
                    uint32_t bytes = (bits + 7) / 8;
                    for (uint32_t b = 0; b < bytes && b < 256; b++) {
                        cluster->centroid[b] = (cluster->centroid[b] * (cluster_size - 1) + t->data[b]) / cluster_size;
                    }
                }

                // Обновляем стабильность
                update_cluster_stability(cluster);
            }
        }
    }

    // 5. Удаляем пустые/старые кластеры
    prune_weak_clusters();

    last_cluster_reorg = now;
}

void update_cluster_stability(MemoryCluster* cluster) {
    if (!cluster) return;

    uint32_t now = (uint32_t)time(NULL);
    uint32_t age = now - cluster->creation_time;

    // Базовая стабильность на основе возраста и активности
    uint8_t age_factor = (age > 3600) ? 200 : (uint8_t)(age / 18);  // 1 час = 200
    uint8_t activity_factor = cluster->activation_level;

    // Вычисляем когерентность кластера
    uint8_t coherence = calculate_cluster_coherence(cluster);

    // Итоговая стабильность
    cluster->stability = (age_factor * 3 + activity_factor * 5 + coherence * 2) / 10;

    if (cluster->stability > 255) cluster->stability = 255;
}

uint8_t calculate_cluster_coherence(MemoryCluster* cluster) {
    if (!cluster || cluster->size < 2) return 0;

    uint32_t total_similarity = 0;
    uint32_t comparisons = 0;

    // Вычисляем среднюю похожесть между всеми тензорами в кластере
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

        // Критерии удаления кластера:
        // 1. Слишком старая и неактивная
        // 2. Очень низкая стабильность
        // 3. Слишком маленький и не используется
        if ((now - cluster->last_access > 7200 && cluster->stability < 50) ||  // 2 часа
            cluster->stability < 20 ||
            (cluster->size < 2 && now - cluster->creation_time > 3600)) {

            printf("[MEM] Удаляем кластер %u (стабильность: %u, размер: %u)\n",
                   cluster->cluster_id, cluster->stability, cluster->size);

            // Освобождаем тензоры от принадлежности к кластеру
            for (uint16_t i = 0; i < cluster->size; i++) {
                uint16_t tensor_idx = cluster->tensor_indices[i];
                if (tensor_idx < tnsr_count) {
                    tnsrs[tensor_idx].cluster_id = 0;
                }
            }

            // Удаляем кластер (заменяем последним)
            if (ci < cluster_count - 1) {
                clusters[ci] = clusters[cluster_count - 1];
            }
            cluster_count--;
            ci--;  // Проверяем снова этот индекс
        }
    }
}

BitTensor* find_center_tensor_in_cluster(MemoryCluster* cluster) {
    if (!cluster || cluster->size == 0) return NULL;

    // Находим тензор с максимальной активностью и связностью
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

    // Ищем существующую связь
    for (uint16_t i = 0; i < lnk_count; i++) {
        if ((lnks[i].src == src && lnks[i].tgt == tgt) ||
            (lnks[i].src == tgt && lnks[i].tgt == src)) {
            return &lnks[i];
        }
    }

    // Создаем новую связь
    return create_link(src, tgt);
}

void semantic_memory_binding(void) {
    if (cluster_count < 2) return;

    uint32_t now = (uint32_t)time(NULL);

    // Создаем семантические связи между кластерами
    for (uint16_t i = 0; i < cluster_count; i++) {
        for (uint16_t j = i + 1; j < cluster_count; j++) {
            create_semantic_links_between_clusters(&clusters[i], &clusters[j]);
        }
    }
}

void create_semantic_links_between_clusters(MemoryCluster* c1, MemoryCluster* c2) {
    if (!c1 || !c2 || c1 == c2) return;

    uint32_t now = (uint32_t)time(NULL);

    // 1. Проверяем временную близость доступа
    uint32_t time_diff = (c1->last_access > c2->last_access) ?
                        (c1->last_access - c2->last_access) :
                        (c2->last_access - c1->last_access);

    // 2. Проверяем overlap через рабочие связи
    uint8_t common_links = 0;
    uint16_t max_check = (c1->size < c2->size) ? c1->size : c2->size;
    max_check = (max_check > 10) ? 10 : max_check;

    for (uint8_t i = 0; i < max_check; i++) {
        uint16_t idx1 = c1->tensor_indices[i];
        uint16_t idx2 = c2->tensor_indices[i];

        if (idx1 < tnsr_count && idx2 < tnsr_count) {
            // Проверяем связи между этими тензорами
            for (uint16_t li = 0; li < lnk_count; li++) {
                if ((lnks[li].src == &tnsrs[idx1] && lnks[li].tgt == &tnsrs[idx2]) ||
                    (lnks[li].src == &tnsrs[idx2] && lnks[li].tgt == &tnsrs[idx1])) {
                    common_links++;
                    break;
                }
            }
        }
    }

    // 3. Вычисляем семантический балл
    uint8_t semantic_score = 0;

    if (time_diff < 5) semantic_score += 80;
    else if (time_diff < 30) semantic_score += 40;

    semantic_score += common_links * 15;

    // 4. Создаем связи при высоком семантическом балле
    if (semantic_score > 70) {
        BitTensor* center1 = find_center_tensor_in_cluster(c1);
        BitTensor* center2 = find_center_tensor_in_cluster(c2);

        if (center1 && center2 && center1 != center2) {
            BitLink* link = find_or_create_link(center1, center2);
            if (link) {
                // Усиливаем связь и отмечаем как семантическую
                link->strength = (link->strength * 7 + 180) >> 3;
                link->res = (link->res * 7 + 200) >> 3;
                link->semantic_type = 2;  // Межкластерная связь
                link->last_act = now;

                // Увеличиваем счетчики связей для кластеров
                c1->link_count++;
                c2->link_count++;
            }
        }
    }
}

void memory_consolidation(void) {
    static uint32_t last_consolidation = 0;
    uint32_t now = (uint32_t)time(NULL);

    // Консолидация по расписанию или при высокой активности
    if (now - last_consolidation < CONSOLIDATION_INTERVAL &&
        interaction_count % 1000 != 0) {
        return;
    }

    last_consolidation = now;

    printf("[MEM] Начало консолидации памяти...\n");
    uint32_t total_before = tnsr_count + lnk_count + memo_size;

    // 1. Рекомбинация слабых тензоров
    for (uint16_t i = 0; i < tnsr_count; i++) {
        for (uint16_t j = i + 1; j < tnsr_count; j++) {
            BitTensor* a = &tnsrs[i];
            BitTensor* b = &tnsrs[j];

            // Критерии для слияния:
            // - Оба слабоактивны
            // - Оба имеют мало связей
            // - Высокая похожесть
            if (a->act < 30 && b->act < 30 &&
                a->conn < 3 && b->conn < 3 &&
                !a->dropout && !b->dropout &&
                !a->is_concept && !b->is_concept) {

                uint8_t sim = calc_bit_sim(a, b);
                if (sim > MERGE_THRESHOLD) {
                    merge_tensors(a, b);
                }
            }
        }
    }

    // 2. Создание концепций из стабильных кластеров
    for (uint16_t ci = 0; ci < cluster_count; ci++) {
        if (clusters[ci].size >= CONCEPT_CREATION_THRESHOLD &&
            clusters[ci].stability > 150) {
            create_concept_from_cluster(&clusters[ci]);
        }
    }

    // 3. Очистка "мусорной" памяти
    aggressive_memory_cleanup();

    // 4. Переиндексация связей
    build_link_index();

    // 5. Создание эпизодной памяти из последовательностей в рабочей памяти
    if (working_mem_count >= EPISODE_MIN_LENGTH) {
        create_episode_from_working_memory();
    }

    // 6. Активация релевантных эпизодов
    activate_relevant_episodes();

    uint32_t total_after = tnsr_count + lnk_count + memo_size;
    printf("[MEM] Консолидация завершена: %u -> %u элементов (экономия: %d%%)\n",
           total_before, total_after,
           (int)(100 - (total_after * 100 / (total_before ? total_before : 1))));
}

void merge_tensors(BitTensor* a, BitTensor* b) {
    if (!a || !b || a == b || !a->data || !b->data) return;

    uint32_t bits_a = a->rows * a->cols;
    uint32_t bits_b = b->rows * b->cols;

    if (bits_a == 0 || bits_b == 0) return;

    // Новый размер - среднее между двумя тензорами
    uint32_t new_bits = (bits_a + bits_b) / 2;
    if (new_bits < 8) new_bits = 8;

    uint16_t new_rows = 16;  // Фиксированная высота
    uint16_t new_cols = (uint16_t)((new_bits + new_rows - 1) / new_rows);

    // Создаем новый объединенный тензор
    BitTensor* merged = create_tnsr(new_rows, new_cols);
    if (!merged) return;

    // Объединяем данные (побитовое ИЛИ + И)
    uint32_t bytes_a = (bits_a + 7) / 8;
    uint32_t bytes_b = (bits_b + 7) / 8;
    uint32_t bytes_merged = (new_bits + 7) / 8;

    // Копируем и комбинируем данные
    for (uint32_t i = 0; i < bytes_merged; i++) {
        uint8_t byte_a = 0, byte_b = 0;

        if (i < bytes_a) byte_a = a->data[i];
        if (i < bytes_b) byte_b = b->data[i];

        // Комбинация: ИЛИ для сильных битов, И для слабых
        uint8_t strong_bits = byte_a | byte_b;
        uint8_t weak_bits = byte_a & byte_b;
        merged->data[i] = strong_bits | (weak_bits >> 2);  // Ослабляем слабые биты
    }

    // Объединяем метаданные
    merged->act = (a->act + b->act) / 2;
    merged->res = (a->res + b->res) / 2;
    merged->ent = calc_bit_ent(merged, merged->cols);
    merged->stab = (a->stab * 3 + b->stab * 2) / 5;  // Взвешенное среднее
    merged->efficiency = calculate_efficiency(merged);
    merged->cluster_id = (a->cluster_id == b->cluster_id) ? a->cluster_id : 0;

    // Переносим связи
    transfer_links(a, merged);
    transfer_links(b, merged);

    // Помечаем старые тензоры для удаления
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

        // Заменяем from на to в качестве источника
        if (link->src == from) {
            link->src = to;
            to->conn++;
            if (from->conn > 0) from->conn--;
        }

        // Заменяем from на to в качестве цели
        if (link->tgt == from) {
            link->tgt = to;
            to->conn++;
            if (from->conn > 0) from->conn--;
        }
    }
}

void create_concept_from_cluster(MemoryCluster* cluster) {
    if (!cluster || cluster->size < CONCEPT_CREATION_THRESHOLD) return;

    // Проверяем, не создана ли уже концепция для этого кластера
    for (uint8_t ci = 0; ci < concept_count; ci++) {
        if (concepts[ci].concept_tensor &&
            concepts[ci].concept_tensor->cluster_id == cluster->cluster_id) {
            return;  // Концепция уже существует
        }
    }

    if (concept_count >= 64) {
        // Удаляем самую старую концепцию
        uint32_t oldest_time = UINT32_MAX;
        uint8_t oldest_idx = 0;

        for (uint8_t ci = 0; ci < concept_count; ci++) {
            if (concepts[ci].last_used < oldest_time) {
                oldest_time = concepts[ci].last_used;
                oldest_idx = ci;
            }
        }

        // Заменяем старую концепцию
        concept_count--;
        for (uint8_t ci = oldest_idx; ci < concept_count; ci++) {
            concepts[ci] = concepts[ci + 1];
        }
    }

    // Создаем тензор-концепцию
    BitTensor* concept = create_tnsr(8, 8);  // Компактный тензор для концепции
    if (!concept) return;

    // Заполняем данными из центроида кластера
    for (uint32_t i = 0; i < 64; i++) {
        if (cluster->centroid[i % 256] > 128) {
            BIT_SET(concept->data[i / 8], i % 8);
        }
    }

    // Устанавливаем метаданные концепции
    concept->act = cluster->stability / 2;  // Средняя активность
    concept->res = 200;
    concept->stab = cluster->stability;
    concept->ent = 60;  // Средняя энтропия для абстракции
    concept->efficiency = 180;
    concept->is_concept = 1;
    concept->cluster_id = cluster->cluster_id;

    // Создаем запись концепции
    MemoryConcept* concept_entry = &concepts[concept_count++];
    concept_entry->concept_tensor = concept;
    concept_entry->member_count = (cluster->size > 32) ? 32 : cluster->size;
    concept_entry->abstraction_level = 1;  // Базовый уровень
    concept_entry->coherence = calculate_cluster_coherence(cluster);
    concept_entry->last_used = (uint32_t)time(NULL);

    // Копируем индексы членов
    for (uint8_t i = 0; i < concept_entry->member_count; i++) {
        concept_entry->member_indices[i] = cluster->tensor_indices[i];
    }

    // Создаем сильные связи с членами кластера
    for (uint8_t i = 0; i < concept_entry->member_count; i++) {
        uint16_t idx = concept_entry->member_indices[i];
        if (idx < tnsr_count) {
            BitLink* link = create_link(concept, &tnsrs[idx]);
            if (link) {
                link->strength = 200;  // Очень сильная связь
                link->res = 220;
                link->semantic_type = 3;  // Концептуальная связь
            }
        }
    }

    // Сохраняем концепцию
    save_concept(concept, cluster->cluster_id);

    printf("[CONCEPT] Создана концепция [%u] из кластера %u (размер: %u, стабильность: %u)\n",
           (uint32_t)(concept - tnsrs), cluster->cluster_id, cluster->size, cluster->stability);
}

void save_concept(BitTensor* concept, uint8_t cluster_id) {
    // Сохраняем концепцию в долговременную память (memo)
    save_tnsr(concept);

    // Можно добавить дополнительное сохранение в специальную структуру
    // Например, в отдельный файл или секцию памяти
}

void create_episode_from_working_memory(void) {
    if (working_mem_count < EPISODE_MIN_LENGTH) return;

    // Проверяем, есть ли новая информация для эпизода
    uint8_t new_episode = 0;
    for (uint8_t i = 0; i < working_mem_count; i++) {
        if (working_mem[i].episode_marker) {
            new_episode = 1;
            break;
        }
    }

    if (!new_episode) return;

    // Освобождаем место для нового эпизода если нужно
    if (episode_count >= MAX_EPISODES) {
        // Ищем наименее важный эпизод для удаления
        uint8_t least_important_idx = 0;
        uint8_t min_importance = 255;

        for (uint16_t i = 0; i < episode_count; i++) {
            if (episodes[i].importance < min_importance) {
                min_importance = episodes[i].importance;
                least_important_idx = i;
            }
        }

        // Удаляем наименее важный эпизод
        for (uint16_t i = least_important_idx; i < episode_count - 1; i++) {
            episodes[i] = episodes[i + 1];
        }
        episode_count--;
    }

    EpisodeMemory* episode = &episodes[episode_count++];
    memset(episode, 0, sizeof(EpisodeMemory));

    // Записываем последовательность из рабочей памяти
    uint8_t seq_len = (working_mem_count < 256) ? working_mem_count : 256;
    episode->length = seq_len;

    for (uint8_t i = 0; i < seq_len; i++) {
        uint16_t tensor_idx = working_mem[i].tensor - tnsrs;
        episode->sequence[i] = tensor_idx;
        working_mem[i].episode_marker = 0;  // Сбрасываем маркер
    }

    // Вычисляем хеш контекста эпизода
    uint32_t context_hash = calculate_context_hash();
    memcpy(episode->context_hash, &context_hash, sizeof(uint32_t));

    episode->start_time = working_mem[0].timestamp;
    episode->end_time = working_mem[seq_len - 1].timestamp;
    episode->success_score = 100;  // Начальная оценка успешности
    episode->importance = 100;     // Начальная важность
    episode->last_recall = 0;
    episode->recall_count = 0;

    printf("[EPISODE] Создан эпизод %u (длина: %u, контекстный хеш: %08X)\n",
           episode_count, seq_len, context_hash);
}

uint32_t calculate_context_hash(void) {
    uint32_t hash = 0x811C9DC5;  // FNV-1a начальное значение

    for (uint8_t i = 0; i < working_mem_count && i < 8; i++) {
        BitTensor* t = working_mem[i].tensor;
        if (t) {
            // Хешируем комбинацию характеристик тензора
            uint32_t tensor_hash = (t->act << 24) | (t->res << 16) | (t->ent << 8) | t->stab;
            hash ^= tensor_hash;
            hash *= 0x01000193;  // FNV-1a prime
        }
    }

    // Добавляем глобальный контекст
    hash ^= global_context_hash;
    hash *= 0x01000193;

    return hash;
}

void activate_relevant_episodes(void) {
    if (episode_count == 0) return;

    uint32_t now = (uint32_t)time(NULL);
    uint32_t current_context = calculate_context_hash();

    for (uint16_t i = 0; i < episode_count; i++) {
        EpisodeMemory* episode = &episodes[i];

        // Извлекаем хеш контекста эпизода
        uint32_t episode_context = *((uint32_t*)episode->context_hash);

        // Вычисляем схожесть контекстов
        uint32_t context_diff = (current_context > episode_context) ?
                               (current_context - episode_context) :
                               (episode_context - current_context);

        // Если контекст похож (первые 24 бита совпадают)
        if (context_diff < 0x100) {  // Порог схожести
            // Активируем последовательность из эпизода
            for (uint8_t j = 0; j < episode->length; j++) {
                uint16_t tensor_idx = episode->sequence[j];
                if (tensor_idx < tnsr_count) {
                    BitTensor* t = &tnsrs[tensor_idx];

                    // Активируем тензор, но не слишком сильно
                    t->act = (t->act * 7 + 120) >> 3;
                    t->lu = now;

                    // Добавляем в рабочую память
                    add_to_working_memory(t);
                }
            }

            // Обновляем статистику эпизода
            episode->last_recall = now;
            episode->recall_count++;

            // Увеличиваем важность часто вспоминаемых эпизодов
            if (episode->recall_count % 5 == 0) {
                episode->importance = (episode->importance * 9 + 110) >> 3;
                if (episode->importance > 200) episode->importance = 200;
            }

            printf("[EPISODE] Активирован эпизод %u (схожесть: %u%%)\n",
                   i, (uint32_t)(100 - (context_diff * 100 / 0x100)));

            // Ограничиваем количество активируемых эпизодов за раз
            if (i > 3) break;
        }
    }
}

void reorganize_clusters_by_activity(void) {
    if (cluster_count < 2) return;

    // Сортируем кластеры по активности (пузырьковая сортировка)
    for (uint16_t i = 0; i < cluster_count - 1; i++) {
        for (uint16_t j = 0; j < cluster_count - i - 1; j++) {
            if (clusters[j].activation_level < clusters[j + 1].activation_level) {
                // Меняем местами
                MemoryCluster temp = clusters[j];
                clusters[j] = clusters[j + 1];
                clusters[j + 1] = temp;
            }
        }
    }

    // Перераспределяем тензоры по кластерам если нужно
    reassign_tensors_to_clusters();
}

void reassign_tensors_to_clusters(void) {
    // Эта функция может быть вызвана при значительном изменении активности
    // Пока оставляем как заглушку - можно расширить для динамической реорганизации
}

// ===== ОБНОВЛЕННАЯ ОСНОВНАЯ ФУНКЦИЯ ОБНОВЛЕНИЯ =====

void update_thought_stream(void) {
    uint32_t now = (uint32_t)time(NULL);

    // 1. Базовая индексация
    build_link_index();

    // 2. Самоорганизация памяти (если включена)
    if (goals.self_organization_enabled) {
        // Самоорганизация кластеров каждые 5 секунд
        if (now - sstate.self_org_timer >= SELF_ORG_INTERVAL) {
            self_organize_memory_clusters();
            semantic_memory_binding();
            sstate.self_org_timer = now;
        }

        // Консолидация памяти каждые 30 секунд
        if (now - sstate.consolidation_timer >= CONSOLIDATION_INTERVAL) {
            memory_consolidation();
            sstate.consolidation_timer = now;
        }
    }

    // 3. Остальная часть существующего кода...
    // [Здесь будет продолжение существующей логики update_thought_stream]

    // Обновляем глобальный контекстный хеш
    global_context_hash = calculate_context_hash();
}

// ===== ПРОДОЛЖЕНИЕ CORE.C =====

// === ОБНОВЛЕННАЯ create_link с ЭКОНОМИЧЕСКОЙ МОДЕЛЬЮ (Loss/Profit) ===

BitLink* create_link(BitTensor* src, BitTensor* tgt) {
    if (!src || !tgt || lnk_count >= MAX_LINKS) return NULL;
    if (src == tgt && src->conn > 20) return NULL;

    uint16_t search_limit = (lnk_count > 1000) ? lnk_count / 4 : lnk_count;
    for (uint16_t i = 0; i < search_limit; i++) {
        if (lnks[i].src == src && lnks[i].tgt == tgt) {
            // Если связь уже есть, обновляем только хронологию
            lnks[i].ts = (uint32_t)time(NULL);
            if (lnks[i].strength < LINK_MAX_STRENGTH) lnks[i].strength += 2; // Минимальный буст
            return &lnks[i];
        }
    }

    // 1. Расчет базовых метрик
    uint8_t sim_score = calc_bit_sim(src, tgt);
    uint32_t now = (uint32_t)time(NULL);
    uint32_t time_diff = (src->lu > tgt->lu) ? (src->lu - tgt->lu) : (tgt->lu - src->lu);
    
    // 2. Расчет "ЦЕНЫ" (Cost) создания связи
    // Чем ниже схожесть и чем больше разброс во времени, тем дороже создавать связь
    // Это наша базовая "Потеря" (Loss)
    uint8_t creation_cost = 255 - sim_score; 
    if (time_diff > 10) creation_cost += 50; // Штраф за разрыв времени
    if (creation_cost > 255) creation_cost = 255;
    
    // Нормализуем цену к диапазону 0-50 (единиц активности/эффективности)
    creation_cost = creation_cost / 5; 

    // 3. Контекстные бонусы (инвестиции в будущее)
    uint8_t context_bonus = (time_diff < 2) ? 80 : 0;
    if (sim_score > 120) context_bonus += 20;

    // 4. Кластерный бонус (снижение стоимости для проверенных групп)
    uint8_t cluster_bonus = 0;
    if (src->cluster_id != 0 && src->cluster_id == tgt->cluster_id) {
        cluster_bonus = 60;
        // Если кластер очень стабилен, цена почти нулевая
        for (uint16_t ci = 0; ci < cluster_count; ci++) {
            if (clusters[ci].cluster_id == src->cluster_id && clusters[ci].stability > 150) {
                cluster_bonus = 150; 
                break;
            }
        }
    }

    // 5. Эффективная похожесть (прогнозируемая прибыль)
    uint8_t effective_sim = sim_score + context_bonus + cluster_bonus;
    if (effective_sim > 255) effective_sim = 255;
    uint8_t res_score = calc_res_match(src, tgt);

    // 6. Инициализация связи
    BitLink* link = &lnks[lnk_count++];
    link->src = src;
    link->tgt = tgt;
    
    // Начальная сила теперь зависит от того, насколько "выгодна" эта связь
    // Profit = Effective_Sim - Cost. Если Profit < 0, связь слабая.
    int32_t net_profit = effective_sim - creation_cost;
    if (net_profit < 0) net_profit = 0;
    
    link->strength = (uint8_t)net_profit;
    if (link->strength < 10) link->strength = 10; // Минимальная сила, даже если убыток
    if (link->strength > 200) link->strength = 200;

    link->res = res_score;
    link->semantic_type = (src->cluster_id != 0 && src->cluster_id == tgt->cluster_id) ? 1 : 
                          (src->is_concept || tgt->is_concept) ? 3 : 
                          (context_bonus > 50) ? 2 : 0;

    // 7. Композиция веса
    uint32_t src_potential = src->act * 10 + (10 - time_diff);
    uint32_t tgt_potential = tgt->act * 10 + (10 - time_diff);
    link->weight = (src_potential * 255) / (src_potential + tgt_potential + 1);

    link->ts = now;
    link->last_act = now;
    link->use_count = 1;
    link->success_count = 0; // Пока не доказала свою ценность

    // 8. Мгновенное влияние на узлы (Инвестиция)
    // Мы сразу "платим" из активности узлов за создание связи
    uint8_t investment = creation_cost / 2;
    if (src->act > investment) src->act -= investment;
    if (tgt->act > investment) tgt->act -= investment;

    // Обновляем счетчики кластеров
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

// === ОБНОВЛЕННАЯ update_link_strength (Реализация Loss/Profit цикла) ===

void update_link_strength(BitLink* link, uint8_t was_successful) {
    if (!link) return;
    link->use_count++;
    link->ts = (uint32_t)time(NULL);

    // --- Коэффициенты на основе типа связи (Риски и Дивиденды) ---
    float ROI_Multiplier = 1.0f; // Множитель возврата инвестиций
    
    if (link->semantic_type == 1) {  // Внутрикластерная (Низкий риск, стабильный доход)
        ROI_Multiplier = 1.5f; 
    } else if (link->semantic_type == 3) { // Концептуальная (Пассивный доход, почти не теряет)
        ROI_Multiplier = 2.0f;
    } else if (link->semantic_type == 2) { // Семантическая (Средний риск)
        ROI_Multiplier = 1.2f;
    } else { // Обычная (Высокий риск, может исчезнуть)
        ROI_Multiplier = 0.8f;
    }

    if (was_successful) {
        // --- УСПЕХ (Прибыль / Profit) ---
        link->success_count++;
        
        // Вычисляем прибыль. 
        // Если связь доказала свою эффективность, она должна вернуть "инвестиции" и дать бонус.
        // "Investment" здесь — это изначальная слабость связи (чем она слабее была, тем больше надо вернуть).
        uint8_t base_investment = 255 - link->strength; 
        
        // Возврат инвестиций + Проценты (Буст силы)
        uint8_t profit = (uint8_t)(base_investment * 0.1f * ROI_Multiplier); 
        
        if (link->strength < LINK_MAX_STRENGTH) {
            link->strength += profit + 2; // +2 за факт успеха
            if (link->strength > LINK_MAX_STRENGTH) link->strength = LINK_MAX_STRENGTH;
        }
        
        if (link->res < RES_MAX) link->res += 5;

        // Дрифт веса (подтверждение направления)
        uint8_t src_act = link->src->act;
        uint8_t tgt_act = link->tgt->act;
        if (src_act > tgt_act) {
            link->weight = (link->weight * 15 + 200) >> 4;
        } else if (tgt_act > src_act) {
            link->weight = (link->weight * 15 + 55) >> 4;
        }

        // --- ВНУТРЕННЯЯ НАГРАДА (Влияние на нейроны) ---
        // Если связь прибыльна, "дофамин" идет в связанные узлы
        // Это создает положительный цикл: удачная связь делает узлы активнее
        uint8_t reward_sharing = (uint8_t)(profit * 0.5f);
        if (reward_sharing > 0) {
            link->src->act = (link->src->act > ACT_MAX - reward_sharing) ? ACT_MAX : link->src->act + reward_sharing;
            link->tgt->act = (link->tgt->act > ACT_MAX - reward_sharing) ? ACT_MAX : link->tgt->act + reward_sharing;
        }

    } else {
        // --- НЕУДАЧА (Убыток / Loss) ---
        
        // Расчет убытка. 
        // Чем нестабильнее тип связи, тем выше убыток.
        float loss_multiplier = 1.0f;
        if (link->semantic_type == 0) loss_multiplier = 2.0f; // Обычные связи быстро рвутся при ошибке
        if (link->semantic_type == 3) loss_multiplier = 0.2f; // Концептуальные почти не теряют

        // Убыток пропорционален текущей силе (чем сильнее связь была, тем больнее её терять)
        uint8_t loss = (uint8_t)(link->strength * 0.05f * loss_multiplier);
        if (loss < 1) loss = 1;

        // Снижаем силу
        if (link->strength > LINK_MIN_STRENGTH + loss) {
            link->strength -= loss;
        } else {
            link->strength = LINK_MIN_STRENGTH;
        }

        // Снижаем резонанс
        if (link->res > 10) link->res -= 2;

        // --- ВНУТРЕННЯЯ "БОЛЬ" (Влияние на нейроны) ---
        // Неудачная связь "тормозит" узлы. Это обучение через боль.
        if (link->src->act > 5) link->src->act -= 2;
        if (link->tgt->act > 5) link->tgt->act -= 2;
    }

    link->last_act = link->ts;
}

void decay_unused_links(void) {
    uint32_t now = (uint32_t)time(NULL);
    for (uint16_t i = 0; i < lnk_count; i++) {
        BitLink* link = &lnks[i];
        uint32_t inactive_time = now - link->last_act;

        // Разное время затухания для разных типов связей
        uint32_t decay_threshold = 30;  // По умолчанию 30 секунд

        if (link->semantic_type == 1) {  // Внутрикластерные
            decay_threshold = 120;  // 2 минуты
        } else if (link->semantic_type == 3) {  // Концептуальные
            decay_threshold = 300;  // 5 минут
        }

        if (inactive_time > decay_threshold) {
            uint8_t decay_amount = LINK_STRENGTH_DEC;

            // Меньше ослабляем семантические связи
            if (link->semantic_type > 0) {
                decay_amount = LINK_STRENGTH_DEC / 2;
                if (decay_amount < 1) decay_amount = 1;
            }

            if (link->strength > decay_amount + LINK_MIN_STRENGTH) {
                link->strength -= decay_amount;
            } else {
                link->strength = LINK_MIN_STRENGTH;
            }

            // Удаляем очень слабые и редко используемые связи
            if (link->strength < LINK_MIN_STRENGTH * 2 && link->use_count < 3) {
                if (link->src && link->src->conn > 0) link->src->conn--;
                if (link->tgt && link->tgt->conn > 0) link->tgt->conn--;

                // Обновляем счетчики связей в кластерах
                if (link->src->cluster_id != 0) {
                    for (uint16_t ci = 0; ci < cluster_count; ci++) {
                        if (clusters[ci].cluster_id == link->src->cluster_id && clusters[ci].link_count > 0) {
                            clusters[ci].link_count--;
                            break;
                        }
                    }
                }

                lnks[i] = lnks[lnk_count - 1];
                lnk_count--;
                i--;
            }
        }
    }
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
        if (i % 2 == 0) new_byte = (new_byte & 0xF0) | BIT_NOT(new_byte & 0x0F);
        target->data[i] = new_byte;
    }

    target->ent = calc_bit_ent(target, target->cols);
    target->act = (target->act > 245) ? ACT_MAX : target->act + 8;
    target->res = (target->res > 250) ? RES_MAX : target->res + 2;
    target->efficiency = calculate_efficiency(target);
    target->lu = (uint32_t)time(NULL);

    // Если тензор стал очень активным, проверяем возможность создания кластера
    if (target->act > 180 && target->cluster_id == 0 && !target->is_concept) {
        // Активируем самоорганизацию при следующем обновлении
        sstate.self_org_timer = 0;
    }

    if (target->act > 220) prevent_overfitting_by_bit_shift(target);
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

    // Если это концепция, обновляем связанные тензоры
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

    // Применяем dropout с учетом кластерной принадлежности
    for (uint16_t i = 0; i < tnsr_count; i++) {
        if (tnsrs[i].is_concept) {
            // Концепции реже получают dropout
            if (rand() % 1000 < (DROPOUT_RATE / 2)) {
                tnsrs[i].dropout = 1;
            }
            continue;
        }

        // Проверяем, активен ли кластер тензора
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
            // Тензоры в активных кластерах быстрее выходят из dropout
            if (rand() % 100 < (cluster_active ? 3 : 1)) {
                tnsrs[i].dropout = 0;
                tnsrs[i].act = 50;  // Базовая активность при восстановлении
            }
        } else {
            // Тензоры в неактивных кластерах чаще получают dropout
            uint8_t dropout_chance = cluster_active ? DROPOUT_RATE : DROPOUT_RATE * 2;
            if (rand() % 1000 < dropout_chance) {
                tnsrs[i].dropout = 1;
            }
        }
    }
}

void save_tnsr(BitTensor* t) {
    if (!t || !t->data || memo_size >= MAX_MEM_ENTRIES) return;
    uint16_t total_bits = t->rows * t->cols;
    uint8_t total_bytes = (uint8_t)((total_bits + 7) / 8);
    if (total_bytes > MAX_PATTERN) return;

    // Проверяем, не является ли это концепцией (концепции сохраняем отдельно)
    if (t->is_concept) {
        // Концепции сохраняем с более высоким приоритетом
        for (uint16_t i = 0; i < memo_size; i++) {
            if (memo[i].len == total_bytes && !memcmp(memo[i].data, t->data, total_bytes)) {
                memo[i].count += 2;  // Двойной счет для концепций
                memo[i].res = (memo[i].res * 230 + t->res * 25) >> 8;
                memo[i].act = (memo[i].act * 230 + t->act * 25) >> 8;
                memo[i].ent = (memo[i].ent * 230 + t->ent * 25) >> 8;
                memo[i].ts = (uint32_t)time(NULL);
                memo[i].cluster_id = t->cluster_id;
                return;
            }
        }
    } else {
        // Обычные тензоры
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
    entry->count = t->is_concept ? 2 : 1;  // Концепции начинают с большего счета
    entry->res = t->res;
    entry->act = t->act;
    entry->ent = t->ent;
    entry->ts = (uint32_t)time(NULL);
    entry->cluster_id = t->cluster_id;
}

BitTensor* find_efficient_match(BitTensor* input) {
    if (!input) return NULL;
    uint16_t best_score = 0;
    BitTensor* best_match = NULL;
    uint16_t best_link_idx = 0xFFFF;
    uint32_t now = (uint32_t)time(NULL);

    // 1. Сначала ищем среди связанных тензоров (уже существующие связи)
    for (uint16_t i = 0; i < lnk_count; i++) {
        if (lnks[i].src == input && lnks[i].strength > 20) {
            BitTensor* t = lnks[i].tgt;

            // Пропускаем концепции если input не концепция
            if (t->is_concept && !input->is_concept) continue;

            uint8_t freshness = (now - t->lu > 300) ? 0 : 255 - (uint8_t)((now - t->lu)/2);
            uint8_t connectivity = (t->conn > 10) ? 255 : (t->conn * 25);
            uint8_t sim = calc_bit_sim(input, t);
            uint8_t res_match = calc_res_match(input, t);
            uint8_t eff_boost = (t->efficiency > goals.target_efficiency) ? 255 : t->efficiency;

            // Бонус за принадлежность к одному кластеру
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
    // 2. Если хорошего совпадения нет, ищем среди всех тензоров
    if (!best_match || best_score < 6000) {
        for (uint16_t i = 0; i < tnsr_count; i++) {
            BitTensor* t = &tnsrs[i];
            if (t == input || t->act < 20 || t->dropout) continue;
            if (t->is_concept && !input->is_concept) continue;  // Пропускаем концепции

            uint8_t freshness = (now - t->lu > 300) ? 0 : 255 - (uint8_t)((now - t->lu)/2);
            uint8_t sim = calc_bit_sim(input, t);
            uint8_t res_match = calc_res_match(input, t);

            // Бонус за кластер
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

        // Активируем кластер если тензор в нем
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

// ОБНОВЛЕННАЯ ФУНКЦИЯ ОБНОВЛЕНИЯ СЕТИ С УЧЕТОМ САМООРГАНИЗАЦИИ
void update_bit_net_with_goals(void) {
    uint32_t now = (uint32_t)time(NULL);

    // 1. Затухание неиспользуемых связей
    if (now % 5 == 0) decay_unused_links();

    // 2. Периодическое сжатие памяти
    if (now - last_mem_check_ts >= MEM_REDUCE_INTERVAL) {
        last_mem_check_ts = now;
        for (uint16_t i = 0; i < tnsr_count; i++) {
            if (tnsrs[i].act < LOW_ACT_THRESHOLD && !tnsrs[i].mem_red) {
                reduce_tnsr_mem(&tnsrs[i]);
            }
        }
    }

    // 3. Обновление целей эффективности и dropout
    if (now % 10 == 0) {
        update_efficiency_goal();
        for (uint16_t i = 0; i < tnsr_count; i++) {
            if (tnsrs[i].efficiency < goals.target_efficiency - 20) {
                tnsrs[i].goal_active = 1;
            }
        }
        apply_dropout();

        // Обновляем активацию кластеров
        for (uint16_t ci = 0; ci < cluster_count; ci++) {
            clusters[ci].activation_level = (clusters[ci].activation_level * 9) >> 3;  // Плавное затухание
        }
    }

    // 4. Оптимизация тензоров с активными целями
    for (uint16_t i = 0; i < tnsr_count; i++) {
        if (tnsrs[i].goal_active == 1) {
            optimize_tensor(&tnsrs[i]);
        }
    }

    // 5. Обновление связей на основе взаимодействий
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

            // Активируем кластеры связанных тензоров
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

    // 6. Создание новых связей для изолированных активных тензоров
    if (now % 15 == 0) {
        for (uint16_t i = 0; i < tnsr_count; i++) {
            if (tnsrs[i].act > 100 && tnsrs[i].conn < 5 && !tnsrs[i].is_concept) {
                // Ищем похожие активные тензоры для связывания
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

    // 7. Обновление системного резонанса
    if (active_links > 0) {
        uint8_t avg_link_res = (uint8_t)(total_res_sum / active_links);
        sys_res = (sys_res * 230 + avg_link_res * 25) >> 8;
    }

    // 8. Обновление истории системы
    sstate.res_hist[sstate.hist_idx] = sys_res;
    sstate.hist_idx = (sstate.hist_idx + 1) % HISTORY_SIZE;

    // 9. Периодическая самоорганизация
    if (goals.self_organization_enabled && now % SELF_ORG_INTERVAL == 0) {
        self_organize_memory_clusters();
    }
}

static uint32_t calculate_resonance_score(BitTensor* t, uint32_t now) {
    if (!t) return 0;

    // Базовый счет на основе характеристик тензора
    uint32_t base_score = (uint32_t)t->act * t->res * t->efficiency;

    // Бонус за кластерную принадлежность
    uint32_t cluster_bonus = 0;
    if (t->cluster_id != 0) {
        for (uint16_t ci = 0; ci < cluster_count; ci++) {
            if (clusters[ci].cluster_id == t->cluster_id) {
                cluster_bonus = clusters[ci].stability * clusters[ci].activation_level;
                break;
            }
        }
    }

    // Бонус за связи
    uint32_t link_score = 0, link_count = 0;
    for (uint16_t i = 0; i < lnk_count; i++) {
        BitLink* l = &lnks[i];
        if ((l->src == t || l->tgt == t) && l->strength > 40 && !l->src->dropout && !l->tgt->dropout) {
            link_score += l->strength * (l->semantic_type + 1);  // Семантические связи весят больше
            link_count++;
        }
    }

    uint32_t avg_link = link_count ? (link_score / link_count) : 0;
    uint32_t freshness = (now - t->lu > 300) ? 0 : 1000 - (now - t->lu) * 3;

    return base_score + cluster_bonus + (uint32_t)t->conn * avg_link * 10 +
           (uint32_t)t->stab * 10 + freshness;
}

// ОБНОВЛЕННАЯ ФУНКЦИЯ ПОИСКА ЗНАЧИМЫХ ТЕНЗОРОВ
BitTensor* find_significant_tensor(SearchStrategy strategy, void* context) {
    BitTensor* best = NULL;
    uint32_t best_score = 0;
    uint32_t now = (uint32_t)time(NULL);
    ScoreFunction custom_func = (strategy == SEARCH_CUSTOM_SCORE) ? (ScoreFunction)context : NULL;

    // Обрабатываем специальные стратегии поиска
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
        // Ищем концепции
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
        // Если концепций нет, продолжаем обычный поиск
        strategy = SEARCH_RESONANT;
    }

    // Стандартный поиск среди всех тензоров
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

    // Также проверяем рабочую память
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

    // Если нашли резонансный тензор, увеличиваем его стабильность
    if (strategy == SEARCH_RESONANT && best && best_score > 20000) {
        best->stab = (best->stab * 7 + 200) >> 3;

        // Если это тензор в кластере, увеличиваем стабильность кластера
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

void build_link_index(void) {
    memset(tensor_links, 0, sizeof(tensor_links));
    for (uint16_t i = 0; i < lnk_count; i++) {
        BitLink* link = &lnks[i];
        uint16_t src_idx = link->src - tnsrs;
        uint16_t tgt_idx = link->tgt - tnsrs;

        if (src_idx < MAX_TENSORS && tensor_links[src_idx].link_count < 32) {
            tensor_links[src_idx].link_indices[tensor_links[src_idx].link_count++] = i;

            // Также добавляем в кластерные связи если тензоры в одном кластере
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

    // Проверяем обратную связь
    for (uint16_t i = 0; i < lnk_count; i++) {
        if (lnks[i].src == b && lnks[i].tgt == a) {
            base = (base + lnks[i].strength + lnks[i].res) / 3;
            break;
        }
    }

    // Бонус за кластерную принадлежность
    if (a->cluster_id != 0 && a->cluster_id == b->cluster_id) {
        base += 40;
    }

    // Бонус за временную близость
    uint32_t ts_diff = (a->lu > b->lu) ? a->lu - b->lu : b->lu - a->lu;
    if (ts_diff < 10) base += 30;
    else if (ts_diff < 60) base += 10;

    return (base > 255) ? 255 : base;
}

void fast_contextual_activation(BitTensor* context) {
    uint16_t ctx_idx = context - tnsrs;
    if (ctx_idx >= MAX_TENSORS) return;

    // Активируем связанные тензоры
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

    // Также активируем тензоры из того же кластера
    if (context->cluster_id != 0) {
        for (uint16_t ci = 0; ci < cluster_count; ci++) {
            if (clusters[ci].cluster_id == context->cluster_id) {
                // Активируем несколько случайных тензоров из кластера
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

// ОБНОВЛЕННАЯ ФУНКЦИЯ ОБНОВЛЕНИЯ ПОТОКА МЫСЛЕЙ

// ОБНОВЛЕННАЯ ФУНКЦИЯ АГРЕССИВНОЙ ОЧИСТКИ
void aggressive_memory_cleanup(void) {
    uint32_t now = (uint32_t)time(NULL);
    uint16_t i;

    // 1. Удаление плохих связей
    for (i = 0; i < lnk_count; i++) {
        BitLink* link = &lnks[i];
        uint32_t inactive_time = now - link->last_act;

        // Разные критерии для разных типов связей
        uint8_t should_remove = 0;

        if (link->semantic_type == 0) {  // Обычные связи
            should_remove = (inactive_time > 3600) ||
                           (link->strength < LINK_MIN_STRENGTH && link->use_count < 3) ||
                           (link->use_count > 0 && link->success_count * 10 < link->use_count);
        } else if (link->semantic_type == 1) {  // Внутрикластерные
            should_remove = (inactive_time > 7200) ||
                           (link->strength < LINK_MIN_STRENGTH / 2 && link->use_count < 5);
        } else {  // Семантические и концептуальные
            should_remove = (inactive_time > 10800) ||  // 3 часа
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

    // 2. Очистка Рабочей Памяти
    for (i = 0; i < working_mem_count; i++) {
        if (working_mem[i].tensor &&
            (now - working_mem[i].timestamp > 300) &&
            working_mem[i].priority < 40) {
            working_mem[i] = working_mem[working_mem_count - 1];
            working_mem_count--;
            i--;
        }
    }

    // 3. Удаление изолированных тензоров (кроме концепций)
    for (int32_t t_idx = tnsr_count - 1; t_idx >= 0; t_idx--) {
        BitTensor* t = &tnsrs[t_idx];

        // Не удаляем концепции
        if (t->is_concept) continue;

        // Проверка на наличие в рабочей памяти
        uint8_t in_working_mem = 0;
        for (uint8_t w = 0; w < working_mem_count; w++) {
            if (working_mem[w].tensor == t) {
                in_working_mem = 1;
                break;
            }
        }

        // Критерии удаления тензора:
        if (t->conn < 2 &&
            t->act < 10 &&
            !in_working_mem &&
            t->efficiency < 30 &&
            (now - t->lu > 600)) {

            // Удаляем из кластера если есть
            if (t->cluster_id != 0) {
                for (uint16_t ci = 0; ci < cluster_count; ci++) {
                    if (clusters[ci].cluster_id == t->cluster_id) {
                        // Удаляем тензор из кластера
                        for (uint16_t k = 0; k < clusters[ci].size; k++) {
                            if (clusters[ci].tensor_indices[k] == (uint16_t)t_idx) {
                                // Сдвигаем остальные элементы
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

            // Удаляем все связи
            for (int32_t l = lnk_count - 1; l >= 0; l--) {
                if (lnks[l].src == t || lnks[l].tgt == t) {
                    lnks[l] = lnks[lnk_count - 1];
                    lnk_count--;
                }
            }

            // Удаляем из рабочей памяти
            for (uint8_t w = 0; w < working_mem_count; w++) {
                if (working_mem[w].tensor == t) {
                    working_mem[w] = working_mem[working_mem_count - 1];
                    working_mem_count--;
                    break;
                }
            }

            // Освобождаем память
            if (t->data) {
                free(t->data);
                t->data = NULL;
            }

            // Перемещаем последний элемент
            if (t_idx < tnsr_count - 1) {
                tnsrs[t_idx] = tnsrs[tnsr_count - 1];

                BitTensor* moved_tensor = &tnsrs[t_idx];
                BitTensor* old_place = &tnsrs[tnsr_count - 1];

                // Обновляем ссылки в кластерах
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

                // Обновляем концепции
                for (uint8_t ci = 0; ci < concept_count; ci++) {
                    for (uint8_t k = 0; k < concepts[ci].member_count; k++) {
                        if (concepts[ci].member_indices[k] == tnsr_count - 1) {
                            concepts[ci].member_indices[k] = t_idx;
                            break;
                        }
                    }
                }

                // Обновляем эпизоды
                for (uint16_t ei = 0; ei < episode_count; ei++) {
                    for (uint8_t k = 0; k < episodes[ei].length; k++) {
                        if (episodes[ei].sequence[k] == tnsr_count - 1) {
                            episodes[ei].sequence[k] = t_idx;
                        }
                    }
                }

                // Обновляем связи
                for (uint16_t l = 0; l < lnk_count; l++) {
                    if (lnks[l].src == old_place) lnks[l].src = moved_tensor;
                    if (lnks[l].tgt == old_place) lnks[l].tgt = moved_tensor;
                }

                // Обновляем рабочую память
                for (uint8_t w = 0; w < working_mem_count; w++) {
                    if (working_mem[w].tensor == old_place) working_mem[w].tensor = moved_tensor;
                }
            }

            tnsr_count--;
        }
    }

    // 4. Сжатие старых тихих тензоров
    for (i = 0; i < tnsr_count; i++) {
        BitTensor* t = &tnsrs[i];
        if (t->act < 30 && (now - t->lu > 1800) &&
            (t->rows * t->cols > 256) && !t->mem_red && !t->is_concept) {
            reduce_tnsr_mem(t);
        }
    }

    // 5. Очистка memo с учетом кластеров
    if (memo_size > 0) {
        for (uint16_t i = 0; i < memo_size; i++) {
            for (uint16_t j = i + 1; j < memo_size; j++) {
                float si = (float)memo[i].count * memo[i].act / ((now - memo[i].ts) + 1);
                float sj = (float)memo[j].count * memo[j].act / ((now - memo[j].ts) + 1);

                // Бонус за принадлежность к активному кластеру
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

        // Оставляем топ-50% или минимум 50 записей
        uint16_t target_size = (MAX_MEM_ENTRIES / 2) > 50 ? (MAX_MEM_ENTRIES / 2) : 50;
        if (memo_size > target_size) {
            memo_size = target_size;
        }
    }

    // 6. Удаление старых/неважных эпизодов
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

uint8_t sync_and_get_recommendation(uint8_t* vector_data, int vector_len, uint8_t last_action) {
    // 1. Обновляем систему с самоорганизацией
    update_thought_stream();

    // 2. Находим самый активный тензор (рекомендация)
    BitTensor* best = find_significant_tensor(SEARCH_MOST_ACTIVE, NULL);

    // 3. Если нет активных тензоров, ищем концепции
    if (!best) {
        best = find_significant_tensor(SEARCH_CONCEPTUAL, NULL);
    }

    // 4. Если все еще нет, возвращаем случайное действие
    if (!best) return rand() % 4;

    // 5. Возвращаем действие на основе характеристик тензора
    uint8_t action = (best->act + best->res + best->efficiency) % 4;

    // Если это концепция, используем более детерминированный выбор
    if (best->is_concept) {
        if (best->data) {
            action = best->data[0] % 4;
        }
    }

    return action;
}

uint8_t get_global_resonance() {
    return sys_res;
}

void learn_from_data(uint8_t* vector_data, int len, uint8_t action_idx, uint8_t reward) {
    // 1. Создаем тензор из данных
    BitTensor* t = create_tnsr(16, 16);
    if (!t) return;

    // 2. Записываем данные вектора
    int bytes_to_copy = (len < (16*16/8)) ? len : (16*16/8);
    for(int i = 0; i < bytes_to_copy; i++) {
        t->data[i] = vector_data[i];
    }

    // 3. Воздействие награды
    if (reward > 128) {  // Положительная награда
        t->act = (t->act > 200) ? 255 : t->act + 50;
        t->res = (t->res > 200) ? 255 : t->res + 20;
        t->stab = (t->stab > 200) ? 255 : t->stab + 10;

        // Создаем связь с действием если возможно
        if (action_idx < 7) {  // Валидное действие
            // Создаем тензор для действия
            BitTensor* action_tensor = create_tnsr(1, 8);
            if (action_tensor) {
                // Кодируем действие в битах
                memset(action_tensor->data, 0, 1);
                BIT_SET(action_tensor->data[0], action_idx % 8);

                // Создаем сильную связь
                BitLink* link = create_link(t, action_tensor);
                if (link) {
                    link->strength = 200;
                    link->res = 220;
                    link->semantic_type = 1;  // Внутрикластерная/семантическая
                }
            }
        }
    } else {  // Отрицательная или низкая награда
        t->act = t->act / 2;
        t->res = t->res / 2;
        t->stab = t->stab / 2;
    }

    t->efficiency = calculate_efficiency(t);

    // 4. Добавляем в рабочую память
    add_to_working_memory(t);

    // 5. Обновляем сеть
    update_thought_stream();
}

// ===== СЕРИАЛИЗАЦИЯ (с учетом самоорганизации) =====

int save_state_to_file(const char* filename) {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open(save)"); return -1; }

#define WRITE(ptr, size, count) do { \
    if (write(fd, (ptr), (size_t)(size)*(count)) != (ssize_t)((size_t)(size)*(count))) { \
        perror("write"); close(fd); return -1; } \
} while (0)

    // Основные счетчики
    WRITE(&tnsr_count, sizeof(uint16_t), 1);
    WRITE(&lnk_count, sizeof(uint16_t), 1);
    WRITE(&memo_size, sizeof(uint16_t), 1);
    WRITE(&working_mem_count, sizeof(uint8_t), 1);
    WRITE(&goals, sizeof(SystemGoals), 1);
    WRITE(&sys_res, sizeof(uint8_t), 1);
    WRITE(&interaction_count, sizeof(uint32_t), 1);
    WRITE(&last_mem_check_ts, sizeof(uint32_t), 1);
    WRITE(&sstate, sizeof(BitSystemState), 1);

    // Самоорганизация
    WRITE(&cluster_count, sizeof(uint16_t), 1);
    WRITE(&episode_count, sizeof(uint16_t), 1);
    WRITE(&concept_count, sizeof(uint8_t), 1);
    WRITE(&global_context_hash, sizeof(uint32_t), 1);
    WRITE(&next_cluster_id, sizeof(uint8_t), 1);

    // Тензоры
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

    // Связи
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

    // Кластеры
    WRITE(clusters, sizeof(MemoryCluster), cluster_count);

    // Эпизоды
    WRITE(episodes, sizeof(EpisodeMemory), episode_count);

    // Концепции
    for (uint8_t i = 0; i < concept_count; i++) {
        uint16_t concept_idx = tensor_to_index(concepts[i].concept_tensor);
        WRITE(&concept_idx, sizeof(uint16_t), 1);
        WRITE(&concepts[i].member_count, sizeof(uint8_t), 1);
        WRITE(concepts[i].member_indices, sizeof(uint16_t), concepts[i].member_count);
        WRITE(&concepts[i].abstraction_level, sizeof(uint8_t), 1);
        WRITE(&concepts[i].coherence, sizeof(uint8_t), 1);
        WRITE(&concepts[i].last_used, sizeof(uint32_t), 1);
    }

    // Долговременная память (memo)
    WRITE(memo, sizeof(BitMemory), memo_size);

    // Рабочая память
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

    // Очистка текущего состояния
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

    // Самоорганизация
    READ(&cluster_count, sizeof(uint16_t), 1);
    READ(&episode_count, sizeof(uint16_t), 1);
    READ(&concept_count, sizeof(uint8_t), 1);
    READ(&global_context_hash, sizeof(uint32_t), 1);
    READ(&next_cluster_id, sizeof(uint8_t), 1);

    if (cluster_count > MAX_CLUSTERS || episode_count > MAX_EPISODES || concept_count > 64) {
        fprintf(stderr, "[ERR] Corrupted self-organization state\n");
        close(fd); return -1;
    }
    // Загрузка тензоров
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
    // Временный буфер для связей
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
    // Кластеры и эпизоды
    READ(clusters, sizeof(MemoryCluster), cluster_count);
    READ(episodes, sizeof(EpisodeMemory), episode_count);
    // Концепции
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
    // Рабочая память
    for (uint8_t i = 0; i < working_mem_count; i++) {
        uint16_t t_idx;
        READ(&t_idx, sizeof(uint16_t), 1);
        READ(&working_mem[i].timestamp, sizeof(uint32_t), 1);
        READ(&working_mem[i].priority, sizeof(uint8_t), 1);
        READ(&working_mem[i].access_count, sizeof(uint8_t), 1);
        READ(&working_mem[i].episode_marker, sizeof(uint8_t), 1);
        working_mem[i].tensor = index_to_tensor(t_idx);
    }
    // Восстановление связей
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

    // Переиндексация после загрузки
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
