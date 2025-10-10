#include "kernel.h"
#include <time.h>
#include <ctype.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============== ОБЩИЕ ФОРМУЛЫ ===============
// ЗАМЕНИТЕ существующую fractal_connectivity на эту:
float fractal_connectivity(float dimension, float intensity, int depth) {
    // Биологически оптимальные диапазоны из PDF (1.7-1.9)
    float biological_boost = 1.0f;
    if (dimension > 1.7f && dimension < 1.9f) {
        biological_boost = 1.3f; // +30% эффективности в оптимальном диапазоне
    }
    
    // Основа из вашего текущего кода
    float base = (dimension * intensity) / (1.0f + logf(1.0f + depth));
    
    return base * biological_boost;
}

float resonance_amplification(float base, float harmonic, float resonance) {
    // Базовая формула (сохраняем вашу текущую логику)
    float base_amplification = base * (1.0f + harmonic * resonance);
    
    // Добавляем умеренный стохастический резонанс из PDF
    // Оптимальный шум для усиления слабых, но значимых сигналов
    float optimal_noise = 0.0f;
    
    // Стохастический резонанс наиболее эффективен для средних уровней активации
    if (base > 0.2f && base < 0.8f) {
        optimal_noise = 0.015f; // Очень умеренный шум
    }
    
    float noise = ((float)rand() / RAND_MAX - 0.5f) * optimal_noise;
    
    // Принцип квантового дарвинизма - усиление устойчивых состояний
    float stability_boost = 1.0f;
    if (harmonic > 0.6f && resonance > 0.6f) {
        stability_boost = 1.15f; // +15% для высоко согласованных состояний
    }
    
    return base_amplification * stability_boost + noise;
}
// =============== МЕТАФОРМУЛЫ ===============
float adaptive_learning_rate(float current_rate, float performance, float stability) {
    float adaptive = current_rate * (1.0f + tanhf(performance * 2.0f - 1.0f));
    return adaptive * (0.5f + stability * 0.5f);
}

float energy_balance(float consumption, float activation, float target_efficiency) {
    float imbalance = fabsf(consumption - activation * 0.7f);
    return target_efficiency * expf(-imbalance * 2.0f);
}

float semantic_coherence(const char** patterns, int pattern_count, float base_coherence) {
    if (pattern_count == 0) return base_coherence;
    
    float coherence = base_coherence;
    for (int i = 0; i < pattern_count - 1; i++) {
        if (patterns[i] && patterns[i+1] && strstr(patterns[i], patterns[i+1])) {
            coherence += 0.1f;
        }
    }
    return fminf(1.0f, coherence);
}

// =============== FractalSpike Implementation ===============
FractalSpike* create_fractal_spike(long timestamp, float intensity, const char* source, float fractalDimension, char** path, int pathSize)
{
    FractalSpike* spike = (FractalSpike*)malloc(sizeof(FractalSpike));
    if (!spike) return NULL;

    spike->timestamp = timestamp;
    spike->intensity = intensity;
    spike->source = strdup(source);
    spike->fractalDimension = fractalDimension;

    spike->intensity = fractal_connectivity(fractalDimension, intensity, pathSize);

    spike->pathSize = pathSize;
    if (pathSize > 0) {
        spike->propagationPath = (char**)malloc(pathSize * sizeof(char*));
        for (int i = 0; i < pathSize; i++) {
            spike->propagationPath[i] = strdup(path[i]);
        }
        
        float coherence = semantic_coherence((const char**)path, pathSize, 0.5f);
        spike->intensity *= (0.7f + coherence * 0.3f);
    } else {
        spike->propagationPath = NULL;
    }

    return spike;
}

void destroy_fractal_spike(FractalSpike* spike)
{
    if (!spike) return;

    free(spike->source);

    if (spike->pathSize > 0 && spike->propagationPath) {
        for (int i = 0; i < spike->pathSize; i++) {
            free(spike->propagationPath[i]);
        }
        free(spike->propagationPath);
    }

    free(spike);
}

void print_fractal_spike(const FractalSpike* spike)
{
    if (!spike) return;

    printf("FractalSpike {\n");
    printf("  timestamp: %ld\n", spike->timestamp);
    printf("  intensity: %.2f\n", spike->intensity);
    printf("  source: %s\n", spike->source);
    printf("  fractalDimension: %.2f\n", spike->fractalDimension);
    printf("  propagationPath: [");
    for (int i = 0; i < spike->pathSize; i++) {
        printf("%s", spike->propagationPath[i]);
        if (i < spike->pathSize - 1) printf(", ");
    }
    printf("]\n}\n");
}

// =============== FractalActivation Implementation ===============
FractalActivation* create_fractal_activation(
    float baseActivation,
    float harmonicActivation,
    float spikeResonance,
    int fractalDepth,
    float energyConsumption
) {
    FractalActivation* act = (FractalActivation*)malloc(sizeof(FractalActivation));
    if (!act) return NULL;

    act->baseActivation = baseActivation;
    act->harmonicActivation = harmonicActivation;
    act->spikeResonance = spikeResonance;
    act->fractalDepth = fractalDepth;
    act->energyConsumption = energyConsumption;

    act->baseActivation = resonance_amplification(baseActivation, harmonicActivation, spikeResonance);

    return act;
}

void destroy_fractal_activation(FractalActivation* act)
{
    free(act);
}

float get_total_activation(const FractalActivation* act)
{
    if (!act) return 0.0f;

    float weighted = act->baseActivation * 0.4f +
                     act->harmonicActivation * 0.3f +
                     act->spikeResonance * 0.3f;

    float efficiency = energy_balance(act->energyConsumption, weighted, 0.8f);
    
    float decay = expf(-act->energyConsumption * (float)act->fractalDepth * 0.1f);
    return weighted * decay * efficiency;
}

void print_fractal_activation(const FractalActivation* act)
{
    if (!act) return;

    printf("FractalActivation {\n");
    printf("  baseActivation: %.3f\n", act->baseActivation);
    printf("  harmonicActivation: %.3f\n", act->harmonicActivation);
    printf("  spikeResonance: %.3f\n", act->spikeResonance);
    printf("  fractalDepth: %d\n", act->fractalDepth);
    printf("  energyConsumption: %.3f\n", act->energyConsumption);
    printf("  totalActivation: %.3f\n", get_total_activation(act));
    printf("}\n");
}

void fractal_gradient_descent(FractalActivation *act, float learning_rate)
{
    if (!act) return;

    float performance = (act->baseActivation + act->harmonicActivation) / 2.0f;
    float stability = 1.0f - fabsf(act->energyConsumption - 0.5f);
    float adaptive_lr = adaptive_learning_rate(learning_rate, performance, stability);

    float base = act->baseActivation;
    float harm = act->harmonicActivation;
    float spike = act->spikeResonance;
    float energy = act->energyConsumption;
    float depth = (float)act->fractalDepth;

    float weighted_sum = base * 0.4f + harm * 0.3f + spike * 0.3f;
    float exp_val = expf(-energy * depth * 0.1f);

    float grad_energy = -weighted_sum * exp_val * depth * 0.1f;
    float grad_base = 0.4f * exp_val;
    float grad_harm = 0.3f * exp_val;
    float grad_spike = 0.3f * exp_val;

    act->baseActivation += adaptive_lr * grad_base;
    act->harmonicActivation += adaptive_lr * grad_harm;
    act->spikeResonance += adaptive_lr * grad_spike;
    act->energyConsumption += adaptive_lr * grad_energy;

    CLAMP(act->baseActivation);
    CLAMP(act->harmonicActivation);
    CLAMP(act->spikeResonance);
    CLAMP(act->energyConsumption);

    float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.005f;
    act->baseActivation += noise;
    act->harmonicActivation += noise;
    act->spikeResonance += noise;
    act->energyConsumption += noise;

    CLAMP(act->baseActivation);
    CLAMP(act->harmonicActivation);
    CLAMP(act->spikeResonance);
    CLAMP(act->energyConsumption);
}

// =============== HierarchicalSpikeSystem Implementation ===============
HierarchicalSpikeSystem* create_hierarchical_spike_system(int max_low_spikes, int max_mid_spikes, int max_high_spikes)
{
    HierarchicalSpikeSystem* system = (HierarchicalSpikeSystem*)malloc(sizeof(HierarchicalSpikeSystem));
    if (!system) return NULL;

    system->max_low_spikes = max_low_spikes;
    system->max_mid_spikes = max_mid_spikes;
    system->max_high_spikes = max_high_spikes;
    
    system->low_level_count = 0;
    system->mid_level_count = 0;
    system->high_level_count = 0;

    system->low_level_spikes = (FractalSpike**)malloc(max_low_spikes * sizeof(FractalSpike*));
    system->mid_level_spikes = (FractalSpike**)malloc(max_mid_spikes * sizeof(FractalSpike*));
    system->high_level_spikes = (FractalSpike**)malloc(max_high_spikes * sizeof(FractalSpike*));

    for (int i = 0; i < max_low_spikes; i++) {
        system->low_level_spikes[i] = NULL;
    }
    for (int i = 0; i < max_mid_spikes; i++) {
        system->mid_level_spikes[i] = NULL;
    }
    for (int i = 0; i < max_high_spikes; i++) {
        system->high_level_spikes[i] = NULL;
    }

    system->low_to_mid_weights = (float*)malloc(max_low_spikes * max_mid_spikes * sizeof(float));
    system->mid_to_high_weights = (float*)malloc(max_mid_spikes * max_high_spikes * sizeof(float));
    system->high_to_mid_weights = (float*)malloc(max_high_spikes * max_mid_spikes * sizeof(float));
    system->mid_to_low_weights = (float*)malloc(max_mid_spikes * max_low_spikes * sizeof(float));

    for (int i = 0; i < max_low_spikes * max_mid_spikes; i++) {
        system->low_to_mid_weights[i] = (float)rand() / RAND_MAX * 0.5f;
    }
    for (int i = 0; i < max_mid_spikes * max_high_spikes; i++) {
        system->mid_to_high_weights[i] = (float)rand() / RAND_MAX * 0.3f;
    }
    for (int i = 0; i < max_high_spikes * max_mid_spikes; i++) {
        system->high_to_mid_weights[i] = (float)rand() / RAND_MAX * 0.2f;
    }
    for (int i = 0; i < max_mid_spikes * max_low_spikes; i++) {
        system->mid_to_low_weights[i] = (float)rand() / RAND_MAX * 0.1f;
    }

    system->cache = create_fractal_hash_cache(500);
    system->last_optimization_time = time(NULL);

    return system;
}

void destroy_hierarchical_spike_system(HierarchicalSpikeSystem* system)
{
    if (!system) return;

    for (int i = 0; i < system->max_low_spikes; i++) {
        if (system->low_level_spikes[i]) {
            destroy_fractal_spike(system->low_level_spikes[i]);
        }
    }
    for (int i = 0; i < system->max_mid_spikes; i++) {
        if (system->mid_level_spikes[i]) {
            destroy_fractal_spike(system->mid_level_spikes[i]);
        }
    }
    for (int i = 0; i < system->max_high_spikes; i++) {
        if (system->high_level_spikes[i]) {
            destroy_fractal_spike(system->high_level_spikes[i]);
        }
    }

    free(system->low_level_spikes);
    free(system->mid_level_spikes);
    free(system->high_level_spikes);
    
    free(system->low_to_mid_weights);
    free(system->mid_to_high_weights);
    free(system->high_to_mid_weights);
    free(system->mid_to_low_weights);

    if (system->cache) {
        destroy_fractal_hash_cache(system->cache);
    }

    free(system);
}

void add_spike_to_hierarchy(HierarchicalSpikeSystem* system, FractalSpike* spike, int level)
{
    if (!system || !spike) return;

    FractalSpike** target_array = NULL;
    int* count = NULL;
    int max_count = 0;

    switch (level) {
        case SPIKE_LEVEL_LOW:
            target_array = system->low_level_spikes;
            count = &system->low_level_count;
            max_count = system->max_low_spikes;
            break;
        case SPIKE_LEVEL_MID:
            target_array = system->mid_level_spikes;
            count = &system->mid_level_count;
            max_count = system->max_mid_spikes;
            break;
        case SPIKE_LEVEL_HIGH:
            target_array = system->high_level_spikes;
            count = &system->high_level_count;
            max_count = system->max_high_spikes;
            break;
        default:
            return;
    }

    if (*count < max_count) {
        target_array[*count] = spike;
        (*count)++;
        
        if (system->cache) {
            hash_cache_store(system->cache, spike->source, 
                           spike->fractalDimension, spike->intensity, 
                           spike->intensity);
        }
    } else {
        if (target_array[0]) {
            destroy_fractal_spike(target_array[0]);
        }
        for (int i = 1; i < max_count; i++) {
            target_array[i-1] = target_array[i];
        }
        target_array[max_count-1] = spike;
        
        if (system->cache) {
            hash_cache_store(system->cache, spike->source, 
                           spike->fractalDimension, spike->intensity, 
                           spike->intensity);
        }
    }
}

float propagate_through_hierarchy(HierarchicalSpikeSystem* system, FractalSpike* input_spike)
{
    if (!system || !input_spike) return 0.0f;

    FractalSpike* spike_copy = create_fractal_spike(
        input_spike->timestamp,
        input_spike->intensity,
        input_spike->source,
        input_spike->fractalDimension,
        input_spike->propagationPath,
        input_spike->pathSize
    );
    add_spike_to_hierarchy(system, spike_copy, SPIKE_LEVEL_LOW);

    float total_activation = 0.0f;
    
    for (int i = 0; i < system->low_level_count; i++) {
        FractalSpike* low_spike = system->low_level_spikes[i];
        if (!low_spike) continue;

        float cached_activation = get_cached_activation(system->cache, low_spike);
        
        for (int j = 0; j < system->mid_level_count; j++) {
            if (system->mid_level_spikes[j]) {
                float weight = system->low_to_mid_weights[i * system->max_mid_spikes + j];
                float activation = cached_activation * weight;
                
                system->mid_level_spikes[j]->intensity += activation * 0.1f;
                CLAMP(system->mid_level_spikes[j]->intensity);
                
                total_activation += activation;
            }
        }
    }

    for (int i = 0; i < system->mid_level_count; i++) {
        FractalSpike* mid_spike = system->mid_level_spikes[i];
        if (!mid_spike) continue;

        float cached_activation = get_cached_activation(system->cache, mid_spike);
        
        for (int j = 0; j < system->high_level_count; j++) {
            if (system->high_level_spikes[j]) {
                float weight = system->mid_to_high_weights[i * system->max_high_spikes + j];
                float activation = cached_activation * weight;
                
                system->high_level_spikes[j]->intensity += activation * 0.05f;
                CLAMP(system->high_level_spikes[j]->intensity);
                
                total_activation += activation;
            }
        }
    }

    for (int i = 0; i < system->high_level_count; i++) {
        FractalSpike* high_spike = system->high_level_spikes[i];
        if (!high_spike || high_spike->intensity < 0.3f) continue;

        float feedback_strength = high_spike->intensity * 0.2f;
        
        for (int j = 0; j < system->mid_level_count; j++) {
            if (system->mid_level_spikes[j]) {
                float weight = system->high_to_mid_weights[i * system->max_mid_spikes + j];
                system->mid_level_spikes[j]->intensity += feedback_strength * weight;
                CLAMP(system->mid_level_spikes[j]->intensity);
            }
        }
    }

    return total_activation;
}

void optimize_hierarchical_connections(HierarchicalSpikeSystem* system)
{
    if (!system) return;

    long current_time = time(NULL);
    if (current_time - system->last_optimization_time < 60) {
        return;
    }

    system->last_optimization_time = current_time;

    for (int i = 0; i < system->low_level_count; i++) {
        for (int j = 0; j < system->mid_level_count; j++) {
            if (system->low_level_spikes[i] && system->mid_level_spikes[j]) {
                FractalHashEntry* entry_low = hash_cache_lookup(system->cache, 
                    system->low_level_spikes[i]->source,
                    system->low_level_spikes[i]->fractalDimension,
                    system->low_level_spikes[i]->intensity);
                    
                FractalHashEntry* entry_mid = hash_cache_lookup(system->cache,
                    system->mid_level_spikes[j]->source,
                    system->mid_level_spikes[j]->fractalDimension,
                    system->mid_level_spikes[j]->intensity);

                if (entry_low && entry_mid && entry_low->access_count > 5 && entry_mid->access_count > 3) {
                    int index = i * system->max_mid_spikes + j;
                    system->low_to_mid_weights[index] *= 1.1f;
                    CLAMP(system->low_to_mid_weights[index]);
                }
            }
        }
    }

    for (int i = 0; i < system->max_low_spikes * system->max_mid_spikes; i++) {
        if (system->low_to_mid_weights[i] < 0.01f) {
            system->low_to_mid_weights[i] *= 0.9f;
        }
    }

    if (system->cache) {
        optimize_hash_energy(system->cache, 0.8f);
    }
}

float get_hierarchical_activation(HierarchicalSpikeSystem* system, const char* pattern)
{
    if (!system || !pattern) return 0.0f;

    float total_activation = 0.0f;
    int matches = 0;

    for (int i = 0; i < system->low_level_count; i++) {
        if (system->low_level_spikes[i] && strstr(system->low_level_spikes[i]->source, pattern)) {
            float activation = get_cached_activation(system->cache, system->low_level_spikes[i]);
            total_activation += activation * 0.3f;
            matches++;
        }
    }

    for (int i = 0; i < system->mid_level_count; i++) {
        if (system->mid_level_spikes[i] && strstr(system->mid_level_spikes[i]->source, pattern)) {
            float activation = get_cached_activation(system->cache, system->mid_level_spikes[i]);
            total_activation += activation * 0.5f;
            matches++;
        }
    }

    for (int i = 0; i < system->high_level_count; i++) {
        if (system->high_level_spikes[i] && strstr(system->high_level_spikes[i]->source, pattern)) {
            float activation = get_cached_activation(system->cache, system->high_level_spikes[i]);
            total_activation += activation * 0.8f;
            matches++;
        }
    }

    return matches > 0 ? total_activation / matches : 0.0f;
}

void print_hierarchical_system_status(const HierarchicalSpikeSystem* system)
{
    if (!system) return;

    printf("=== Hierarchical Spike System Status ===\n");
    printf("Low-level spikes: %d/%d\n", system->low_level_count, system->max_low_spikes);
    printf("Mid-level spikes: %d/%d\n", system->mid_level_count, system->max_mid_spikes);
    printf("High-level spikes: %d/%d\n", system->high_level_count, system->max_high_spikes);
    
    float avg_low_intensity = 0.0f;
    for (int i = 0; i < system->low_level_count; i++) {
        if (system->low_level_spikes[i]) {
            avg_low_intensity += system->low_level_spikes[i]->intensity;
        }
    }
    if (system->low_level_count > 0) avg_low_intensity /= system->low_level_count;

    float avg_mid_intensity = 0.0f;
    for (int i = 0; i < system->mid_level_count; i++) {
        if (system->mid_level_spikes[i]) {
            avg_mid_intensity += system->mid_level_spikes[i]->intensity;
        }
    }
    if (system->mid_level_count > 0) avg_mid_intensity /= system->mid_level_count;

    float avg_high_intensity = 0.0f;
    for (int i = 0; i < system->high_level_count; i++) {
        if (system->high_level_spikes[i]) {
            avg_high_intensity += system->high_level_spikes[i]->intensity;
        }
    }
    if (system->high_level_count > 0) avg_high_intensity /= system->high_level_count;

    printf("Average intensities - Low: %.3f, Mid: %.3f, High: %.3f\n", 
           avg_low_intensity, avg_mid_intensity, avg_high_intensity);
    
    if (system->cache) {
        printf("Cache size: %d/%d\n", system->cache->size, system->cache->capacity);
    } else {
        printf("Cache: not initialized\n");
    }
    printf("========================================\n");
}

// =============== FractalHashCache Implementation (с CURE-подобной кластеризацией) ===============
static unsigned long simple_hash(const char *str) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c;
    return hash;
}

FractalHashCache* create_fractal_hash_cache(int capacity) {
    FractalHashCache* cache = (FractalHashCache*)malloc(sizeof(FractalHashCache));
    if (!cache) return NULL;
    
    cache->capacity = capacity;
    cache->size = 0;
    cache->global_learning_rate = 0.01f;
    cache->decay_factor = 0.95f;
    cache->resonance_threshold = 0.7f;
    
    cache->entries = (FractalHashEntry**)calloc(capacity, sizeof(FractalHashEntry*));
    if (!cache->entries) {
        free(cache);
        return NULL;
    }
    
    return cache;
}

void destroy_fractal_hash_cache(FractalHashCache* cache) {
    if (!cache) return;
    
    for (int i = 0; i < cache->capacity; i++) {
        FractalHashEntry* entry = cache->entries[i];
        if (entry) {
            free(entry->pattern_hash);
            free(entry);
        }
    }
    
    free(cache->entries);
    free(cache);
}

char* generate_fractal_hash(const char* pattern, float dimension, float intensity) {
    int hash_size = snprintf(NULL, 0, "%s:%.2f:%.2f:%ld", pattern, dimension, intensity, time(NULL));
    char* hash = (char*)malloc(hash_size + 1);
    if (hash) {
        snprintf(hash, hash_size + 1, "%s:%.2f:%.2f:%ld", pattern, dimension, intensity, time(NULL));
    }
    return hash;
}

FractalHashEntry* hash_cache_lookup(FractalHashCache* cache, const char* pattern, 
                                   float dimension, float intensity) {
    if (!cache || !pattern) return NULL;
    
    char* target_hash = generate_fractal_hash(pattern, dimension, intensity);
    if (!target_hash) return NULL;
    
    unsigned long hash_val = simple_hash(target_hash);
    int index = hash_val % cache->capacity;
    
    FractalHashEntry* entry = cache->entries[index];
    
    int found = 0;
    if (entry && strcmp(entry->pattern_hash, target_hash) == 0) {
        entry->last_accessed = time(NULL);
        entry->access_count++;
        found = 1;
    }
    
    free(target_hash);
    return found ? entry : NULL;
}

void hash_cache_store(FractalHashCache* cache, const char* pattern, 
                     float dimension, float intensity, float activation) {
    if (!cache || !pattern || cache->size >= cache->capacity) return;
    
    char* hash = generate_fractal_hash(pattern, dimension, intensity);
    if (!hash) return;
    
    unsigned long hash_val = simple_hash(hash);
    int index = hash_val % cache->capacity;
    
    if (cache->entries[index]) {
        free(cache->entries[index]->pattern_hash);
        free(cache->entries[index]);
        cache->size--;
    }
    
    FractalHashEntry* entry = (FractalHashEntry*)malloc(sizeof(FractalHashEntry));
    if (!entry) {
        free(hash);
        return;
    }
    
    entry->pattern_hash = hash;
    entry->cached_activation = activation;
    entry->adaptive_learning_rate = cache->global_learning_rate;
    entry->fractal_coherence = 0.5f;
    entry->last_accessed = time(NULL);
    entry->access_count = 1;
    entry->spike_resonance_level = 0.7f;
    entry->energy_efficiency = 0.8f;
    
    // === НОВОЕ: CURE-подобные поля ===
    entry->fractal_dimension = dimension;
    entry->is_cluster_representative = 0;
    entry->cluster_radius = 0.0f;
    
    cache->entries[index] = entry;
    cache->size++;
}

// === НОВОЕ: CURE-подобная кластеризация ===
void hash_cache_clusterize(FractalHashCache* cache, float similarity_threshold) {
    if (!cache || similarity_threshold <= 0) return;
    
    for (int i = 0; i < cache->capacity; i++) {
        FractalHashEntry* entry_i = cache->entries[i];
        if (!entry_i) continue;
        
        for (int j = i + 1; j < cache->capacity; j++) {
            FractalHashEntry* entry_j = cache->entries[j];
            if (!entry_j) continue;
            
            float dim_diff = fabsf(entry_i->fractal_dimension - entry_j->fractal_dimension);
            float act_diff = fabsf(entry_i->cached_activation - entry_j->cached_activation);
            
            if (dim_diff < similarity_threshold && act_diff < 0.2f) {
                // entry_i становится представителем
                entry_i->is_cluster_representative = 1;
                entry_i->cluster_radius = fmaxf(dim_diff, act_diff);
                
                // entry_j удаляется
                free(entry_j->pattern_hash);
                free(entry_j);
                cache->entries[j] = NULL;
                cache->size--;
            }
        }
    }
}

// === НОВОЕ: найти ближайший кластер-представитель ===
FractalHashEntry* find_closest_representative(FractalHashCache* cache, float dimension, float intensity) {
    if (!cache) return NULL;
    
    FractalHashEntry* best = NULL;
    float best_distance = FLT_MAX;
    
    for (int i = 0; i < cache->capacity; i++) {
        FractalHashEntry* entry = cache->entries[i];
        if (!entry || !entry->is_cluster_representative) continue;
        
        float d_dim = fabsf(entry->fractal_dimension - dimension);
        float d_act = fabsf(entry->cached_activation - intensity);
        float distance = d_dim * 0.7f + d_act * 0.3f;
        
        if (distance < best_distance) {
            best_distance = distance;
            best = entry;
        }
    }
    
    return best;
}

float get_adaptive_learning_rate(FractalHashCache* cache, const char* pattern, 
                                float dimension, float intensity, float base_rate) {
    FractalHashEntry* entry = hash_cache_lookup(cache, pattern, dimension, intensity);
    if (entry) {
        return base_rate * (1.0f + entry->fractal_coherence * 0.5f);
    }
    return base_rate;
}

void update_hash_learning_rates(FractalHashCache* cache, float performance_factor) {
    if (!cache) return;
    
    for (int i = 0; i < cache->capacity; i++) {
        FractalHashEntry* entry = cache->entries[i];
        if (entry) {
            entry->adaptive_learning_rate *= (1.0f + performance_factor * 0.1f);
            CLAMP(entry->adaptive_learning_rate);
        }
    }
}

float calculate_hash_resonance(FractalHashEntry* entry1, FractalHashEntry* entry2) {
    if (!entry1 || !entry2) return 0.0f;
    
    float time_diff = fabsf((float)(entry1->last_accessed - entry2->last_accessed));
    float time_similarity = expf(-time_diff / 3600.0f);
    
    float activation_similarity = 1.0f - fabsf(entry1->cached_activation - entry2->cached_activation);
    
    return (time_similarity + activation_similarity) * 0.5f;
}

void optimize_hash_energy(FractalHashCache* cache, float target_efficiency) {
    if (!cache) return;
    
    long current_time = time(NULL);
    for (int i = 0; i < cache->capacity; i++) {
        FractalHashEntry* entry = cache->entries[i];
        if (entry) {
            float age = (float)(current_time - entry->last_accessed) / 3600.0f;
            if (age > 24.0f) {
                free(entry->pattern_hash);
                free(entry);
                cache->entries[i] = NULL;
                cache->size--;
            } else {
                entry->energy_efficiency = target_efficiency * (1.0f - age / 24.0f);
                CLAMP(entry->energy_efficiency);
            }
        }
    }
}

// =============== NeuralResonance Implementation ===============
NeuralResonance* create_neural_resonance(float frequency, float amplitude, float damping) {
    NeuralResonance* resonance = (NeuralResonance*)malloc(sizeof(NeuralResonance));
    if (!resonance) return NULL;
    
    resonance->frequency = frequency;
    resonance->amplitude = amplitude;
    resonance->phase = 0.0f;
    resonance->damping = damping;
    resonance->resonance_mode = 0;
    
    return resonance;
}

void destroy_neural_resonance(NeuralResonance* resonance) {
    free(resonance);
}

float apply_resonance(NeuralResonance* resonance, float input_signal, float time_delta) {
    if (!resonance) return input_signal;
    
    resonance->phase += resonance->frequency * time_delta;
    if (resonance->phase > 2 * M_PI) {
        resonance->phase -= 2 * M_PI;
    }
    
    float resonance_factor = sinf(resonance->phase) * resonance->amplitude;
    float damped_signal = input_signal * (1.0f - resonance->damping);
    
    return damped_signal + resonance_factor * input_signal;
}

void update_resonance_parameters(NeuralResonance* resonance, float learning_signal) {
    if (!resonance) return;
    
    resonance->frequency *= (1.0f + learning_signal * 0.01f);
    resonance->amplitude *= (1.0f + learning_signal * 0.02f);
    resonance->damping *= (1.0f - learning_signal * 0.005f);
    
    CLAMP(resonance->frequency);
    CLAMP(resonance->amplitude);
    CLAMP(resonance->damping);
}

float calculate_resonance_match(NeuralResonance* res1, NeuralResonance* res2) {
    if (!res1 || !res2) return 0.0f;
    
    float freq_match = 1.0f - fabsf(res1->frequency - res2->frequency);
    float amp_match = 1.0f - fabsf(res1->amplitude - res2->amplitude);
    
    return (freq_match + amp_match) * 0.5f;
}

// =============== НОВЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С ПАМЯТЬЮ ===============
float analyze_memory_patterns(FractalSpike** spikes, int spike_count) {
    if (spike_count == 0) return 0.5f;
    
    float total_coherence = 0.0f;
    int comparisons = 0;
    
    for (int i = 0; i < spike_count; i++) {
        for (int j = i + 1; j < spike_count; j++) {
            if (spikes[i] && spikes[j]) {
                const char* patterns[] = {spikes[i]->source, spikes[j]->source};
                float coherence = semantic_coherence(patterns, 2, 0.3f);
                total_coherence += coherence;
                comparisons++;
            }
        }
    }
    
    return comparisons > 0 ? total_coherence / comparisons : 0.5f;
}

void optimize_energy_usage(FractalActivation* act, float recent_activity) {
    if (!act) return;
    
    float target_consumption = 0.3f + recent_activity * 0.4f;
    float adjustment = (target_consumption - act->energyConsumption) * 0.1f;
    act->energyConsumption += adjustment;
    
    CLAMP(act->energyConsumption);
}

// =============== РЕЗОНАНСНЫЕ ФУНКЦИИ ДЛЯ АКТИВАЦИИ ===============
void apply_resonance_to_activation(FractalActivation* act, NeuralResonance* resonance) {
    if (!act || !resonance) return;
    
    float time_delta = 0.1f;
    act->baseActivation = apply_resonance(resonance, act->baseActivation, time_delta);
    act->harmonicActivation = apply_resonance(resonance, act->harmonicActivation, time_delta);
    
    CLAMP(act->baseActivation);
    CLAMP(act->harmonicActivation);
}

float calculate_network_resonance(FractalSpike** spikes, int count, float base_frequency) {
    if (count == 0) return base_frequency;
    
    float total_resonance = 0.0f;
    for (int i = 0; i < count; i++) {
        if (spikes[i]) {
            total_resonance += spikes[i]->intensity;
        }
    }
    
    return base_frequency * (1.0f + total_resonance / count * 0.5f);
}

// =============== ХЕШ-ИНТЕГРИРОВАННЫЕ ФУНКЦИИ ===============
float get_cached_activation(FractalHashCache* cache, FractalSpike* spike) {
    if (!cache || !spike) return 0.0f;
    
    FractalHashEntry* entry = hash_cache_lookup(cache, spike->source, 
                                              spike->fractalDimension, spike->intensity);
    if (entry) {
        return entry->cached_activation;
    }
    return spike->intensity;
}

void update_spike_learning_with_hash(FractalHashCache* cache, FractalSpike* spike, 
                                   FractalActivation* act) {
    if (!cache || !spike || !act) return;
    
    float learning_rate = get_adaptive_learning_rate(cache, spike->source,
                                                   spike->fractalDimension, spike->intensity,
                                                   0.01f);
    fractal_gradient_descent(act, learning_rate);
}

float calculate_fractal_hash_similarity(const char* hash1, const char* hash2) {
    if (!hash1 || !hash2) return 0.0f;
    
    int matches = 0;
    int length = MIN(strlen(hash1), strlen(hash2));
    
    for (int i = 0; i < length; i++) {
        if (hash1[i] == hash2[i]) {
            matches++;
        }
    }
    
    return length > 0 ? (float)matches / length : 0.0f;
}

// =============== ФРАКТАЛЬНОЕ ОБРАТНОЕ РАСПРОСТРАНЕНИЕ ===============
FractalBackprop* create_fractal_backprop(int max_depth) {
    FractalBackprop* bp = (FractalBackprop*)malloc(sizeof(FractalBackprop));
    if (!bp) return NULL;
    
    bp->depth = max_depth;
    bp->learning_rate = 0.01f;
    bp->momentum = 0.9f;
    bp->spike_error = 0.0f;
    
    bp->error_signals = (float*)malloc(max_depth * sizeof(float));
    bp->fractal_gradients = (float*)malloc(max_depth * sizeof(float));
    
    for (int i = 0; i < max_depth; i++) {
        bp->error_signals[i] = 0.0f;
        bp->fractal_gradients[i] = 0.0f;
    }
    
    return bp;
}

void destroy_fractal_backprop(FractalBackprop* bp) {
    if (!bp) return;
    
    free(bp->error_signals);
    free(bp->fractal_gradients);
    free(bp);
}

void fractal_backward_pass(FractalBackprop* bp, FractalActivation* act, 
                          float target_error, float current_activation) {
    if (!bp || !act) return;
    
    float output_error = target_error - current_activation;
    bp->spike_error = output_error;
    
    for (int d = 0; d < bp->depth; d++) {
        float depth_decay = expf(-d * 0.5f);
        bp->error_signals[d] = output_error * depth_decay;
        
        float fractal_grad = bp->error_signals[d] * 
                           (0.4f * act->baseActivation +
                            0.3f * act->harmonicActivation + 
                            0.3f * act->spikeResonance);
        
        bp->fractal_gradients[d] = bp->momentum * bp->fractal_gradients[d] + 
                                  (1.0f - bp->momentum) * fractal_grad;
    }
}

void apply_fractal_gradients(FractalActivation* act, FractalBackprop* bp) {
    if (!act || !bp) return;
    
    float total_gradient = 0.0f;
    for (int d = 0; d < bp->depth; d++) {
        total_gradient += bp->fractal_gradients[d] * expf(-d * 0.3f);
    }
    
    act->baseActivation += bp->learning_rate * total_gradient * 0.4f;
    act->harmonicActivation += bp->learning_rate * total_gradient * 0.3f;
    act->spikeResonance += bp->learning_rate * total_gradient * 0.3f;
    
    act->energyConsumption -= bp->learning_rate * bp->spike_error * 0.1f;
    
    CLAMP(act->baseActivation);
    CLAMP(act->harmonicActivation);
    CLAMP(act->spikeResonance);
    CLAMP(act->energyConsumption);
}

float calculate_fractal_error(FractalSpike* output, FractalSpike* target) {
    if (!output || !target) return 1.0f;
    
    float intensity_error = fabsf(output->intensity - target->intensity);
    float dimension_error = fabsf(output->fractalDimension - target->fractalDimension);
    
    return (intensity_error * 0.7f + dimension_error * 0.3f) * 
           (1.0f + output->fractalDimension * 0.2f);
}

// =============== ОБУЧЕНИЕ НА ЛЕТУ ===============
void fractal_online_learning(FractalHashCache* cache, NeuralResonance* resonance,
                           const char* input_pattern, float actual_output, 
                           float expected_output, float dimension) {
    if (!cache || !resonance) return;
    
    float error = expected_output - actual_output;
    float abs_error = fabsf(error);
    
    float adaptive_lr = 0.01f * (1.0f + abs_error * 2.0f);
    
    FractalHashEntry* entry = hash_cache_lookup(cache, input_pattern, dimension, actual_output);
    if (entry) {
        entry->cached_activation += adaptive_lr * error;
        CLAMP(entry->cached_activation);
        
        entry->adaptive_learning_rate *= (1.0f + error * 0.1f);
        CLAMP(entry->adaptive_learning_rate);
    }
    
    resonance->frequency += adaptive_lr * error * 0.1f;
    resonance->amplitude += adaptive_lr * error * 0.05f;
    resonance->damping -= adaptive_lr * error * 0.02f;
    
    CLAMP(resonance->frequency);
    CLAMP(resonance->amplitude);
    CLAMP(resonance->damping);
    
    optimize_hash_energy(cache, 0.8f - abs_error * 0.3f);
}
