#include "kernel.h"
#include <time.h>
#include <ctype.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============== ОБЩИЕ ФОРМУЛЫ ===============

// 1. Формула фрактальной связности (улучшает работу с памятью)
float fractal_connectivity(float dimension, float intensity, int depth) {
    return (dimension * intensity) / (1.0f + logf(1.0f + depth));
}

// 2. Формула резонансного усиления (для активации нейронов)
float resonance_amplification(float base, float harmonic, float resonance) {
    return base * (1.0f + harmonic * resonance);
}

// =============== МЕТАФОРМУЛЫ ===============

// 1. Метаформула адаптивного обучения (динамический learning rate)
float adaptive_learning_rate(float current_rate, float performance, float stability) {
    float adaptive = current_rate * (1.0f + tanhf(performance * 2.0f - 1.0f));
    return adaptive * (0.5f + stability * 0.5f);
}

// 2. Метаформула энергетического баланса (оптимизация энергии)
float energy_balance(float consumption, float activation, float target_efficiency) {
    float imbalance = fabsf(consumption - activation * 0.7f);
    return target_efficiency * expf(-imbalance * 2.0f);
}

// 3. Метаформула семантической когерентности (для работы с памятью)
float semantic_coherence(const char** patterns, int pattern_count, float base_coherence) {
    if (pattern_count == 0) return base_coherence;
    
    // Простая имитация анализа паттернов
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

    // ПРИМЕНЕНИЕ ФОРМУЛЫ: коррекция интенсивности на основе фрактальной связности
    spike->intensity = fractal_connectivity(fractalDimension, intensity, pathSize);

    spike->pathSize = pathSize;
    if (pathSize > 0) {
        spike->propagationPath = (char**)malloc(pathSize * sizeof(char*));
        for (int i = 0; i < pathSize; i++) {
            spike->propagationPath[i] = strdup(path[i]);
        }
        
        // ПРИМЕНЕНИЕ МЕТАФОРМУЛЫ: оценка семантической когерентности пути
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

    // ПРИМЕНЕНИЕ ФОРМУЛЫ: резонансное усиление активации
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

    // ПРИМЕНЕНИЕ МЕТАФОРМУЛЫ: энергетический баланс
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

    // ПРИМЕНЕНИЕ МЕТАФОРМУЛЫ: адаптивный learning rate
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

    // Ограничение [0, 1]
    CLAMP(act->baseActivation);
    CLAMP(act->harmonicActivation);
    CLAMP(act->spikeResonance);
    CLAMP(act->energyConsumption);

    // Фрактальный шум (уменьшен благодаря адаптивному обучению)
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

// =============== FractalHashCache Implementation ===============

// Простая хеш-функция для строк
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
    // Простая реализация генерации хеша
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
    
    // Сравниваем до освобождения памяти
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
    
    // Освобождаем старую запись если есть
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
    
    cache->entries[index] = entry;
    cache->size++;
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
    float time_similarity = expf(-time_diff / 3600.0f); // 1 час
    
    float activation_similarity = 1.0f - fabsf(entry1->cached_activation - entry2->cached_activation);
    
    return (time_similarity + activation_similarity) * 0.5f;
}

void optimize_hash_energy(FractalHashCache* cache, float target_efficiency) {
    if (!cache) return;
    
    long current_time = time(NULL);
    for (int i = 0; i < cache->capacity; i++) {
        FractalHashEntry* entry = cache->entries[i];
        if (entry) {
            float age = (float)(current_time - entry->last_accessed) / 3600.0f; // в часах
            if (age > 24.0f) { // Удаляем старые записи (старше 24 часов)
                free(entry->pattern_hash);
                free(entry);
                cache->entries[i] = NULL;
                cache->size--;
            } else {
                // Оптимизируем энергоэффективность
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
    
    // Простая модель резонанса - гармонический осциллятор
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
    
    // Адаптация параметров на основе сигнала обучения
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

// Анализ паттернов памяти (использует семантическую когерентность)
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

// Оптимизация энергопотребления на основе активности
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
    
    float time_delta = 0.1f; // Фиксированный временной шаг
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
    
    // Вычисляем ошибку на выходе
    float output_error = target_error - current_activation;
    bp->spike_error = output_error;
    
    // Распространяем ошибку по фрактальным глубинам
    for (int d = 0; d < bp->depth; d++) {
        // Фрактальное затухание ошибки с глубиной
        float depth_decay = expf(-d * 0.5f);
        bp->error_signals[d] = output_error * depth_decay;
        
        // Градиенты учитывают фрактальную размерность
        float fractal_grad = bp->error_signals[d] * 
                           (0.4f * act->baseActivation +
                            0.3f * act->harmonicActivation + 
                            0.3f * act->spikeResonance);
        
        // Применяем моментум
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
    
    // Применяем градиенты к параметрам активации
    act->baseActivation += bp->learning_rate * total_gradient * 0.4f;
    act->harmonicActivation += bp->learning_rate * total_gradient * 0.3f;
    act->spikeResonance += bp->learning_rate * total_gradient * 0.3f;
    
    // Адаптируем энергопотребление на основе ошибки
    act->energyConsumption -= bp->learning_rate * bp->spike_error * 0.1f;
    
    // Ограничиваем значения
    CLAMP(act->baseActivation);
    CLAMP(act->harmonicActivation);
    CLAMP(act->spikeResonance);
    CLAMP(act->energyConsumption);
}

float calculate_fractal_error(FractalSpike* output, FractalSpike* target) {
    if (!output || !target) return 1.0f; // Максимальная ошибка
    
    float intensity_error = fabsf(output->intensity - target->intensity);
    float dimension_error = fabsf(output->fractalDimension - target->fractalDimension);
    
    // Взвешенная ошибка с учетом фрактальной сложности
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
    
    // Адаптируем скорость обучения на основе ошибки
    float adaptive_lr = 0.01f * (1.0f + abs_error * 2.0f);
    
    // Обновляем кэш на основе ошибки
    FractalHashEntry* entry = hash_cache_lookup(cache, input_pattern, dimension, actual_output);
    if (entry) {
        // Корректируем кэшированную активацию
        entry->cached_activation += adaptive_lr * error;
        CLAMP(entry->cached_activation);
        
        // Адаптируем скорость обучения записи
        entry->adaptive_learning_rate *= (1.0f + error * 0.1f);
        CLAMP(entry->adaptive_learning_rate);
    }
    
    // Обновляем резонансные параметры на основе ошибки
    resonance->frequency += adaptive_lr * error * 0.1f;
    resonance->amplitude += adaptive_lr * error * 0.05f;
    resonance->damping -= adaptive_lr * error * 0.02f;
    
    CLAMP(resonance->frequency);
    CLAMP(resonance->amplitude);
    CLAMP(resonance->damping);
    
    // Оптимизируем энергоэффективность кэша
    optimize_hash_energy(cache, 0.8f - abs_error * 0.3f);
}