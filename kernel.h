#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> // для expf(), tanhf(), logf(), fabsf()

// =============== FractalSpike ===============
typedef struct {
    long timestamp;
    float intensity;
    char* source;
    float fractalDimension;
    char** propagationPath;
    int pathSize;
} FractalSpike;

FractalSpike* create_fractal_spike(long timestamp, float intensity, const char* source, float fractalDimension, char** path, int pathSize);
void destroy_fractal_spike(FractalSpike* spike);
void print_fractal_spike(const FractalSpike* spike);

// =============== FractalActivation ===============
typedef struct {
    float baseActivation;
    float harmonicActivation;
    float spikeResonance;
    int fractalDepth;
    float energyConsumption;
} FractalActivation;

FractalActivation* create_fractal_activation(
    float baseActivation,
    float harmonicActivation,
    float spikeResonance,
    int fractalDepth,
    float energyConsumption
);

void destroy_fractal_activation(FractalActivation* act);
float get_total_activation(const FractalActivation* act);
void print_fractal_activation(const FractalActivation* act);

// =============== Fractal Learning ===============
void fractal_gradient_descent(FractalActivation *act, float learning_rate);

// =============== ГИБРИДНЫЙ ХЕШ (КЭШ + LEARNING RATE) ===============
typedef struct {
    char* pattern_hash;          // Хеш паттерна
    float cached_activation;     // Кэшированная активация
    float adaptive_learning_rate;// Адаптивная скорость обучения
    float fractal_coherence;     // Фрактальная согласованность
    long last_accessed;          // Время последнего доступа
    int access_count;            // Счетчик обращений
    float spike_resonance_level; // Уровень резонанса спайков
    float energy_efficiency;     // Энергетическая эффективность
} FractalHashEntry;

typedef struct {
    FractalHashEntry** entries;
    int capacity;
    int size;
    float global_learning_rate;  // Глобальная базовая скорость обучения
    float decay_factor;          // Фактор затухания кэша
    float resonance_threshold;   // Порог резонанса для обновления
} FractalHashCache;

// Создание и управление хеш-кэшем
FractalHashCache* create_fractal_hash_cache(int capacity);
void destroy_fractal_hash_cache(FractalHashCache* cache);

// Основные операции хеширования
char* generate_fractal_hash(const char* pattern, float dimension, float intensity);
FractalHashEntry* hash_cache_lookup(FractalHashCache* cache, const char* pattern, 
                                   float dimension, float intensity);
void hash_cache_store(FractalHashCache* cache, const char* pattern, 
                     float dimension, float intensity, float activation);

// Адаптивное обучение через хеш
float get_adaptive_learning_rate(FractalHashCache* cache, const char* pattern, 
                                float dimension, float intensity, float base_rate);
void update_hash_learning_rates(FractalHashCache* cache, float performance_factor);

// Фрактальные операции с хешем
float calculate_hash_resonance(FractalHashEntry* entry1, FractalHashEntry* entry2);
void optimize_hash_energy(FractalHashCache* cache, float target_efficiency);

// =============== НЕЙРОННЫЙ РЕЗОНАНС ===============
typedef struct {
    float frequency;
    float amplitude;
    float phase;
    float damping;
    int resonance_mode;
} NeuralResonance;

NeuralResonance* create_neural_resonance(float frequency, float amplitude, float damping);
void destroy_neural_resonance(NeuralResonance* resonance);
float apply_resonance(NeuralResonance* resonance, float input_signal, float time_delta);
void update_resonance_parameters(NeuralResonance* resonance, float learning_signal);
float calculate_resonance_match(NeuralResonance* res1, NeuralResonance* res2);

// =============== ОБЩИЕ ФОРМУЛЫ ===============
float fractal_connectivity(float dimension, float intensity, int depth);
float resonance_amplification(float base, float harmonic, float resonance);

// =============== МЕТАФОРМУЛЫ ===============
float adaptive_learning_rate(float current_rate, float performance, float stability);
float energy_balance(float consumption, float activation, float target_efficiency);
float semantic_coherence(const char** patterns, int pattern_count, float base_coherence);

// =============== ФУНКЦИИ ДЛЯ РАБОТЫ С ПАМЯТЬЮ ===============
float analyze_memory_patterns(FractalSpike** spikes, int spike_count);
void optimize_energy_usage(FractalActivation* act, float recent_activity);

// =============== РЕЗОНАНСНЫЕ ФУНКЦИИ ДЛЯ АКТИВАЦИИ ===============
void apply_resonance_to_activation(FractalActivation* act, NeuralResonance* resonance);
float calculate_network_resonance(FractalSpike** spikes, int count, float base_frequency);

// =============== ХЕШ-ИНТЕГРИРОВАННЫЕ ФУНКЦИИ ===============
float get_cached_activation(FractalHashCache* cache, FractalSpike* spike);
void update_spike_learning_with_hash(FractalHashCache* cache, FractalSpike* spike, 
                                   FractalActivation* act);
float calculate_fractal_hash_similarity(const char* hash1, const char* hash2);

// =============== ВСПОМОГАТЕЛЬНЫЕ МАКРОСЫ ===============
#define CLAMP(x) do { \
    if ((x) > 1.0f) (x) = 1.0f; \
    else if ((x) < 0.0f) (x) = 0.0f; \
} while(0)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP_01(x) (MAX(0.0f, MIN(1.0f, (x))))

#define HASH_CACHE_HIT_BONUS 0.1f
#define FRACTAL_COHERENCE_BOOST 0.15f

typedef struct {
    float* error_signals;       // Сигналы ошибок по глубине
    float* fractal_gradients;   // Градиенты по фрактальным измерениям
    int depth;                  // Глубина распространения
    float learning_rate;        // Динамическая скорость обучения
    float momentum;             // Моментум для устойчивости
    float spike_error;          // Ошибка спайковой активности
} FractalBackprop;

// Прототипы функций (ДОБАВЛЯЕМ ПЕРЕД #endif)
FractalBackprop* create_fractal_backprop(int max_depth);
void destroy_fractal_backprop(FractalBackprop* bp);
void fractal_backward_pass(FractalBackprop* bp, FractalActivation* act, 
                          float target_error, float current_activation);
void apply_fractal_gradients(FractalActivation* act, FractalBackprop* bp);
float calculate_fractal_error(FractalSpike* output, FractalSpike* target);
void fractal_online_learning(FractalHashCache* cache, NeuralResonance* resonance,
                           const char* input_pattern, float actual_output, 
                           float expected_output, float dimension);

#endif // KERNEL_H