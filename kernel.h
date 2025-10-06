#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Макросы
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef CLAMP
#define CLAMP(x) do { if ((x) < 0.0f) (x) = 0.0f; if ((x) > 1.0f) (x) = 1.0f; } while(0)
#endif

#ifndef HASH_CACHE_HIT_BONUS
#define HASH_CACHE_HIT_BONUS 0.1f
#endif

// Типы для иерархии
#define SPIKE_LEVEL_LOW 0
#define SPIKE_LEVEL_MID 1
#define SPIKE_LEVEL_HIGH 2

// === СТРУКТУРЫ ===

// FractalSpike — фрактальный спайк
typedef struct {
    long timestamp;
    float intensity;
    char* source;
    float fractalDimension;
    char** propagationPath;
    int pathSize;
} FractalSpike;

// FractalActivation — фрактальная активация
typedef struct {
    float baseActivation;
    float harmonicActivation;
    float spikeResonance;
    int fractalDepth;
    float energyConsumption;
} FractalActivation;

// FractalHashEntry — элемент кэша
typedef struct {
    char* pattern_hash;
    float cached_activation;
    float adaptive_learning_rate;
    float fractal_coherence;
    float spike_resonance_level;
    float energy_efficiency;
    time_t last_accessed;
    int access_count;

    // === НОВОЕ: CURE-подобные поля ===
    float fractal_dimension;            // Фрактальная размерность
    int is_cluster_representative;      // Является ли кластерным представителем
    float cluster_radius;               // Радиус кластера
} FractalHashEntry;

// FractalHashCache — кэш с хэшем
typedef struct {
    FractalHashEntry** entries;
    int capacity;
    int size;
    float global_learning_rate;
    float decay_factor;
    float resonance_threshold;
} FractalHashCache;

// NeuralResonance — нейронный резонанс
typedef struct {
    float frequency;
    float amplitude;
    float phase;
    float damping;
    int resonance_mode;
} NeuralResonance;

// FractalBackprop — фрактальное обратное распространение
typedef struct {
    int depth;
    float learning_rate;
    float momentum;
    float spike_error;
    float* error_signals;
    float* fractal_gradients;
} FractalBackprop;

// HierarchicalSpikeSystem — иерархическая система спайков
typedef struct {
    FractalSpike** low_level_spikes;
    FractalSpike** mid_level_spikes;
    FractalSpike** high_level_spikes;
    int low_level_count;
    int mid_level_count;
    int high_level_count;
    int max_low_spikes;
    int max_mid_spikes;
    int max_high_spikes;

    float* low_to_mid_weights;
    float* mid_to_high_weights;
    float* high_to_mid_weights;
    float* mid_to_low_weights;

    FractalHashCache* cache;
    long last_optimization_time;
} HierarchicalSpikeSystem;

// === ФУНКЦИИ FractalSpike ===
FractalSpike* create_fractal_spike(long timestamp, float intensity, const char* source, float fractalDimension, char** path, int pathSize);
void destroy_fractal_spike(FractalSpike* spike);
void print_fractal_spike(const FractalSpike* spike);

// === ФУНКЦИИ FractalActivation ===
FractalActivation* create_fractal_activation(float baseActivation, float harmonicActivation, float spikeResonance, int fractalDepth, float energyConsumption);
void destroy_fractal_activation(FractalActivation* act);
float get_total_activation(const FractalActivation* act);
void print_fractal_activation(const FractalActivation* act);
void fractal_gradient_descent(FractalActivation* act, float learning_rate);

// === ФУНКЦИИ FractalHashCache (с CURE-подобной кластеризацией) ===
FractalHashCache* create_fractal_hash_cache(int capacity);
void destroy_fractal_hash_cache(FractalHashCache* cache);
void hash_cache_store(FractalHashCache* cache, const char* pattern, float dimension, float intensity, float activation);
FractalHashEntry* hash_cache_lookup(FractalHashCache* cache, const char* pattern, float dimension, float intensity);
float get_adaptive_learning_rate(FractalHashCache* cache, const char* pattern, float dimension, float intensity, float base_rate);
void update_hash_learning_rates(FractalHashCache* cache, float performance_factor);
void optimize_hash_energy(FractalHashCache* cache, float target_efficiency);

// === НОВОЕ: CURE-подобные функции ===
void hash_cache_clusterize(FractalHashCache* cache, float similarity_threshold);
FractalHashEntry* find_closest_representative(FractalHashCache* cache, float dimension, float intensity);

// === ФУНКЦИИ NeuralResonance ===
NeuralResonance* create_neural_resonance(float frequency, float amplitude, float damping);
void destroy_neural_resonance(NeuralResonance* resonance);
float apply_resonance(NeuralResonance* resonance, float input_signal, float time_delta);
void update_resonance_parameters(NeuralResonance* resonance, float learning_signal);
float calculate_resonance_match(NeuralResonance* res1, NeuralResonance* res2);

// === ФУНКЦИИ FractalBackprop ===
FractalBackprop* create_fractal_backprop(int max_depth);
void destroy_fractal_backprop(FractalBackprop* bp);
void fractal_backward_pass(FractalBackprop* bp, FractalActivation* act, float target_error, float current_activation);
void apply_fractal_gradients(FractalActivation* act, FractalBackprop* bp);
float calculate_fractal_error(FractalSpike* output, FractalSpike* target);

// === ФУНКЦИИ HierarchicalSpikeSystem ===
HierarchicalSpikeSystem* create_hierarchical_spike_system(int max_low_spikes, int max_mid_spikes, int max_high_spikes);
void destroy_hierarchical_spike_system(HierarchicalSpikeSystem* system);
void add_spike_to_hierarchy(HierarchicalSpikeSystem* system, FractalSpike* spike, int level);
float propagate_through_hierarchy(HierarchicalSpikeSystem* system, FractalSpike* input_spike);
void optimize_hierarchical_connections(HierarchicalSpikeSystem* system);
float get_hierarchical_activation(HierarchicalSpikeSystem* system, const char* pattern);
void print_hierarchical_system_status(const HierarchicalSpikeSystem* system);

// === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
float fractal_connectivity(float dimension, float intensity, int depth);
float resonance_amplification(float base, float harmonic, float resonance);
float adaptive_learning_rate(float current_rate, float performance, float stability);
float energy_balance(float consumption, float activation, float target_efficiency);
float semantic_coherence(const char** patterns, int pattern_count, float base_coherence);
float get_cached_activation(FractalHashCache* cache, FractalSpike* spike);
void update_spike_learning_with_hash(FractalHashCache* cache, FractalSpike* spike, FractalActivation* act);
void fractal_online_learning(FractalHashCache* cache, NeuralResonance* resonance, const char* input_pattern, float actual_output, float expected_output, float dimension);
void apply_resonance_to_activation(FractalActivation* act, NeuralResonance* resonance);

#endif // KERNEL_H