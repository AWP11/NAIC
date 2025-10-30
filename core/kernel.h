#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "fractal_tensor.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef CHECK_MEMORY_STABILITY
#define CHECK_MEMORY_STABILITY(memory, threshold) do { \
    if ((memory)->count > (threshold)) { \
        printf("[Memory] Стабилизация памяти: %d > %d\n", (memory)->count, (int)(threshold)); \
        compact_memory(memory); \
    } \
} while(0)
#endif

#ifndef CLAMP
#define CLAMP(x) do { if ((x) < 0.0f) (x) = 0.0f; if ((x) > 1.0f) (x) = 1.0f; } while(0)
#endif

#define SPIKE_LEVEL_LOW 0
#define SPIKE_LEVEL_MID 1
#define SPIKE_LEVEL_HIGH 2

typedef struct FractalNeuron FractalNeuron;
typedef struct SynapseGate SynapseGate;
typedef struct Connection Connection;
typedef struct FractalField FractalField;

typedef struct FractalSpike FractalSpike;
typedef struct FractalActivation FractalActivation;
typedef struct FractalHashEntry FractalHashEntry;
typedef struct FractalHashCache FractalHashCache;
typedef struct NeuralResonance NeuralResonance;
typedef struct FractalBackprop FractalBackprop;
typedef struct HierarchicalSpikeSystem HierarchicalSpikeSystem;

typedef struct {
    long timestamp;
    float intensity;
    float fractalDimension;
    int pathSize;
    int source_len;
} MemoryHeader;

typedef struct NeuralMemory {
    FractalSpike** neurons;
    int count;
    int capacity;
    time_t last_update;
} NeuralMemory;

struct FractalNeuron {
    long timestamp;
    float potential;
    float threshold;
    int fired;
};

struct SynapseGate {
    float weight;
    int state;
    float eligibility_trace;
    float neuromodulator_level;
    float resonance_factor;
};

struct Connection {
    int pre_neuron_id;
    int post_neuron_id;
    SynapseGate* gate;
};

struct FractalField {
    FractalNeuron** neurons;
    int neuron_count;
    int neuron_capacity;

    Connection** connections;
    int connection_count;
    int connection_capacity;

    float growth_threshold;
    int max_neurons;
    int max_connections;

    float global_reward_signal;
    int is_critical;
    long last_growth_time;
};

struct FractalSpike {
    long timestamp;
    float intensity;
    char* source;
    float fractalDimension;
    char** propagationPath;
    int pathSize;
};

struct FractalActivation {
    float baseActivation;
    float harmonicActivation;
    float spikeResonance;
    int fractalDepth;
    float energyConsumption;
};

struct FractalHashEntry {
    char* pattern_hash;
    float cached_activation;
    float adaptive_learning_rate;
    float fractal_coherence;
    float spike_resonance_level;
    float energy_efficiency;
    time_t last_accessed;
    int access_count;

    float fractal_dimension;
    int is_cluster_representative;
    float cluster_radius;
};

struct FractalHashCache {
    FractalHashEntry** entries;
    int capacity;
    int size;
    float global_learning_rate;
    float decay_factor;
    float resonance_threshold;
};

struct NeuralResonance {
    float frequency;
    float amplitude;
    float phase;
    float damping;
    int resonance_mode;
};

struct FractalBackprop {
    int depth;
    float learning_rate;
    float momentum;
    float spike_error;
    float* error_signals;
    float* fractal_gradients;
};

struct HierarchicalSpikeSystem {
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
};

void forget_old_neurons(NeuralMemory* memory, float forget_ratio);
void update_neuron_importance(NeuralMemory* memory);
float calculate_neuron_importance(FractalSpike* neuron, long current_time);

FractalField* create_fractal_field(int initial_neurons, int initial_connections);
void destroy_fractal_field(FractalField* field);
FractalNeuron* add_fractal_neuron(FractalField* field);
Connection* add_connection(FractalField* field, int pre_id, int post_id);
void propagate_fractal_field(FractalField* field, float global_reward);
void update_fractal_field(FractalField* field);
void check_growth_conditions(FractalField* field);
void grow_fractal_field(FractalField* field);

FractalSpike* create_fractal_spike(long timestamp, float intensity, const char* source, float fractalDimension, char** path, int pathSize);
void destroy_fractal_spike(FractalSpike* spike);
void print_fractal_spike(const FractalSpike* spike);

FractalActivation* create_fractal_activation(float baseActivation, float harmonicActivation, float spikeResonance, int fractalDepth, float energyConsumption);
void destroy_fractal_activation(FractalActivation* act);
float get_total_activation(const FractalActivation* act);
void print_fractal_activation(const FractalActivation* act);
void fractal_gradient_descent(FractalActivation* act, float learning_rate);

FractalHashCache* create_fractal_hash_cache(int capacity);
void destroy_fractal_hash_cache(FractalHashCache* cache);
void hash_cache_store(FractalHashCache* cache, const char* pattern, float dimension, float intensity, float activation);
FractalHashEntry* hash_cache_lookup(FractalHashCache* cache, const char* pattern, float dimension, float intensity);
float get_adaptive_learning_rate(FractalHashCache* cache, const char* pattern, float dimension, float intensity, float base_rate);
void update_hash_learning_rates(FractalHashCache* cache, float performance_factor);
void optimize_hash_energy(FractalHashCache* cache, float target_efficiency);
void hash_cache_clusterize(FractalHashCache* cache, float similarity_threshold);
FractalHashEntry* find_closest_representative(FractalHashCache* cache, float dimension, float intensity);

NeuralResonance* create_neural_resonance(float frequency, float amplitude, float damping);
void destroy_neural_resonance(NeuralResonance* resonance);
float apply_resonance(NeuralResonance* resonance, float input_signal, float time_delta);
void update_resonance_parameters(NeuralResonance* resonance, float learning_signal);
float calculate_resonance_match(NeuralResonance* res1, NeuralResonance* res2);

FractalBackprop* create_fractal_backprop(int max_depth);
void destroy_fractal_backprop(FractalBackprop* bp);
void fractal_backward_pass(FractalBackprop* bp, FractalActivation* act, float target_error, float current_activation);
void apply_fractal_gradients(FractalActivation* act, FractalBackprop* bp);
float calculate_fractal_error(FractalSpike* output, FractalSpike* target);

HierarchicalSpikeSystem* create_hierarchical_spike_system(int max_low_spikes, int max_mid_spikes, int max_high_spikes);
void destroy_hierarchical_spike_system(HierarchicalSpikeSystem* system);
void add_spike_to_hierarchy(HierarchicalSpikeSystem* system, FractalSpike* spike, int level);
float propagate_through_hierarchy(HierarchicalSpikeSystem* system, FractalSpike* input_spike);
void optimize_hierarchical_connections(HierarchicalSpikeSystem* system);
float get_hierarchical_activation(HierarchicalSpikeSystem* system, const char* pattern);
void print_hierarchical_system_status(const HierarchicalSpikeSystem* system);

NeuralMemory* create_neural_memory(int capacity);
void destroy_neural_memory(NeuralMemory* memory);
void add_neuron_to_memory(NeuralMemory* memory, FractalSpike* neuron);

NeuralMemory* load_memory_from_file(const char* filename);
void save_memory_to_file(NeuralMemory* memory, const char* filename);

void compact_memory(NeuralMemory* memory);
void optimize_memory_structure(NeuralMemory* memory);
int get_memory_stats(NeuralMemory* memory);

float fractal_connectivity(float dimension, float intensity, int depth);
float resonance_amplification(float base, float harmonic, float resonance);
float adaptive_learning_rate(float current_rate, float performance, float stability);
float energy_balance(float consumption, float activation, float target_efficiency);
float semantic_coherence(const char** patterns, int pattern_count, float base_coherence);
float get_cached_activation(FractalHashCache* cache, FractalSpike* spike);
void update_spike_learning_with_hash(FractalHashCache* cache, FractalSpike* spike, FractalActivation* act);
void fractal_online_learning(FractalHashCache* cache, NeuralResonance* resonance, const char* input_pattern, float actual_output, float expected_output, float dimension);
void apply_resonance_to_activation(FractalActivation* act, NeuralResonance* resonance);

#endif