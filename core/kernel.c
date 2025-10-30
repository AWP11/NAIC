#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <fractal_tensor.h>

// === ПЛАВНОЕ ЗАБЫВАНИЕ НЕЙРОНОВ ===

void update_neuron_importance(NeuralMemory* memory) {
    if (!memory || memory->count == 0) return;
    long current_time = time(NULL);

    for (int i = 0; i < memory->count; i++) {
        if (memory->neurons[i]) {
            float importance = calculate_neuron_importance(memory->neurons[i], current_time);
            // В простейшем случае, importance может быть просто intensity
            // В более сложных, это может быть функция intensity, fractalDimension, времени и др.
            // Для совместимости, обновляем intensity как "рабочую" важность
            memory->neurons[i]->intensity = importance;
        }
    }
}

float calculate_neuron_importance(FractalSpike* neuron, long current_time) {
    if (!neuron) return 0.0f;

    float base_importance = neuron->intensity;
    float time_factor = 1.0f;
    float stability_factor = 1.0f + neuron->fractalDimension; // Более высокая фрактальная размерность может означать более стабильную структуру

    // Пример: снижение важности со временем, но с учётом стабильности
    long time_diff = current_time - neuron->timestamp;
    if (time_diff > 3600) { // Если прошло больше часа
        time_factor = fmaxf(0.1f, 1.0f - (float)time_diff / (3600.0f * 24.0f)); // Плавное уменьшение в течение суток
    }

    float importance = base_importance * time_factor * stability_factor;
    CLAMP(importance);
    return importance;
}

void forget_old_neurons(NeuralMemory* memory, float forget_ratio) {
    if (!memory || memory->count <= 10) return; // Минимум 10 нейронов

    // Сначала обновим важность
    update_neuron_importance(memory);

    // Определяем сколько забывать
    int target_count = (int)(memory->count * (1.0f - forget_ratio));
    target_count = target_count < 10 ? 10 : target_count; // Не ниже минимума
    if (memory->count <= target_count) return;

    // Сортируем нейроны по важности (intensity)
    for (int i = 0; i < memory->count - 1; i++) {
        for (int j = i + 1; j < memory->count; j++) {
            if (memory->neurons[i] && memory->neurons[j]) {
                if (memory->neurons[i]->intensity < memory->neurons[j]->intensity) {
                    FractalSpike* temp = memory->neurons[i];
                    memory->neurons[i] = memory->neurons[j];
                    memory->neurons[j] = temp;
                }
            }
        }
    }

    // Удаляем наименее важные
    int removed_count = 0;
    for (int i = target_count; i < memory->count; i++) {
        if (memory->neurons[i]) {
            printf("[Забывание] Удален: '%s' (важность: %.3f)\n", memory->neurons[i]->source, memory->neurons[i]->intensity);
            destroy_fractal_spike(memory->neurons[i]);
            memory->neurons[i] = NULL;
            removed_count++;
        }
    }
    memory->count = target_count;
    printf("[Забывание] Удалено %d нейронов. Осталось: %d\n", removed_count, memory->count);
}

// === ФУНДАМЕНТ: FractalField ===

SynapseGate* create_synapse_gate(void) {
    SynapseGate* gate = (SynapseGate*)malloc(sizeof(SynapseGate));
    if (!gate) return NULL;
    gate->weight = (float)rand() / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
    gate->state = 1; // Открыто
    gate->eligibility_trace = 0.0f;
    gate->neuromodulator_level = 0.0f;
    gate->resonance_factor = 1.0f;
    return gate;
}

void destroy_synapse_gate(SynapseGate* gate) {
    if (gate) free(gate);
}

FractalField* create_fractal_field(int initial_neurons, int initial_connections) {
    FractalField* field = (FractalField*)malloc(sizeof(FractalField));
    if (!field) return NULL;

    field->neurons = (FractalNeuron**)malloc(initial_neurons * sizeof(FractalNeuron*));
    field->neuron_capacity = initial_neurons;
    field->neuron_count = 0;

    field->connections = (Connection**)malloc(initial_connections * sizeof(Connection*));
    field->connection_capacity = initial_connections;
    field->connection_count = 0;

    field->growth_threshold = 0.1f;
    field->max_neurons = 1000;
    field->max_connections = 5000;

    field->global_reward_signal = 0.0f;
    field->is_critical = 0;
    field->last_growth_time = time(NULL);

    if (!field->neurons || !field->connections) {
        free(field->neurons);
        free(field->connections);
        free(field);
        return NULL;
    }

    for (int i = 0; i < field->neuron_capacity; i++) {
        field->neurons[i] = NULL;
    }
    for (int i = 0; i < field->connection_capacity; i++) {
        field->connections[i] = NULL;
    }

    return field;
}

void destroy_fractal_field(FractalField* field) {
    if (!field) return;

    for (int i = 0; i < field->neuron_count; i++) {
        if (field->neurons[i]) {
            // FractalNeuron может не требовать сложной очистки, просто free
            free(field->neurons[i]);
        }
    }
    free(field->neurons);

    for (int i = 0; i < field->connection_count; i++) {
        if (field->connections[i]) {
            destroy_synapse_gate(field->connections[i]->gate);
            free(field->connections[i]);
        }
    }
    free(field->connections);
    free(field);
}

FractalNeuron* add_fractal_neuron(FractalField* field) {
    if (!field || field->neuron_count >= field->neuron_capacity) {
        // Попытка расширить
        if (field->neuron_count >= field->max_neurons) return NULL; // Достигнут лимит
        int new_capacity = field->neuron_capacity * 2;
        if (new_capacity > field->max_neurons) new_capacity = field->max_neurons;
        FractalNeuron** temp = (FractalNeuron**)realloc(field->neurons, new_capacity * sizeof(FractalNeuron*));
        if (!temp) return NULL;
        field->neurons = temp;
        field->neuron_capacity = new_capacity;
    }

    FractalNeuron* neuron = (FractalNeuron*)malloc(sizeof(FractalNeuron));
    if (!neuron) return NULL;

    neuron->timestamp = time(NULL);
    neuron->potential = 0.0f;
    neuron->threshold = 0.5f + (float)rand() / RAND_MAX * 0.3f; // [0.5, 0.8]
    neuron->fired = 0;

    field->neurons[field->neuron_count] = neuron;
    field->neuron_count++;
    return neuron;
}

Connection* add_connection(FractalField* field, int pre_id, int post_id) {
    if (!field || pre_id < 0 || pre_id >= field->neuron_count || post_id < 0 || post_id >= field->neuron_count) {
        return NULL;
    }

    if (field->connection_count >= field->connection_capacity) {
        // Попытка расширить
        if (field->connection_count >= field->max_connections) return NULL; // Достигнут лимит
        int new_capacity = field->connection_capacity * 2;
        if (new_capacity > field->max_connections) new_capacity = field->max_connections;
        Connection** temp = (Connection**)realloc(field->connections, new_capacity * sizeof(Connection*));
        if (!temp) return NULL;
        field->connections = temp;
        field->connection_capacity = new_capacity;
    }

    Connection* conn = (Connection*)malloc(sizeof(Connection));
    if (!conn) return NULL;
    conn->pre_neuron_id = pre_id;
    conn->post_neuron_id = post_id;
    conn->gate = create_synapse_gate();
    if (!conn->gate) { free(conn); return NULL; }

    field->connections[field->connection_count] = conn;
    field->connection_count++;
    return conn;
}

void propagate_fractal_field(FractalField* field, float global_reward) {
    if (!field) return;

    // Обновляем глобальный сигнал вознаграждения
    field->global_reward_signal = global_reward;

    // Простая модель распространения: обновляем состояние синапсов
    for (int i = 0; i < field->connection_count; i++) {
        Connection* conn = field->connections[i];
        if (conn && conn->gate) {
            // R-STDP: обновление веса на основе global_reward
            // Упрощённо: вес изменяется пропорционально награде и следу eligibilit
            float delta_w = global_reward * conn->gate->eligibility_trace * 0.01f;
            conn->gate->weight += delta_w;
            CLAMP(conn->gate->weight);

            // Обновление следа eligibilit (STDP)
            // Упрощённо: след уменьшается со временем
            conn->gate->eligibility_trace *= 0.95f; // Декремент
            // Можно добавить "наращивание" следа при спайке
        }
    }
}

void update_fractal_field(FractalField* field) {
    if (!field) return;

    // Обновление состояния нейронов (LIF или другая модель)
    for (int i = 0; i < field->neuron_count; i++) {
        if (field->neurons[i]) {
            FractalNeuron* neuron = field->neurons[i];
            // Простое накопление потенциала
            // В реальности, нужно учитывать входящие связи и веса
            neuron->potential += 0.01f; // Импульс

            if (neuron->potential >= neuron->threshold) {
                neuron->fired = 1;
                neuron->potential = 0.0f; // Сброс
                // Здесь можно запустить распространение спайка по связям
            } else {
                neuron->fired = 0;
            }
        }
    }
}

void check_growth_conditions(FractalField* field) {
    if (!field) return;

    // Пример условия: низкая награда, прошло время, есть ресурсы
    if (field->global_reward_signal < -0.1f &&
        (time(NULL) - field->last_growth_time) > 10 &&
        field->neuron_count < field->max_neurons &&
        field->connection_count < field->max_connections) {
        grow_fractal_field(field);
    }
}

void grow_fractal_field(FractalField* field) {
    if (!field) return;

    printf("[Фундамент] Условия для роста выполнены. Пытаюсь добавить нейрон и связи...\n");
    FractalNeuron* new_neuron = add_fractal_neuron(field);
    if (!new_neuron) {
        printf("[Фундамент] Ошибка: не удалось добавить новый нейрон\n");
        return;
    }
    int new_neuron_id = field->neuron_count - 1;
    printf("[Фундамент] Добавлен нейрон с ID %d.\n", new_neuron_id);

    // Пример добавления связи к новому нейрону
    if (field->neuron_count > 1) {
        int random_pre_id = rand() % (field->neuron_count - 1);
        Connection* new_conn = add_connection(field, random_pre_id, new_neuron_id);
        if (new_conn) {
            printf("[Фундамент] Добавлена связь от %d к %d.\n", random_pre_id, new_neuron_id);
        } else {
            printf("[Фундамент] Ошибка: не удалось добавить новую связь\n");
        }
    }

    field->last_growth_time = time(NULL);
    printf("[Фундамент] Рост завершён. Нейронов: %d, Связей: %d\n", field->neuron_count, field->connection_count);
}


// =============== FractalSpike Implementation ===============

FractalSpike* create_fractal_spike(long timestamp, float intensity, const char* source, float fractalDimension, char** path, int pathSize) {
    FractalSpike* spike = (FractalSpike*)malloc(sizeof(FractalSpike));
    if (!spike) return NULL;

    spike->timestamp = timestamp;
    spike->intensity = intensity;
    spike->source = strdup(source);
    if (!spike->source) { free(spike); return NULL; }
    spike->fractalDimension = fractalDimension;

    spike->pathSize = pathSize;
    if (pathSize > 0 && path) {
        spike->propagationPath = (char**)malloc(pathSize * sizeof(char*));
        if (!spike->propagationPath) {
            free(spike->source);
            free(spike);
            return NULL;
        }
        for (int i = 0; i < pathSize; i++) {
            spike->propagationPath[i] = strdup(path[i]);
            if (!spike->propagationPath[i]) {
                // Ошибка, очищаем уже выделенное
                for (int j = 0; j < i; j++) {
                    free(spike->propagationPath[j]);
                }
                free(spike->propagationPath);
                free(spike->source);
                free(spike);
                return NULL;
            }
        }
    } else {
        spike->propagationPath = NULL;
    }

    return spike;
}

void destroy_fractal_spike(FractalSpike* spike) {
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

void print_fractal_spike(const FractalSpike* spike) {
    if (!spike) return;
    printf("FractalSpike {");
    printf(" timestamp: %ld", spike->timestamp);
    printf(" intensity: %.2f", spike->intensity);
    printf(" source: %s", spike->source);
    printf(" fractalDimension: %.2f", spike->fractalDimension);
    printf(" propagationPath: [");
    for (int i = 0; i < spike->pathSize; i++) {
        printf("%s", spike->propagationPath[i]);
        if (i < spike->pathSize - 1) printf(", ");
    }
    printf("] }");
}

// =============== FractalActivation Implementation ===============

FractalActivation* create_fractal_activation(float baseActivation, float harmonicActivation, float spikeResonance, int fractalDepth, float energyConsumption) {
    FractalActivation* act = (FractalActivation*)malloc(sizeof(FractalActivation));
    if (!act) return NULL;

    act->baseActivation = baseActivation;
    act->harmonicActivation = harmonicActivation;
    act->spikeResonance = spikeResonance;
    act->fractalDepth = fractalDepth;
    act->energyConsumption = energyConsumption;

    return act;
}

void destroy_fractal_activation(FractalActivation* act) {
    if (act) free(act);
}

float get_total_activation(const FractalActivation* act) {
    if (!act) return 0.0f;
    return act->baseActivation + act->harmonicActivation + act->spikeResonance;
}

void print_fractal_activation(const FractalActivation* act) {
    if (!act) return;
    printf("FractalActivation { base: %.2f, harmonic: %.2f, resonance: %.2f, depth: %d, energy: %.2f }\n",
           act->baseActivation, act->harmonicActivation, act->spikeResonance, act->fractalDepth, act->energyConsumption);
}

void fractal_gradient_descent(FractalActivation* act, float learning_rate) {
    if (!act) return;
    // Простой GD для демонстрации
    act->baseActivation -= learning_rate * act->baseActivation;
    act->harmonicActivation -= learning_rate * act->harmonicActivation;
    act->spikeResonance -= learning_rate * act->spikeResonance;
    CLAMP(act->baseActivation);
    CLAMP(act->harmonicActivation);
    CLAMP(act->spikeResonance);
}

// =============== FractalHashCache Implementation ===============

FractalHashCache* create_fractal_hash_cache(int capacity) {
    FractalHashCache* cache = (FractalHashCache*)malloc(sizeof(FractalHashCache));
    if (!cache) return NULL;

    cache->entries = (FractalHashEntry**)calloc(capacity, sizeof(FractalHashEntry*));
    if (!cache->entries) { free(cache); return NULL; }

    cache->capacity = capacity;
    cache->size = 0;
    cache->global_learning_rate = 0.01f;
    cache->decay_factor = 0.99f;
    cache->resonance_threshold = 0.5f;

    return cache;
}

void destroy_fractal_hash_cache(FractalHashCache* cache) {
    if (!cache) return;

    for (int i = 0; i < cache->capacity; i++) {
        if (cache->entries[i]) {
            free(cache->entries[i]->pattern_hash);
            free(cache->entries[i]);
        }
    }
    free(cache->entries);
    free(cache);
}

void hash_cache_store(FractalHashCache* cache, const char* pattern, float dimension, float intensity, float activation) {
    if (!cache || !pattern) return;

    // Простой хэш (для реальной системы нужен лучше)
    unsigned long hash = 0;
    for (const char* c = pattern; *c; c++) {
        hash = hash * 31 + *c;
    }
    int index = hash % cache->capacity;

    // Линейное пробирование
    int start_index = index;
    while (cache->entries[index] != NULL && strcmp(cache->entries[index]->pattern_hash, pattern) != 0) {
        index = (index + 1) % cache->capacity;
        if (index == start_index) return; // Кэш полон для этого паттерна
    }

    FractalHashEntry* entry;
    if (cache->entries[index] == NULL) {
        // Новый элемент
        entry = (FractalHashEntry*)malloc(sizeof(FractalHashEntry));
        if (!entry) return;
        entry->pattern_hash = strdup(pattern);
        if (!entry->pattern_hash) { free(entry); return; }
        cache->entries[index] = entry;
        cache->size++;
    } else {
        // Обновление существующего
        entry = cache->entries[index];
    }

    entry->cached_activation = activation;
    entry->fractal_dimension = dimension;
    entry->spike_resonance_level = intensity; // Используем intensity как proxy для резонанса
    entry->last_accessed = time(NULL);
    entry->access_count++;
    entry->adaptive_learning_rate = cache->global_learning_rate;
    entry->fractal_coherence = 0.5f + (dimension - 1.5f) * 0.2f; // Пример
    entry->energy_efficiency = 0.8f; // Пример
    CLAMP(entry->fractal_coherence);
    CLAMP(entry->energy_efficiency);
}

FractalHashEntry* hash_cache_lookup(FractalHashCache* cache, const char* pattern, float dimension, float intensity) {
    if (!cache || !pattern) return NULL;

    unsigned long hash = 0;
    for (const char* c = pattern; *c; c++) {
        hash = hash * 31 + *c;
    }
    int index = hash % cache->capacity;

    int start_index = index;
    while (cache->entries[index] != NULL) {
        if (strcmp(cache->entries[index]->pattern_hash, pattern) == 0) {
            cache->entries[index]->last_accessed = time(NULL);
            cache->entries[index]->access_count++;
            return cache->entries[index];
        }
        index = (index + 1) % cache->capacity;
        if (index == start_index) break; // Обошли весь кольцевой буфер
    }
    return NULL;
}

float get_adaptive_learning_rate(FractalHashCache* cache, const char* pattern, float dimension, float intensity, float base_rate) {
    FractalHashEntry* entry = hash_cache_lookup(cache, pattern, dimension, intensity);
    if (entry) {
        return entry->adaptive_learning_rate;
    }
    return base_rate; // Если нет в кэше, используем базовый
}

void update_hash_learning_rates(FractalHashCache* cache, float performance_factor) {
    if (!cache) return;

    for (int i = 0; i < cache->capacity; i++) {
        if (cache->entries[i]) {
            // Адаптируем скорость обучения на основе фактора производительности
            cache->entries[i]->adaptive_learning_rate *= (0.9f + performance_factor * 0.2f);
            cache->entries[i]->adaptive_learning_rate = fmaxf(0.001f, fminf(0.1f, cache->entries[i]->adaptive_learning_rate));
        }
    }
}

void optimize_hash_energy(FractalHashCache* cache, float target_efficiency) {
    if (!cache) return;

    long current_time = time(NULL);
    for (int i = 0; i < cache->capacity; i++) {
        FractalHashEntry* entry = cache->entries[i];
        if (entry) {
            float age = (float)(current_time - entry->last_accessed) / 3600.0f; // Возраст в часах
            if (age > 24.0f) { // Удаляем старые неиспользуемые
                free(entry->pattern_hash);
                free(entry);
                cache->entries[i] = NULL;
                cache->size--;
            } else {
                // Обновляем эффективность
                entry->energy_efficiency = target_efficiency * (1.0f - age / 24.0f);
                CLAMP(entry->energy_efficiency);
            }
        }
    }
}

void hash_cache_clusterize(FractalHashCache* cache, float similarity_threshold) {
    if (!cache) return;

    for (int i = 0; i < cache->capacity; i++) {
        FractalHashEntry* entry_i = cache->entries[i];
        if (!entry_i) continue;

        for (int j = i + 1; j < cache->capacity; j++) {
            FractalHashEntry* entry_j = cache->entries[j];
            if (!entry_j) continue;

            float dim_diff = fabsf(entry_i->fractal_dimension - entry_j->fractal_dimension);
            float act_diff = fabsf(entry_i->cached_activation - entry_j->cached_activation);

            if (dim_diff < similarity_threshold && act_diff < 0.2f) {
                entry_i->is_cluster_representative = 1;
                entry_i->cluster_radius = fmaxf(dim_diff, act_diff);
                // Удаляем entry_j как дубликат
                free(entry_j->pattern_hash);
                free(entry_j);
                cache->entries[j] = NULL;
                cache->size--;
            }
        }
    }
}

FractalHashEntry* find_closest_representative(FractalHashCache* cache, float dimension, float intensity) {
    if (!cache) return NULL;
    FractalHashEntry* closest = NULL;
    float min_dist = FLT_MAX;

    for (int i = 0; i < cache->capacity; i++) {
        FractalHashEntry* entry = cache->entries[i];
        if (entry && entry->is_cluster_representative) {
            float dist = fabsf(entry->fractal_dimension - dimension) + fabsf(entry->spike_resonance_level - intensity);
            if (dist < min_dist) {
                min_dist = dist;
                closest = entry;
            }
        }
    }
    return closest;
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
    if (resonance) free(resonance);
}

float apply_resonance(NeuralResonance* resonance, float input_signal, float time_delta) {
    if (!resonance) return input_signal;

    // Простая модель: осциллятор
    resonance->phase += resonance->frequency * time_delta * 2.0f * M_PI;
    float resonance_wave = resonance->amplitude * sinf(resonance->phase);
    float output = input_signal + resonance_wave;

    // Затухание
    resonance->amplitude *= (1.0f - resonance->damping * time_delta);
    CLAMP(resonance->amplitude);

    return output;
}

void update_resonance_parameters(NeuralResonance* resonance, float learning_signal) {
    if (!resonance) return;

    resonance->frequency += learning_signal * 0.001f;
    resonance->amplitude += learning_signal * 0.005f;
    resonance->damping += learning_signal * 0.0005f;

    CLAMP(resonance->frequency);
    CLAMP(resonance->amplitude);
    CLAMP(resonance->damping);
}

float calculate_resonance_match(NeuralResonance* res1, NeuralResonance* res2) {
    if (!res1 || !res2) return 0.0f;

    float freq_diff = fabsf(res1->frequency - res2->frequency);
    float amp_diff = fabsf(res1->amplitude - res2->amplitude);
    float damp_diff = fabsf(res1->damping - res2->damping);

    float match = 1.0f - (freq_diff * 0.4f + amp_diff * 0.4f + damp_diff * 0.2f);
    return fmaxf(0.0f, fminf(1.0f, match));
}

// =============== FractalBackprop Implementation ===============

FractalBackprop* create_fractal_backprop(int max_depth) {
    FractalBackprop* bp = (FractalBackprop*)malloc(sizeof(FractalBackprop));
    if (!bp) return NULL;

    bp->depth = max_depth;
    bp->learning_rate = 0.01f;
    bp->momentum = 0.9f;
    bp->spike_error = 0.0f;
    bp->error_signals = (float*)malloc(max_depth * sizeof(float));
    bp->fractal_gradients = (float*)malloc(max_depth * sizeof(float));
    if (!bp->error_signals || !bp->fractal_gradients) {
        free(bp->error_signals);
        free(bp->fractal_gradients);
        free(bp);
        return NULL;
    }

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

void fractal_backward_pass(FractalBackprop* bp, FractalActivation* act, float target_error, float current_activation) {
    if (!bp || !act) return;

    float output_error = target_error - current_activation;
    bp->spike_error = output_error;

    for (int d = 0; d < bp->depth; d++) {
        float depth_decay = expf(-d * 0.5f);
        bp->error_signals[d] = output_error * depth_decay;

        float fractal_grad = bp->error_signals[d] * (0.4f * act->baseActivation + 0.3f * act->harmonicActivation + 0.3f * act->spikeResonance);
        bp->fractal_gradients[d] = fractal_grad;
    }
}

void apply_fractal_gradients(FractalActivation* act, FractalBackprop* bp) {
    if (!act || !bp) return;

    for (int d = 0; d < bp->depth; d++) {
        act->baseActivation += bp->learning_rate * bp->fractal_gradients[d];
        act->harmonicActivation += bp->learning_rate * bp->fractal_gradients[d] * 0.8f;
        act->spikeResonance += bp->learning_rate * bp->fractal_gradients[d] * 0.6f;
    }
    CLAMP(act->baseActivation);
    CLAMP(act->harmonicActivation);
    CLAMP(act->spikeResonance);
}

float calculate_fractal_error(FractalSpike* output, FractalSpike* target) {
    if (!output || !target) return 0.0f;
    float intensity_error = fabsf(output->intensity - target->intensity);
    float dimension_error = fabsf(output->fractalDimension - target->fractalDimension);
    return (intensity_error + dimension_error) * 0.5f;
}

// =============== HierarchicalSpikeSystem Implementation ===============

HierarchicalSpikeSystem* create_hierarchical_spike_system(int max_low_spikes, int max_mid_spikes, int max_high_spikes) {
    HierarchicalSpikeSystem* system = (HierarchicalSpikeSystem*)malloc(sizeof(HierarchicalSpikeSystem));
    if (!system) return NULL;

    system->low_level_spikes = (FractalSpike**)malloc(max_low_spikes * sizeof(FractalSpike*));
    system->mid_level_spikes = (FractalSpike**)malloc(max_mid_spikes * sizeof(FractalSpike*));
    system->high_level_spikes = (FractalSpike**)malloc(max_high_spikes * sizeof(FractalSpike*));

    if (!system->low_level_spikes || !system->mid_level_spikes || !system->high_level_spikes) {
        free(system->low_level_spikes);
        free(system->mid_level_spikes);
        free(system->high_level_spikes);
        free(system);
        return NULL;
    }

    for (int i = 0; i < max_low_spikes; i++) system->low_level_spikes[i] = NULL;
    for (int i = 0; i < max_mid_spikes; i++) system->mid_level_spikes[i] = NULL;
    for (int i = 0; i < max_high_spikes; i++) system->high_level_spikes[i] = NULL;

    system->max_low_spikes = max_low_spikes;
    system->max_mid_spikes = max_mid_spikes;
    system->max_high_spikes = max_high_spikes;
    system->low_level_count = 0;
    system->mid_level_count = 0;
    system->high_level_count = 0;

    system->low_to_mid_weights = (float*)malloc(max_low_spikes * max_mid_spikes * sizeof(float));
    system->mid_to_high_weights = (float*)malloc(max_mid_spikes * max_high_spikes * sizeof(float));
    system->high_to_mid_weights = (float*)malloc(max_high_spikes * max_mid_spikes * sizeof(float));
    system->mid_to_low_weights = (float*)malloc(max_mid_spikes * max_low_spikes * sizeof(float));

    if (!system->low_to_mid_weights || !system->mid_to_high_weights || !system->high_to_mid_weights || !system->mid_to_low_weights) {
        free(system->low_level_spikes);
        free(system->mid_level_spikes);
        free(system->high_level_spikes);
        free(system->low_to_mid_weights);
        free(system->mid_to_high_weights);
        free(system->high_to_mid_weights);
        free(system->mid_to_low_weights);
        free(system);
        return NULL;
    }

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

void destroy_hierarchical_spike_system(HierarchicalSpikeSystem* system) {
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

    destroy_fractal_hash_cache(system->cache);

    free(system);
}

void add_spike_to_hierarchy(HierarchicalSpikeSystem* system, FractalSpike* spike, int level) {
    if (!system || !spike) return;

    FractalSpike** target_array = NULL;
    int* count_ptr = NULL;
    int max_count = 0;

    switch (level) {
        case SPIKE_LEVEL_LOW:
            target_array = system->low_level_spikes;
            count_ptr = &system->low_level_count;
            max_count = system->max_low_spikes;
            break;
        case SPIKE_LEVEL_MID:
            target_array = system->mid_level_spikes;
            count_ptr = &system->mid_level_count;
            max_count = system->max_mid_spikes;
            break;
        case SPIKE_LEVEL_HIGH:
            target_array = system->high_level_spikes;
            count_ptr = &system->high_level_count;
            max_count = system->max_high_spikes;
            break;
        default:
            return;
    }

    if (*count_ptr < max_count) {
        target_array[*count_ptr] = spike;
        (*count_ptr)++;
    }
}

float propagate_through_hierarchy(HierarchicalSpikeSystem* system, FractalSpike* input_spike) {
    if (!system || !input_spike) return 0.0f;

    float total_activation = 0.0f;

    // Propagate Low -> Mid
    for (int i = 0; i < system->low_level_count; i++) {
        FractalSpike* low_spike = system->low_level_spikes[i];
        if (!low_spike) continue;

        float cached_activation = low_spike->intensity;
        if (system->cache) {
            FractalHashEntry* entry = hash_cache_lookup(system->cache, low_spike->source, low_spike->fractalDimension, low_spike->intensity);
            if (entry) {
                cached_activation = entry->cached_activation;
            }
        }

        for (int j = 0; j < system->mid_level_count; j++) {
            if (system->mid_level_spikes[j]) {
                int weight_index = i * system->max_mid_spikes + j;
                float weight = system->low_to_mid_weights[weight_index];
                float activation = cached_activation * weight;
                system->mid_level_spikes[j]->intensity += activation * 0.1f;
                CLAMP(system->mid_level_spikes[j]->intensity);
                total_activation += activation;
            }
        }
    }

    // Propagate Mid -> High
    for (int i = 0; i < system->mid_level_count; i++) {
        FractalSpike* mid_spike = system->mid_level_spikes[i];
        if (!mid_spike) continue;

        float cached_activation = mid_spike->intensity;
        if (system->cache) {
            FractalHashEntry* entry = hash_cache_lookup(system->cache, mid_spike->source, mid_spike->fractalDimension, mid_spike->intensity);
            if (entry) {
                cached_activation = entry->cached_activation;
            }
        }

        for (int j = 0; j < system->high_level_count; j++) {
            if (system->high_level_spikes[j]) {
                int weight_index = i * system->max_high_spikes + j;
                float weight = system->mid_to_high_weights[weight_index];
                float activation = cached_activation * weight;
                system->high_level_spikes[j]->intensity += activation * 0.05f;
                CLAMP(system->high_level_spikes[j]->intensity);
                total_activation += activation;
            }
        }
    }

    // Propagate High -> Mid (feedback)
    for (int i = 0; i < system->high_level_count; i++) {
        FractalSpike* high_spike = system->high_level_spikes[i];
        if (!high_spike || high_spike->intensity < 0.3f) continue; // Only strong high-level spikes feedback

        float feedback_strength = high_spike->intensity * 0.2f;
        for (int j = 0; j < system->mid_level_count; j++) {
            if (system->mid_level_spikes[j]) {
                int weight_index = i * system->max_mid_spikes + j;
                float weight = system->high_to_mid_weights[weight_index];
                float feedback = feedback_strength * weight;
                system->mid_level_spikes[j]->intensity += feedback;
                CLAMP(system->mid_level_spikes[j]->intensity);
                total_activation += feedback;
            }
        }
    }

    return total_activation;
}

void optimize_hierarchical_connections(HierarchicalSpikeSystem* system) {
    if (!system) return;

    // Простая оптимизация: обновление кэша
    if (system->cache) {
        optimize_hash_energy(system->cache, 0.7f);
    }
}

float get_hierarchical_activation(HierarchicalSpikeSystem* system, const char* pattern) {
    if (!system || !pattern) return 0.0f;

    // Ищем в кэше
    if (system->cache) {
        FractalHashEntry* entry = hash_cache_lookup(system->cache, pattern, 1.5f, 0.5f); // Примерные значения
        if (entry) {
            return entry->cached_activation;
        }
    }
    return 0.0f;
}

void print_hierarchical_system_status(const HierarchicalSpikeSystem* system) {
    if (!system) return;

    printf("=== HierarchicalSpikeSystem Status ===\n");
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
    printf("Avg Low Intensity: %.3f\n", avg_low_intensity);

    float avg_mid_intensity = 0.0f;
    for (int i = 0; i < system->mid_level_count; i++) {
        if (system->mid_level_spikes[i]) {
            avg_mid_intensity += system->mid_level_spikes[i]->intensity;
        }
    }
    if (system->mid_level_count > 0) avg_mid_intensity /= system->mid_level_count;
    printf("Avg Mid Intensity: %.3f\n", avg_mid_intensity);

    printf("=======================================\n");
}


// =============== NeuralMemory Implementation ===============

NeuralMemory* create_neural_memory(int capacity) {
    NeuralMemory* memory = (NeuralMemory*)malloc(sizeof(NeuralMemory));
    if (!memory) return NULL;

    memory->neurons = (FractalSpike**)malloc(capacity * sizeof(FractalSpike*));
    if (!memory->neurons) {
        free(memory);
        return NULL;
    }

    memory->capacity = capacity;
    memory->count = 0;
    memory->last_update = time(NULL);

    for (int i = 0; i < capacity; i++) {
        memory->neurons[i] = NULL;
    }

    return memory;
}

void destroy_neural_memory(NeuralMemory* memory) {
    if (!memory) return;

    for (int i = 0; i < memory->count; i++) {
        if (memory->neurons[i]) {
            destroy_fractal_spike(memory->neurons[i]);
        }
    }
    free(memory->neurons);
    free(memory);
}

void add_neuron_to_memory(NeuralMemory* memory, FractalSpike* neuron) {
    if (!memory || !neuron) return;

    if (memory->count >= memory->capacity) {
        // Попытка расширить
        int new_capacity = memory->capacity * 2;
        FractalSpike** temp = (FractalSpike**)realloc(memory->neurons, new_capacity * sizeof(FractalSpike*));
        if (!temp) return; // Не удалось расширить
        memory->neurons = temp;
        memory->capacity = new_capacity;
    }

    // Создаём копию нейрона
    FractalSpike* neuron_copy = create_fractal_spike(
        neuron->timestamp,
        neuron->intensity,
        neuron->source,
        neuron->fractalDimension,
        neuron->propagationPath,
        neuron->pathSize
    );

    if (neuron_copy) {
        memory->neurons[memory->count] = neuron_copy;
        memory->count++;
        memory->last_update = time(NULL);
    }
}

void compact_memory(NeuralMemory* memory) {
    if (!memory || memory->count < 2) return;

    int removed = 0;
    for (int i = 0; i < memory->count - 1; i++) {
        if (!memory->neurons[i]) continue;
        for (int j = i + 1; j < memory->count; j++) {
            if (!memory->neurons[j]) continue;
            // Простое сравнение: если source и intensity близки, считаем дубликатом
            if (strcmp(memory->neurons[i]->source, memory->neurons[j]->source) == 0 &&
                fabs(memory->neurons[i]->intensity - memory->neurons[j]->intensity) < 0.1f) {
                memory->neurons[i]->intensity = (memory->neurons[i]->intensity + memory->neurons[j]->intensity) / 2.0f;
                destroy_fractal_spike(memory->neurons[j]);
                memory->neurons[j] = NULL;
                removed++;
            }
        }
    }

    int new_index = 0;
    for (int i = 0; i < memory->count; i++) {
        if (memory->neurons[i]) {
            memory->neurons[new_index++] = memory->neurons[i];
        }
    }
    memory->count = new_index;
    printf("[Kernel] Оптимизация памяти: удалено %d дубликатов\n", removed);
}

void optimize_memory_structure(NeuralMemory* memory) {
    if (!memory) return;

    printf("[Optimize Memory] Запуск анализа и оптимизации памяти...\n");

    // --- 1. Глобальный фрактальный анализ ---
    int series_length;
    float* intensity_series = NULL;
    float* dimension_series = NULL;

    if (memory->count > 10) { // Минимальная длина для анализа
        intensity_series = (float*)malloc(memory->count * sizeof(float));
        dimension_series = (float*)malloc(memory->count * sizeof(float));
        if (intensity_series && dimension_series) {
            int valid_count = 0;
            for (int i = 0; i < memory->count; i++) {
                if (memory->neurons[i]) {
                    intensity_series[valid_count] = memory->neurons[i]->intensity;
                    dimension_series[valid_count] = memory->neurons[i]->fractalDimension;
                    valid_count++;
                }
            }
            series_length = valid_count;

            if (series_length > 10) {
                float correlation_dim = calculate_correlation_dimension(intensity_series, series_length, 3);
                float entropy = calculate_kolmogorov_entropy(intensity_series, series_length);
                int is_chaotic = detect_chaotic_behavior_simple(intensity_series, series_length);

                printf("[Optimize Memory] Анализ памяти: CorrDim=%.3f, Entropy=%.3f, Chaotic=%s\n",
                       correlation_dim, entropy, is_chaotic ? "YES" : "NO");

                // --- 2. Принятие решений на основе анализа ---
                if (is_chaotic) {
                    printf("[Optimize Memory] Обнаружен высокий уровень хаоса. Запуск агрессивного забывания...\n");
                    forget_old_neurons(memory, 0.5f);
                } else if (entropy < 0.2f) {
                    printf("[Optimize Memory] Обнаружена низкая энтропия (застой). Запуск консервативного забывания...\n");
                    forget_old_neurons(memory, 0.1f);
                }
            } else {
                 printf("[Optimize Memory] Недостаточно валидных данных для анализа.\n");
            }
        } else {
             printf("[Optimize Memory] Ошибка выделения памяти для анализа.\n");
        }
    } else {
        printf("[Optimize Memory] Недостаточно данных для глобального анализа. Пропуск.\n");
    }

    free(intensity_series);
    free(dimension_series);

    // --- 3. Стандартная сортировка по важности ---
    for (int i = 0; i < memory->count - 1; i++) {
        for (int j = i + 1; j < memory->count; j++) {
            if (memory->neurons[i] && memory->neurons[j]) {
                if (memory->neurons[i]->intensity < memory->neurons[j]->intensity) {
                    FractalSpike* temp = memory->neurons[i];
                    memory->neurons[i] = memory->neurons[j];
                    memory->neurons[j] = temp;
                }
            }
        }
    }

    // --- 4. Дополнительная оптимизация ---
    compact_memory(memory);

    printf("[Optimize Memory] Структура памяти оптимизирована на основе фрактального анализа.\n");
}

int get_memory_stats(NeuralMemory* memory) {
    if (!memory) return 0;

    float total_intensity = 0.0f;
    float total_dimension = 0.0f;
    int total_path_elements = 0;

    for (int i = 0; i < memory->count; i++) {
        if (memory->neurons[i]) {
            total_intensity += memory->neurons[i]->intensity;
            total_dimension += memory->neurons[i]->fractalDimension;
            total_path_elements += memory->neurons[i]->pathSize;
        }
    }

    printf("[Memory Stats] Count: %d, Total Intensity: %.2f, Avg Intensity: %.2f, Total Fractal Dim: %.2f, Avg Dim: %.2f, Total Path Elements: %d\n",
           memory->count, total_intensity, memory->count > 0 ? total_intensity / memory->count : 0.0f,
           total_dimension, memory->count > 0 ? total_dimension / memory->count : 0.0f, total_path_elements);

    return memory->count;
}


// =============== REAL MEMORY FILE IMPLEMENTATION ===============

NeuralMemory* load_memory_from_file(const char* filename) {
    printf("[Kernel] Загрузка памяти из файла '%s'\n", filename);
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("[Kernel] Файл не существует, создается новая память\n");
        return create_neural_memory(100);
    }

    int neuron_count = 0;
    if (fread(&neuron_count, sizeof(int), 1, file) != 1) {
        printf("[Kernel] Ошибка чтения количества нейронов\n");
        fclose(file);
        return create_neural_memory(100);
    }

    NeuralMemory* memory = create_neural_memory(neuron_count + 10); // Запас на всякий
    if (!memory) {
        fclose(file);
        return NULL;
    }

    printf("[Kernel] Загружаем %d нейронов...\n", neuron_count);
    int loaded_count = 0;
    for (int i = 0; i < neuron_count; i++) {
        MemoryHeader header;
        if (fread(&header, sizeof(MemoryHeader), 1, file) != 1) {
            printf("[Kernel] Ошибка чтения заголовка нейрона %d\n", i);
            break;
        }

        char* source = (char*)malloc(header.source_len + 1);
        if (!source) {
            printf("[Kernel] Ошибка выделения памяти для source\n");
            break;
        }
        if (fread(source, 1, header.source_len, file) != header.source_len) {
            printf("[Kernel] Ошибка чтения source нейрона %d\n", i);
            free(source);
            break;
        }
        source[header.source_len] = '\0';

        char** path = NULL;
        if (header.pathSize > 0) {
            path = (char**)malloc(header.pathSize * sizeof(char*));
            if (!path) {
                printf("[Kernel] Ошибка выделения памяти для path\n");
                free(source);
                break;
            }
            for (int j = 0; j < header.pathSize; j++) {
                int path_len;
                if (fread(&path_len, sizeof(int), 1, file) != 1) {
                    printf("[Kernel] Ошибка чтения длины path элемента\n");
                    for (int k = 0; k < j; k++) free(path[k]);
                    free(path);
                    free(source);
                    goto cleanup;
                }
                path[j] = (char*)malloc(path_len + 1);
                if (!path[j]) {
                    printf("[Kernel] Ошибка выделения памяти для path элемента\n");
                    for (int k = 0; k < j; k++) free(path[k]);
                    free(path);
                    free(source);
                    goto cleanup;
                }
                if (fread(path[j], 1, path_len, file) != path_len) {
                    printf("[Kernel] Ошибка чтения path элемента\n");
                    free(path[j]);
                    for (int k = 0; k < j; k++) free(path[k]);
                    free(path);
                    free(source);
                    goto cleanup;
                }
                path[j][path_len] = '\0';
            }
        }

        FractalSpike* neuron = create_fractal_spike(
            header.timestamp,
            header.intensity,
            source,
            header.fractalDimension,
            path,
            header.pathSize
        );

        if (neuron) {
            add_neuron_to_memory(memory, neuron);
            destroy_fractal_spike(neuron); // add_neuron_to_memory создаёт копию
            loaded_count++;
        }

        free(source);
        if (path) {
            for (int j = 0; j < header.pathSize; j++) {
                free(path[j]);
            }
            free(path);
        }
    }

cleanup:
    fclose(file);
    printf("[Kernel] Успешно загружено %d/%d нейронов\n", loaded_count, neuron_count);
    return memory;
}

void save_memory_to_file(NeuralMemory* memory, const char* filename) {
    if (!memory || !filename) return;

    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("[Kernel] Ошибка создания файла '%s'\n", filename);
        return;
    }

    if (fwrite(&memory->count, sizeof(int), 1, file) != 1) {
        printf("[Kernel] Ошибка записи количества нейронов\n");
        fclose(file);
        return;
    }

    printf("[Kernel] Сохраняем %d нейронов в файл '%s'...\n", memory->count, filename);
    int saved_count = 0;
    for (int i = 0; i < memory->count; i++) {
        FractalSpike* neuron = memory->neurons[i];
        if (!neuron) continue;

        MemoryHeader header;
        header.timestamp = neuron->timestamp;
        header.intensity = neuron->intensity;
        header.fractalDimension = neuron->fractalDimension;
        header.pathSize = neuron->pathSize;
        header.source_len = strlen(neuron->source);

        if (fwrite(&header, sizeof(MemoryHeader), 1, file) != 1) {
            printf("[Kernel] Ошибка записи заголовка нейрона %d\n", i);
            break;
        }
        if (fwrite(neuron->source, 1, header.source_len, file) != header.source_len) {
            printf("[Kernel] Ошибка записи source нейрона %d\n", i);
            break;
        }

        for (int j = 0; j < neuron->pathSize; j++) {
            int path_len = strlen(neuron->propagationPath[j]);
            if (fwrite(&path_len, sizeof(int), 1, file) != 1) {
                printf("[Kernel] Ошибка записи длины path элемента %d\n", j);
                goto cleanup;
            }
            if (fwrite(neuron->propagationPath[j], 1, path_len, file) != path_len) {
                printf("[Kernel] Ошибка записи path элемента %d\n", j);
                goto cleanup;
            }
        }
        saved_count++;
    }

cleanup:
    fclose(file);
    printf("[Kernel] Успешно сохранено %d/%d нейронов\n", saved_count, memory->count);
}


// =============== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===============

float fractal_connectivity(float dimension, float intensity, int depth) {
    float base = (dimension * 0.5f + intensity * 0.3f + (float)depth * 0.2f);
    float correlation_dim = 0.0f;
    float biological_boost = 1.0f;

    float series[] = {dimension, intensity, (float)depth};
    correlation_dim = calculate_correlation_dimension(series, 3, 2);

    if (dimension > 1.4f && dimension < 2.4f) {
        float optimal_range_center = 1.8f;
        float distance_from_optimal = fabsf(dimension - optimal_range_center);
        float range_factor = 1.0f - (distance_from_optimal / 0.5f);
        range_factor = fmaxf(0.0f, fminf(1.0f, range_factor));
        biological_boost += 0.3f * range_factor;
    }

    base = fmaxf(0.0f, fminf(2.0f, base * biological_boost));

    return base;
}

float resonance_amplification(float base, float harmonic, float resonance) {
    float resonance_params[3] = {base, harmonic, resonance};
    float hurst_exponent = calculate_hurst_exponent(resonance_params, 3);

    float amplification = base + harmonic * 0.5f + resonance * 0.3f;
    amplification *= (1.0f + hurst_exponent * 0.1f); // Модуляция на основе H
    CLAMP(amplification);
    return amplification;
}

float adaptive_learning_rate(float current_rate, float performance, float stability) {
    float perf_factor = fmaxf(0.1f, fminf(1.5f, 1.0f + performance));
    float stab_factor = fmaxf(0.5f, fminf(1.2f, stability));
    return current_rate * perf_factor * stab_factor;
}

float energy_balance(float consumption, float activation, float target_efficiency) {
    float efficiency = activation / (consumption + 1.0f);
    float error = target_efficiency - efficiency;
    return error;
}

float semantic_coherence(const char** patterns, int pattern_count, float base_coherence) {
    if (pattern_count < 2) return base_coherence;

    float total_similarity = 0.0f;
    int comparisons = 0;
    for (int i = 0; i < pattern_count - 1; i++) {
        for (int j = i + 1; j < pattern_count; j++) {
            if (patterns[i] && patterns[j]) {
                // Простое сравнение длины строк как пример
                int len1 = strlen(patterns[i]);
                int len2 = strlen(patterns[j]);
                float len_diff = fabsf((float)len1 - (float)len2);
                float similarity = 1.0f / (1.0f + len_diff);
                total_similarity += similarity;
                comparisons++;
            }
        }
    }
    float avg_similarity = comparisons > 0 ? total_similarity / comparisons : 1.0f;
    return (base_coherence + avg_similarity) * 0.5f;
}

float get_cached_activation(FractalHashCache* cache, FractalSpike* spike) {
    if (!cache || !spike) return 0.0f;

    FractalHashEntry* entry = hash_cache_lookup(cache, spike->source, spike->fractalDimension, spike->intensity);
    return entry ? entry->cached_activation : 0.0f;
}

void update_spike_learning_with_hash(FractalHashCache* cache, FractalSpike* spike, FractalActivation* act) {
    if (!cache || !spike || !act) return;

    float cached_activation = get_cached_activation(cache, spike);
    float error = act->baseActivation - cached_activation;
    float new_activation = cached_activation + error * 0.1f; // Простой фактор обучения

    hash_cache_store(cache, spike->source, spike->fractalDimension, spike->intensity, new_activation);
}

void fractal_online_learning(FractalHashCache* cache, NeuralResonance* resonance,
                            const char* input_pattern, float actual_output,
                            float expected_output, float dimension) {
    if (!cache || !resonance || !input_pattern) return;

    float error = expected_output - actual_output;
    float abs_error = fabsf(error);
    float sign_error = (error >= 0.0f) ? 1.0f : -1.0f;

    float base_lr = 0.01f;
    float optimal_fractal_dim = 1.4f;
    float dim_deviation = fabsf(dimension - optimal_fractal_dim);
    float dim_factor = 1.0f / (1.0f + dim_deviation * 2.0f); // Снижение lr при отклонении от оптимума
    float adaptive_lr = base_lr * dim_factor;

    FractalHashEntry* entry = hash_cache_lookup(cache, input_pattern, dimension, actual_output);
    if (entry) {
        entry->cached_activation += adaptive_lr * error;
        CLAMP(entry->cached_activation);

        entry->fractal_coherence = fmaxf(0.1f, entry->fractal_coherence - abs_error * 0.05f);
    }

    float resonance_change_factor = 1.0f;
    if (resonance->amplitude > 0.1f) {
        float freq_change = adaptive_lr * error * 0.1f * resonance_change_factor;
        float amp_change = adaptive_lr * error * 0.05f * resonance_change_factor;
        float damp_change = adaptive_lr * error * 0.02f * resonance_change_factor;

        freq_change = fmaxf(-0.05f, fminf(0.05f, freq_change));
        amp_change = fmaxf(-0.03f, fminf(0.03f, amp_change));
        damp_change = fmaxf(-0.01f, fminf(0.01f, damp_change));

        resonance->frequency += freq_change;
        resonance->amplitude += amp_change;
        resonance->damping -= damp_change;

        CLAMP(resonance->frequency);
        CLAMP(resonance->amplitude);
        CLAMP(resonance->damping);
    }

    float target_efficiency_base = 0.8f;
    float error_penalty = abs_error * 0.1f;
    float resonance_load = fmaxf(0.0f, resonance->amplitude - 0.5f) * 0.1f;
    float target_efficiency = target_efficiency_base - error_penalty - resonance_load;
    target_efficiency = fmaxf(0.1f, fminf(0.9f, target_efficiency));

    if (cache) {
        optimize_hash_energy(cache, target_efficiency);
    }
}

void apply_resonance_to_activation(FractalActivation* act, NeuralResonance* resonance) {
    if (!act || !resonance) return;

    act->baseActivation *= resonance->amplitude;
    act->harmonicActivation *= resonance->frequency;
    act->spikeResonance *= (1.0f - resonance->damping);
    CLAMP(act->baseActivation);
    CLAMP(act->harmonicActivation);
    CLAMP(act->spikeResonance);
}