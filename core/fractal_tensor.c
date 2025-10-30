#include "fractal_tensor.h"
#include <string.h>
#include <float.h>
#include <math.h>

#ifndef isnan
#define isnan(x) ((x) != (x))
#endif

// === БАЗОВОЕ СОЗДАНИЕ ТЕНЗОРА ===

FractalTensor* fractal_tensor_create(int rows, int cols) {
    FractalTensor* tensor = (FractalTensor*)malloc(sizeof(FractalTensor));
    if (!tensor) return NULL;

    tensor->rows = rows;
    tensor->cols = cols;

    // Выделение памяти для данных
    tensor->data = (float**)malloc(rows * sizeof(float*));
    if (!tensor->data) {
        free(tensor);
        return NULL;
    }
    
    for (int i = 0; i < rows; i++) {
        tensor->data[i] = (float*)calloc(cols, sizeof(float));
        if (!tensor->data[i]) {
            for (int j = 0; j < i; j++) free(tensor->data[j]);
            free(tensor->data);
            free(tensor);
            return NULL;
        }
    }

    // Выделение памяти для собственных значений и векторов
    tensor->eigenvalues = (float*)calloc(rows, sizeof(float));
    tensor->eigenvectors = (float*)calloc(rows * cols, sizeof(float));
    
    if (!tensor->eigenvalues || !tensor->eigenvectors) {
        free(tensor->eigenvalues);
        free(tensor->eigenvectors);
        for (int j = 0; j < rows; j++) free(tensor->data[j]);
        free(tensor->data);
        free(tensor);
        return NULL;
    }

    // Инициализация значений по умолчанию
    tensor->fractal_dimension = 1.5f;
    tensor->correlation_dimension = 0.0f;
    tensor->entropy = 0.0f;
    tensor->lyapunov_exponent = 0.0f;
    tensor->is_chaotic = 0;
    
    tensor->timestamp = time(NULL);
    tensor->activity_sum = 0.0f;
    tensor->activity_variance = 0.0f;
    tensor->temporal_correlation = 0.0f;
    
    // Инициализация спайковых компонентов как NULL (для базовой версии)
    tensor->spike_resonance_weights = NULL;
    tensor->neuromodulator_levels = NULL;
    tensor->eligibility_traces = NULL;
    tensor->last_activation_time = NULL;
    tensor->global_reward_signal = 0.0f;
    tensor->is_critical = 0;
    
    tensor->low_level_tensor = NULL;
    tensor->mid_level_tensor = NULL;
    tensor->high_level_tensor = NULL;
    
    tensor->learning_rate = 0.01f;
    tensor->performance_score = 0.5f;
    tensor->activation_count = 0;

    return tensor;
}

// === УЛУЧШЕННОЕ СОЗДАНИЕ ТЕНЗОРА СО СПАЙКОВЫМИ СИСТЕМАМИ ===

FractalTensor* fractal_tensor_create_enhanced(int rows, int cols) {
    FractalTensor* tensor = fractal_tensor_create(rows, cols);
    if (!tensor) return NULL;
    
    // Инициализация спайковых компонентов
    tensor->spike_resonance_weights = (float**)malloc(rows * sizeof(float*));
    tensor->neuromodulator_levels = (float*)calloc(rows, sizeof(float));
    tensor->eligibility_traces = (float*)calloc(rows, sizeof(float));
    tensor->last_activation_time = (int*)calloc(rows, sizeof(int));
    
    if (!tensor->spike_resonance_weights || !tensor->neuromodulator_levels || 
        !tensor->eligibility_traces || !tensor->last_activation_time) {
        fractal_tensor_destroy(tensor);
        return NULL;
    }
    
    for (int i = 0; i < rows; i++) {
        tensor->spike_resonance_weights[i] = (float*)calloc(cols, sizeof(float));
        if (!tensor->spike_resonance_weights[i]) {
            for (int j = 0; j < i; j++) free(tensor->spike_resonance_weights[j]);
            fractal_tensor_destroy(tensor);
            return NULL;
        }
        
        // Инициализация резонансных весов случайными значениями
        for (int j = 0; j < cols; j++) {
            tensor->spike_resonance_weights[i][j] = 0.1f + (float)rand() / RAND_MAX * 0.8f;
        }
        
        tensor->neuromodulator_levels[i] = 0.5f;
        tensor->eligibility_traces[i] = 0.0f;
        tensor->last_activation_time[i] = 0;
    }
    
    tensor->global_reward_signal = 0.0f;
    tensor->is_critical = 0;
    
    // Создание иерархических тензоров (рекурсивно)
    if (rows >= 4 && cols >= 4) {
        tensor->low_level_tensor = fractal_tensor_create_enhanced(rows / 2, cols / 2);
        tensor->mid_level_tensor = fractal_tensor_create_enhanced(rows / 4, cols / 4);
        if (rows >= 8 && cols >= 8) {
            tensor->high_level_tensor = fractal_tensor_create_enhanced(rows / 8, cols / 8);
        }
    }
    
    return tensor;
}

// === ОЧИСТКА ПАМЯТИ ===

void fractal_tensor_destroy(FractalTensor* tensor) {
    if (!tensor) return;

    // Очистка основных данных
    for (int i = 0; i < tensor->rows; i++) {
        free(tensor->data[i]);
    }
    free(tensor->data);
    free(tensor->eigenvalues);
    free(tensor->eigenvectors);

    // Очистка спайковых компонентов
    if (tensor->spike_resonance_weights) {
        for (int i = 0; i < tensor->rows; i++) {
            if (tensor->spike_resonance_weights[i]) {
                free(tensor->spike_resonance_weights[i]);
            }
        }
        free(tensor->spike_resonance_weights);
    }
    
    free(tensor->neuromodulator_levels);
    free(tensor->eligibility_traces);
    free(tensor->last_activation_time);
    
    // Рекурсивная очистка иерархических тензоров
    if (tensor->low_level_tensor) fractal_tensor_destroy(tensor->low_level_tensor);
    if (tensor->mid_level_tensor) fractal_tensor_destroy(tensor->mid_level_tensor);
    if (tensor->high_level_tensor) fractal_tensor_destroy(tensor->high_level_tensor);
    
    free(tensor);
}

// === ФРАКТАЛЬНЫЙ АНАЛИЗ ===

float calculate_correlation_dimension(float* time_series, int length, int embedding_dim) {
    if (length < 100 || embedding_dim < 2) return 1.0f;

    int m = length - embedding_dim + 1;
    float* embedded_data = (float*)malloc(m * embedding_dim * sizeof(float));
    if (!embedded_data) return 1.0f;

    // Создание вложенного пространства
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < embedding_dim; j++) {
            embedded_data[i * embedding_dim + j] = time_series[i + j];
        }
    }

    // Используем несколько значений r для оценки наклона
    float r_values[] = {0.01f, 0.02f, 0.05f, 0.1f, 0.2f};
    int num_r = sizeof(r_values) / sizeof(r_values[0]);
    float log_C_r[num_r];
    float log_r[num_r];

    for (int r_idx = 0; r_idx < num_r; r_idx++) {
        float r = r_values[r_idx];
        float correlation_sum = 0.0f;
        int pairs = 0;

        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                float distance = 0.0f;
                for (int k = 0; k < embedding_dim; k++) {
                    float diff = embedded_data[i * embedding_dim + k] - embedded_data[j * embedding_dim + k];
                    distance += diff * diff;
                }
                distance = sqrtf(distance);

                if (distance < r) {
                    correlation_sum += 1.0f;
                }
                pairs++;
            }
        }

        if (pairs > 0) {
            float C_r = correlation_sum / pairs;
            log_C_r[r_idx] = logf(C_r > 0 ? C_r : 1e-10f);
            log_r[r_idx] = logf(r);
        } else {
            log_C_r[r_idx] = -FLT_MAX;
            log_r[r_idx] = logf(r);
        }
    }

    // Линейная регрессия для получения D
    float sum_log_r = 0.0f, sum_log_C_r = 0.0f, sum_log_r_sq = 0.0f, sum_log_r_log_C_r = 0.0f;
    for (int i = 0; i < num_r; i++) {
        sum_log_r += log_r[i];
        sum_log_C_r += log_C_r[i];
        sum_log_r_sq += log_r[i] * log_r[i];
        sum_log_r_log_C_r += log_r[i] * log_C_r[i];
    }
    
    float n = (float)num_r;
    float denominator = n * sum_log_r_sq - sum_log_r * sum_log_r;
    float correlation_dim = 0.0f;
    
    if (fabsf(denominator) > 1e-6f) {
        correlation_dim = (n * sum_log_r_log_C_r - sum_log_r * sum_log_C_r) / denominator;
        correlation_dim = -correlation_dim;
    } else {
        correlation_dim = 1.0f;
    }

    free(embedded_data);
    return correlation_dim > 0 ? correlation_dim : 1.0f;
}

float calculate_kolmogorov_entropy(float* time_series, int length) {
    if (length < 50) return 0.0f;

    int bins = 50;
    if (length < bins) bins = length;
    int* histogram = (int*)calloc(bins, sizeof(int));
    if (!histogram) return 0.0f;

    float min_val = time_series[0], max_val = time_series[0];
    for (int i = 1; i < length; i++) {
        if (time_series[i] < min_val) min_val = time_series[i];
        if (time_series[i] > max_val) max_val = time_series[i];
    }

    float range = max_val - min_val;
    if (range < 1e-6f) range = 1.0f;

    for (int i = 0; i < length; i++) {
        int bin = (int)((time_series[i] - min_val) / range * (bins - 1));
        if (bin >= 0 && bin < bins) {
            histogram[bin]++;
        }
    }

    float entropy = 0.0f;
    float total = (float)length;
    for (int i = 0; i < bins; i++) {
        if (histogram[i] > 0) {
            float p = (float)histogram[i] / total;
            entropy -= p * logf(p);
        }
    }

    free(histogram);
    return entropy / logf(2.0f);
}

float calculate_hurst_exponent(float* time_series, int length) {
    if (length < 20) return 0.5f;

    float mean = 0.0f;
    for (int i = 0; i < length; i++) {
        mean += time_series[i];
    }
    mean /= length;

    float* deviations = (float*)malloc(length * sizeof(float));
    if (!deviations) return 0.5f;
    
    deviations[0] = time_series[0] - mean;
    for (int i = 1; i < length; i++) {
        deviations[i] = deviations[i-1] + (time_series[i] - mean);
    }

    float min_dev = deviations[0], max_dev = deviations[0];
    for (int i = 1; i < length; i++) {
        if (deviations[i] < min_dev) min_dev = deviations[i];
        if (deviations[i] > max_dev) max_dev = deviations[i];
    }
    float R = max_dev - min_dev;

    float variance = 0.0f;
    for (int i = 0; i < length; i++) {
        float diff = time_series[i] - mean;
        variance += diff * diff;
    }
    float S = sqrtf(variance / length);

    free(deviations);

    if (S > 1e-6f) {
        float rs = R / S;
        if (rs > 0) {
            return logf(rs) / logf(length);
        }
    }

    return 0.5f;
}

int detect_chaotic_behavior_simple(float* series, int length) {
    if (length < 3) return 0;

    float total_divergence = 0.0f;
    int divergence_pairs = 0;

    for (int i = 1; i < length - 1; i++) {
        float local_diff = fabsf(series[i] - series[i-1]);
        float next_diff = fabsf(series[i+1] - series[i]);

        if (local_diff > 1e-6f) {
            float divergence_ratio = next_diff / local_diff;
            if (divergence_ratio > 1.1f) {
                total_divergence += divergence_ratio;
                divergence_pairs++;
            }
        }
    }

    if (divergence_pairs > 0) {
        float avg_divergence = total_divergence / divergence_pairs;
        return (avg_divergence > 1.2f) ? 1 : 0;
    }

    return 0;
}

int detect_chaotic_behavior_advanced(float* series, int length) {
    if (length < 5) return 0;
    
    float max_lyapunov = estimate_max_lyapunov(series, length);
    float correlation_dim = calculate_correlation_dimension(series, length, 3);
    float entropy = calculate_kolmogorov_entropy(series, length);
    
    int chaos_score = 0;
    
    if (max_lyapunov > 0.05f) chaos_score++;
    if (correlation_dim > 1.2f && correlation_dim < 2.8f) chaos_score++;
    if (entropy > 0.1f) chaos_score++;
    
    return (chaos_score >= 2) ? 1 : 0;
}

int detect_chaotic_behavior_with_resonance(FractalTensor* tensor, float* series, int length) {
    if (!tensor || length < 5) return 0;
    
    int base_chaos = detect_chaotic_behavior_advanced(series, length);
    
    if (!tensor->spike_resonance_weights) return base_chaos;
    
    float resonance_stability = 0.0f;
    int count = 0;
    
    for (int i = 0; i < tensor->rows && i < length; i++) {
        for (int j = 0; j < tensor->cols; j++) {
            resonance_stability += fabsf(tensor->spike_resonance_weights[i][j] - 1.0f);
            count++;
        }
    }
    
    if (count > 0) {
        resonance_stability /= count;
        if (resonance_stability < 0.2f) {
            return 0; // Стабильная система менее склонна к хаосу
        }
    }
    
    return base_chaos;
}

int check_self_organized_criticality(float* activity_series, int length) {
    if (length < 100) return 0;

    int* avalanche_sizes = (int*)calloc(length, sizeof(int));
    if (!avalanche_sizes) return 0;
    
    int current_avalanche = 0;
    int avalanche_count = 0;
    float threshold = 0.5f;

    for (int i = 0; i < length; i++) {
        if (activity_series[i] > threshold) {
            current_avalanche++;
        } else {
            if (current_avalanche > 0) {
                avalanche_sizes[avalanche_count++] = current_avalanche;
                current_avalanche = 0;
            }
        }
    }

    int power_law_like = 0;
    if (avalanche_count > 10) {
        for (int i = 1; i < avalanche_count; i++) {
            if (avalanche_sizes[i] < avalanche_sizes[i-1]) {
                power_law_like++;
            }
        }

        float power_law_ratio = (float)power_law_like / avalanche_count;
        free(avalanche_sizes);
        return (power_law_ratio > 0.7f) ? 1 : 0;
    }

    free(avalanche_sizes);
    return 0;
}

// === ИНТЕГРИРОВАННЫЕ СПАЙКОВЫЕ ФУНКЦИИ ===

void fractal_tensor_analyze_spike(FractalTensor* tensor, const char* source, float intensity, float fractalDimension) {
    if (!tensor || !source) return;

    // Базовая версия - просто добавляем данные
    if (tensor->cols >= 2) {
        tensor->data[0][0] = intensity;
        tensor->data[0][1] = fractalDimension;
        fractal_tensor_update_from_data(tensor);
    }
}

void fractal_tensor_analyze_spike_enhanced(FractalTensor* tensor, const char* source, 
                                         float intensity, float fractalDimension) {
    if (!tensor || !source) return;
    
    long current_time = time(NULL);
    tensor->activation_count++;
    
    // === L1: БЫСТРОЕ ОБНОВЛЕНИЕ С R-STDP ===
    if (tensor->cols >= 2) {
        // Сдвиг данных с учетом резонансных весов
        for (int i = tensor->rows - 1; i > 0; i--) {
            for (int j = tensor->cols - 1; j > 0; j--) {
                if (i < tensor->rows && j < tensor->cols) {
                    float resonance_factor = 1.0f;
                    if (tensor->spike_resonance_weights && 
                        i < tensor->rows && j < tensor->cols) {
                        resonance_factor = 1.0f + tensor->spike_resonance_weights[i][j] * 0.5f;
                    }
                    tensor->data[i][j] = tensor->data[i-1][j-1] * resonance_factor;
                }
            }
        }
        
        // Добавление новых данных с нейромодуляцией
        float neuromod_factor = 1.0f;
        if (tensor->neuromodulator_levels) {
            neuromod_factor = 1.0f + tensor->neuromodulator_levels[0] * 0.3f;
        }
        
        tensor->data[0][0] = intensity * neuromod_factor;
        tensor->data[0][1] = fractalDimension * neuromod_factor;
        
        // Обновление следов eligibility (STDP)
        update_eligibility_traces(tensor, intensity, current_time);
    }
    
    // === L2: ФРАКТАЛЬНЫЙ АНАЛИЗ С РЕЗОНАНСНОЙ СИНХРОНИЗАЦИЕЙ ===
    if (tensor->rows >= 10) {
        float* intensity_series = (float*)malloc(tensor->rows * sizeof(float));
        float* dimension_series = (float*)malloc(tensor->rows * sizeof(float));
        
        if (intensity_series && dimension_series) {
            int valid_count = 0;
            for (int i = 0; i < tensor->rows; i++) {
                if (!isnan(tensor->data[i][0]) && !isnan(tensor->data[i][1])) {
                    intensity_series[valid_count] = tensor->data[i][0];
                    dimension_series[valid_count] = tensor->data[i][1];
                    valid_count++;
                }
            }
            
            if (valid_count >= 10) {
                // Фрактальный анализ с резонансной коррекцией
                float base_correlation_dim = calculate_correlation_dimension(
                    intensity_series, valid_count, 3);
                
                float resonance_correction = calculate_resonance_correction(tensor, intensity_series, valid_count);
                tensor->correlation_dimension = base_correlation_dim * resonance_correction;
                
                tensor->entropy = calculate_kolmogorov_entropy(intensity_series, valid_count);
                tensor->lyapunov_exponent = calculate_hurst_exponent(intensity_series, valid_count);
                tensor->is_chaotic = detect_chaotic_behavior_with_resonance(tensor, intensity_series, valid_count);
                
                // Проверка самоорганизованной критичности
                tensor->is_critical = check_self_organized_criticality(intensity_series, valid_count);
                
                // Адаптивное обновление фрактальной размерности
                update_fractal_dimension_with_feedback(tensor, intensity_series, valid_count);
                
                // Обновление резонансных весов на основе анализа
                if (tensor->spike_resonance_weights) {
                    update_resonance_weights(tensor, intensity_series, dimension_series, valid_count);
                }
                
                // Распространение в иерархические тензоры
                propagate_to_hierarchical_tensors(tensor, intensity, fractalDimension, source);
                
                // Обновление метаданных
                update_tensor_metadata(tensor, intensity_series, dimension_series, valid_count);
            }
            
            free(intensity_series);
            free(dimension_series);
        }
    }
    
    // === L3: СТРАТЕГИЧЕСКОЕ УПРАВЛЕНИЕ И ОБУЧЕНИЕ ===
    perform_strategic_learning(tensor, intensity, fractalDimension, source);
    tensor->timestamp = current_time;
}

// === R-STDP И РЕЗОНАНСНЫЕ ФУНКЦИИ ===

void update_eligibility_traces(FractalTensor* tensor, float intensity, long current_time) {
    if (!tensor || !tensor->eligibility_traces || !tensor->last_activation_time) return;
    
    for (int i = 0; i < tensor->rows; i++) {
        if (tensor->last_activation_time[i] > 0) {
            float time_diff = current_time - tensor->last_activation_time[i];
            tensor->eligibility_traces[i] *= expf(-time_diff * 0.1f);
        }
    }
    tensor->eligibility_traces[0] += intensity * 0.1f;
    tensor->last_activation_time[0] = current_time;
}

void update_resonance_weights(FractalTensor* tensor, float* intensity_series, 
                            float* dimension_series, int length) {
    if (!tensor || !tensor->spike_resonance_weights || length < 2) return;
    
    for (int i = 0; i < length - 1 && i < tensor->rows; i++) {
        for (int j = 0; j < tensor->cols && j < length; j++) {
            float correlation = fabsf(intensity_series[i] - intensity_series[i+1]) * 
                              fabsf(dimension_series[j] - (j < length ? dimension_series[j] : 0.0f));
            
            float delta_weight = tensor->global_reward_signal * correlation * 
                               tensor->eligibility_traces[i] * 0.01f;
            
            tensor->spike_resonance_weights[i][j] += delta_weight;
            tensor->spike_resonance_weights[i][j] = fmaxf(0.1f, fminf(2.0f, 
                tensor->spike_resonance_weights[i][j]));
        }
    }
}

float calculate_resonance_correction(FractalTensor* tensor, float* series, int length) {
    if (!tensor || !tensor->spike_resonance_weights || length < 5) return 1.0f;
    
    float avg_resonance = 0.0f;
    int count = 0;
    
    for (int i = 0; i < tensor->rows && i < length; i++) {
        for (int j = 0; j < tensor->cols; j++) {
            avg_resonance += tensor->spike_resonance_weights[i][j];
            count++;
        }
    }
    
    if (count > 0) {
        avg_resonance /= count;
        return 0.8f + avg_resonance * 0.4f;
    }
    
    return 1.0f;
}

// === ИЕРАРХИЧЕСКАЯ ОБРАБОТКА ===

void propagate_to_hierarchical_tensors(FractalTensor* tensor, float intensity, 
                                     float fractalDimension, const char* source) {
    if (!tensor) return;
    
    // L1: Быстрая обработка
    if (tensor->low_level_tensor) {
        fractal_tensor_analyze_spike_enhanced(tensor->low_level_tensor, source, intensity, fractalDimension);
    }
    
    // L2: Алгоритмическая обработка
    if (tensor->mid_level_tensor && intensity > 0.3f) {
        float processed_intensity = intensity * (1.0f + tensor->global_reward_signal);
        fractal_tensor_analyze_spike_enhanced(tensor->mid_level_tensor, source, processed_intensity, fractalDimension);
    }
    
    // L3: Стратегическая обработка
    if (tensor->high_level_tensor && intensity > 0.6f && fractalDimension > 1.5f) {
        float strategic_intensity = intensity * fractalDimension * (1.0f + tensor->global_reward_signal);
        fractal_tensor_analyze_spike_enhanced(tensor->high_level_tensor, source, strategic_intensity, fractalDimension);
    }
}

// === СТРАТЕГИЧЕСКОЕ ОБУЧЕНИЕ ===

void perform_strategic_learning(FractalTensor* tensor, float intensity, 
                              float fractalDimension, const char* source) {
    if (!tensor) return;
    
    // Расчет производительности и обновление нейромодуляторов
    float performance = calculate_tensor_performance(tensor);
    tensor->performance_score = performance;
    
    if (tensor->neuromodulator_levels) {
        for (int i = 0; i < tensor->rows; i++) {
            if (performance > 0.7f) {
                tensor->neuromodulator_levels[i] += 0.01f;
            } else if (performance < 0.3f) {
                tensor->neuromodulator_levels[i] -= 0.01f;
            }
            tensor->neuromodulator_levels[i] = fmaxf(0.1f, fminf(1.0f, tensor->neuromodulator_levels[i]));
        }
    }
    
    // Обновление глобального сигнала вознаграждения
    tensor->global_reward_signal = calculate_global_reward(tensor, intensity, fractalDimension);
    
    // Адаптация learning rate
    tensor->learning_rate = 0.01f * (0.5f + performance * 0.5f);
}

float calculate_tensor_performance(FractalTensor* tensor) {
    if (!tensor) return 0.0f;
    
    float performance = 0.0f;
    
    if (tensor->fractal_dimension > 1.2f && tensor->fractal_dimension < 2.5f) {
        performance += 0.3f;
    }
    
    if (tensor->entropy > 0.1f && tensor->entropy < 0.8f) {
        performance += 0.3f;
    }
    
    if (tensor->temporal_correlation > 0.1f) {
        performance += 0.2f;
    }
    
    if (tensor->is_critical) {
        performance += 0.2f;
    }
    
    return fminf(1.0f, performance);
}

float calculate_global_reward(FractalTensor* tensor, float intensity, float fractalDimension) {
    if (!tensor) return 0.0f;
    
    float reward = 0.0f;
    
    reward += intensity * 0.4f;
    reward += (fractalDimension - 1.0f) * 0.3f;
    
    if (tensor->is_critical) {
        reward += 0.3f;
    }
    
    if (tensor->is_chaotic && tensor->entropy > 1.0f) {
        reward -= 0.2f;
    }
    
    return fmaxf(-1.0f, fminf(1.0f, reward));
}

void update_fractal_dimension_with_feedback(FractalTensor* tensor, float* series, int length) {
    if (!tensor || length < 10) return;
    
    float new_fractal_dim = tensor->correlation_dimension;
    float performance_factor = calculate_tensor_performance(tensor);
    float learning_rate = 0.05f * performance_factor;
    
    tensor->fractal_dimension = (1.0f - learning_rate) * tensor->fractal_dimension + 
                               learning_rate * new_fractal_dim;
    
    tensor->fractal_dimension = fmaxf(1.0f, fminf(3.0f, tensor->fractal_dimension));
}

// === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

void fractal_tensor_print(const FractalTensor* tensor) {
    if (!tensor) return;

    printf("=== Fractal Tensor Analysis ===\n");
    printf("Dimensions: %d x %d\n", tensor->rows, tensor->cols);
    printf("Timestamp: %ld\n", tensor->timestamp);
    printf("Fractal Dimension: %.3f\n", tensor->fractal_dimension);
    printf("Correlation Dimension: %.3f\n", tensor->correlation_dimension);
    printf("Kolmogorov Entropy: %.3f\n", tensor->entropy);
    printf("Lyapunov Exponent: %.3f\n", tensor->lyapunov_exponent);
    printf("Chaotic Behavior: %s\n", tensor->is_chaotic ? "YES" : "NO");
    printf("Self-Organized Criticality: %s\n", tensor->is_critical ? "YES" : "NO");
    printf("Activity Sum: %.3f\n", tensor->activity_sum);
    printf("Activity Variance: %.3f\n", tensor->activity_variance);
    printf("Temporal Correlation: %.3f\n", tensor->temporal_correlation);
    printf("Global Reward Signal: %.3f\n", tensor->global_reward_signal);
    printf("Performance Score: %.3f\n", tensor->performance_score);
    printf("Activation Count: %d\n", tensor->activation_count);
    
    if (tensor->spike_resonance_weights) {
        printf("Spike Resonance: ENABLED\n");
    }
    if (tensor->low_level_tensor) {
        printf("Hierarchical Structure: ENABLED\n");
    }
    printf("===============================\n");
}

void fractal_tensor_update_from_data(FractalTensor* tensor) {
    if (!tensor || !tensor->data) return;

    if (tensor->rows > 0 && tensor->cols > 0) {
        float* series = tensor->data[0];
        int length = tensor->cols;

        tensor->correlation_dimension = calculate_correlation_dimension(series, length, 2);
        tensor->entropy = calculate_kolmogorov_entropy(series, length);
        tensor->lyapunov_exponent = calculate_hurst_exponent(series, length);
        tensor->is_chaotic = detect_chaotic_behavior_simple(series, length);

        // Расчет статистик
        tensor->activity_sum = 0.0f;
        float mean = 0.0f;
        for (int i = 0; i < length; i++) {
            tensor->activity_sum += series[i];
        }
        mean = tensor->activity_sum / length;
        
        tensor->activity_variance = 0.0f;
        for (int i = 0; i < length; i++) {
            float diff = series[i] - mean;
            tensor->activity_variance += diff * diff;
        }
        tensor->activity_variance /= length;

        // Временная корреляция
        if (tensor->rows > 1) {
            float* series2 = tensor->data[1];
            float mean2 = 0.0f, sum_prod = 0.0f, var1 = 0.0f, var2 = 0.0f;
            
            for (int i = 0; i < length; i++) {
                mean2 += series2[i];
            }
            mean2 /= length;
            
            for (int i = 0; i < length; i++) {
                sum_prod += (series[i] - mean) * (series2[i] - mean2);
                var1 += (series[i] - mean) * (series[i] - mean);
                var2 += (series2[i] - mean2) * (series2[i] - mean2);
            }
            
            var1 /= length; 
            var2 /= length;
            float denom = sqrtf(var1 * var2);
            
            if (denom > 1e-6f) {
                tensor->temporal_correlation = sum_prod / (length * denom);
            } else {
                tensor->temporal_correlation = 0.0f;
            }
        }
    }
}

void update_tensor_metadata(FractalTensor* tensor, float* intensity_series, float* dimension_series, int length) {
    if (!tensor || !intensity_series || length == 0) return;
    
    tensor->activity_sum = 0.0f;
    float mean_intensity = 0.0f;
    float mean_dimension = 0.0f;
    
    for (int i = 0; i < length; i++) {
        tensor->activity_sum += intensity_series[i];
        mean_intensity += intensity_series[i];
        mean_dimension += dimension_series[i];
    }
    
    mean_intensity /= length;
    mean_dimension /= length;
    
    tensor->activity_variance = 0.0f;
    float dimension_variance = 0.0f;
    
    for (int i = 0; i < length; i++) {
        float intensity_diff = intensity_series[i] - mean_intensity;
        float dimension_diff = dimension_series[i] - mean_dimension;
        tensor->activity_variance += intensity_diff * intensity_diff;
        dimension_variance += dimension_diff * dimension_diff;
    }
    
    tensor->activity_variance /= length;
    dimension_variance /= length;
    
    if (length > 1) {
        float autocorr_intensity = 0.0f;
        float autocorr_dimension = 0.0f;
        
        for (int i = 0; i < length - 1; i++) {
            autocorr_intensity += (intensity_series[i] - mean_intensity) * 
                                 (intensity_series[i+1] - mean_intensity);
            autocorr_dimension += (dimension_series[i] - mean_dimension) * 
                                 (dimension_series[i+1] - mean_dimension);
        }
        
        autocorr_intensity /= (length - 1) * tensor->activity_variance;
        autocorr_dimension /= (length - 1) * dimension_variance;
        
        tensor->temporal_correlation = (autocorr_intensity + autocorr_dimension) * 0.5f;
    }
}

float analyze_semantic_complexity(const char* source) {
    if (!source) return 0.0f;
    
    int length = strlen(source);
    if (length == 0) return 0.0f;
    
    int char_counts[256] = {0};
    int unique_chars = 0;
    
    for (int i = 0; i < length; i++) {
        unsigned char c = source[i];
        if (char_counts[c] == 0) unique_chars++;
        char_counts[c]++;
    }
    
    float entropy = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (char_counts[i] > 0) {
            float probability = (float)char_counts[i] / length;
            entropy -= probability * logf(probability);
        }
    }
    
    float max_entropy = logf(256.0f);
    float normalized_entropy = entropy / max_entropy;
    float length_factor = fminf(1.0f, (float)length / 100.0f);
    
    return normalized_entropy * 0.7f + length_factor * 0.3f;
}

float estimate_max_lyapunov(float* series, int length) {
    if (length < 20) return 0.0f;
    
    float total_divergence = 0.0f;
    int divergence_pairs = 0;
    
    for (int i = 1; i < length - 1; i++) {
        float local_diff = fabsf(series[i] - series[i-1]);
        float next_diff = fabsf(series[i+1] - series[i]);
        
        if (local_diff > 1e-6f && next_diff > 1e-6f) {
            float divergence = logf(next_diff / local_diff);
            total_divergence += divergence;
            divergence_pairs++;
        }
    }
    
    return (divergence_pairs > 0) ? total_divergence / divergence_pairs : 0.0f;
}