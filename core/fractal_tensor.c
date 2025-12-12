#include "fractal_tensor.h"
#include <string.h>
#include <float.h>
#include <math.h>

#ifndef isnan
#define isnan(x) ((x) != (x))
#endif

// Используем другое имя для константы, чтобы избежать конфликта с макросом
static const float GOLDEN_SECTION = 1.61803398875f;
static const float DEFAULT_FIB_X = 1.0f;
static const float DEFAULT_FIB_Y = 2.0f;

// Вспомогательные функции
static void fill_spiral_fibonacci(float** data, int rows, int cols);
static void fill_fractal_recursive(float** data, int row_start, int row_end, 
                                 int col_start, int col_end, int level, float base_value);

// === БАЗОВОЕ СОЗДАНИЕ ТЕНЗОРА С ГРАДИЕНТНЫМ ФРАКТАЛЬНЫМ ЗАПОЛНЕНИЕМ ===

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
    
    // ГРАДИЕНТНОЕ ЗАПОЛНЕНИЕ: создаём только необходимые ячейки
    for (int i = 0; i < rows; i++) {
        tensor->data[i] = (float*)calloc(cols, sizeof(float));
        if (!tensor->data[i]) {
            for (int j = 0; j < i; j++) free(tensor->data[j]);
            free(tensor->data);
            free(tensor);
            return NULL;
        }
    }

    // === ФРАКТАЛЬНОЕ ГРАДИЕНТНОЕ ЗАПОЛНЕНИЕ ===
    // Используем принцип "только по X Y" + вычисление остальных
    if (rows >= 2 && cols >= 2) {
        // Инициализируем только граничные значения (X и Y оси)
        float fib1 = DEFAULT_FIB_X, fib2 = DEFAULT_FIB_X;
        
        // 1. Заполняем первую строку (ось X) последовательностью Фибоначчи
        for (int j = 0; j < cols; j++) {
            tensor->data[0][j] = fib1;
            float next_fib = fib1 + fib2;
            fib1 = fib2;
            fib2 = next_fib;
        }
        
        // 2. Заполняем первый столбец (ось Y) последовательностью Фибоначчи
        fib1 = DEFAULT_FIB_Y;
        fib2 = DEFAULT_FIB_X + DEFAULT_FIB_Y; // Начинаем с другого начального значения для разнообразия
        for (int i = 0; i < rows; i++) {
            tensor->data[i][0] = fib1;
            float next_fib = fib1 + fib2;
            fib1 = fib2;
            fib2 = next_fib;
        }
        
        // 3. Вычисляем остальные значения по фрактальному правилу:
        // Каждая ячейка = среднее значение от соседей слева и сверху * золотое сечение
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < cols; j++) {
                float left = tensor->data[i][j-1];
                float top = tensor->data[i-1][j];
                
                // Базовое правило: среднее арифметическое
                float base_value = (left + top) * 0.5f;
                
                // Добавляем фрактальную компоненту: чем дальше от центра, тем больше влияние золотого сечения
                float distance_from_center = sqrtf(powf(i - rows/2.0f, 2) + powf(j - cols/2.0f, 2));
                float max_distance = sqrtf(powf(rows/2.0f, 2) + powf(cols/2.0f, 2));
                float fractal_factor = 1.0f + (GOLDEN_SECTION - 1.0f) * (distance_from_center / max_distance);
                
                tensor->data[i][j] = base_value * fractal_factor;
            }
        }
        
        // 4. Нормализуем матрицу для создания устойчивых паттернов
        float matrix_sum = 0.0f;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix_sum += tensor->data[i][j];
            }
        }
        
        float avg_value = matrix_sum / (rows * cols);
        if (avg_value > 0.001f) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    tensor->data[i][j] /= avg_value;
                }
            }
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
    
    tensor->learning_rate = DEFAULT_LEARNING_RATE;
    tensor->performance_score = 0.5f;
    tensor->activation_count = 0;
    tensor->stability_factor = 0.5f;

    return tensor;
}

// === УЛУЧШЕННОЕ СОЗДАНИЕ ТЕНЗОРА СО СПАЙКОВЫМИ СИСТЕМАМИ И ГРАДИЕНТНЫМ ЗАПОЛНЕНИЕМ ===

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
    
    // === ГРАДИЕНТНОЕ ЗАПОЛНЕНИЕ ДЛЯ СПАЙКОВЫХ КОМПОНЕНТОВ ===
    for (int i = 0; i < rows; i++) {
        tensor->spike_resonance_weights[i] = (float*)calloc(cols, sizeof(float));
        if (!tensor->spike_resonance_weights[i]) {
            for (int j = 0; j < i; j++) free(tensor->spike_resonance_weights[j]);
            fractal_tensor_destroy(tensor);
            return NULL;
        }
        
        // Градиентное заполнение резонансных весов (радиальный градиент)
        float center_i = rows / 2.0f;
        float center_j = cols / 2.0f;
        
        for (int j = 0; j < cols; j++) {
            // Расстояние от центра
            float distance = sqrtf(powf(i - center_i, 2) + powf(j - center_j, 2));
            float max_distance = sqrtf(powf(center_i, 2) + powf(center_j, 2));
            
            // Градиент: максимальное значение в центре, уменьшается к краям
            float base_weight = 0.5f * (1.0f - distance / max_distance) + 0.1f;
            
            // Добавляем фрактальную компоненту
            float fractal_mod = sinf(i * GOLDEN_SECTION) * cosf(j * GOLDEN_SECTION) * 0.1f;
            
            tensor->spike_resonance_weights[i][j] = base_weight + fractal_mod;
            CLAMP(tensor->spike_resonance_weights[i][j]);
        }
        
        // Градиент для нейромодуляторов (увеличивается к краям)
        float neuromod_gradient = (float)i / rows * 0.5f + 0.3f;
        tensor->neuromodulator_levels[i] = neuromod_gradient;
        
        tensor->eligibility_traces[i] = 0.0f;
        tensor->last_activation_time[i] = 0;
    }
    
    tensor->global_reward_signal = 0.0f;
    tensor->is_critical = 0;
    
    // === РЕКУРСИВНОЕ СОЗДАНИЕ ИЕРАРХИЧЕСКИХ ТЕНЗОРОВ С ГРАДИЕНТНЫМ ЗАПОЛНЕНИЕМ ===
    if (rows >= 4 && cols >= 4) {
        tensor->low_level_tensor = fractal_tensor_create_enhanced(rows / 2, cols / 2);
        tensor->mid_level_tensor = fractal_tensor_create_enhanced(rows / 4, cols / 4);
        if (rows >= 8 && cols >= 8) {
            tensor->high_level_tensor = fractal_tensor_create_enhanced(rows / 8, cols / 8);
        }
    }
    
    // === ВЫЧИСЛЕНИЕ ФРАКТАЛЬНЫХ ХАРАКТЕРИСТИК ===
    fractal_tensor_analyze_structure(tensor);
    
    return tensor;
}

// === ФУНКЦИЯ ДЛЯ АНАЛИЗА ФРАКТАЛЬНОЙ СТРУКТУРЫ ТЕНЗОРА ===

void fractal_tensor_analyze_structure(FractalTensor* tensor) {
    if (!tensor || !tensor->data) return;
    
    // Анализ самоподобия
    float total_sum = 0.0f;
    float sub_sum = 0.0f;
    
    // Сумма всех элементов
    for (int i = 0; i < tensor->rows; i++) {
        for (int j = 0; j < tensor->cols; j++) {
            total_sum += tensor->data[i][j];
        }
    }
    
    // Сумма элементов в первой четверти
    int half_rows = tensor->rows / 2;
    int half_cols = tensor->cols / 2;
    for (int i = 0; i < half_rows; i++) {
        for (int j = 0; j < half_cols; j++) {
            sub_sum += tensor->data[i][j];
        }
    }
    
    // Вычисление коэффициента самоподобия
    if (sub_sum > 0 && total_sum > 0) {
        float scaling_factor = total_sum / (4.0f * sub_sum);
        if (scaling_factor > 0 && !isnan(scaling_factor)) {
            tensor->fractal_dimension = logf(4.0f) / logf(scaling_factor);
            CLAMP_FRACTAL_DIM(tensor->fractal_dimension);
        }
    }
    
    // Анализ корреляционной размерности
    if (tensor->rows > 1 && tensor->cols > 1) {
        // Преобразуем матрицу в временной ряд для анализа
        float* time_series = (float*)malloc(tensor->rows * tensor->cols * sizeof(float));
        if (time_series) {
            int index = 0;
            for (int i = 0; i < tensor->rows; i++) {
                for (int j = 0; j < tensor->cols; j++) {
                    time_series[index++] = tensor->data[i][j];
                }
            }
            
            tensor->correlation_dimension = calculate_correlation_dimension(
                time_series, tensor->rows * tensor->cols, 3);
            tensor->entropy = calculate_kolmogorov_entropy(
                time_series, tensor->rows * tensor->cols);
            tensor->lyapunov_exponent = calculate_hurst_exponent(
                time_series, tensor->rows * tensor->cols);
            tensor->is_chaotic = detect_chaotic_behavior_simple(
                time_series, tensor->rows * tensor->cols);
            
            free(time_series);
        }
    }
}

// === ОЧИСТКА ПАМЯТИ ===

void fractal_tensor_destroy(FractalTensor* tensor) {
    if (!tensor) return;

    // Очистка основных данных (градиентно заполненных)
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

// === ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ЗАПОЛНЕНИЯ ===

FractalTensor* fractal_tensor_create_gradient_fibonacci(int rows, int cols) {
    FractalTensor* tensor = fractal_tensor_create(rows, cols);
    if (!tensor) return NULL;
    
    fractal_tensor_fill_gradient_fibonacci(tensor);
    return tensor;
}

FractalTensor* fractal_tensor_create_fractal_gradient(int rows, int cols, int levels) {
    FractalTensor* tensor = fractal_tensor_create(rows, cols);
    if (!tensor) return NULL;
    
    fractal_tensor_fill_fractal_quadrants(tensor, levels);
    return tensor;
}

void fractal_tensor_fill_gradient_fibonacci(FractalTensor* tensor) {
    if (!tensor || !tensor->data) return;
    
    fill_spiral_fibonacci(tensor->data, tensor->rows, tensor->cols);
}

void fractal_tensor_fill_spiral_pattern(FractalTensor* tensor) {
    if (!tensor || !tensor->data) return;
    
    fill_spiral_fibonacci(tensor->data, tensor->rows, tensor->cols);
}

void fractal_tensor_fill_radial_gradient(FractalTensor* tensor) {
    if (!tensor || !tensor->data) return;
    
    float center_value = 1.0f;
    fill_radial_gradient(tensor->data, tensor->rows, tensor->cols, center_value);
}

void fractal_tensor_fill_fractal_quadrants(FractalTensor* tensor, int levels) {
    if (!tensor || !tensor->data || levels <= 0) return;
    
    levels = MIN(levels, MAX_FRACTAL_LEVELS);
    fill_fractal_recursive(tensor->data, 0, tensor->rows-1, 0, tensor->cols-1, 
                          levels, 1.0f);
}

// === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЗАПОЛНЕНИЯ ===

static void fill_spiral_fibonacci(float** data, int rows, int cols) {
    if (rows == 0 || cols == 0) return;
    
    int top = 0, bottom = rows - 1;
    int left = 0, right = cols - 1;
    float fib1 = DEFAULT_FIB_X, fib2 = DEFAULT_FIB_X;
    
    while (top <= bottom && left <= right) {
        // Заполняем верхнюю строку
        for (int j = left; j <= right; j++) {
            data[top][j] = fib1;
            float temp = fib1 + fib2;
            fib1 = fib2;
            fib2 = temp;
        }
        top++;
        
        // Заполняем правый столбец
        for (int i = top; i <= bottom; i++) {
            data[i][right] = fib1;
            float temp = fib1 + fib2;
            fib1 = fib2;
            fib2 = temp;
        }
        right--;
        
        // Заполняем нижнюю строку (если есть)
        if (top <= bottom) {
            for (int j = right; j >= left; j--) {
                data[bottom][j] = fib1;
                float temp = fib1 + fib2;
                fib1 = fib2;
                fib2 = temp;
            }
            bottom--;
        }
        
        // Заполняем левый столбец (если есть)
        if (left <= right) {
            for (int i = bottom; i >= top; i--) {
                data[i][left] = fib1;
                float temp = fib1 + fib2;
                fib1 = fib2;
                fib2 = temp;
            }
            left++;
        }
    }
}

void fill_radial_gradient(float** data, int rows, int cols, float center_value) {
    float center_i = rows / 2.0f;
    float center_j = cols / 2.0f;
    float max_distance = sqrtf(powf(center_i, 2) + powf(center_j, 2));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float distance = sqrtf(powf(i - center_i, 2) + powf(j - center_j, 2));
            float gradient = center_value * (1.0f - distance / max_distance);
            data[i][j] = fmaxf(0.1f, gradient);
        }
    }
}

static void fill_fractal_recursive(float** data, int row_start, int row_end, 
                                 int col_start, int col_end, int level, float base_value) {
    if (level <= 0 || row_start > row_end || col_start > col_end) return;
    
    int rows = row_end - row_start + 1;
    int cols = col_end - col_start + 1;
    
    if (rows <= 2 || cols <= 2) {
        // Базовый случай: заполняем простым градиентом
        for (int i = row_start; i <= row_end; i++) {
            float row_factor = (float)(i - row_start) / (rows - 1);
            for (int j = col_start; j <= col_end; j++) {
                float col_factor = (float)(j - col_start) / (cols - 1);
                data[i][j] = base_value * (1.0f + row_factor + col_factor);
            }
        }
        return;
    }
    
    // Разделяем на 4 квадранта (фрактальное разделение)
    int row_mid = row_start + rows / 2;
    int col_mid = col_start + cols / 2;
    
    // Рекурсивно заполняем квадранты с разными базовыми значениями
    float quadrant_values[4] = {
        base_value,                    // Верхний левый
        base_value * GOLDEN_SECTION,   // Верхний правый (золотое сечение)
        base_value * 0.618f,           // Нижний левый
        base_value * 1.0f              // Нижний правый
    };
    
    fill_fractal_recursive(data, row_start, row_mid-1, col_start, col_mid-1, 
                         level-1, quadrant_values[0]);
    fill_fractal_recursive(data, row_start, row_mid-1, col_mid, col_end, 
                         level-1, quadrant_values[1]);
    fill_fractal_recursive(data, row_mid, row_end, col_start, col_mid-1, 
                         level-1, quadrant_values[2]);
    fill_fractal_recursive(data, row_mid, row_end, col_mid, col_end, 
                         level-1, quadrant_values[3]);
}

// ==================== ФУНКЦИИ АНАЛИЗА ====================

float calculate_similarity_coefficient(FractalTensor* tensor) {
    if (!tensor || tensor->rows < 4 || tensor->cols < 4) return 0.0f;
    
    float quadrant_sums[4] = {0};
    int quad_rows = tensor->rows / 2;
    int quad_cols = tensor->cols / 2;
    
    for (int quad = 0; quad < 4; quad++) {
        int row_start = (quad / 2) * quad_rows;
        int col_start = (quad % 2) * quad_cols;
        
        for (int i = row_start; i < row_start + quad_rows; i++) {
            for (int j = col_start; j < col_start + quad_cols; j++) {
                quadrant_sums[quad] += tensor->data[i][j];
            }
        }
    }
    
    float avg_sum = (quadrant_sums[0] + quadrant_sums[1] + 
                    quadrant_sums[2] + quadrant_sums[3]) / 4.0f;
    
    if (avg_sum == 0) return 0.0f;
    
    float similarity = 1.0f - (fabs(quadrant_sums[0] - avg_sum) + 
                              fabs(quadrant_sums[1] - avg_sum) + 
                              fabs(quadrant_sums[2] - avg_sum) + 
                              fabs(quadrant_sums[3] - avg_sum)) / (4.0f * avg_sum);
    
    return fmaxf(0.0f, fminf(1.0f, similarity));
}

// === ВИЗУАЛИЗАЦИЯ ===

void print_fractal_tensor_gradient(FractalTensor* tensor) {
    if (!tensor || !tensor->data) return;
    
    printf("=== Фрактальный тензор с градиентным заполнением ===\n");
    printf("Размер: %d x %d\n", tensor->rows, tensor->cols);
    printf("Фрактальная размерность: %.3f\n", tensor->fractal_dimension);
    printf("Паттерн заполнения:\n");
    
    // Простая ASCII визуализация
    for (int i = 0; i < tensor->rows; i++) {
        for (int j = 0; j < tensor->cols; j++) {
            float value = tensor->data[i][j];
            char symbol = ' ';
            
            if (value < 0.2f) symbol = '.';
            else if (value < 0.4f) symbol = ':';
            else if (value < 0.6f) symbol = 'o';
            else if (value < 0.8f) symbol = 'O';
            else symbol = '@';
            
            printf("%c ", symbol);
        }
        printf("\n");
    }
    
    // Статистика градиента
    float min_val = tensor->data[0][0];
    float max_val = tensor->data[0][0];
    float center_val = tensor->data[tensor->rows/2][tensor->cols/2];
    
    for (int i = 0; i < tensor->rows; i++) {
        for (int j = 0; j < tensor->cols; j++) {
            if (tensor->data[i][j] < min_val) min_val = tensor->data[i][j];
            if (tensor->data[i][j] > max_val) max_val = tensor->data[i][j];
        }
    }
    
    printf("\nСтатистика градиента:\n");
    printf("Минимальное значение: %.3f\n", min_val);
    printf("Максимальное значение: %.3f\n", max_val);
    printf("Центральное значение: %.3f\n", center_val);
    printf("Градиентный диапазон: %.3f\n", max_val - min_val);
    
    // Проверка самоподобия
    float similarity = calculate_similarity_coefficient(tensor);
    printf("Коэффициент самоподобия: %.3f\n", similarity);
}

// ==================== ИНТЕГРИРОВАННЫЕ СПАЙКОВЫЕ ФУНКЦИИ ====================

void fractal_tensor_analyze_spike(FractalTensor* tensor, const char* source, 
                                  float intensity, float fractalDimension) {
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

// ==================== R-STDP И РЕЗОНАНСНЫЕ ФУНКЦИИ ====================

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
            CLAMP(tensor->spike_resonance_weights[i][j]);
        }
    }
}

float calculate_resonance_correction(FractalTensor* tensor, float* series, int length) {
    (void)series; // Не используется, убираем предупреждение
    
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

// ==================== ИЕРАРХИЧЕСКАЯ ОБРАБОТКА ====================

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

// ==================== СТРАТЕГИЧЕСКОЕ ОБУЧЕНИЕ ====================

void perform_strategic_learning(FractalTensor* tensor, float intensity, 
                              float fractalDimension, const char* source) {
    (void)source; // Не используется, убираем предупреждение
    
    if (!tensor) return;
    
    // Расчет производительности и обновление нейромодуляторов
    float performance = calculate_tensor_performance(tensor);
    tensor->performance_score = performance;
    
    if (tensor->neuromodulator_levels) {
        for (int i = 0; i < tensor->rows; i++) {
            if (performance > PERFORMANCE_THRESHOLD_HIGH) {
                tensor->neuromodulator_levels[i] += 0.01f;
            } else if (performance < PERFORMANCE_THRESHOLD_LOW) {
                tensor->neuromodulator_levels[i] -= 0.01f;
            }
            CLAMP(tensor->neuromodulator_levels[i]);
        }
    }
    
    // Обновление глобального сигнала вознаграждения
    tensor->global_reward_signal = calculate_global_reward(tensor, intensity, fractalDimension);
    
    // Адаптация learning rate
    tensor->learning_rate = DEFAULT_LEARNING_RATE * (0.5f + performance * 0.5f);
    tensor->learning_rate = fmaxf(MIN_LEARNING_RATE, fminf(MAX_LEARNING_RATE, tensor->learning_rate));
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
    (void)series; // Не используется, убираем предупреждение
    (void)length; // Не используется, убираем предупреждение
    
    if (!tensor) return;
    
    float new_fractal_dim = tensor->correlation_dimension;
    float performance_factor = calculate_tensor_performance(tensor);
    float learning_rate = 0.05f * performance_factor;
    
    tensor->fractal_dimension = (1.0f - learning_rate) * tensor->fractal_dimension + 
                               learning_rate * new_fractal_dim;
    
    CLAMP_FRACTAL_DIM(tensor->fractal_dimension);
}

// ==================== ФРАКТАЛЬНЫЙ АНАЛИЗ ====================

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

    int bins = MIN(50, length);
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

// ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

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
    printf("Learning Rate: %.4f\n", tensor->learning_rate);
    printf("Stability Factor: %.3f\n", tensor->stability_factor);
    
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