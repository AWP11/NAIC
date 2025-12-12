#ifndef FRACTAL_TENSOR_H
#define FRACTAL_TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Структура для фрактального тензора с интеграцией спайковых систем
typedef struct FractalTensor {
    // Основные данные тензора
    float** data;
    float* eigenvalues;
    float* eigenvectors;
    int rows;
    int cols;
    
    // Фрактальные характеристики
    float fractal_dimension;
    float correlation_dimension;
    float entropy;
    float lyapunov_exponent;
    int is_chaotic;
    
    // Мета-характеристики и временные данные
    long timestamp;
    float activity_sum;
    float activity_variance;
    float temporal_correlation;
    
    // === ИНТЕГРАЦИЯ СПАЙКОВЫХ СИСТЕМ ===
    float** spike_resonance_weights;    // Резонансные веса между измерениями
    float* neuromodulator_levels;       // Уровни нейромодуляторов для каждого ряда
    float* eligibility_traces;          // Следы eligibility для STDP
    int* last_activation_time;          // Время последней активации
    float global_reward_signal;         // Глобальный сигнал вознаграждения
    int is_critical;                    // Флаг самоорганизованной критичности
    
    // Иерархические уровни (L1, L2, L3)
    struct FractalTensor* low_level_tensor;    // L1: Быстрые операции
    struct FractalTensor* mid_level_tensor;    // L2: Алгоритмические паттерны  
    struct FractalTensor* high_level_tensor;   // L3: Стратегическое управление
    
    // Статистика обучения и производительности
    float learning_rate;
    float performance_score;
    int activation_count;
    float stability_factor;
    
} FractalTensor;

// ==================== ОСНОВНЫЕ ФУНКЦИИ ====================

// Создание и уничтожение тензоров
FractalTensor* fractal_tensor_create(int rows, int cols);
FractalTensor* fractal_tensor_create_enhanced(int rows, int cols);
FractalTensor* fractal_tensor_create_gradient_fibonacci(int rows, int cols);
FractalTensor* fractal_tensor_create_fractal_gradient(int rows, int cols, int levels);
void fractal_tensor_destroy(FractalTensor* tensor);

// ==================== ФРАКТАЛЬНЫЙ АНАЛИЗ ====================

// Базовые метрики фрактального анализа
float calculate_correlation_dimension(float* time_series, int length, int embedding_dim);
float calculate_kolmogorov_entropy(float* time_series, int length);
float calculate_hurst_exponent(float* time_series, int length);

// Детектирование хаотического поведения
int detect_chaotic_behavior_simple(float* series, int length);
int detect_chaotic_behavior_advanced(float* series, int length);
int detect_chaotic_behavior_with_resonance(FractalTensor* tensor, float* series, int length);

// Анализ критических состояний
int check_self_organized_criticality(float* activity_series, int length);

// ==================== ИНТЕГРИРОВАННЫЕ СПАЙКОВЫЕ ФУНКЦИИ ====================

// Анализ спайков с разной степенью сложности
void fractal_tensor_analyze_spike(FractalTensor* tensor, const char* source, 
                                  float intensity, float fractalDimension);
void fractal_tensor_analyze_spike_enhanced(FractalTensor* tensor, const char* source, 
                                         float intensity, float fractalDimension);

// R-STDP и резонансные механизмы
void update_resonance_weights(FractalTensor* tensor, float* intensity_series, 
                            float* dimension_series, int length);
void update_eligibility_traces(FractalTensor* tensor, float intensity, long current_time);
float calculate_resonance_correction(FractalTensor* tensor, float* series, int length);

// Иерархическая обработка
void propagate_to_hierarchical_tensors(FractalTensor* tensor, float intensity, 
                                     float fractalDimension, const char* source);

// Стратегическое обучение и управление
void perform_strategic_learning(FractalTensor* tensor, float intensity, 
                              float fractalDimension, const char* source);

// ==================== УПРАВЛЕНИЕ И ОБУЧЕНИЕ ====================

// Оценка производительности и вознаграждения
float calculate_tensor_performance(FractalTensor* tensor);
float calculate_global_reward(FractalTensor* tensor, float intensity, float fractalDimension);

// Адаптивное обновление параметров
void update_fractal_dimension_with_feedback(FractalTensor* tensor, float* series, int length);
void adapt_learning_parameters(FractalTensor* tensor, float performance);

// ==================== ФУНКЦИИ ГРАДИЕНТНОГО ЗАПОЛНЕНИЯ ====================

// Функции градиентного заполнения
void fractal_tensor_fill_gradient_fibonacci(FractalTensor* tensor);
void fractal_tensor_fill_spiral_pattern(FractalTensor* tensor);
void fractal_tensor_fill_radial_gradient(FractalTensor* tensor);
void fractal_tensor_fill_fractal_quadrants(FractalTensor* tensor, int levels);

// Вспомогательные функции для заполнения (не экспортируем наружу)
void fill_xy_axes_fibonacci(float** data, int rows, int cols);

// Анализ градиентных паттернов
void fractal_tensor_analyze_structure(FractalTensor* tensor);
float calculate_similarity_coefficient(FractalTensor* tensor);
float calculate_gradient_variance(FractalTensor* tensor);
void check_fractal_self_similarity(FractalTensor* tensor);

// ==================== ВИЗУАЛИЗАЦИЯ И ДИАГНОСТИКА ====================

// Ввод-вывод и диагностика
void fractal_tensor_print(const FractalTensor* tensor);
void fractal_tensor_print_detailed(const FractalTensor* tensor);
void print_fractal_tensor_gradient(FractalTensor* tensor);
void print_fractal_tensor_pattern(FractalTensor* tensor);
void fractal_tensor_save_to_file(const FractalTensor* tensor, const char* filename);
FractalTensor* fractal_tensor_load_from_file(const char* filename);

// Обновление и анализ данных
void fractal_tensor_update_from_data(FractalTensor* tensor);
void update_tensor_metadata(FractalTensor* tensor, float* intensity_series, 
                          float* dimension_series, int length);

// Семантический и статистический анализ
float analyze_semantic_complexity(const char* source);
float estimate_max_lyapunov(float* series, int length);
float calculate_tensor_stability(FractalTensor* tensor);

// Утилиты для работы с временными рядами
void normalize_time_series(float* series, int length);
void apply_exponential_smoothing(float* series, int length, float alpha);
float calculate_autocorrelation(float* series, int length, int lag);

// ==================== МАКРОСЫ И КОНСТАНТЫ ====================

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef CLAMP
#define CLAMP(x) do { if ((x) < 0.0f) (x) = 0.0f; if ((x) > 1.0f) (x) = 1.0f; } while(0)
#endif

#ifndef CLAMP_FRACTAL_DIM
#define CLAMP_FRACTAL_DIM(x) do { if ((x) < 1.0f) (x) = 1.0f; if ((x) > 3.0f) (x) = 3.0f; } while(0)
#endif

#ifndef CLAMP_GRADIENT
#define CLAMP_GRADIENT(x, min, max) do { \
    if ((x) < (min)) (x) = (min); \
    if ((x) > (max)) (x) = (max); \
} while(0)
#endif

// Уровни иерархической обработки
#define TENSOR_LEVEL_LOW 0
#define TENSOR_LEVEL_MID 1
#define TENSOR_LEVEL_HIGH 2

// Режимы заполнения
#define FILL_MODE_FIBONACCI 0
#define FILL_MODE_SPIRAL 1
#define FILL_MODE_RADIAL 2
#define FILL_MODE_FRACTAL 3
#define FILL_MODE_HYBRID 4

// Пороговые значения для принятия решений
#define CRITICALITY_THRESHOLD 0.7f
#define CHAOS_THRESHOLD 0.5f
#define PERFORMANCE_THRESHOLD_HIGH 0.7f
#define PERFORMANCE_THRESHOLD_LOW 0.3f
#define GRADIENT_SIMILARITY_THRESHOLD 0.8f

// Константы обучения
#define DEFAULT_LEARNING_RATE 0.01f
#define MAX_LEARNING_RATE 0.1f
#define MIN_LEARNING_RATE 0.001f

// Константы градиентного заполнения
#define GOLDEN_RATIO 1.61803398875f
#define FIBONACCI_START_X 1.0f
#define FIBONACCI_START_Y 2.0f
#define MAX_FRACTAL_LEVELS 8

// Флаги для режимов работы
#define TENSOR_MODE_BASIC 0
#define TENSOR_MODE_ENHANCED 1
#define TENSOR_MODE_HIERARCHICAL 2
#define TENSOR_MODE_GRADIENT 3

// ==================== СТРУКТУРЫ ДЛЯ АНАЛИЗА ====================

typedef struct {
    float fractal_dimension;
    float correlation_dimension;
    float entropy;
    float lyapunov_exponent;
    int is_chaotic;
    int is_critical;
    float performance_score;
    float stability_factor;
    float gradient_variance;
    float self_similarity;
    int fill_mode;
} TensorAnalysisResult;

typedef struct {
    float* intensity_series;
    float* dimension_series;
    int length;
    float semantic_complexity;
    long analysis_timestamp;
} SpikeAnalysisData;

// Структура для анализа градиентных паттернов
typedef struct {
    float min_value;
    float max_value;
    float center_value;
    float gradient_range;
    float diagonal_symmetry;
    float quadrant_similarity[4];
    float fractal_scaling_factor;
} GradientPatternAnalysis;

// Функции для работы со структурами анализа
TensorAnalysisResult analyze_tensor_comprehensive(FractalTensor* tensor);
SpikeAnalysisData prepare_spike_analysis_data(FractalTensor* tensor, const char* source);
GradientPatternAnalysis analyze_gradient_pattern(FractalTensor* tensor);
void free_analysis_data(SpikeAnalysisData* data);

// ==================== ФУНКЦИИ ОПТИМИЗАЦИИ ====================

// Оптимизация градиентных паттернов
void optimize_gradient_pattern(FractalTensor* tensor, float target_similarity);
void refill_tensor_with_optimization(FractalTensor* tensor, int fill_mode);
void blend_gradient_patterns(FractalTensor* tensor, int mode1, int mode2, float blend_factor);

#endif // FRACTAL_TENSOR_H