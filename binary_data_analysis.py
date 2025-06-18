import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Путь к датасету
DATASET_PATH = "data/csv_spotify/csv/sampled_dataset_PE_C_fd.csv"

# Список колонок с энтропиями и сложностью
ENTROPY_COLS = [
    'entropy_amplitude', 'complexity_amplitude',
    'entropy_flux', 'complexity_flux',
    'entropy_harmony', 'complexity_harmony',
    'entropy_spectral', 'complexity_spectral'
]

# Список колонок с фрактальными размерностями
FRACTAL_COLS = ['higuchi_fd', 'box_counting_fd']

# Список колонок с mode
MODE_COLS = ['mode']

def cohens_d_welch(group1, group2):
    """
    Вычисляет эффект-размер по Cohen's d с учетом неравных дисперсий (Welch).
    
    Parameters:
    -----------
    group1, group2 : array-like
        Две группы данных для сравнения
        
    Returns:
    --------
    float
        Значение Cohen's d по Welch
    """
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    return (np.mean(group1) - np.mean(group2)) / np.sqrt((s1 + s2) / 2)

def cliffs_delta(group1, group2):
    """
    Вычисляет эффект-размер по Cliff's delta (непараметрический аналог Cohen's d).
    
    Parameters:
    -----------
    group1, group2 : array-like
        Две группы данных для сравнения
        
    Returns:
    --------
    float
        Значение Cliff's delta
    """
    # Convert to numpy arrays to avoid pandas indexing issues
    group1 = np.array(group1)
    group2 = np.array(group2)
    n1, n2 = len(group1), len(group2)
    # Создаем матрицу сравнений
    comparisons = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            if group1[i] > group2[j]:
                comparisons[i, j] = 1
            elif group1[i] < group2[j]:
                comparisons[i, j] = -1
    return np.mean(comparisons)

def check_normality(data, alpha=0.05):
    """
    Проверяет нормальность распределения с помощью теста Шапиро-Уилка.
    
    Parameters:
    -----------
    data : array-like
        Данные для проверки
    alpha : float
        Уровень значимости
        
    Returns:
    --------
    bool
        True если распределение нормальное, False иначе
    """
    if len(data) < 3:  # Минимальный размер выборки для теста
        return True
    _, p_value = stats.shapiro(data)
    return p_value > alpha

def analyze_effect_sizes(df, metrics, genre_column='track_genre', mode_column='mode', plot_histograms=True):
    """
    Анализ различий между жанрами и mode с использованием t-test и эффект-размеров.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame с данными
    metrics : list
        Список метрик для анализа (энтропии, сложности, размерности)
    genre_column : str
        Название колонки с жанрами
    mode_column : str
        Название колонки с mode
    plot_histograms : bool
        Строить ли гистограммы распределений
    """
    # Создаем базовые директории для сохранения результатов
    welch_dir = 'plots/binary/effect_sizes/welch'
    mannwhitney_dir = 'plots/binary/effect_sizes/mannwhitney'
    os.makedirs(welch_dir, exist_ok=True)
    os.makedirs(mannwhitney_dir, exist_ok=True)
    
    # Создаем директории для каждой метрики внутри папок методов
    for metric in metrics:
        os.makedirs(os.path.join(welch_dir, metric), exist_ok=True)
        os.makedirs(os.path.join(mannwhitney_dir, metric), exist_ok=True)
    
    # Получаем уникальные жанры и значения mode
    genres = df[genre_column].unique()
    modes = df[mode_column].unique()
    
    # Создаем DataFrame для хранения результатов
    results = []
    
    # Для каждой метрики
    for metric in metrics:
        # Подготовка данных для метрики
        metric_data = df[metric].dropna()
        
        # Анализ по жанрам
        for genre in genres:
            # Создаем бинарную целевую переменную для жанра
            genre_mask = pd.Series(df[genre_column] == genre, index=df.index)
            genre_mask_clean = genre_mask[metric_data.index]
            
            # Разделяем данные на две группы
            group1 = metric_data[genre_mask_clean]  # Треки данного жанра
            group2 = metric_data[~genre_mask_clean]  # Треки других жанров
            
            # Проверяем нормальность распределений
            is_normal1 = check_normality(group1)
            is_normal2 = check_normality(group2)
            
            # Выбираем тест в зависимости от нормальности
            # if is_normal1 and is_normal2:
            #     # t-test для нормальных распределений
            #     t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            #     effect_size = cohens_d_welch(group1, group2)
            #     test_type = 'Welch t-test'
            #     output_dir = os.path.join(welch_dir, metric)
            # else:
            # Mann-Whitney U для ненормальных распределений
            t_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            effect_size = cliffs_delta(group1, group2)
            test_type = 'Mann-Whitney U'
            output_dir = os.path.join(mannwhitney_dir, metric)
            
            # Сохраняем результаты
            results.append({
                'metric': metric,
                'comparison_type': 'genre',
                'comparison_value': genre,
                'test_type': test_type,
                'statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'mean_group1': np.mean(group1),
                'mean_group2': np.mean(group2),
                'std_group1': np.std(group1),
                'std_group2': np.std(group2),
                'n_group1': len(group1),
                'n_group2': len(group2),
                'is_normal_group1': is_normal1,
                'is_normal_group2': is_normal2
            })
            
            # Строим график распределения только если включено
            if plot_histograms:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=pd.DataFrame({
                    'value': pd.concat([group1, group2]),
                    'group': ['Genre'] * len(group1) + ['Other'] * len(group2)
                }), x='value', hue='group', bins=30, stat='density', kde=True, 
                           common_norm=False, alpha=0.6)
                
                plt.axvline(np.mean(group1), color='blue', linestyle='dashed',
                           label=f'{genre} mean')
                plt.axvline(np.mean(group2), color='orange', linestyle='dashed',
                           label='Other genres mean')
                
                plt.xlabel(metric)
                plt.ylabel('Probability Density')
                plt.title(f'Distribution of {metric} for {genre} vs Other Genres\n'
                         f'{test_type}: p={p_value:.2e}, effect={effect_size:.2f}')
                plt.legend()
                plt.savefig(os.path.join(output_dir, f'distribution_genre_{genre}.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # Анализ по mode
        for mode in modes:
            # Создаем бинарную целевую переменную для mode
            mode_mask = pd.Series(df[mode_column] == mode, index=df.index)
            mode_mask_clean = mode_mask[metric_data.index]
            
            # Разделяем данные на две группы
            group1 = metric_data[mode_mask_clean]  # Треки с данным mode
            group2 = metric_data[~mode_mask_clean]  # Треки с другими mode
            
            # Проверяем нормальность распределений
            is_normal1 = check_normality(group1)
            is_normal2 = check_normality(group2)
            
            # Выбираем тест в зависимости от нормальности
            # if is_normal1 and is_normal2:

            #     # t-test для нормальных распределений
        #   t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        #   effect_size = cohens_d_welch(group1, group2)
        #   test_type = 'Welch t-test'
        #   output_dir = os.path.join(welch_dir, metric)
            # else:
            
            # Mann-Whitney U для ненормальных распределений
            t_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            effect_size = cliffs_delta(group1, group2)
            test_type = 'Mann-Whitney U'
            output_dir = os.path.join(mannwhitney_dir, metric)
            
            # Сохраняем результаты
            results.append({
                'metric': metric,
                'comparison_type': 'mode',
                'comparison_value': mode,
                'test_type': test_type,
                'statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'mean_group1': np.mean(group1),
                'mean_group2': np.mean(group2),
                'std_group1': np.std(group1),
                'std_group2': np.std(group2),
                'n_group1': len(group1),
                'n_group2': len(group2),
                'is_normal_group1': is_normal1,
                'is_normal_group2': is_normal2
            })
            
            # Строим график распределения только если включено
            if plot_histograms:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=pd.DataFrame({
                    'value': pd.concat([group1, group2]),
                    'group': ['Mode'] * len(group1) + ['Other'] * len(group2)
                }), x='value', hue='group', bins=30, stat='density', kde=True, 
                           common_norm=False, alpha=0.6)
                
                plt.axvline(np.mean(group1), color='blue', linestyle='dashed',
                           label=f'Mode {mode} mean')
                plt.axvline(np.mean(group2), color='orange', linestyle='dashed',
                           label='Other modes mean')
                
                plt.xlabel(metric)
                plt.ylabel('Probability Density')
                plt.title(f'Distribution of {metric} for Mode {mode} vs Other Modes\n'
                         f'{test_type}: p={p_value:.2e}, effect={effect_size:.2f}')
                plt.legend()
                plt.savefig(os.path.join(output_dir, f'distribution_mode_{mode}.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
        
    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    
    # Добавляем FDR-коррекцию для p-values
    results_df['p_adj'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
    
    # Сортируем по абсолютному значению эффект-размера
    results_df['abs_effect_size'] = results_df['effect_size'].abs()
    results_df = results_df.sort_values('abs_effect_size', ascending=False)
    results_df = results_df.drop('abs_effect_size', axis=1)
    
    # Сохраняем результаты в CSV для каждого типа теста и метрики
    for test_type, base_dir in [('Welch t-test', welch_dir), ('Mann-Whitney U', mannwhitney_dir)]:
        test_results = results_df[results_df['test_type'] == test_type]
        
        # Сохраняем общие результаты для теста
        test_results.to_csv(os.path.join(base_dir, 'effect_sizes_results.csv'), index=False)
        
        # Создаем общие тепловые карты для всех метрик против жанров и mode
        overall_heatmaps_dir = os.path.join(base_dir, 'overall_heatmaps')
        os.makedirs(overall_heatmaps_dir, exist_ok=True)
        
        # Тепловые карты для всех метрик против жанров
        overall_genre_results = test_results[test_results['comparison_type'] == 'genre']
        if not overall_genre_results.empty:
            # P-value heatmap
            plt.figure(figsize=(15, 10))
            pivot_table_p = overall_genre_results.pivot(index='comparison_value', columns='metric', values='p_adj').reindex(columns=metrics)
            if not pivot_table_p.empty:
                pivot_table_p = -np.log10(pivot_table_p)
                sns.heatmap(pivot_table_p.T, annot=True, cmap='YlOrRd', vmin=0)
                plt.title(f'-log10(adjusted p-value) Heatmap - All Metrics vs Genres ({test_type})')
                plt.tight_layout()
                plt.savefig(os.path.join(overall_heatmaps_dir, 'pvalue_heatmap_all_genres.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Effect size heatmap
            plt.figure(figsize=(15, 10))
            pivot_table_e = overall_genre_results.pivot(index='comparison_value', columns='metric', values='effect_size').reindex(columns=metrics)
            if not pivot_table_e.empty:
                sns.heatmap(pivot_table_e.T, annot=True, cmap='coolwarm', center=0)
                plt.title(f"Effect Size Heatmap - All Metrics vs Genres ({test_type})")
                plt.tight_layout()
                plt.savefig(os.path.join(overall_heatmaps_dir, 'effect_size_heatmap_all_genres.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        # Тепловые карты для всех метрик против mode
        overall_mode_results = test_results[test_results['comparison_type'] == 'mode']
        if not overall_mode_results.empty:
            # P-value heatmap
            plt.figure(figsize=(15, 10))
            pivot_table_p = overall_mode_results.pivot(index='comparison_value', columns='metric', values='p_adj').reindex(columns=metrics)
            if not pivot_table_p.empty:
                pivot_table_p = -np.log10(pivot_table_p)
                sns.heatmap(pivot_table_p.T, annot=True, cmap='YlOrRd', vmin=0)
                plt.title(f'-log10(adjusted p-value) Heatmap - All Metrics vs Modes ({test_type})')
                plt.tight_layout()
                plt.savefig(os.path.join(overall_heatmaps_dir, 'pvalue_heatmap_all_modes.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Effect size heatmap
            plt.figure(figsize=(15, 10))
            pivot_table_e = overall_mode_results.pivot(index='comparison_value', columns='metric', values='effect_size').reindex(columns=metrics)
            if not pivot_table_e.empty:
                sns.heatmap(pivot_table_e.T, annot=True, cmap='coolwarm', center=0)
                plt.title(f"Effect Size Heatmap - All Metrics vs Modes ({test_type})")
                plt.tight_layout()
                plt.savefig(os.path.join(overall_heatmaps_dir, 'effect_size_heatmap_all_modes.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Сохраняем результаты для каждой метрики (без тепловых карт)
        for metric in metrics:
            metric_results = test_results[test_results['metric'] == metric]
            metric_dir = os.path.join(base_dir, metric)
            metric_results.to_csv(os.path.join(metric_dir, 'results.csv'), index=False)

def analyze_correlations():
    # Создаем директории для графиков
    os.makedirs('plots/binary/effect_sizes', exist_ok=True)

    # Загружаем данные
    print("Загрузка данных...")
    df = pd.read_csv(DATASET_PATH)
    
    print("Выполнение анализа эффект-размеров для метрик...")
    all_metrics = ENTROPY_COLS + FRACTAL_COLS
    analyze_effect_sizes(df, all_metrics, plot_histograms=True)
    
    print("Готово! Графики сохранены в директории 'plots/binary/effect_sizes'")

if __name__ == "__main__":
    analyze_correlations() 