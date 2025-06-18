import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import hdbscan
import ordpy
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import json
import itertools

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

def calculate_correlation_and_pvalue(x, y):
    """Вычисляет корреляцию Пирсона и p-value для двух серий."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) < 2:
        return 0.0, 1.0
    return stats.spearmanr(x[mask], y[mask])

def create_correlation_matrices(df, columns):
    """Создает матрицы корреляций и p-values."""
    n = len(columns)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                corr, p_val = calculate_correlation_and_pvalue(df[columns[i]], df[columns[j]])
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p_val
    
    return pd.DataFrame(corr_matrix, index=columns, columns=columns), \
           pd.DataFrame(p_matrix, index=columns, columns=columns)

def plot_entropy_correlations(df, entropy_cols):
    """Построение тепловой карты корреляций между энтропиями и фрактальными размерностями"""
    plt.figure(figsize=(12, 10))
    
    # Объединяем энтропии и фрактальные размерности
    all_metrics = entropy_cols + FRACTAL_COLS
    
    # Вычисляем корреляции и p-values
    corr_matrix, p_matrix = create_correlation_matrices(df, all_metrics)
    
    # Сохраняем в CSV
    os.makedirs('plots/spearman/csv', exist_ok=True)
    corr_matrix.to_csv('plots/spearman/csv/entropy_correlations.csv', index=False)
    p_matrix.to_csv('plots/spearman/csv/entropy_pvalues.csv', index=False)
    
    # Строим тепловую карту
    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1)
    
    plt.title('Correlations between entropy measures, complexity and fractal dimensions')
    plt.tight_layout()
    plt.savefig('plots/spearman/entropy_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_numerical_correlations(df, entropy_cols):
    """Построение тепловой карты корреляций между числовыми параметрами и метриками"""
    # Выбираем числовые колонки (кроме метрик и id)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols 
                     if col not in ENTROPY_COLS + FRACTAL_COLS 
                     and 'id' not in col.lower()]
    
    # Объединяем все метрики
    all_metrics = entropy_cols + FRACTAL_COLS
    
    # Вычисляем корреляции и p-values
    all_cols = all_metrics + numerical_cols
    corr_matrix, p_matrix = create_correlation_matrices(df, all_cols)
    
    # Выбираем только нужные части матриц
    corr_with_entropy = corr_matrix.loc[all_metrics, numerical_cols]
    p_with_entropy = p_matrix.loc[all_metrics, numerical_cols]
    
    # Сохраняем в CSV
    os.makedirs('plots/spearman/csv', exist_ok=True)
    corr_with_entropy.to_csv('plots/spearman/csv/numerical_correlations.csv', index=False)
    p_with_entropy.to_csv('plots/spearman/csv/numerical_pvalues.csv', index=False)
    
    # Строим тепловую карту
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_with_entropy,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1)
    
    # Поворачиваем метки на оси X для лучшей читаемости
    plt.xticks(rotation=45, ha='right')
    
    plt.title('Correlations between entropy measures, complexity, fractal dimensions and numerical parameters')
    plt.tight_layout()
    plt.savefig('plots/spearman/numerical_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_genre_correlations(df, entropy_cols):
    """Построение тепловой карты корреляций между энтропиями и жанрами"""
    # Создаем dummy-переменные для жанров и преобразуем в float
    genres_dummies = pd.get_dummies(df['track_genre'], prefix='genre').astype(float)
    
    # Объединяем все метрики
    all_metrics = entropy_cols + FRACTAL_COLS
    
    # Объединяем данные
    combined_df = pd.concat([df[all_metrics], genres_dummies], axis=1)
    
    # Вычисляем корреляции и p-values
    all_cols = all_metrics + list(genres_dummies.columns)
    corr_matrix, p_matrix = create_correlation_matrices(combined_df, all_cols)
    
    # Выбираем только нужные части матриц
    corr_with_genres = corr_matrix.loc[all_metrics, genres_dummies.columns]
    p_with_genres = p_matrix.loc[all_metrics, genres_dummies.columns]
    
    # Сохраняем в CSV
    os.makedirs('plots/spearman/csv', exist_ok=True)
    corr_with_genres.to_csv('plots/spearman/csv/genre_correlations.csv', index=False)
    p_with_genres.to_csv('plots/spearman/csv/genre_pvalues.csv', index=False)
    
    # Строим тепловую карту
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_with_genres,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1)
    
    # Поворачиваем метки на оси X для лучшей читаемости
    plt.xticks(rotation=45, ha='right')
    
    plt.title('Correlations between entropy measures, complexity, fractal dimensions and genres')
    plt.tight_layout()
    plt.savefig('plots/spearman/genre_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_complete_correlation_matrix(df, entropy_cols):
    """Построение полной корреляционной матрицы"""
    # Получаем числовые колонки
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if 'id' not in col.lower()]
    
    # Создаем dummy-переменные для жанров и преобразуем в float
    genres_dummies = pd.get_dummies(df['track_genre'], prefix='genre').astype(float)
    
    # Объединяем все признаки
    features_df = pd.concat([
        df[numerical_cols],
        genres_dummies
    ], axis=1)
    
    # Вычисляем корреляции и p-values
    corr_matrix, p_matrix = create_correlation_matrices(features_df, features_df.columns.tolist())
    
    # Сохраняем в CSV
    os.makedirs('plots/spearman/csv', exist_ok=True)
    corr_matrix.to_csv('plots/spearman/csv/complete_correlation_matrix.csv', index=False)
    p_matrix.to_csv('plots/spearman/csv/complete_pvalues_matrix.csv', index=False)
    
    # Строим большую тепловую карту
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1)
    
    # Поворачиваем метки на оси X для лучшей читаемости
    plt.xticks(rotation=90)
    
    plt.title('Complete Correlation Matrix of All Features')
    plt.tight_layout()
    plt.savefig('plots/spearman/complete_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(df, target_cols):
    """Построение графиков важности признаков для предсказания энтропий и сложностей"""
    # Создаем директории для графиков
    os.makedirs('plots/spearman/features', exist_ok=True)
    
    # Группируем метрики по типам
    metric_pairs = [
        ('Amplitude', ['entropy_amplitude', 'complexity_amplitude']),
        ('Flux', ['entropy_flux', 'complexity_flux']),
        ('Harmony', ['entropy_harmony', 'complexity_harmony']),
        ('Spectral', ['entropy_spectral', 'complexity_spectral'])
    ]
    
    # Для каждой пары метрик создаем отдельный график
    for metric_type, metrics in metric_pairs:
        # Подготовка признаков
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols 
                         if col not in ENTROPY_COLS + FRACTAL_COLS and 'id' not in col.lower()]
        
        # Кодируем категориальные признаки
        le = LabelEncoder()
        df_encoded = df.copy()
        df_encoded['track_genre'] = le.fit_transform(df['track_genre'])
        
        # Формируем матрицу признаков
        feature_cols = numerical_cols + ['track_genre']
        
        # Создаем фигуру с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        for idx, metric in enumerate(metrics):
            # Выбираем текущую ось
            ax = ax1 if idx == 0 else ax2
            
            # Подготовка данных без NaN
            data = df_encoded[feature_cols + [metric]].dropna()
            X = data[feature_cols]
            y = data[metric]
            
            # Обучаем Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Получаем важности признаков
            importances = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            })
            importances = importances.sort_values('importance', ascending=True)
            
            # Строим график
            bars = ax.barh(
                y=np.arange(len(feature_cols)),
                width=importances['importance'],
                height=0.5,
                label=metric.replace('_', ' ').title()
            )
            
            # Добавляем значения на графике
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', 
                       ha='left', va='center', fontsize=8)
            
            # Настраиваем оси и подписи
            ax.set_yticks(np.arange(len(feature_cols)))
            ax.set_yticklabels(importances['feature'])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel('Feature Importance')
            if idx == 0:  # Только для левого графика
                ax.set_ylabel('Features')
        
        plt.suptitle(f'Feature Importance for {metric_type} Metrics')
        plt.tight_layout()
        plt.savefig(f'plots/spearman/features/feature_importance_{metric_type.lower()}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def plot_fractal_feature_importance(df):
    """Построение графиков важности признаков для предсказания фрактальных размерностей"""
    # Создаем директорию для графиков
    os.makedirs('plots/spearman/features', exist_ok=True)
    
    # Подготовка признаков
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols 
                     if col not in ENTROPY_COLS + FRACTAL_COLS and 'id' not in col.lower()]
    
    # Кодируем категориальные признаки
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded['track_genre'] = le.fit_transform(df['track_genre'])
    
    # Формируем матрицу признаков
    feature_cols = numerical_cols + ['track_genre']
    
    # Создаем фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for idx, metric in enumerate(FRACTAL_COLS):
        # Выбираем текущую ось
        ax = ax1 if idx == 0 else ax2
        
        # Подготовка данных без NaN
        data = df_encoded[feature_cols + [metric]].dropna()
        X = data[feature_cols]
        y = data[metric]
        
        # Обучаем Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Получаем важности признаков
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=True)
        
        # Строим график
        bars = ax.barh(
            y=np.arange(len(feature_cols)),
            width=importances['importance'],
            height=0.5,
            label=metric.replace('_', ' ').title()
        )
        
        # Добавляем значения на графике
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', 
                   ha='left', va='center', fontsize=8)
        
        # Настраиваем оси и подписи
        ax.set_yticks(np.arange(len(feature_cols)))
        ax.set_yticklabels(importances['feature'])
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Feature Importance')
        if idx == 0:  # Только для левого графика
            ax.set_ylabel('Features')
    
    plt.suptitle('Feature Importance for Fractal Dimensions')
    plt.tight_layout()
    plt.savefig('plots/spearman/features/feature_importance_fractal.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_top_feature_scatter(df, target_cols):
    """Построение scatter plots для важнейших признаков с использованием Lowess и hexbin визуализации"""
    # Создаем директорию для графиков
    os.makedirs('plots/spearman/features', exist_ok=True)
    
    # Подготовка числовых колонок
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols 
                     if col not in ENTROPY_COLS + FRACTAL_COLS and 'id' not in col.lower()]
    
    # Кодируем категориальные признаки
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded['track_genre'] = le.fit_transform(df['track_genre'])
    
    # Формируем матрицу признаков
    feature_cols = numerical_cols + ['track_genre']
    
    # Группируем метрики по типу
    metric_groups = {
        'Entropy': [col for col in target_cols if 'entropy' in col],
        'Complexity': [col for col in target_cols if 'complexity' in col],
        'Fractal': [col for col in target_cols if 'fractal' in col]
    }
    
    # Для каждой группы метрик
    for group_name, metrics in metric_groups.items():
        if not metrics:
            continue
            
        # Для каждой метрики в группе
        for metric in metrics:
            # Подготовка данных без NaN
            data = df_encoded[feature_cols + [metric]].dropna()
            X = data[feature_cols]
            y = data[metric]
            
            # Обучаем Random Forest для определения важности
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Находим самую важную характеристику
            importances = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            })
            top_feature = importances.nlargest(1, 'importance')['feature'].iloc[0]
            
            # Создаем DataFrame с рангами
            rank_df = pd.DataFrame({
                "X": df_encoded[top_feature],
                "Y": df_encoded[metric]
            })
            rank_df["X_rank"] = rank_df["X"].rank(method="average")
            rank_df["Y_rank"] = rank_df["Y"].rank(method="average")
            
            # Создаем фигуру с тремя подграфиками
            plt.figure(figsize=(18, 6))
            
            # 1. Scatter plot по рангам
            plt.subplot(1, 3, 1)
            plt.scatter(rank_df["X_rank"], rank_df["Y_rank"], alpha=0.5)
            plt.xlabel(f'Rank of {top_feature}')
            plt.ylabel(f'Rank of {metric}')
            plt.title('Rank-based Scatter')
            
            # 2. Lowess smoothing
            plt.subplot(1, 3, 2)
            sns.regplot(x='X_rank', y='Y_rank', data=rank_df, lowess=True,
                       scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
            plt.xlabel(f'Rank of {top_feature}')
            plt.ylabel(f'Rank of {metric}')
            plt.title('Lowess Smoothing')
            
            # 3. Hexbin density plot
            plt.subplot(1, 3, 3)
            plt.hexbin(rank_df['X_rank'], rank_df['Y_rank'], 
                      gridsize=30, cmap='Blues', mincnt=1)
            plt.colorbar(label='Counts')
            plt.xlabel(f'Rank of {top_feature}')
            plt.ylabel(f'Rank of {metric}')
            plt.title('Rank Density')
            
            # Добавляем информацию о корреляции
            corr, p_val = stats.spearmanr(rank_df["X_rank"], rank_df["Y_rank"])
            if p_val < 0.001:
                p_text = "p < 0.001"
            else:
                p_text = f"p = {p_val:.3f}"
            
            plt.suptitle(f'Top Feature Analysis for {metric}\n' +
                        f'Spearman\'s ρ = {corr:.3f}, {p_text}', y=1.02)
            
            plt.tight_layout()
            plt.savefig(f'plots/spearman/features/top_feature_{metric}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

def plot_fractal_scatter(df):
    """Построение scatter plots для фрактальных размерностей с использованием Lowess и hexbin визуализации"""
    # Создаем директорию для графиков
    os.makedirs('plots/spearman/features', exist_ok=True)
    
    # Подготовка числовых колонок
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols 
                     if col not in ENTROPY_COLS + FRACTAL_COLS and 'id' not in col.lower()]
    
    # Кодируем категориальные признаки
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded['track_genre'] = le.fit_transform(df['track_genre'])
    
    # Формируем матрицу признаков
    feature_cols = numerical_cols + ['track_genre']
    
    # Для каждой фрактальной размерности
    for target in FRACTAL_COLS:
        # Подготовка данных без NaN
        data = df_encoded[feature_cols + [target]].dropna()
        X = data[feature_cols]
        y = data[target]
        
        # Обучаем Random Forest для определения важности
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Находим самую важную характеристику
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        })
        top_feature = importances.nlargest(1, 'importance')['feature'].iloc[0]
        
        # Создаем DataFrame с рангами
        rank_df = pd.DataFrame({
            "X": df_encoded[top_feature],
            "Y": df_encoded[target]
        })
        rank_df["X_rank"] = rank_df["X"].rank(method="average")
        rank_df["Y_rank"] = rank_df["Y"].rank(method="average")
        
        # Создаем фигуру с тремя подграфиками
        plt.figure(figsize=(18, 6))
        
        # 1. Scatter plot по рангам
        plt.subplot(1, 3, 1)
        plt.scatter(rank_df["X_rank"], rank_df["Y_rank"], alpha=0.5)
        plt.xlabel(f'Rank of {top_feature}')
        plt.ylabel(f'Rank of {target}')
        plt.title('Rank-based Scatter')
        
        # 2. Lowess smoothing
        plt.subplot(1, 3, 2)
        sns.regplot(x='X_rank', y='Y_rank', data=rank_df, lowess=True,
                   scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        plt.xlabel(f'Rank of {top_feature}')
        plt.ylabel(f'Rank of {target}')
        plt.title('Lowess Smoothing')
        
        # 3. Hexbin density plot
        plt.subplot(1, 3, 3)
        plt.hexbin(rank_df['X_rank'], rank_df['Y_rank'], 
                  gridsize=30, cmap='Blues', mincnt=1)
        plt.colorbar(label='Counts')
        plt.xlabel(f'Rank of {top_feature}')
        plt.ylabel(f'Rank of {target}')
        plt.title('Rank Density')
        
        # Добавляем информацию о корреляции
        corr, p_val = stats.spearmanr(rank_df["X_rank"], rank_df["Y_rank"])
        if p_val < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {p_val:.3f}"
        
        plt.suptitle(f'Top Feature Analysis for {target}\n' +
                    f'Spearman\'s ρ = {corr:.3f}, {p_text}', y=1.02)
        
        plt.tight_layout()
        plt.savefig(f'plots/spearman/features/top_feature_{target}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_most_correlated_scatter(df, metric_name, output_dir):
    """Построение scatter plots для самых коррелированных признаков с использованием различных техник визуализации Spearman корреляции"""
    # Создаем директорию для графиков
    os.makedirs(output_dir, exist_ok=True)
    
    # Подготовка числовых колонок
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols 
                     if col not in ENTROPY_COLS + FRACTAL_COLS and 'id' not in col.lower()]
    
    # Кодируем категориальные признаки
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded['track_genre'] = le.fit_transform(df['track_genre'])
    
    # Формируем матрицу признаков
    feature_cols = numerical_cols + ['track_genre']
    
    # Вычисляем корреляции и p-values
    correlations = []
    for col in feature_cols:
        if col != metric_name:
            corr, p_val = stats.spearmanr(df_encoded[metric_name], df_encoded[col])
            correlations.append((col, corr, p_val))
    
    # Сортируем по абсолютному значению корреляции
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Берем топ-2 положительных и отрицательных корреляций
    top_correlations = []
    for col, corr, p_val in correlations:
        if len(top_correlations) < 4:
            if corr > 0 and len([x for x in top_correlations if x[1] > 0]) < 2:
                top_correlations.append((col, corr, p_val))
            elif corr < 0 and len([x for x in top_correlations if x[1] < 0]) < 2:
                top_correlations.append((col, corr, p_val))
    
    # Создаем графики для каждой топовой корреляции
    for idx, (col, corr, p_val) in enumerate(top_correlations):
        plt.figure(figsize=(20, 5))
        
        # 1. Scatter plot по рангам
        plt.subplot(1, 4, 1)
        rank_df = pd.DataFrame({
            "X": df_encoded[col],
            "Y": df_encoded[metric_name]
        })
        rank_df["X_rank"] = rank_df["X"].rank(method="average")
        rank_df["Y_rank"] = rank_df["Y"].rank(method="average")
        
        plt.scatter(rank_df["X_rank"], rank_df["Y_rank"], alpha=0.5)
        plt.xlabel(f'Rank of {col}')
        plt.ylabel(f'Rank of {metric_name}')
        plt.title(f'Rank-based Scatter\nSpearman\'s ρ = {corr:.3f}')
        
        # 2. Lowess smoothing
        plt.subplot(1, 4, 2)
        sns.regplot(x='X_rank', y='Y_rank', data=rank_df, lowess=True,
                   scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        plt.xlabel(f'Rank of {col}')
        plt.ylabel(f'Rank of {metric_name}')
        plt.title('Lowess Smoothing')
        
        # 3. Hexbin plot
        plt.subplot(1, 4, 3)
        plt.hexbin(rank_df['X_rank'], rank_df['Y_rank'], 
                  gridsize=30, cmap='Blues', mincnt=1)
        plt.colorbar(label='Counts')
        plt.xlabel(f'Rank of {col}')
        plt.ylabel(f'Rank of {metric_name}')
        plt.title('Rank Density')
        
        # 4. Binned ranks with error bars
        plt.subplot(1, 4, 4)
        bins = np.linspace(1, len(rank_df), 10)
        rank_df['bin'] = pd.cut(rank_df['X_rank'], bins)
        grouped = rank_df.groupby('bin')['Y_rank'].agg(['mean','sem']).reset_index()
        
        plt.errorbar(x=range(len(grouped)), y=grouped['mean'],
                    yerr=grouped['sem'], fmt='o-', capsize=3)
        plt.xticks(range(len(grouped)), 
                  [f"{int(iv.left)}–{int(iv.right)}" for iv in grouped['bin']], 
                  rotation=45)
        plt.xlabel('Rank bins')
        plt.ylabel('Mean rank ± SEM')
        plt.title('Binned Rank Relationship')
        
        # Общий заголовок
        plt.suptitle(f'Spearman Correlation Analysis: {metric_name} vs {col}\n' +
                    f'ρ = {corr:.3f}, p-value = {p_val:.3e}', y=1.02)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{metric_name}_vs_{col}_spearman.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_strong_genre_correlations(df, metrics, threshold=0.4):
    """Построение box plots для сильных корреляций между жанрами и метриками (|corr| > threshold)"""
    # Создаем dummy-переменные для жанров
    genres_dummies = pd.get_dummies(df['track_genre'], prefix='genre').astype(float)
    
    # Создаем директорию для сохранения графиков
    output_dir = 'plots/spearman/genre_correlations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Для каждой метрики ищем сильные корреляции с жанрами
    strong_correlations = []
    for metric in metrics:
        for genre in genres_dummies.columns:
            mask = ~(np.isnan(df[metric]) | np.isnan(genres_dummies[genre]))
            if mask.sum() < 2:
                continue
            
            corr, p_val = stats.spearmanr(df[metric][mask], genres_dummies[genre][mask])
            
            if abs(corr) >= threshold:
                strong_correlations.append({
                    'metric': metric,
                    'genre': genre,
                    'correlation': corr,
                    'p_value': p_val
                })
    
    # Если найдены сильные корреляции, строим для них графики
    if strong_correlations:
        # Сортируем по абсолютному значению корреляции
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Создаем boxplot для каждой сильной корреляции
        for corr_info in strong_correlations:
            plt.figure(figsize=(10, 6))
            
            # Подготовка данных для boxplot
            genre_name = corr_info['genre'].replace('genre_', '')
            metric_data = []
            labels = []
            
            # Данные для жанра с сильной корреляцией
            genre_mask = df['track_genre'] == genre_name
            metric_data.append(df[corr_info['metric']][genre_mask].dropna())
            labels.append(genre_name)
            
            # Данные для остальных жанров (объединенные)
            other_mask = ~genre_mask
            metric_data.append(df[corr_info['metric']][other_mask].dropna())
            labels.append('Other Genres')
            
            # Создаем boxplot
            bp = plt.boxplot(metric_data, labels=labels, patch_artist=True)
            
            # Настраиваем цвета
            colors = ['lightblue', 'lightgray']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            # Добавляем точки выбросов с прозрачностью
            for i, data in enumerate(metric_data):
                x = np.random.normal(i + 1, 0.04, size=len(data))
                plt.scatter(x, data, alpha=0.2, color='blue' if i == 0 else 'gray')
            
            # Форматируем названия
            metric_title = corr_info['metric'].replace('_', ' ').title()
            plt.title(f'{metric_title} Distribution by Genre')
            plt.ylabel(metric_title)
            
            # Добавляем информацию о корреляции и p-value
            if corr_info['p_value'] < 0.001:
                p_text = "p < 0.001"
            else:
                p_text = f"p = {corr_info['p_value']:.3f}"
            
            plt.text(0.05, 0.95, 
                    f'Correlation: {corr_info["correlation"]:.3f}\n{p_text}',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8))
            
            # Добавляем статистическую информацию под графиком
            genre_data = metric_data[0]
            other_data = metric_data[1]
            
            stats_text = (
                f'{genre_name}:\n'
                f'Mean: {np.mean(genre_data):.3f}\n'
                f'Median: {np.median(genre_data):.3f}\n'
                f'Std: {np.std(genre_data):.3f}\n\n'
                f'Other Genres:\n'
                f'Mean: {np.mean(other_data):.3f}\n'
                f'Median: {np.median(other_data):.3f}\n'
                f'Std: {np.std(other_data):.3f}'
            )
            
            plt.text(1.3, 0.5, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.8))
            
            # Настраиваем размер графика с учетом статистики справа
            plt.subplots_adjust(right=0.8)
            
            # Сохраняем график
            safe_genre_name = genre_name.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(output_dir, 
                       f'genre_correlation_{corr_info["metric"]}_{safe_genre_name}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Создаем сводную таблицу сильных корреляций
        correlations_df = pd.DataFrame(strong_correlations)
        correlations_df['genre'] = correlations_df['genre'].str.replace('genre_', '')
        correlations_df = correlations_df.sort_values('correlation', ascending=False)
        
        # Сохраняем таблицу
        correlations_df.to_csv(os.path.join(output_dir, 'strong_genre_correlations.csv'), 
                             index=False)

def get_complexity_bounds():
    """Получает теоретические ограничения сложности для энтропии используя ordpy"""
    # Получаем максимальные и минимальные значения из ordpy
    max_HC = ordpy.maximum_complexity_entropy(6, 1)
    min_HC = ordpy.minimum_complexity_entropy(6, 1)
    
    # Возвращаем координаты для построения границ
    return max_HC[0], min_HC[2], max_HC[2]

def plot_entropy_complexity_by_genre(df):
    """Построение scatter plots в пространстве энтропия-сложность для каждого типа метрик, с цветовой кодировкой жанров"""
    # Создаем директорию для сохранения графиков
    output_dir = 'plots/spearman/entropy_complexity_space'
    os.makedirs(output_dir, exist_ok=True)
    
    # Определяем пары энтропия-сложность
    metric_pairs = [
        ('entropy_amplitude', 'complexity_amplitude', 'Amplitude'),
        ('entropy_flux', 'complexity_flux', 'Flux'),
        ('entropy_harmony', 'complexity_harmony', 'Harmony'),
        ('entropy_spectral', 'complexity_spectral', 'Spectral')
    ]
    
    # Получаем уникальные жанры
    genres = df['track_genre'].unique()
    
    # Создаем цветовую палитру для жанров
    colors = plt.cm.tab20(np.linspace(0, 1, len(genres)))
    color_dict = dict(zip(genres, colors))
    
    # Получаем теоретические ограничения из ordpy
    max_HC = ordpy.maximum_complexity_entropy(6, 1)
    min_HC = ordpy.minimum_complexity_entropy(6, 1)
    
    # Создаем общую сетку x-координат
    x_grid = np.linspace(0, 1, 100)
    
    # Интерполируем кривые максимальной и минимальной сложности
    from scipy.interpolate import interp1d
    max_interp = interp1d(max_HC[:, 0], max_HC[:, 1], kind='linear', fill_value='extrapolate')
    min_interp = interp1d(min_HC[:, 0], min_HC[:, 1], kind='linear', fill_value='extrapolate')
    
    max_curve = max_interp(x_grid)
    min_curve = min_interp(x_grid)
    
    # Устанавливаем фиксированные пределы для всех графиков
    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 0.5
    
    # Для каждой пары метрик создаем два графика
    for entropy_col, complexity_col, metric_type in metric_pairs:
        # 1. Общий график для всех жанров
        plt.figure(figsize=(12, 8))
        
        # Рисуем теоретические ограничения
        plt.plot(x_grid, max_curve, 'k--', alpha=0.5, label='Max Complexity')
        plt.plot(x_grid, min_curve, 'k:', alpha=0.5, label='Min Complexity')
        plt.fill_between(x_grid, min_curve, max_curve, alpha=0.1, color='gray')
        
        for genre in genres:
            mask = df['track_genre'] == genre
            plt.scatter(df[entropy_col][mask], 
                       df[complexity_col][mask],
                       c=[color_dict[genre]],
                       label=genre,
                       alpha=0.6)
        
        plt.xlabel(f'{entropy_col.replace("_", " ").title()}')
        plt.ylabel(f'{complexity_col.replace("_", " ").title()}')
        plt.title(f'{metric_type} Entropy-Complexity Space by Genre')
        
        # Устанавливаем масштаб по осям
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect('equal', 'box')
        
        # Добавляем легенду справа от графика
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig(os.path.join(output_dir, f'entropy_complexity_{metric_type.lower()}_all_genres.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Сетка графиков по жанрам
        n_genres = len(genres)
        n_cols = 4  # Количество колонок в сетке
        n_rows = (n_genres + n_cols - 1) // n_cols  # Округление вверх
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        fig.suptitle(f'{metric_type} Entropy-Complexity Space by Genre', fontsize=16, y=1.02)
        
        # Уплощаем массив осей для удобства
        axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        # Строим график для каждого жанра
        for ax, genre in zip(axes_flat, genres):
            # Рисуем теоретические ограничения
            ax.plot(x_grid, max_curve, 'k--', alpha=0.5)
            ax.plot(x_grid, min_curve, 'k:', alpha=0.5)
            ax.fill_between(x_grid, min_curve, max_curve, alpha=0.1, color='gray')
            
            mask = df['track_genre'] == genre
            genre_data = df[mask]
            
            # Scatter plot только для текущего жанра
            ax.scatter(genre_data[entropy_col], 
                      genre_data[complexity_col],
                      c=[color_dict[genre]],
                      alpha=0.6)
            
            # Добавляем заголовок и подписи осей
            ax.set_title(genre)
            ax.set_xlabel(f'{entropy_col.replace("_", " ").title()}')
            ax.set_ylabel(f'{complexity_col.replace("_", " ").title()}')
            
            # Устанавливаем масштаб по осям
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal', 'box')
            
            # Добавляем статистическую информацию
            mean_entropy = genre_data[entropy_col].mean()
            mean_complexity = genre_data[complexity_col].mean()
            std_entropy = genre_data[entropy_col].std()
            std_complexity = genre_data[complexity_col].std()
            
            stats_text = (
                f'Mean E: {mean_entropy:.3f}\n'
                f'Std E: {std_entropy:.3f}\n'
                f'Mean C: {mean_complexity:.3f}\n'
                f'Std C: {std_complexity:.3f}'
            )
            
            ax.text(0.05, 0.95, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Скрываем пустые подграфики
        for ax in axes_flat[len(genres):]:
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'entropy_complexity_{metric_type.lower()}_by_genre.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def plot_fractal_dimensions_by_genre(df):
    """Построение боксплотов фрактальных размерностей по жанрам"""
    # Создаем директорию для сохранения графиков
    output_dir = 'plots/spearman/fractal_dimensions'
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем общий график для всех фрактальных размерностей
    plt.figure(figsize=(15, 8))
    
    # Создаем подграфики для каждой фрактальной размерности
    for i, fd_col in enumerate(FRACTAL_COLS, 1):
        plt.subplot(1, len(FRACTAL_COLS), i)
        
        # Создаем боксплот
        sns.boxplot(data=df, x='track_genre', y=fd_col, palette='tab20')
        
        # Поворачиваем метки на оси X для лучшей читаемости
        plt.xticks(rotation=45, ha='right')
        
        # Добавляем заголовок и метки осей
        plt.title(f'{fd_col.replace("_", " ").title()} by Genre')
        plt.xlabel('Genre')
        plt.ylabel('Fractal Dimension')
        
        # Добавляем статистическую информацию для каждого жанра
        for genre in df['track_genre'].unique():
            genre_data = df[df['track_genre'] == genre][fd_col]
            mean_val = genre_data.mean()
            std_val = genre_data.std()
            print(f"{fd_col} - {genre}:")
            print(f"Mean: {mean_val:.3f}")
            print(f"Std: {std_val:.3f}\n")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fractal_dimensions_boxplot.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Создаем отдельные графики для каждой фрактальной размерности
    for fd_col in FRACTAL_COLS:
        plt.figure(figsize=(12, 6))
        
        # Создаем боксплот
        sns.boxplot(data=df, x='track_genre', y=fd_col, palette='tab20')
        
        # Поворачиваем метки на оси X для лучшей читаемости
        plt.xticks(rotation=45, ha='right')
        
        # Добавляем заголовок и метки осей
        plt.title(f'{fd_col.replace("_", " ").title()} Distribution by Genre')
        plt.xlabel('Genre')
        plt.ylabel('Fractal Dimension')
        
        # Добавляем статистическую информацию
        stats_text = ""
        for genre in df['track_genre'].unique():
            genre_data = df[df['track_genre'] == genre][fd_col]
            stats_text += f"{genre}:\n"
            stats_text += f"Mean: {genre_data.mean():.3f}\n"
            stats_text += f"Std: {genre_data.std():.3f}\n\n"
        
        # Добавляем текст со статистикой справа от графика
        plt.figtext(1.02, 0.5, stats_text,
                   fontsize=8, va='center',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{fd_col}_boxplot.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

def analyze_correlations():
    # Создаем директории для графиков
    os.makedirs('plots/spearman', exist_ok=True)
    os.makedirs('plots/spearman/features', exist_ok=True)
    os.makedirs('plots/spearman/most_correlated', exist_ok=True)

    # Загружаем данные
    print("Загрузка данных...")
    df = pd.read_csv(DATASET_PATH)
    
    print("Построение графиков пространства энтропия-сложность...")
    plot_entropy_complexity_by_genre(df)


    print("Построение боксплотов фрактальных размерностей...")
    plot_fractal_dimensions_by_genre(df)
    
    # Строим корреляционные карты
    print("Построение корреляций между энтропиями...")
    plot_entropy_correlations(df, ENTROPY_COLS)
    
    print("Построение корреляций с числовыми параметрами...")
    plot_numerical_correlations(df, ENTROPY_COLS)
    
    print("Построение корреляций с жанрами...")
    plot_genre_correlations(df, ENTROPY_COLS)
    
    print("Построение сильных корреляций с жанрами...")
    all_metrics = ENTROPY_COLS + FRACTAL_COLS
    plot_strong_genre_correlations(df, all_metrics, threshold=0.4)
    
    print("Построение полной корреляционной матрицы...")
    plot_complete_correlation_matrix(df, ENTROPY_COLS)
    
    print("Построение графиков важности признаков для энтропий...")
    plot_feature_importance(df, ENTROPY_COLS)
    
    print("Построение scatter plots для важнейших признаков энтропий...")
    plot_top_feature_scatter(df, ENTROPY_COLS)
    
    print("Построение графиков важности признаков для фрактальных размерностей...")
    plot_fractal_feature_importance(df)
    
    print("Построение scatter plots для важнейших признаков фрактальных размерностей...")
    plot_fractal_scatter(df)
    
    print("Построение scatter plots для самых коррелированных признаков...")
    # Для энтропий
    for metric in ['entropy_amplitude', 'entropy_flux', 'entropy_harmony', 'entropy_spectral']:
        plot_most_correlated_scatter(df, metric, 'plots/spearman/most_correlated')
    
    # Для сложностей
    for metric in ['complexity_amplitude', 'complexity_flux', 'complexity_harmony', 'complexity_spectral']:
        plot_most_correlated_scatter(df, metric, 'plots/spearman/most_correlated')
    
    # Для фрактальных размерностей
    for metric in FRACTAL_COLS:
        plot_most_correlated_scatter(df, metric, 'plots/spearman/most_correlated')
    
    print("Готово! Графики сохранены в директории 'plots/spearman'")

if __name__ == "__main__":
    analyze_correlations() 