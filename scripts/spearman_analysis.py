import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

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
    """
    Calculate Spearman correlation coefficient and p-value between two variables
    """
    # Remove NaN values from both series simultaneously
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return np.nan, np.nan # Not enough data for correlation

    correlation, p_value = stats.spearmanr(x_clean, y_clean)
    return correlation, p_value

def create_correlation_matrices(df, columns):
    """
    Create correlation matrices using Spearman correlation
    """
    n = len(columns)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            col_i = columns[i]
            col_j = columns[j]
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                corr, p_val = calculate_correlation_and_pvalue(df[col_i], df[col_j])
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p_val
    
    return pd.DataFrame(corr_matrix, index=columns, columns=columns), \
           pd.DataFrame(p_matrix, index=columns, columns=columns)

def plot_entropy_correlations(df, entropy_cols):
    """
    Построение тепловой карты корреляций между энтропиями и фрактальными размерностями
    """
    plt.figure(figsize=(12, 10))
    
    # Объединяем энтропии и фрактальные размерности
    all_metrics = entropy_cols + FRACTAL_COLS
    
    # Вычисляем корреляции и p-values
    corr_matrix, p_matrix = create_correlation_matrices(df, all_metrics)
    
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
    """
    Построение тепловой карты корреляций между числовыми параметрами и метриками
    """
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
    """
    Построение тепловой карты корреляций между энтропиями и жанрами
    """
    # Создаем dummy-переменные для жанров и преобразуем в float
    genres_dummies = pd.get_dummies(df['track_genre'], prefix='genre').astype(float)
    
    genre_cols_from_dummies = list(genres_dummies.columns)
    if not genre_cols_from_dummies:
        print("No genre columns found after dummy encoding. Skipping genre correlation analysis.")
        return
        
    # Объединяем все метрики
    all_metrics = entropy_cols + FRACTAL_COLS
    
    # Объединяем данные
    combined_df = pd.concat([df[all_metrics], genres_dummies], axis=1)
    
    # Вычисляем корреляции и p-values
    all_cols = all_metrics + list(genres_dummies.columns)
    corr_matrix, p_matrix = create_correlation_matrices(combined_df, all_cols)
    
    # Выбираем только нужные части матриц
    corr_with_genres = corr_matrix.loc[all_metrics, genres_dummies.columns]
    
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
    """
    Построение полной корреляционной матрицы
    """
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
    """
    Построение графиков важности признаков для предсказания энтропий и сложностей
    """
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
        if 'track_genre' in df.columns:
            df_encoded['track_genre'] = le.fit_transform(df['track_genre'])
        
        # Формируем матрицу признаков
        feature_cols = numerical_cols + ['track_genre'] if 'track_genre' in df.columns else numerical_cols
        
        # Создаем фигуру с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        for idx, metric in enumerate(metrics):
            # Выбираем текущую ось
            ax = ax1 if idx == 0 else ax2
            
            # Подготовка данных без NaN
            data = df_encoded[feature_cols + [metric]].dropna()
            X = data[feature_cols]
            y = data[metric]
            
            if X.empty or y.empty:
                print(f"Not enough data for {metric}. Skipping feature importance plot.")
                continue

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
    """
    Построение графиков важности признаков для предсказания фрактальных размерностей
    """
    # Создаем директорию для графиков
    os.makedirs('plots/spearman/features', exist_ok=True)
    
    # Подготовка признаков
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols 
                     if col not in ENTROPY_COLS + FRACTAL_COLS and 'id' not in col.lower()]
    
    # Кодируем категориальные признаки
    le = LabelEncoder()
    df_encoded = df.copy()
    if 'track_genre' in df.columns:
        df_encoded['track_genre'] = le.fit_transform(df['track_genre'])
    
    # Формируем матрицу признаков
    feature_cols = numerical_cols + ['track_genre'] if 'track_genre' in df.columns else numerical_cols
    
    # Создаем фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for idx, metric in enumerate(FRACTAL_COLS):
        # Выбираем текущую ось
        ax = ax1 if idx == 0 else ax2
        
        # Подготовка данных без NaN
        data = df_encoded[feature_cols + [metric]].dropna()
        X = data[feature_cols]
        y = data[metric]
        
        if X.empty or y.empty:
            print(f"Not enough data for {metric}. Skipping fractal feature importance plot.")
            continue

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
    """
    Построение scatter plots между каждой метрикой и её самой важной характеристикой
    """
    # Создаем директорию для графиков
    os.makedirs('plots/spearman/features', exist_ok=True)
    
    # Подготовка признаков
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols 
                     if col not in ENTROPY_COLS + FRACTAL_COLS and 'id' not in col.lower()]
    
    # Кодируем категориальные признаки
    le = LabelEncoder()
    df_encoded = df.copy()
    if 'track_genre' in df.columns:
        df_encoded['track_genre'] = le.fit_transform(df['track_genre'])
    
    # Формируем матрицу признаков
    feature_cols = numerical_cols + ['track_genre'] if 'track_genre' in df.columns else numerical_cols
    
    # Группируем метрики по типам
    metric_pairs = [
        ('Amplitude', ['entropy_amplitude', 'complexity_amplitude']),
        ('Flux', ['entropy_flux', 'complexity_flux']),
        ('Harmony', ['entropy_harmony', 'complexity_harmony']),
        ('Spectral', ['entropy_spectral', 'complexity_spectral'])
    ]
    
    # Для каждой пары метрик создаем отдельный график
    for metric_type, metrics in metric_pairs:
        plt.figure(figsize=(15, 6))
        
        for idx, target in enumerate(metrics):
            # Подготовка данных без NaN
            data = df_encoded[feature_cols + [target]].dropna()
            X = data[feature_cols]
            y = data[target]
            
            if X.empty or y.empty:
                print(f"Not enough data for {target}. Skipping top feature scatter plot.")
                continue

            # Обучаем Random Forest для определения важности
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Находим самую важную характеристику
            importances = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            })
            top_feature = importances.nlargest(1, 'importance')['feature'].iloc[0]
            
            # Строим scatter plot
            plt.subplot(1, 2, idx + 1)
            
            # Если это жанр, используем оригинальные метки
            if top_feature == 'track_genre':
                x_data = df['track_genre']
                plt.xticks(rotation=45)
            else:
                x_data = df[top_feature]
            
            plt.scatter(x_data, df[target], alpha=0.5)
            
            # Добавляем линию тренда
            if top_feature != 'track_genre':
                # Ensure we only use non-NaN values for polyfit
                valid_mask_for_polyfit = ~(np.isnan(x_data) | np.isnan(df[target]))
                if valid_mask_for_polyfit.sum() > 1:
                    z = np.polyfit(x_data[valid_mask_for_polyfit], 
                                 df[target][valid_mask_for_polyfit], 1)
                    p = np.poly1d(z)
                    plt.plot(x_data, p(x_data), "r--", alpha=0.8)
            
            # Форматируем название метрики для заголовка
            metric_name = target.replace('_', ' ').title()
            feature_name = top_feature.replace('_', ' ').title()
            
            plt.title(f'{metric_name} vs {feature_name}')
            plt.xlabel(feature_name)
            plt.ylabel(metric_name)
            
            # Добавляем коэффициент корреляции и p-value
            if top_feature != 'track_genre':
                valid_mask = ~(np.isnan(x_data) | np.isnan(df[target]))
                x_clean = x_data[valid_mask]
                y_clean = df[target][valid_mask]
                
                corr, p_val = stats.spearmanr(x_clean, y_clean)
                
                if p_val < 0.001:
                    p_text = "p < 0.001"
                else:
                    p_text = f"p = {p_val:.3f}"
                
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}\n{p_text}', 
                        transform=plt.gca().transAxes, 
                        verticalalignment='top')
        
        plt.suptitle(f'Top Feature Relationships for {metric_type} Metrics')
        plt.tight_layout()
        plt.savefig(f'plots/spearman/features/top_feature_scatter_{metric_type.lower()}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def plot_fractal_scatter(df):
    """
    Построение scatter plots между фрактальными размерностями и их важнейшими характеристиками
    """
    # Создаем директорию для графиков
    os.makedirs('plots/spearman/features', exist_ok=True)
    
    # Подготовка признаков
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols 
                     if col not in ENTROPY_COLS + FRACTAL_COLS and 'id' not in col.lower()]
    
    # Кодируем категориальные признаки
    le = LabelEncoder()
    df_encoded = df.copy()
    if 'track_genre' in df.columns:
        df_encoded['track_genre'] = le.fit_transform(df['track_genre'])
    
    # Формируем матрицу признаков
    feature_cols = numerical_cols + ['track_genre'] if 'track_genre' in df.columns else numerical_cols
    
    plt.figure(figsize=(15, 6))
    
    for idx, target in enumerate(FRACTAL_COLS):
        # Подготовка данных без NaN
        data = df_encoded[feature_cols + [target]].dropna()
        X = data[feature_cols]
        y = data[target]
        
        if X.empty or y.empty:
            print(f"Not enough data for {target}. Skipping fractal scatter plot.")
            continue

        # Обучаем Random Forest для определения важности
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Находим самую важную характеристику
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        })
        top_feature = importances.nlargest(1, 'importance')['feature'].iloc[0]
        
        # Строим scatter plot
        plt.subplot(1, 2, idx + 1)
        
        # Если это жанр, используем оригинальные метки
        if top_feature == 'track_genre':
            x_data = df['track_genre']
            plt.xticks(rotation=45)
        else:
            x_data = df[top_feature]
        
        plt.scatter(x_data, df[target], alpha=0.5)
        
        # Добавляем линию тренда
        if top_feature != 'track_genre':
            valid_mask_for_polyfit = ~(np.isnan(x_data) | np.isnan(df[target]))
            if valid_mask_for_polyfit.sum() > 1:
                z = np.polyfit(x_data[valid_mask_for_polyfit], 
                             df[target][valid_mask_for_polyfit], 1)
                p = np.poly1d(z)
                plt.plot(x_data, p(x_data), "r--", alpha=0.8)
        
        # Форматируем название метрики для заголовка
        metric_name = target.replace('_', ' ').title()
        feature_name = top_feature.replace('_', ' ').title()
        
        plt.title(f'{metric_name} vs {feature_name}')
        plt.xlabel(feature_name)
        plt.ylabel(metric_name)
        
        # Добавляем коэффициент корреляции и p-value
        if top_feature != 'track_genre':
            valid_mask = ~(np.isnan(x_data) | np.isnan(df[target]))
            x_clean = x_data[valid_mask]
            y_clean = df[target][valid_mask]
            
            corr, p_val = stats.spearmanr(x_clean, y_clean)
            
            if p_val < 0.001:
                p_text = "p < 0.001"
            else:
                p_text = f"p = {p_val:.3f}"
            
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}\n{p_text}', 
                    transform=plt.gca().transAxes, 
                    verticalalignment='top')
    
    plt.suptitle('Top Feature Relationships for Fractal Dimensions')
    plt.tight_layout()
    plt.savefig('plots/spearman/features/top_feature_scatter_fractal.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_most_correlated_scatter(df, metric_name, output_dir):
    """
    Построение scatter plots для метрики с самыми сильно коррелирующими признаками
    """
    # Подготовка признаков
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols 
                     if col not in ENTROPY_COLS + FRACTAL_COLS and 'id' not in col.lower() and col != metric_name]
    
    # Вычисляем корреляции и p-values
    correlations = {}
    p_values = {}
    for col in numerical_cols:
        mask = ~(np.isnan(df[col]) | np.isnan(df[metric_name]))
        if mask.sum() < 2:
            continue
        corr, p_val = stats.spearmanr(df[col][mask], df[metric_name][mask])
        correlations[col] = corr
        p_values[col] = p_val
    
    # Находим самую положительную и самую отрицательную корреляцию
    if not correlations:
        print(f"No numerical features found to correlate with {metric_name}. Skipping most correlated scatter plot.")
        return

    sorted_correlations = sorted(correlations.items(), key=lambda x: x[1])
    most_negative = (sorted_correlations[0][0], sorted_correlations[0][1], p_values[sorted_correlations[0][0]])
    most_positive = (sorted_correlations[-1][0], sorted_correlations[-1][1], p_values[sorted_correlations[-1][0]])
    
    # Создаем график с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Строим scatter plot для отрицательной корреляции
    x_data = df[most_negative[0]]
    y_data = df[metric_name]
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    ax1.scatter(x_data[mask], y_data[mask], alpha=0.5)
    
    # Добавляем линию тренда
    if mask.sum() > 1:
        z = np.polyfit(x_data[mask], y_data[mask], 1)
        p = np.poly1d(z)
        ax1.plot(x_data[mask], p(x_data[mask]), "r--", alpha=0.8)
    
    # Форматируем название для заголовка
    metric_title = metric_name.replace('_', ' ').title()
    feature_title = most_negative[0].replace('_', ' ').title()
    ax1.set_title(f'{metric_title} vs {feature_title}')
    ax1.set_xlabel(feature_title)
    ax1.set_ylabel(metric_title)
    
    # Добавляем информацию о корреляции и p-value
    if most_negative[2] < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {most_negative[2]:.3f}"
    ax1.text(0.05, 0.95, f'Correlation: {most_negative[1]:.3f}\n{p_text}', 
             transform=ax1.transAxes, 
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Строим scatter plot для положительной корреляции
    x_data = df[most_positive[0]]
    y_data = df[metric_name]
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    ax2.scatter(x_data[mask], y_data[mask], alpha=0.5)
    
    # Добавляем линию тренда
    if mask.sum() > 1:
        z = np.polyfit(x_data[mask], y_data[mask], 1)
        p = np.poly1d(z)
        ax2.plot(x_data[mask], p(x_data[mask]), "r--", alpha=0.8)
    
    # Форматируем название для заголовка
    feature_title = most_positive[0].replace('_', ' ').title()
    ax2.set_title(f'{metric_title} vs {feature_title}')
    ax2.set_xlabel(feature_title)
    ax2.set_ylabel(metric_title)
    
    # Добавляем информацию о корреляции и p-value
    if most_positive[2] < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {most_positive[2]:.3f}"
    ax2.text(0.05, 0.95, f'Correlation: {most_positive[1]:.3f}\n{p_text}', 
             transform=ax2.transAxes, 
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Most Correlated Features for {metric_title}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'most_correlated_{metric_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_strong_genre_correlations(df, metrics, threshold=0.4):
    """
    Plot strong genre correlations using Spearman correlation
    """
    genre_cols = [col for col in df.columns if col.startswith('genre_')]
    
    if not genre_cols:
        print("No genre columns found in the dataset. Skipping strong genre correlation analysis.")
        return
    
    for metric in metrics:
        correlations = []
        for genre in genre_cols:
            correlation, _ = calculate_correlation_and_pvalue(df[genre], df[metric])
            correlations.append((genre, correlation))
        
        strong_correlations = [(g, c) for g, c in correlations if abs(c) >= threshold]
        if strong_correlations:
            strong_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            plt.figure(figsize=(12, 6))
            genres = [g.replace('genre_', '') for g, _ in strong_correlations]
            corrs = [c for _, c in strong_correlations]
            
            sns.barplot(x=genres, y=corrs)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Strong Genre Correlations with {metric} (Spearman)')
            plt.tight_layout()
            plt.savefig(f'plots/spearman/strong_genre_correlations_{metric}.png')
            plt.close()

def analyze_correlations():
    """
    Main function to run all correlation analyses using Spearman correlation
    """
    # Create directories for plots
    plots_dir = 'plots/spearman'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.join(plots_dir, 'features'), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, 'most_correlated'), exist_ok=True)
    
    # Load the data
    df = pd.read_csv('data/csv_spotify/csv/sampled_dataset_PE_C_fd.csv')
    
    # Run all analyses
    plot_entropy_correlations(df, ENTROPY_COLS)
    plot_numerical_correlations(df, ENTROPY_COLS)
    plot_genre_correlations(df, ENTROPY_COLS)
    plot_complete_correlation_matrix(df, ENTROPY_COLS)
    
    plot_feature_importance(df, ENTROPY_COLS)
    plot_fractal_feature_importance(df)
    plot_top_feature_scatter(df, ENTROPY_COLS)
    plot_fractal_scatter(df)
    
    # For most correlated scatter plots, iterate over all entropy and fractal dimensions
    all_metrics_for_scatter = ENTROPY_COLS + FRACTAL_COLS
    for metric in all_metrics_for_scatter:
        plot_most_correlated_scatter(df, metric, os.path.join(plots_dir, 'most_correlated'))
    
    # For strong genre correlations, use all metrics
    plot_strong_genre_correlations(df, all_metrics_for_scatter)

if __name__ == "__main__":
    analyze_correlations()
