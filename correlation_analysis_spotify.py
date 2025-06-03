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
    return stats.pearsonr(x[mask], y[mask])

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
    os.makedirs('plots/sampled_dataset_PE_C/csv', exist_ok=True)
    corr_matrix.to_csv('plots/sampled_dataset_PE_C/csv/entropy_correlations.csv', index=False)
    p_matrix.to_csv('plots/sampled_dataset_PE_C/csv/entropy_pvalues.csv', index=False)
    
    # Строим тепловую карту
    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1)
    
    plt.title('Correlations between entropy measures, complexity and fractal dimensions')
    plt.tight_layout()
    plt.savefig('plots/sampled_dataset_PE_C/entropy_correlations.png', dpi=300, bbox_inches='tight')
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
    os.makedirs('plots/sampled_dataset_PE_C/csv', exist_ok=True)
    corr_with_entropy.to_csv('plots/sampled_dataset_PE_C/csv/numerical_correlations.csv', index=False)
    p_with_entropy.to_csv('plots/sampled_dataset_PE_C/csv/numerical_pvalues.csv', index=False)
    
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
    plt.savefig('plots/sampled_dataset_PE_C/numerical_correlations.png', dpi=300, bbox_inches='tight')
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
    os.makedirs('plots/sampled_dataset_PE_C/csv', exist_ok=True)
    corr_with_genres.to_csv('plots/sampled_dataset_PE_C/csv/genre_correlations.csv', index=False)
    p_with_genres.to_csv('plots/sampled_dataset_PE_C/csv/genre_pvalues.csv', index=False)
    
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
    plt.savefig('plots/sampled_dataset_PE_C/genre_correlations.png', dpi=300, bbox_inches='tight')
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
    os.makedirs('plots/sampled_dataset_PE_C/csv', exist_ok=True)
    corr_matrix.to_csv('plots/sampled_dataset_PE_C/csv/complete_correlation_matrix.csv', index=False)
    p_matrix.to_csv('plots/sampled_dataset_PE_C/csv/complete_pvalues_matrix.csv', index=False)
    
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
    plt.savefig('plots/sampled_dataset_PE_C/complete_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(df, target_cols):
    """Построение графиков важности признаков для предсказания энтропий и сложностей"""
    # Создаем директории для графиков
    os.makedirs('plots/sampled_dataset_PE_C/features', exist_ok=True)
    
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
        plt.savefig(f'plots/sampled_dataset_PE_C/features/feature_importance_{metric_type.lower()}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def plot_fractal_feature_importance(df):
    """Построение графиков важности признаков для предсказания фрактальных размерностей"""
    # Создаем директорию для графиков
    os.makedirs('plots/sampled_dataset_PE_C/features', exist_ok=True)
    
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
    plt.savefig('plots/sampled_dataset_PE_C/features/feature_importance_fractal.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_top_feature_scatter(df, target_cols):
    """Построение scatter plots между каждой метрикой и её самой важной характеристикой"""
    # Создаем директорию для графиков
    os.makedirs('plots/sampled_dataset_PE_C/features', exist_ok=True)
    
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
                z = np.polyfit(x_data[~np.isnan(df[target])], 
                             df[target][~np.isnan(df[target])], 1)
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
                
                corr, p_val = stats.pearsonr(x_clean, y_clean)
                
                if p_val < 0.001:
                    p_text = "p < 0.001"
                else:
                    p_text = f"p = {p_val:.3f}"
                
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}\n{p_text}', 
                        transform=plt.gca().transAxes, 
                        verticalalignment='top')
        
        plt.suptitle(f'Top Feature Relationships for {metric_type} Metrics')
        plt.tight_layout()
        plt.savefig(f'plots/sampled_dataset_PE_C/features/top_feature_scatter_{metric_type.lower()}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def plot_fractal_scatter(df):
    """Построение scatter plots между фрактальными размерностями и их важнейшими характеристиками"""
    # Создаем директорию для графиков
    os.makedirs('plots/sampled_dataset_PE_C/features', exist_ok=True)
    
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
    
    plt.figure(figsize=(15, 6))
    
    for idx, target in enumerate(FRACTAL_COLS):
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
            z = np.polyfit(x_data[~np.isnan(df[target])], 
                         df[target][~np.isnan(df[target])], 1)
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
            
            corr, p_val = stats.pearsonr(x_clean, y_clean)
            
            if p_val < 0.001:
                p_text = "p < 0.001"
            else:
                p_text = f"p = {p_val:.3f}"
            
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}\n{p_text}', 
                    transform=plt.gca().transAxes, 
                    verticalalignment='top')
    
    plt.suptitle('Top Feature Relationships for Fractal Dimensions')
    plt.tight_layout()
    plt.savefig('plots/sampled_dataset_PE_C/features/top_feature_scatter_fractal.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_most_correlated_scatter(df, metric_name, output_dir):
    """Построение scatter plots для метрики с самыми сильно коррелирующими признаками"""
    # Подготовка признаков
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols 
                     if col not in ENTROPY_COLS + FRACTAL_COLS and 'id' not in col.lower()]
    
    # Вычисляем корреляции и p-values
    correlations = {}
    p_values = {}
    for col in numerical_cols:
        mask = ~(np.isnan(df[col]) | np.isnan(df[metric_name]))
        if mask.sum() < 2:
            continue
        corr, p_val = stats.pearsonr(df[col][mask], df[metric_name][mask])
        correlations[col] = corr
        p_values[col] = p_val
    
    # Находим самую положительную и самую отрицательную корреляцию
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
    """Построение box plots для сильных корреляций между жанрами и метриками (|corr| > threshold)"""
    # Создаем dummy-переменные для жанров
    genres_dummies = pd.get_dummies(df['track_genre'], prefix='genre').astype(float)
    
    # Создаем директорию для сохранения графиков
    output_dir = 'plots/sampled_dataset_PE_C/genre_correlations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Для каждой метрики ищем сильные корреляции с жанрами
    strong_correlations = []
    for metric in metrics:
        for genre in genres_dummies.columns:
            mask = ~(np.isnan(df[metric]) | np.isnan(genres_dummies[genre]))
            if mask.sum() < 2:
                continue
            
            corr, p_val = stats.pearsonr(df[metric][mask], genres_dummies[genre][mask])
            
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
    output_dir = 'plots/sampled_dataset_PE_C/entropy_complexity_space'
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

def get_smart_annotation_position(x, y, text_length, cluster_idx, total_clusters, ax):
    """Определяет оптимальное положение аннотации с учетом длины текста и доступного пространства"""
    # Нормализуем координаты точки
    x_norm = (x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
    y_norm = (y - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    
    # Базовое расстояние зависит от длины текста, но имеет ограничение
    base_distance = min(100 + text_length * 1.5, 200)  # Ограничиваем максимальное расстояние
    
    # Определяем основные направления в зависимости от положения точки
    directions = []
    
    # Правая часть графика
    if x_norm > 0.8:
        if y_norm > 0.7:
            # Верхний правый угол - размещаем слева или снизу
            directions.extend([
                ((-base_distance, 0), 'right', 'center', 'arc3,rad=0'),
                ((0, -base_distance), 'center', 'top', 'arc3,rad=-0.2')
            ])
        elif y_norm < 0.3:
            # Нижний правый угол - размещаем слева или сверху
            directions.extend([
                ((-base_distance, 0), 'right', 'center', 'arc3,rad=0'),
                ((0, base_distance), 'center', 'bottom', 'arc3,rad=0.2')
            ])
        else:
            # Середина справа - размещаем слева
            directions.extend([
                ((-base_distance, 0), 'right', 'center', 'arc3,rad=0'),
                ((-base_distance, base_distance/2), 'right', 'bottom', 'arc3,rad=0.2')
            ])
    
    # Левая часть графика
    elif x_norm < 0.2:
        if y_norm > 0.7:
            # Верхний левый угол - размещаем справа или снизу
            directions.extend([
                ((base_distance, 0), 'left', 'center', 'arc3,rad=0'),
                ((0, -base_distance), 'center', 'top', 'arc3,rad=0.2')
            ])
        elif y_norm < 0.3:
            # Нижний левый угол - размещаем справа или сверху
            directions.extend([
                ((base_distance, 0), 'left', 'center', 'arc3,rad=0'),
                ((0, base_distance), 'center', 'bottom', 'arc3,rad=-0.2')
            ])
        else:
            # Середина слева - размещаем справа
            directions.extend([
                ((base_distance, 0), 'left', 'center', 'arc3,rad=0'),
                ((base_distance, base_distance/2), 'left', 'bottom', 'arc3,rad=-0.2')
            ])
    
    # Центральная часть
    else:
        if y_norm > 0.7:
            # Верхняя часть - размещаем снизу
            directions.extend([
                ((0, -base_distance), 'center', 'top', 'arc3,rad=0'),
                ((base_distance/2, -base_distance), 'left', 'top', 'arc3,rad=-0.2')
            ])
        elif y_norm < 0.3:
            # Нижняя часть - размещаем сверху
            directions.extend([
                ((0, base_distance), 'center', 'bottom', 'arc3,rad=0'),
                ((base_distance/2, base_distance), 'left', 'bottom', 'arc3,rad=0.2')
            ])
        else:
            # Центральная часть - выбираем направление в зависимости от индекса кластера
            directions.extend([
                ((base_distance, 0), 'left', 'center', 'arc3,rad=0'),
                ((-base_distance, 0), 'right', 'center', 'arc3,rad=0'),
                ((0, base_distance), 'center', 'bottom', 'arc3,rad=0'),
                ((0, -base_distance), 'center', 'top', 'arc3,rad=0')
            ])
    
    # Если список направлений пуст, добавляем стандартные направления
    if not directions:
        directions = [
            ((base_distance, 0), 'left', 'center', 'arc3,rad=0'),
            ((-base_distance, 0), 'right', 'center', 'arc3,rad=0'),
            ((0, base_distance), 'center', 'bottom', 'arc3,rad=0'),
            ((0, -base_distance), 'center', 'top', 'arc3,rad=0')
        ]
    
    # Выбираем направление на основе индекса кластера
    direction_idx = cluster_idx % len(directions)
    offset, ha, va, connection_style = directions[direction_idx]
    
    # Добавляем небольшое случайное смещение для предотвращения наложений
    jitter = 10 * (cluster_idx // len(directions) + 1)
    offset = (offset[0] + jitter, offset[1] + jitter)
    
    return offset, ha, va, connection_style

def find_representative_tracks(df, X, labels, metric_type, algorithm='kmeans', centers=None):
    """Находит самые репрезентативные треки для каждого кластера"""
    representative_tracks = []
    
    # Для каждого кластера (исключая шум для HDBSCAN)
    unique_labels = np.unique(labels)
    if algorithm == 'hdbscan':
        unique_labels = unique_labels[unique_labels != -1]  # Исключаем шум
    
    for label in unique_labels:
        # Получаем индексы точек в текущем кластере
        cluster_mask = labels == label
        cluster_points = X[cluster_mask]
        
        if len(cluster_points) == 0:
            continue
        
        # Находим центр кластера
        if algorithm == 'kmeans' and centers is not None:
            cluster_center = centers[label]
        else:
            cluster_center = np.mean(cluster_points, axis=0)
        
        # Находим точку, ближайшую к центру кластера
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        closest_point_idx = np.argmin(distances)
        
        # Получаем индекс трека в исходном датафрейме
        original_idx = df.index[cluster_mask].values[closest_point_idx]
        
        # Собираем информацию о треке
        track_info = {
            'cluster': label,
            'algorithm': algorithm,
            'metric_type': metric_type,
            'track_id': df.loc[original_idx, 'track_id'],
            'track_name': df.loc[original_idx, 'track_name'],
            'artists': df.loc[original_idx, 'artists'],
            'track_genre': df.loc[original_idx, 'track_genre'],
            'distance_to_center': distances[closest_point_idx]
        }
        representative_tracks.append(track_info)
    
    return representative_tracks

def plot_clustering_analysis(df):
    """Построение кластерного анализа для каждого пространства энтропия-сложность"""
    # Создаем директории для сохранения графиков и результатов
    output_dir = 'plots/sampled_dataset_PE_C/clusters'
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем список для хранения всех репрезентативных треков
    all_representative_tracks = []
    
    # Определяем пары энтропия-сложность
    metric_pairs = [
        ('entropy_amplitude', 'complexity_amplitude', 'Amplitude'),
        ('entropy_flux', 'complexity_flux', 'Flux'),
        ('entropy_harmony', 'complexity_harmony', 'Harmony'),
        ('entropy_spectral', 'complexity_spectral', 'Spectral')
    ]
    
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
    
    # Для каждой пары метрик выполняем кластеризацию
    for entropy_col, complexity_col, metric_type in metric_pairs:
        # Подготовка данных для кластеризации
        X = df[[entropy_col, complexity_col]].dropna()
        
        # Масштабирование данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. KMeans кластеризация
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # Находим репрезентативные треки для KMeans
        kmeans_tracks = find_representative_tracks(
            df.loc[X.index], 
            X_scaled, 
            kmeans_labels, 
            metric_type,
            'kmeans',
            kmeans.cluster_centers_
        )
        all_representative_tracks.extend(kmeans_tracks)
        
        # Построение графика KMeans
        plt.figure(figsize=(15, 10))  # Увеличиваем размер для лучшей читаемости аннотаций
        
        # Рисуем теоретические ограничения
        plt.plot(x_grid, max_curve, 'k--', alpha=0.5, label='Max Complexity')
        plt.plot(x_grid, min_curve, 'k:', alpha=0.5, label='Min Complexity')
        plt.fill_between(x_grid, min_curve, max_curve, alpha=0.1, color='gray')
        
        # Рисуем точки с цветами кластеров
        scatter = plt.scatter(X[entropy_col], X[complexity_col], 
                            c=kmeans_labels, cmap='viridis',
                            alpha=0.6)
        
        # Рисуем центры кластеров и добавляем аннотации для репрезентативных треков
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        plt.scatter(centers[:, 0], centers[:, 1], c='red',
                   marker='x', s=200, linewidth=3,
                   label='Cluster centers')
        
        # Добавляем аннотации для репрезентативных треков
        for track in kmeans_tracks:
            idx = df.index[df['track_id'] == track['track_id']].values[0]
            x, y = X.loc[idx, entropy_col], X.loc[idx, complexity_col]
            
            # Создаем текст аннотации
            annotation_text = f"Cluster {track['cluster']}:\n{track['track_name']}\n{track['artists']}\n{track['track_genre']}"
            
            # Получаем оптимальное положение для аннотации
            offset, ha, va, connection_style = get_smart_annotation_position(
                x, y, len(annotation_text), track['cluster'], 4, plt.gca()
            )
            
            # Добавляем аннотацию со стрелкой
            plt.annotate(annotation_text, 
                        xy=(x, y),
                        xytext=offset,
                        textcoords='offset points',
                        ha=ha,
                        va=va,
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', 
                                      connectionstyle=connection_style,
                                      alpha=0.5))
        
        plt.xlabel(f'{entropy_col.replace("_", " ").title()}')
        plt.ylabel(f'{complexity_col.replace("_", " ").title()}')
        plt.title(f'{metric_type} Entropy-Complexity Space - KMeans Clustering')
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect('equal', 'box')
        
        # Добавляем легенды
        # Легенда для пространства энтропии-сложности слева
        plt.legend(['Max Complexity', 'Min Complexity', 'Cluster centers'],
                  loc='center left', bbox_to_anchor=(-0.15, 0.5))
        
        # Легенда для кластеров справа
        legend_clusters = plt.legend(*scatter.legend_elements(),
                                   title="Clusters",
                                   loc='center left', 
                                   bbox_to_anchor=(1.02, 0.5))
        plt.gca().add_artist(legend_clusters)
        
        # Увеличиваем отступы для размещения легенд и аннотаций
        plt.subplots_adjust(left=0.15, right=0.85)
        
        plt.savefig(os.path.join(output_dir, f'kmeans_{metric_type.lower()}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. HDBSCAN кластеризация
        clusterer = hdbscan.HDBSCAN(min_cluster_size=40,
                                   min_samples=5,
                                   metric='euclidean')
        hdbscan_labels = clusterer.fit_predict(X_scaled)
        
        # Находим репрезентативные треки для HDBSCAN
        hdbscan_tracks = find_representative_tracks(
            df.loc[X.index], 
            X_scaled, 
            hdbscan_labels, 
            metric_type,
            'hdbscan'
        )
        all_representative_tracks.extend(hdbscan_tracks)
        
        # Построение графика HDBSCAN
        plt.figure(figsize=(15, 10))  # Увеличиваем размер для лучшей читаемости аннотаций
        
        # Рисуем теоретические ограничения
        plt.plot(x_grid, max_curve, 'k--', alpha=0.5, label='Max Complexity')
        plt.plot(x_grid, min_curve, 'k:', alpha=0.5, label='Min Complexity')
        plt.fill_between(x_grid, min_curve, max_curve, alpha=0.1, color='gray')
        
        # Рисуем точки с цветами кластеров
        # Шум (метка -1) отображаем серым цветом
        noise_mask = hdbscan_labels == -1
        cluster_mask = ~noise_mask
        
        # Сначала рисуем шум
        plt.scatter(X[entropy_col][noise_mask], X[complexity_col][noise_mask],
                   c='gray', alpha=0.3, label='Noise')
        
        # Затем рисуем кластеры
        if np.any(cluster_mask):
            scatter = plt.scatter(X[entropy_col][cluster_mask], 
                                X[complexity_col][cluster_mask],
                                c=hdbscan_labels[cluster_mask], 
                                cmap='viridis',
                                alpha=0.6)
            
            # Добавляем аннотации для репрезентативных треков HDBSCAN
            unique_clusters = np.unique(hdbscan_labels[cluster_mask])
            for track in hdbscan_tracks:
                idx = df.index[df['track_id'] == track['track_id']].values[0]
                x, y = X.loc[idx, entropy_col], X.loc[idx, complexity_col]
                
                # Создаем текст аннотации
                annotation_text = f"Cluster {track['cluster']}:\n{track['track_name']}\n{track['artists']}\n{track['track_genre']}"
                
                # Получаем оптимальное положение для аннотации
                offset, ha, va, connection_style = get_smart_annotation_position(
                    x, y, len(annotation_text), track['cluster'], len(unique_clusters), plt.gca()
                )
                
                # Добавляем аннотацию со стрелкой
                plt.annotate(annotation_text, 
                            xy=(x, y),
                            xytext=offset,
                            textcoords='offset points',
                            ha=ha,
                            va=va,
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', 
                                          connectionstyle=connection_style,
                                          alpha=0.5))
        
        plt.xlabel(f'{entropy_col.replace("_", " ").title()}')
        plt.ylabel(f'{complexity_col.replace("_", " ").title()}')
        plt.title(f'{metric_type} Entropy-Complexity Space - HDBSCAN Clustering')
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect('equal', 'box')
        
        # Добавляем легенды
        # Легенда для пространства энтропии-сложности слева
        plt.legend(['Max Complexity', 'Min Complexity', 'Noise'],
                  loc='center left', bbox_to_anchor=(-0.15, 0.5))
        
        # Легенда для кластеров справа
        if np.any(cluster_mask):
            legend_clusters = plt.legend(*scatter.legend_elements(),
                                       title="Clusters",
                                       loc='center left', 
                                       bbox_to_anchor=(1.02, 0.5))
            plt.gca().add_artist(legend_clusters)
        
        # Увеличиваем отступы для размещения легенд и аннотаций
        plt.subplots_adjust(left=0.15, right=0.85)
        
        # Увеличиваем размер фигуры и отступы для лучшего размещения аннотаций
        plt.gcf().set_size_inches(24, 18)  # Больший размер для лучшего размещения
        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        
        plt.savefig(os.path.join(output_dir, f'hdbscan_{metric_type.lower()}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Сохраняем информацию о репрезентативных треках в CSV
    representative_df = pd.DataFrame(all_representative_tracks)
    representative_df.to_csv(os.path.join(output_dir, 'representative_tracks.csv'), 
                           index=False)

def plot_fractal_dimensions_by_genre(df):
    """Построение боксплотов фрактальных размерностей по жанрам"""
    # Создаем директорию для сохранения графиков
    output_dir = 'plots/sampled_dataset_PE_C/fractal_dimensions'
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
    os.makedirs('plots/sampled_dataset_PE_C', exist_ok=True)
    os.makedirs('plots/sampled_dataset_PE_C/features', exist_ok=True)
    os.makedirs('plots/sampled_dataset_PE_C/most_correlated', exist_ok=True)
    
    # Загружаем данные
    print("Загрузка данных...")
    df = pd.read_csv(DATASET_PATH)
    
    print("Построение графиков пространства энтропия-сложность...")
    plot_entropy_complexity_by_genre(df)
    
    print("Построение кластерного анализа...")
    plot_clustering_analysis(df)
    
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
        plot_most_correlated_scatter(df, metric, 'plots/sampled_dataset_PE_C/most_correlated')
    
    # Для сложностей
    for metric in ['complexity_amplitude', 'complexity_flux', 'complexity_harmony', 'complexity_spectral']:
        plot_most_correlated_scatter(df, metric, 'plots/sampled_dataset_PE_C/most_correlated')
    
    # Для фрактальных размерностей
    for metric in FRACTAL_COLS:
        plot_most_correlated_scatter(df, metric, 'plots/sampled_dataset_PE_C/most_correlated')
    
    print("Готово! Графики сохранены в директории 'plots/sampled_dataset_PE_C'")

if __name__ == "__main__":
    analyze_correlations() 