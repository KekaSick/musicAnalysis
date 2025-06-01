import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
print("Загрузка данных...")
df = pd.read_csv('dataforGithub/csv/sampled_dataset_PE_C_fd.csv')

# Проверка на пропущенные значения
print(f"\nПроверка пропущенных значений:")
print(f"Всего строк: {len(df)}")
print(f"Строк с пропущенными значениями: {df.isnull().any(axis=1).sum()}")
print(f"Колонки с пропущенными значениями:")
for col in df.columns:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        print(f"  {col}: {null_count} пропущенных значений")

# Определение наборов признаков
spotify_features = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 
                   'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
                   'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

entropy_complexity_features = ['entropy_amplitude', 'complexity_amplitude', 'entropy_flux', 
                              'complexity_flux', 'entropy_harmony', 'complexity_harmony',
                              'entropy_spectral', 'complexity_spectral', 'higuchi_fd', 'box_counting_fd']

all_features = spotify_features + entropy_complexity_features

feature_sets = {
    'Spotify Features': spotify_features,
    'Entropy/Complexity/Fractal': entropy_complexity_features,
    'All Features': all_features
}

print(f"\nДоступные жанры: {sorted(df['track_genre'].unique())}")
print(f"\nНаборы признаков:")
for name, features in feature_sets.items():
    print(f"{name}: {len(features)} признаков")

# Подготовка данных
y = df['track_genre'].values

# Кодирование меток
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nРаспределение жанров:")
for i, genre in enumerate(le.classes_):
    count = np.sum(y_encoded == i)
    print(f"{genre}: {count} треков")

# Равномерное разделение данных (50 треков на жанр для тренировки, 50 для теста)
print("\nСоздание равномерного разделения данных...")

# Создаем списки для хранения индексов
train_indices = []
test_indices = []

# Для каждого жанра берем первые 50 треков для тренировки, остальные 50 для теста
for genre in le.classes_:
    genre_indices = df[df['track_genre'] == genre].index.tolist()
    train_indices.extend(genre_indices[:50])
    test_indices.extend(genre_indices[50:])

# Проверяем равномерность разделения
print("\nПроверка равномерности разделения:")
for i, genre in enumerate(le.classes_):
    train_count = np.sum(y_encoded[train_indices] == i)
    test_count = np.sum(y_encoded[test_indices] == i)
    print(f"{genre}: {train_count} тренировка, {test_count} тест")

# Создание и обучение моделей для каждого набора признаков
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

all_results = {}

print("\n" + "="*80)
print("ОБУЧЕНИЕ МОДЕЛЕЙ ДЛЯ РАЗНЫХ НАБОРОВ ПРИЗНАКОВ")
print("="*80)

for feature_set_name, feature_columns in feature_sets.items():
    print(f"\n{'='*20} {feature_set_name} {'='*20}")
    
    # Подготовка данных для текущего набора признаков
    X = df[feature_columns].values
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]
    
    print(f"Размер тренировочной выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    # Проверка на пропущенные значения в текущем наборе признаков
    train_nulls = pd.DataFrame(X_train, columns=feature_columns).isnull().sum().sum()
    test_nulls = pd.DataFrame(X_test, columns=feature_columns).isnull().sum().sum()
    print(f"Пропущенные значения в тренировочной выборке: {train_nulls}")
    print(f"Пропущенные значения в тестовой выборке: {test_nulls}")
    
    feature_set_results = {}
    
    for model_name, model in models.items():
        print(f"\nОбучение {model_name}...")
        
        # Создаем pipeline с обработкой пропущенных значений, масштабированием и моделью
        pipeline = make_pipeline(
            SimpleImputer(strategy='mean'),  # Заменяем NaN на средние значения
            StandardScaler(),
            model
        )
        
        # Обучаем модель
        pipeline.fit(X_train, y_train)
        
        # Предсказания
        y_pred = pipeline.predict(X_test)
        
        # Оценка качества
        accuracy = accuracy_score(y_test, y_pred)
        feature_set_results[model_name] = {
            'model': pipeline,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"Точность {model_name}: {accuracy:.4f}")
    
    all_results[feature_set_name] = feature_set_results

# Сравнение результатов
print("\n" + "="*80)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*80)

# Создаем таблицу сравнения
comparison_data = []
for feature_set_name, feature_set_results in all_results.items():
    for model_name, result in feature_set_results.items():
        comparison_data.append({
            'Feature Set': feature_set_name,
            'Model': model_name,
            'Accuracy': result['accuracy']
        })

comparison_df = pd.DataFrame(comparison_data)
print("\nСравнительная таблица результатов:")
print(comparison_df.to_string(index=False))

# Находим лучшую модель для каждого набора признаков
print("\nЛучшие модели для каждого набора признаков:")
for feature_set_name, feature_set_results in all_results.items():
    best_model_name = max(feature_set_results.keys(), 
                         key=lambda x: feature_set_results[x]['accuracy'])
    best_accuracy = feature_set_results[best_model_name]['accuracy']
    print(f"{feature_set_name}: {best_model_name} - {best_accuracy:.4f}")

# Находим абсолютно лучшую модель
best_overall = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
print(f"\nАбсолютно лучшая модель:")
print(f"Набор признаков: {best_overall['Feature Set']}")
print(f"Модель: {best_overall['Model']}")
print(f"Точность: {best_overall['Accuracy']:.4f}")

# Детальный анализ лучшей модели
best_feature_set = best_overall['Feature Set']
best_model_name = best_overall['Model']
best_model = all_results[best_feature_set][best_model_name]['model']
best_predictions = all_results[best_feature_set][best_model_name]['predictions']

# Получаем данные для лучшей модели
X_best = df[feature_sets[best_feature_set]].values
X_train_best, X_test_best = X_best[train_indices], X_best[test_indices]
y_train_best, y_test_best = y_encoded[train_indices], y_encoded[test_indices]

print("\n" + "="*80)
print(f"ДЕТАЛЬНЫЙ АНАЛИЗ ЛУЧШЕЙ МОДЕЛИ: {best_model_name} ({best_feature_set})")
print("="*80)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test_best, best_predictions, target_names=le.classes_))

# Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test_best, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {best_model_name} ({best_feature_set})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Анализ по жанрам
print("\nАнализ точности по жанрам:")
genre_accuracy = {}
for i, genre in enumerate(le.classes_):
    genre_mask = y_test_best == i
    if np.sum(genre_mask) > 0:
        genre_acc = accuracy_score(y_test_best[genre_mask], best_predictions[genre_mask])
        genre_accuracy[genre] = genre_acc
        print(f"{genre}: {genre_acc:.4f}")

# Визуализация точности по жанрам
plt.figure(figsize=(12, 6))
genres = list(genre_accuracy.keys())
accuracies = list(genre_accuracy.values())

plt.bar(genres, accuracies, color='skyblue', edgecolor='navy')
plt.title(f'Точность классификации по жанрам - {best_model_name} ({best_feature_set})')
plt.xlabel('Жанр')
plt.ylabel('Точность')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)

# Добавляем значения на столбцы
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('genre_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# Кросс-валидация для лучшей модели
print("\n" + "="*80)
print("КРОСС-ВАЛИДАЦИЯ")
print("="*80)

# Используем все данные для кросс-валидации
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_best, y_encoded, cv=skf, scoring='accuracy')

print(f"5-fold CV scores: {cv_scores}")
print(f"Средняя точность CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Анализ важности признаков (только для Random Forest)
if best_model_name == 'Random Forest':
    print("\n" + "="*80)
    print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
    print("="*80)
    
    # Получаем модель из pipeline
    rf_model = best_model.named_steps['randomforestclassifier']
    feature_importance = rf_model.feature_importances_
    
    # Создаем DataFrame с важностью признаков
    importance_df = pd.DataFrame({
        'feature': feature_sets[best_feature_set],
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("Топ-15 важных признаков:")
    print(importance_df.head(15))
    
    # Визуализация важности признаков
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Важность')
    plt.title(f'Топ-15 важных признаков для классификации жанров ({best_feature_set})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Сравнительная визуализация результатов
print("\n" + "="*80)
print("СРАВНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ")
print("="*80)

# Создаем heatmap для сравнения всех моделей
plt.figure(figsize=(14, 8))
pivot_df = comparison_df.pivot(index='Model', columns='Feature Set', values='Accuracy')
sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Accuracy'})
plt.title('Сравнение точности моделей для разных наборов признаков')
plt.tight_layout()
plt.savefig('model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# График сравнения лучших моделей для каждого набора признаков
plt.figure(figsize=(12, 6))
best_models_data = []
for feature_set_name, feature_set_results in all_results.items():
    best_model_name = max(feature_set_results.keys(), 
                         key=lambda x: feature_set_results[x]['accuracy'])
    best_accuracy = feature_set_results[best_model_name]['accuracy']
    best_models_data.append({
        'Feature Set': feature_set_name,
        'Best Model': best_model_name,
        'Accuracy': best_accuracy
    })

best_models_df = pd.DataFrame(best_models_data)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = plt.bar(best_models_df['Feature Set'], best_models_df['Accuracy'], color=colors)

plt.title('Сравнение лучших моделей для каждого набора признаков')
plt.xlabel('Набор признаков')
plt.ylabel('Точность')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)

# Добавляем значения на столбцы
for bar, acc, model in zip(bars, best_models_df['Accuracy'], best_models_df['Best Model']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}\n({model})', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('best_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Сохранение результатов
print("\n" + "="*80)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*80)

# Сохраняем результаты в CSV
comparison_df.to_csv('classification_results_comparison.csv', index=False)
print("Результаты сохранены в classification_results_comparison.csv")

# Сохраняем лучшую модель
import joblib
joblib.dump(best_model, 'best_genre_classifier.pkl')
print("Лучшая модель сохранена в best_genre_classifier.pkl")

# Сохраняем информацию о лучшей модели
best_model_info = {
    'feature_set': best_feature_set,
    'model_name': best_model_name,
    'accuracy': best_overall['Accuracy'],
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std()
}

import json
with open('best_model_info.json', 'w') as f:
    json.dump(best_model_info, f, indent=2)
print("Информация о лучшей модели сохранена в best_model_info.json")

print("\n" + "="*80)
print("ЗАКЛЮЧЕНИЕ")
print("="*80)
print(f"Лучшая модель: {best_model_name}")
print(f"Набор признаков: {best_feature_set}")
print(f"Точность на тестовой выборке: {best_overall['Accuracy']:.4f}")
print(f"Средняя точность кросс-валидации: {cv_scores.mean():.4f}")
print(f"Стандартное отклонение CV: {cv_scores.std():.4f}")

# Сравнение наборов признаков
print(f"\nСравнение наборов признаков:")
for feature_set_name, feature_set_results in all_results.items():
    best_acc = max(result['accuracy'] for result in feature_set_results.values())
    print(f"- {feature_set_name}: {best_acc:.4f}")

# Анализ ошибок
print(f"\nНаиболее проблемные жанры:")
worst_genres = sorted(genre_accuracy.items(), key=lambda x: x[1])[:3]
for genre, acc in worst_genres:
    print(f"- {genre}: {acc:.4f}")

print(f"\nНаиболее точные жанры:")
best_genres = sorted(genre_accuracy.items(), key=lambda x: x[1], reverse=True)[:3]
for genre, acc in best_genres:
    print(f"- {genre}: {acc:.4f}")

print("\nАнализ завершен!") 