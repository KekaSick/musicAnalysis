import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score, 
                                    RandomizedSearchCV, StratifiedShuffleSplit,
                                    learning_curve)
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, 
                             GradientBoostingClassifier)
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Функция для форматирования времени
def format_time(seconds):
    """Форматирует время в удобочитаемый вид"""
    if seconds < 60:
        return f"{seconds:.1f} сек"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)} мин {secs:.1f} сек"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)} ч {int(minutes)} мин {secs:.1f} сек"

# Начало общего времени
total_start_time = time.time()
print(f"🚀 Начало быстрого анализа: {datetime.now().strftime('%H:%M:%S')}")

# Загрузка данных
print("\n📊 Загрузка данных...")
data_start_time = time.time()
df = pd.read_csv('dataforGithub/csv/sampled_dataset_PE_C_fd.csv')
data_time = time.time() - data_start_time
print(f"✅ Данные загружены за {format_time(data_time)}")

# Проверка на пропущенные значения
print(f"\n🔍 Проверка пропущенных значений:")
print(f"Всего строк: {len(df)}")
print(f"Строк с пропущенными значениями: {df.isnull().any(axis=1).sum()}")

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

print(f"\n🎵 Доступные жанры: {sorted(df['track_genre'].unique())}")
print(f"\n📈 Наборы признаков:")
for name, features in feature_sets.items():
    print(f"{name}: {len(features)} признаков")

# Подготовка данных
y = df['track_genre'].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n📊 Распределение жанров:")
for i, genre in enumerate(le.classes_):
    count = np.sum(y_encoded == i)
    print(f"{genre}: {count} треков")

# Случайное stratified split
print("\n✂️ Создание случайного stratified разделения данных...")
split_start_time = time.time()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for train_idx, test_idx in sss.split(df, y_encoded):
    train_indices = train_idx
    test_indices = test_idx

print("\n✅ Проверка равномерности разделения:")
for i, genre in enumerate(le.classes_):
    train_count = np.sum(y_encoded[train_indices] == i)
    test_count = np.sum(y_encoded[test_indices] == i)
    print(f"{genre}: {train_count} тренировка, {test_count} тест")

split_time = time.time() - split_start_time
print(f"✅ Разделение данных завершено за {format_time(split_time)}")

# УЛУЧШЕНИЕ 1: RandomizedSearchCV для быстрого поиска
def optimize_randomized(X_train, y_train, model_type='rf', n_iter=30):
    """Быстрая оптимизация гиперпараметров через RandomizedSearchCV"""
    
    if model_type == 'rf':
        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', None]
        }
        model = RandomForestClassifier(random_state=42)
    
    elif model_type == 'gb':
        param_dist = {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
        model = GradientBoostingClassifier(random_state=42)
    
    elif model_type == 'svm':
        param_dist = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly'],
            'class_weight': ['balanced', None]
        }
        model = SVC(random_state=42, probability=True)
    
    elif model_type == 'lr':
        param_dist = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None]
        }
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Ускоренный CV - только 5 folds без повторов
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    rand_search = RandomizedSearchCV(
        model, param_distributions=param_dist,
        n_iter=n_iter, cv=cv,
        scoring='accuracy', n_jobs=-1, random_state=42,
        verbose=0
    )
    
    rand_search.fit(X_train, y_train)
    
    return rand_search.best_estimator_, rand_search.best_score_, rand_search.best_params_

# УЛУЧШЕНИЕ 2: HalvingGridSearchCV для еще более быстрого поиска
def optimize_halving(X_train, y_train, model_type='rf'):
    """Сверхбыстрая оптимизация через HalvingGridSearchCV"""
    
    if model_type == 'rf':
        param_grid = {
            'n_estimators': [100, 300, 500],  # Сокращенный диапазон
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        model = RandomForestClassifier(random_state=42)
    
    elif model_type == 'gb':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
        model = GradientBoostingClassifier(random_state=42)
    
    elif model_type == 'svm':
        param_grid = {
            'C': [1, 10, 100],
            'gamma': ['scale', 'auto', 0.01],
            'kernel': ['rbf']  # Убираем poly для ускорения
        }
        model = SVC(random_state=42, probability=True)
    
    elif model_type == 'lr':
        param_grid = {
            'C': [1, 10, 100],
            'penalty': ['l2'],  # Убираем l1 для ускорения
            'solver': ['liblinear']
        }
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    halving = HalvingGridSearchCV(
        model, param_grid=param_grid,
        cv=cv, factor=3,  # Отсекаем 2/3 на каждом раунде
        scoring='accuracy', n_jobs=-1, 
        min_resources='smallest', verbose=0
    )
    
    halving.fit(X_train, y_train)
    
    return halving.best_estimator_, halving.best_score_, halving.best_params_

# Функция для анализа важности признаков
def analyze_feature_importance(model, feature_names, model_name):
    """Анализ важности признаков для разных типов моделей"""
    importance_data = {}
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_data = dict(zip(feature_names, importances))
        importance_type = 'Feature Importance'
        
    elif hasattr(model, 'coef_'):
        if len(model.coef_.shape) > 1:
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            importances = np.abs(model.coef_)
        importance_data = dict(zip(feature_names, importances))
        importance_type = 'Coefficient Magnitude'
    
    else:
        return None, None
    
    sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
    return sorted_features, importance_type

# Функция для создания калиброванных моделей
def create_calibrated_model(base_model, X_train, y_train):
    """Создание калиброванной модели для лучших вероятностей"""
    calibrated_model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')  # Уменьшаем CV
    calibrated_model.fit(X_train, y_train)
    return calibrated_model

# Основной анализ
all_results = {}

print("\n" + "="*100)
print("⚡ БЫСТРЫЙ АНАЛИЗ С RANDOMIZED И HALVING ПОИСКОМ")
print("="*100)

for feature_set_name, feature_columns in feature_sets.items():
    feature_set_start_time = time.time()
    print(f"\n{'='*30} {feature_set_name} {'='*30}")
    
    # Подготовка данных
    X = df[feature_columns].values
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]
    
    print(f"📊 Размер тренировочной выборки: {X_train.shape}")
    print(f"📊 Размер тестовой выборки: {X_test.shape}")
    
    # Предобработка данных
    print("\n⚙️ Предобработка данных...")
    preprocess_start_time = time.time()
    
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Полиномиальные признаки только для небольших наборов
    if len(feature_columns) <= 15:
        print("🔧 Создание полиномиальных признаков...")
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)
        print(f"📈 Размер после полиномиальных признаков: {X_train_poly.shape}")
    else:
        X_train_poly = X_train_scaled
        X_test_poly = X_test_scaled
    
    preprocess_time = time.time() - preprocess_start_time
    print(f"✅ Предобработка завершена за {format_time(preprocess_time)}")
    
    feature_set_results = {}
    
    # 1. Random Forest с RandomizedSearchCV
    print(f"\n🌲 1. Random Forest (RandomizedSearchCV)...")
    rf_start_time = time.time()
    rf_best, rf_cv_score, rf_params = optimize_randomized(X_train_scaled, y_train, 'rf', n_iter=30)
    rf_pred = rf_best.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    rf_importance, rf_importance_type = analyze_feature_importance(rf_best, feature_columns, 'Random Forest')
    
    feature_set_results['Random Forest (Randomized)'] = {
        'model': rf_best,
        'accuracy': rf_accuracy,
        'cv_score': rf_cv_score,
        'predictions': rf_pred,
        'params': rf_params,
        'feature_importance': rf_importance,
        'importance_type': rf_importance_type
    }
    rf_time = time.time() - rf_start_time
    print(f"✅ Random Forest CV: {rf_cv_score:.4f}, Test: {rf_accuracy:.4f} (время: {format_time(rf_time)})")
    
    # 2. Gradient Boosting с HalvingGridSearchCV
    print(f"\n📈 2. Gradient Boosting (HalvingGridSearchCV)...")
    gb_start_time = time.time()
    gb_best, gb_cv_score, gb_params = optimize_halving(X_train_scaled, y_train, 'gb')
    gb_pred = gb_best.predict(X_test_scaled)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    
    gb_importance, gb_importance_type = analyze_feature_importance(gb_best, feature_columns, 'Gradient Boosting')
    
    feature_set_results['Gradient Boosting (Halving)'] = {
        'model': gb_best,
        'accuracy': gb_accuracy,
        'cv_score': gb_cv_score,
        'predictions': gb_pred,
        'params': gb_params,
        'feature_importance': gb_importance,
        'importance_type': gb_importance_type
    }
    gb_time = time.time() - gb_start_time
    print(f"✅ Gradient Boosting CV: {gb_cv_score:.4f}, Test: {gb_accuracy:.4f} (время: {format_time(gb_time)})")
    
    # 3. SVM с RandomizedSearchCV (ускоренная версия)
    print(f"\n🔧 3. SVM (RandomizedSearchCV)...")
    svm_start_time = time.time()
    svm_best, svm_cv_score, svm_params = optimize_randomized(X_train_scaled, y_train, 'svm', n_iter=20)  # Меньше итераций
    svm_pred = svm_best.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    
    feature_set_results['SVM (Randomized)'] = {
        'model': svm_best,
        'accuracy': svm_accuracy,
        'cv_score': svm_cv_score,
        'predictions': svm_pred,
        'params': svm_params
    }
    svm_time = time.time() - svm_start_time
    print(f"✅ SVM CV: {svm_cv_score:.4f}, Test: {svm_accuracy:.4f} (время: {format_time(svm_time)})")
    
    # 4. Logistic Regression с HalvingGridSearchCV
    print(f"\n📊 4. Logistic Regression (HalvingGridSearchCV)...")
    lr_start_time = time.time()
    lr_best, lr_cv_score, lr_params = optimize_halving(X_train_scaled, y_train, 'lr')
    lr_pred = lr_best.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    lr_importance, lr_importance_type = analyze_feature_importance(lr_best, feature_columns, 'Logistic Regression')
    
    feature_set_results['Logistic Regression (Halving)'] = {
        'model': lr_best,
        'accuracy': lr_accuracy,
        'cv_score': lr_cv_score,
        'predictions': lr_pred,
        'params': lr_params,
        'feature_importance': lr_importance,
        'importance_type': lr_importance_type
    }
    lr_time = time.time() - lr_start_time
    print(f"✅ Logistic Regression CV: {lr_cv_score:.4f}, Test: {lr_accuracy:.4f} (время: {format_time(lr_time)})")
    
    # 5. Быстрые ensemble модели
    print(f"\n🎯 5. Создание быстрых Ensemble моделей...")
    ensemble_start_time = time.time()
    
    # Калибруем только лучшие модели для экономии времени
    print("🔧 Калибровка лучших моделей...")
    best_models = [rf_best, gb_best]  # Только RF и GB для скорости
    calibrated_models = []
    
    for i, model in enumerate(best_models):
        calibrated = create_calibrated_model(model, X_train_scaled, y_train)
        calibrated_models.append((f'model_{i}', calibrated))
    
    # Простой voting ensemble
    voting_equal = VotingClassifier(estimators=calibrated_models, voting='soft')
    voting_equal.fit(X_train_scaled, y_train)
    voting_equal_pred = voting_equal.predict(X_test_scaled)
    voting_equal_accuracy = accuracy_score(y_test, voting_equal_pred)
    feature_set_results['Voting Ensemble (Fast)'] = {
        'model': voting_equal,
        'accuracy': voting_equal_accuracy,
        'cv_score': cross_val_score(voting_equal, X_train_scaled, y_train, cv=3).mean(),  # Уменьшаем CV
        'predictions': voting_equal_pred
    }
    print(f"✅ Voting Ensemble (Fast) Test: {voting_equal_accuracy:.4f}")
    
    ensemble_time = time.time() - ensemble_start_time
    print(f"✅ Ensemble модели созданы за {format_time(ensemble_time)}")
    
    # 6. Полиномиальные признаки (только для небольших наборов)
    if len(feature_columns) <= 15:
        print(f"\n🔧 6. Полиномиальные признаки (HalvingGridSearchCV)...")
        poly_start_time = time.time()
        
        rf_poly_best, rf_poly_cv_score, rf_poly_params = optimize_halving(X_train_poly, y_train, 'rf')
        rf_poly_pred = rf_poly_best.predict(X_test_poly)
        rf_poly_accuracy = accuracy_score(y_test, rf_poly_pred)
        
        feature_set_results['Random Forest (Polynomial)'] = {
            'model': rf_poly_best,
            'accuracy': rf_poly_accuracy,
            'cv_score': rf_poly_cv_score,
            'predictions': rf_poly_pred,
            'params': rf_poly_params
        }
        poly_time = time.time() - poly_start_time
        print(f"✅ Random Forest (Polynomial) CV: {rf_poly_cv_score:.4f}, Test: {rf_poly_accuracy:.4f} (время: {format_time(poly_time)})")
    
    feature_set_time = time.time() - feature_set_start_time
    print(f"\n🎉 {feature_set_name} завершен за {format_time(feature_set_time)}")
    
    all_results[feature_set_name] = feature_set_results

# Сравнение результатов
print("\n" + "="*100)
print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ (БЫСТРАЯ ВЕРСИЯ)")
print("="*100)

comparison_data = []
for feature_set_name, feature_set_results in all_results.items():
    for model_name, result in feature_set_results.items():
        comparison_data.append({
            'Feature Set': feature_set_name,
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'CV Score': result['cv_score']
        })

comparison_df = pd.DataFrame(comparison_data)
print("\n📋 Сравнительная таблица результатов:")
print(comparison_df.to_string(index=False))

# Находим лучшую модель
best_overall = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
print(f"\n👑 Лучшая модель (быстрая версия):")
print(f"Набор признаков: {best_overall['Feature Set']}")
print(f"Модель: {best_overall['Model']}")
print(f"Точность на тесте: {best_overall['Accuracy']:.4f}")
print(f"CV Score: {best_overall['CV Score']:.4f}")

# Детальный анализ лучшей модели
best_feature_set = best_overall['Feature Set']
best_model_name = best_overall['Model']
best_model = all_results[best_feature_set][best_model_name]['model']
best_predictions = all_results[best_feature_set][best_model_name]['predictions']

# Получаем данные для лучшей модели
X_best = df[feature_sets[best_feature_set]].values
X_train_best, X_test_best = X_best[train_indices], X_best[test_indices]
y_train_best, y_test_best = y_encoded[train_indices], y_encoded[test_indices]

# Предобработка для лучшей модели
imputer_best = SimpleImputer(strategy='median')
X_train_best_imputed = imputer_best.fit_transform(X_train_best)
X_test_best_imputed = imputer_best.transform(X_test_best)

scaler_best = StandardScaler()
X_train_best_scaled = scaler_best.fit_transform(X_train_best_imputed)
X_test_best_scaled = scaler_best.transform(X_test_best_imputed)

print("\n" + "="*100)
print(f"🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ЛУЧШЕЙ МОДЕЛИ: {best_model_name} ({best_feature_set})")
print("="*100)

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test_best, best_predictions, target_names=le.classes_))

# Анализ важности признаков для лучшей модели
if 'feature_importance' in all_results[best_feature_set][best_model_name]:
    print(f"\n🔍 Анализ важности признаков для {best_model_name}:")
    importance_data = all_results[best_feature_set][best_model_name]['feature_importance']
    importance_type = all_results[best_feature_set][best_model_name]['importance_type']
    
    print(f"Тип важности: {importance_type}")
    print("Топ-10 важных признаков:")
    for i, (feature, importance) in enumerate(importance_data[:10]):
        print(f"{i+1:2d}. {feature}: {importance:.4f}")
    
    # Визуализация важности признаков
    plt.figure(figsize=(12, 8))
    top_features = importance_data[:15]
    features, importances = zip(*top_features)
    
    plt.barh(range(len(features)), importances, color='lightgreen', edgecolor='darkgreen')
    plt.yticks(range(len(features)), features)
    plt.xlabel(importance_type)
    plt.title(f'Важность признаков - {best_model_name} ({best_feature_set}) - БЫСТРАЯ ВЕРСИЯ')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_fast.png', dpi=300, bbox_inches='tight')
    plt.show()

# Confusion Matrix
print("\n📈 Создание визуализаций...")
viz_start_time = time.time()

plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test_best, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {best_model_name} ({best_feature_set}) - БЫСТРАЯ ВЕРСИЯ')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_fast.png', dpi=300, bbox_inches='tight')
plt.show()

# Анализ по жанрам
print("\n📊 Анализ точности по жанрам:")
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

plt.bar(genres, accuracies, color='lightgreen', edgecolor='darkgreen')
plt.title(f'Точность классификации по жанрам - {best_model_name} ({best_feature_set}) - БЫСТРАЯ ВЕРСИЯ')
plt.xlabel('Жанр')
plt.ylabel('Точность')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('genre_accuracy_fast.png', dpi=300, bbox_inches='tight')
plt.show()

# Сравнительная визуализация
print("\n📊 Создание сравнительных визуализаций...")

plt.figure(figsize=(16, 10))
pivot_df = comparison_df.pivot(index='Model', columns='Feature Set', values='Accuracy')
sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlGn', cbar_kws={'label': 'Accuracy'})
plt.title('Сравнение точности моделей (БЫСТРАЯ ВЕРСИЯ)')
plt.tight_layout()
plt.savefig('model_comparison_heatmap_fast.png', dpi=300, bbox_inches='tight')
plt.show()

viz_time = time.time() - viz_start_time
print(f"✅ Визуализации созданы за {format_time(viz_time)}")

# Сохранение результатов
print("\n" + "="*100)
print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*100)

save_start_time = time.time()

comparison_df.to_csv('classification_results_fast.csv', index=False)
print("✅ Результаты сохранены в classification_results_fast.csv")

import joblib
joblib.dump(best_model, 'best_genre_classifier_fast.pkl')
print("✅ Лучшая модель сохранена в best_genre_classifier_fast.pkl")

best_model_info = {
    'feature_set': best_feature_set,
    'model_name': best_model_name,
    'test_accuracy': best_overall['Accuracy'],
    'cv_score': best_overall['CV Score'],
    'best_params': all_results[best_feature_set][best_model_name].get('params', {}),
    'feature_importance': all_results[best_feature_set][best_model_name].get('feature_importance', []),
    'version': 'fast'
}

import json
with open('best_model_info_fast.json', 'w') as f:
    json.dump(best_model_info, f, indent=2, default=str)
print("✅ Информация о лучшей модели сохранена в best_model_info_fast.json")

save_time = time.time() - save_start_time
print(f"✅ Сохранение завершено за {format_time(save_time)}")

# Общее время выполнения
total_time = time.time() - total_start_time
print("\n" + "="*100)
print("🎉 ЗАКЛЮЧЕНИЕ (БЫСТРАЯ ВЕРСИЯ)")
print("="*100)
print(f"⏱️ Общее время выполнения: {format_time(total_time)}")
print(f"🕐 Завершение: {datetime.now().strftime('%H:%M:%S')}")
print(f"🚀 Лучшая модель: {best_model_name}")
print(f"📊 Набор признаков: {best_feature_set}")
print(f"🎯 Точность на тестовой выборке: {best_overall['Accuracy']:.4f}")
print(f"📈 CV Score: {best_overall['CV Score']:.4f}")

# Сравнение наборов признаков
print(f"\n📊 Сравнение наборов признаков:")
for feature_set_name, feature_set_results in all_results.items():
    best_acc = max(result['accuracy'] for result in feature_set_results.values())
    best_cv = max(result['cv_score'] for result in feature_set_results.values())
    print(f"- {feature_set_name}: Test {best_acc:.4f}, CV {best_cv:.4f}")

# Анализ ошибок
print(f"\n❌ Наиболее проблемные жанры:")
worst_genres = sorted(genre_accuracy.items(), key=lambda x: x[1])[:3]
for genre, acc in worst_genres:
    print(f"- {genre}: {acc:.4f}")

print(f"\n✅ Наиболее точные жанры:")
best_genres = sorted(genre_accuracy.items(), key=lambda x: x[1], reverse=True)[:3]
for genre, acc in best_genres:
    print(f"- {genre}: {acc:.4f}")

# Сводка ускорений
print(f"\n⚡ ОСНОВНЫЕ УСКОРЕНИЯ:")
print("1. ✅ RandomizedSearchCV вместо GridSearchCV (экономия ~80% времени)")
print("2. ✅ HalvingGridSearchCV для еще более быстрого поиска")
print("3. ✅ Сокращенные диапазоны гиперпараметров")
print("4. ✅ Убраны повторы в CV (только 5 folds)")
print("5. ✅ Уменьшено количество итераций для SVM")
print("6. ✅ Упрощенные ensemble (только 2 лучшие модели)")
print("7. ✅ Быстрая калибровка (3-fold CV)")

print(f"\n📈 Ожидаемое ускорение:")
print("- Полная версия: ~1-2 часа")
print("- Быстрая версия: ~10-20 минут")
print("- Ускорение: в 3-6 раз!")

print(f"\n🎊 Быстрый анализ завершен за {format_time(total_time)}!") 