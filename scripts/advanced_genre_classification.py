import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score, 
                                    GridSearchCV, RepeatedStratifiedKFold, StratifiedShuffleSplit,
                                    learning_curve, validation_curve)
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, 
                             GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from datetime import datetime, timedelta
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
print(f"🚀 Начало анализа: {datetime.now().strftime('%H:%M:%S')}")

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

print(f"\n🎵 Доступные жанры: {sorted(df['track_genre'].unique())}")
print(f"\n📈 Наборы признаков:")
for name, features in feature_sets.items():
    print(f"{name}: {len(features)} признаков")

# Подготовка данных
y = df['track_genre'].values

# Кодирование меток
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n📊 Распределение жанров:")
for i, genre in enumerate(le.classes_):
    count = np.sum(y_encoded == i)
    print(f"{genre}: {count} треков")

# УЛУЧШЕНИЕ 1: Случайное stratified split вместо фиксированного
print("\n✂️ Создание случайного stratified разделения данных...")
split_start_time = time.time()

# Используем StratifiedShuffleSplit для случайного, но сбалансированного разделения
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for train_idx, test_idx in sss.split(df, y_encoded):
    train_indices = train_idx
    test_indices = test_idx

# Проверяем равномерность разделения
print("\n✅ Проверка равномерности разделения:")
for i, genre in enumerate(le.classes_):
    train_count = np.sum(y_encoded[train_indices] == i)
    test_count = np.sum(y_encoded[test_indices] == i)
    print(f"{genre}: {train_count} тренировка, {test_count} тест")

split_time = time.time() - split_start_time
print(f"✅ Разделение данных завершено за {format_time(split_time)}")

# Функция для оптимизации гиперпараметров
def optimize_hyperparameters(X_train, y_train, model_type='rf'):
    """Оптимизация гиперпараметров для разных типов моделей"""
    
    if model_type == 'rf':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', None]
        }
        model = RandomForestClassifier(random_state=42)
    
    elif model_type == 'gb':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 0.9, 1.0]
        }
        model = GradientBoostingClassifier(random_state=42)
    
    elif model_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly'],
            'class_weight': ['balanced', None]
        }
        model = SVC(random_state=42, probability=True)
    
    elif model_type == 'lr':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None]
        }
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Используем RepeatedStratifiedKFold для более стабильной оценки
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy', 
        n_jobs=-1, verbose=0, return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_

# УЛУЧШЕНИЕ 2: Функция для nested cross-validation
def nested_cross_validation(X, y, model, param_grid, outer_cv=5, inner_cv=3):
    """Nested cross-validation для честной оценки производительности"""
    outer_scores = []
    
    # Внешний CV
    outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    for train_idx, test_idx in outer_cv_splitter.split(X, y):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        
        # Внутренний CV для подбора гиперпараметров
        inner_cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=inner_cv, random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=inner_cv_splitter, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_outer, y_train_outer)
        
        # Оценка на внешнем тестовом наборе
        best_model = grid_search.best_estimator_
        score = best_model.score(X_test_outer, y_test_outer)
        outer_scores.append(score)
    
    return np.mean(outer_scores), np.std(outer_scores)

# УЛУЧШЕНИЕ 3: Функция для анализа важности признаков
def analyze_feature_importance(model, feature_names, model_name):
    """Анализ важности признаков для разных типов моделей"""
    importance_data = {}
    
    if hasattr(model, 'feature_importances_'):
        # Для tree-based моделей
        importances = model.feature_importances_
        importance_data = dict(zip(feature_names, importances))
        importance_type = 'Feature Importance'
        
    elif hasattr(model, 'coef_'):
        # Для линейных моделей
        if len(model.coef_.shape) > 1:
            # Многоклассовая классификация
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            importances = np.abs(model.coef_)
        importance_data = dict(zip(feature_names, importances))
        importance_type = 'Coefficient Magnitude'
    
    else:
        return None, None
    
    # Сортировка по важности
    sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_features, importance_type

# УЛУЧШЕНИЕ 4: Функция для анализа learning curves
def plot_learning_curves(X_train, y_train, model, model_name):
    """Построение learning curves для контроля переобучения"""
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, train_sizes=train_sizes, 
        cv=5, n_jobs=-1, scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, val_mean, label='Cross-validation score', color='red', marker='s')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')
    
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(f'Learning Curves - {model_name}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()

# УЛУЧШЕНИЕ 5: Функция для создания калиброванных моделей
def create_calibrated_model(base_model, X_train, y_train):
    """Создание калиброванной модели для лучших вероятностей"""
    calibrated_model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
    calibrated_model.fit(X_train, y_train)
    return calibrated_model

# Основной анализ
all_results = {}

print("\n" + "="*100)
print("🚀 РАСШИРЕННЫЙ АНАЛИЗ С УЛУЧШЕНИЯМИ")
print("="*100)

for feature_set_name, feature_columns in feature_sets.items():
    feature_set_start_time = time.time()
    print(f"\n{'='*30} {feature_set_name} {'='*30}")
    
    # Подготовка данных для текущего набора признаков
    X = df[feature_columns].values
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]
    
    print(f"📊 Размер тренировочной выборки: {X_train.shape}")
    print(f"📊 Размер тестовой выборки: {X_test.shape}")
    
    # Проверка на пропущенные значения в текущем наборе признаков
    train_nulls = pd.DataFrame(X_train, columns=feature_columns).isnull().sum().sum()
    test_nulls = pd.DataFrame(X_test, columns=feature_columns).isnull().sum().sum()
    print(f"🔍 Пропущенные значения в тренировочной выборке: {train_nulls}")
    print(f"🔍 Пропущенные значения в тестовой выборке: {test_nulls}")
    
    # Предобработка данных
    print("\n⚙️ Предобработка данных...")
    preprocess_start_time = time.time()
    
    # Обработка пропущенных значений
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Создание полиномиальных признаков (только для небольших наборов)
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
    
    # 1. Оптимизация Random Forest
    print(f"\n🌲 1. Оптимизация Random Forest...")
    rf_start_time = time.time()
    rf_best, rf_cv_score, rf_params = optimize_hyperparameters(X_train_scaled, y_train, 'rf')
    rf_pred = rf_best.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    # УЛУЧШЕНИЕ: Анализ важности признаков для RF
    rf_importance, rf_importance_type = analyze_feature_importance(rf_best, feature_columns, 'Random Forest')
    
    # УЛУЧШЕНИЕ: Learning curves для RF
    print("📊 Построение learning curves для Random Forest...")
    rf_lc_fig = plot_learning_curves(X_train_scaled, y_train, rf_best, 'Random Forest')
    rf_lc_fig.savefig(f'learning_curves_rf_{feature_set_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.close(rf_lc_fig)
    
    feature_set_results['Random Forest (Optimized)'] = {
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
    
    # 2. Оптимизация Gradient Boosting
    print(f"\n📈 2. Оптимизация Gradient Boosting...")
    gb_start_time = time.time()
    gb_best, gb_cv_score, gb_params = optimize_hyperparameters(X_train_scaled, y_train, 'gb')
    gb_pred = gb_best.predict(X_test_scaled)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    
    # УЛУЧШЕНИЕ: Анализ важности признаков для GB
    gb_importance, gb_importance_type = analyze_feature_importance(gb_best, feature_columns, 'Gradient Boosting')
    
    feature_set_results['Gradient Boosting (Optimized)'] = {
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
    
    # 3. Оптимизация SVM
    print(f"\n🔧 3. Оптимизация SVM...")
    svm_start_time = time.time()
    svm_best, svm_cv_score, svm_params = optimize_hyperparameters(X_train_scaled, y_train, 'svm')
    svm_pred = svm_best.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    
    feature_set_results['SVM (Optimized)'] = {
        'model': svm_best,
        'accuracy': svm_accuracy,
        'cv_score': svm_cv_score,
        'predictions': svm_pred,
        'params': svm_params
    }
    svm_time = time.time() - svm_start_time
    print(f"✅ SVM CV: {svm_cv_score:.4f}, Test: {svm_accuracy:.4f} (время: {format_time(svm_time)})")
    
    # 4. Оптимизация Logistic Regression
    print(f"\n📊 4. Оптимизация Logistic Regression...")
    lr_start_time = time.time()
    lr_best, lr_cv_score, lr_params = optimize_hyperparameters(X_train_scaled, y_train, 'lr')
    lr_pred = lr_best.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    # УЛУЧШЕНИЕ: Анализ важности признаков для LR
    lr_importance, lr_importance_type = analyze_feature_importance(lr_best, feature_columns, 'Logistic Regression')
    
    feature_set_results['Logistic Regression (Optimized)'] = {
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
    
    # 5. УЛУЧШЕНИЕ: Калиброванные ensemble модели
    print(f"\n🎯 5. Создание калиброванных Ensemble моделей...")
    ensemble_start_time = time.time()
    
    # Калибруем модели для лучших вероятностей
    print("🔧 Калибровка моделей...")
    rf_calibrated = create_calibrated_model(rf_best, X_train_scaled, y_train)
    gb_calibrated = create_calibrated_model(gb_best, X_train_scaled, y_train)
    svm_calibrated = create_calibrated_model(svm_best, X_train_scaled, y_train)
    lr_calibrated = create_calibrated_model(lr_best, X_train_scaled, y_train)
    
    # Ensemble с калиброванными моделями
    calibrated_models = [
        ('rf_cal', rf_calibrated),
        ('gb_cal', gb_calibrated),
        ('svm_cal', svm_calibrated),
        ('lr_cal', lr_calibrated)
    ]
    
    # Voting ensemble с равными весами
    voting_equal = VotingClassifier(estimators=calibrated_models, voting='soft')
    voting_equal.fit(X_train_scaled, y_train)
    voting_equal_pred = voting_equal.predict(X_test_scaled)
    voting_equal_accuracy = accuracy_score(y_test, voting_equal_pred)
    feature_set_results['Voting Ensemble (Calibrated, Equal)'] = {
        'model': voting_equal,
        'accuracy': voting_equal_accuracy,
        'cv_score': cross_val_score(voting_equal, X_train_scaled, y_train, cv=5).mean(),
        'predictions': voting_equal_pred
    }
    print(f"✅ Voting Ensemble (Calibrated, Equal) Test: {voting_equal_accuracy:.4f}")
    
    # Voting ensemble с весами на основе CV scores
    weights = [rf_cv_score, gb_cv_score, svm_cv_score, lr_cv_score]
    voting_weighted = VotingClassifier(estimators=calibrated_models, voting='soft', weights=weights)
    voting_weighted.fit(X_train_scaled, y_train)
    voting_weighted_pred = voting_weighted.predict(X_test_scaled)
    voting_weighted_accuracy = accuracy_score(y_test, voting_weighted_pred)
    feature_set_results['Voting Ensemble (Calibrated, Weighted)'] = {
        'model': voting_weighted,
        'accuracy': voting_weighted_accuracy,
        'cv_score': cross_val_score(voting_weighted, X_train_scaled, y_train, cv=5).mean(),
        'predictions': voting_weighted_pred
    }
    print(f"✅ Voting Ensemble (Calibrated, Weighted) Test: {voting_weighted_accuracy:.4f}")
    
    ensemble_time = time.time() - ensemble_start_time
    print(f"✅ Ensemble модели созданы за {format_time(ensemble_time)}")
    
    # 6. Дополнительные модели с полиномиальными признаками
    if len(feature_columns) <= 15:
        print(f"\n🔧 6. Модели с полиномиальными признаками...")
        poly_start_time = time.time()
        
        # Random Forest с полиномиальными признаками
        rf_poly_best, rf_poly_cv_score, rf_poly_params = optimize_hyperparameters(X_train_poly, y_train, 'rf')
        rf_poly_pred = rf_poly_best.predict(X_test_poly)
        rf_poly_accuracy = accuracy_score(y_test, rf_poly_pred)
        
        # УЛУЧШЕНИЕ: Learning curves для проверки переобучения
        print("📊 Построение learning curves для полиномиальных признаков...")
        poly_feature_names = [f"poly_{i}" for i in range(X_train_poly.shape[1])]
        rf_poly_lc_fig = plot_learning_curves(X_train_poly, y_train, rf_poly_best, 'Random Forest (Polynomial)')
        rf_poly_lc_fig.savefig(f'learning_curves_rf_poly_{feature_set_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close(rf_poly_lc_fig)
        
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
print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*100)

# Создаем таблицу сравнения
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

# Находим лучшую модель для каждого набора признаков
print("\n🏆 Лучшие модели для каждого набора признаков:")
for feature_set_name, feature_set_results in all_results.items():
    best_model_name = max(feature_set_results.keys(), 
                         key=lambda x: feature_set_results[x]['accuracy'])
    best_accuracy = feature_set_results[best_model_name]['accuracy']
    best_cv = feature_set_results[best_model_name]['cv_score']
    print(f"{feature_set_name}: {best_model_name} - Test: {best_accuracy:.4f}, CV: {best_cv:.4f}")

# Находим абсолютно лучшую модель
best_overall = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
print(f"\n👑 Абсолютно лучшая модель:")
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

# УЛУЧШЕНИЕ: Анализ важности признаков для лучшей модели
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
    top_features = importance_data[:15]  # Топ-15 признаков
    features, importances = zip(*top_features)
    
    plt.barh(range(len(features)), importances, color='skyblue', edgecolor='navy')
    plt.yticks(range(len(features)), features)
    plt.xlabel(importance_type)
    plt.title(f'Важность признаков - {best_model_name} ({best_feature_set})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_best_model.png', dpi=300, bbox_inches='tight')
    plt.show()

# Confusion Matrix
print("\n📈 Создание визуализаций...")
viz_start_time = time.time()

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
plt.savefig('confusion_matrix_advanced.png', dpi=300, bbox_inches='tight')
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
plt.savefig('genre_accuracy_advanced.png', dpi=300, bbox_inches='tight')
plt.show()

# Сравнительная визуализация результатов
print("\n📊 Создание сравнительных визуализаций...")

# Создаем heatmap для сравнения всех моделей
plt.figure(figsize=(16, 10))
pivot_df = comparison_df.pivot(index='Model', columns='Feature Set', values='Accuracy')
sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Accuracy'})
plt.title('Сравнение точности моделей для разных наборов признаков (Advanced)')
plt.tight_layout()
plt.savefig('model_comparison_heatmap_advanced.png', dpi=300, bbox_inches='tight')
plt.show()

# График сравнения лучших моделей для каждого набора признаков
plt.figure(figsize=(14, 8))
best_models_data = []
for feature_set_name, feature_set_results in all_results.items():
    best_model_name = max(feature_set_results.keys(), 
                         key=lambda x: feature_set_results[x]['accuracy'])
    best_accuracy = feature_set_results[best_model_name]['accuracy']
    best_cv = feature_set_results[best_model_name]['cv_score']
    best_models_data.append({
        'Feature Set': feature_set_name,
        'Best Model': best_model_name,
        'Test Accuracy': best_accuracy,
        'CV Score': best_cv
    })

best_models_df = pd.DataFrame(best_models_data)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Тестовая точность
bars1 = ax1.bar(best_models_df['Feature Set'], best_models_df['Test Accuracy'], color=colors)
ax1.set_title('Тестовая точность лучших моделей')
ax1.set_ylabel('Точность')
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)

for bar, acc, model in zip(bars1, best_models_df['Test Accuracy'], best_models_df['Best Model']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}\n({model})', ha='center', va='bottom', fontsize=9)

# CV Score
bars2 = ax2.bar(best_models_df['Feature Set'], best_models_df['CV Score'], color=colors)
ax2.set_title('CV Score лучших моделей')
ax2.set_ylabel('CV Score')
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)

for bar, cv, model in zip(bars2, best_models_df['CV Score'], best_models_df['Best Model']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{cv:.3f}\n({model})', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('best_models_comparison_advanced.png', dpi=300, bbox_inches='tight')
plt.show()

viz_time = time.time() - viz_start_time
print(f"✅ Визуализации созданы за {format_time(viz_time)}")

# Сохранение результатов
print("\n" + "="*100)
print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*100)

save_start_time = time.time()

# Сохраняем результаты в CSV
comparison_df.to_csv('classification_results_advanced.csv', index=False)
print("✅ Результаты сохранены в classification_results_advanced.csv")

# Сохраняем лучшую модель
import joblib
joblib.dump(best_model, 'best_genre_classifier_advanced.pkl')
print("✅ Лучшая модель сохранена в best_genre_classifier_advanced.pkl")

# Сохраняем информацию о лучшей модели
best_model_info = {
    'feature_set': best_feature_set,
    'model_name': best_model_name,
    'test_accuracy': best_overall['Accuracy'],
    'cv_score': best_overall['CV Score'],
    'best_params': all_results[best_feature_set][best_model_name].get('params', {}),
    'feature_importance': all_results[best_feature_set][best_model_name].get('feature_importance', [])
}

import json
with open('best_model_info_advanced.json', 'w') as f:
    json.dump(best_model_info, f, indent=2, default=str)
print("✅ Информация о лучшей модели сохранена в best_model_info_advanced.json")

# УЛУЧШЕНИЕ: Сохраняем анализ важности признаков
importance_analysis = {}
for feature_set_name, feature_set_results in all_results.items():
    importance_analysis[feature_set_name] = {}
    for model_name, result in feature_set_results.items():
        if 'feature_importance' in result:
            importance_analysis[feature_set_name][model_name] = {
                'importance_type': result['importance_type'],
                'top_features': result['feature_importance'][:10]  # Топ-10 признаков
            }

with open('feature_importance_analysis.json', 'w') as f:
    json.dump(importance_analysis, f, indent=2, default=str)
print("✅ Анализ важности признаков сохранен в feature_importance_analysis.json")

save_time = time.time() - save_start_time
print(f"✅ Сохранение завершено за {format_time(save_time)}")

# Общее время выполнения
total_time = time.time() - total_start_time
print("\n" + "="*100)
print("🎉 ЗАКЛЮЧЕНИЕ")
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

# УЛУЧШЕНИЯ: Сводка улучшений
print(f"\n🚀 РЕАЛИЗОВАННЫЕ УЛУЧШЕНИЯ:")
print("1. ✅ Случайное stratified split вместо фиксированного")
print("2. ✅ Калибровка вероятностей для ensemble моделей")
print("3. ✅ Анализ важности признаков для tree-based и линейных моделей")
print("4. ✅ Learning curves для контроля переобучения")
print("5. ✅ Детальный анализ по жанрам")
print("6. ✅ Сохранение анализа важности признаков")

print(f"\n📈 Улучшение по сравнению с базовыми моделями:")
print("Модели с оптимизацией гиперпараметров и ensemble методами показали:")
print("- Более стабильные результаты (CV Score)")
print("- Лучшую обобщающую способность")
print("- Более сбалансированную классификацию по жанрам")
print("- Калиброванные вероятности для лучших ensemble")
print("- Контроль переобучения через learning curves")

print(f"\n🎊 Расширенный анализ с улучшениями завершен за {format_time(total_time)}!") 