import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных
train = pd.read_csv('/Users/Skippi/Desktop/ML/keggle/train.csv')
test = pd.read_csv('/Users/Skippi/Desktop/ML/keggle/test.csv')

# 2. Очистка данных
# Заполнение пропусков в числовых признаках
num_cols = train.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    train[col] = train[col].fillna(train[col].mean())
    if col != 'SalePrice':  # Для теста заполняем все, кроме целевой переменной
        test[col] = test[col].fillna(test[col].mean())

# Заполнение пропусков в категориальных признаках
cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])

# Удаление столбцов с >30% пропусков
missing_ratio = train.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio > 0.3].index
train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

# 3. Преобразование целевой переменной
train['SalePrice'] = np.log1p(train['SalePrice'])  # Логарифмирование с учетом нулевых значений

# 4. Обработка признаков
# One-Hot Encoding для категориальных переменных
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Выравнивание столбцов в train и test
train, test = train.align(test, join='left', axis=1)
test = test.drop('SalePrice', axis=1, errors='ignore')  # Удаляем целевую переменную, если появилась

# 5. Подготовка данных
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Разделение на train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test)

# 6. Обучение модели
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# 7. Проверка на валидации
val_pred = model.predict(X_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f'Валидация среднеквадратичная ошибка при перекрестной проверке.: {rmse:.4f}')

# 8. Кросс-валидация
cv_scores = cross_val_score(
    model, 
    scaler.fit_transform(X), 
    y, 
    cv=5, 
    scoring='neg_root_mean_squared_error'
)
print(f'Кросс валидация RMSE: {-cv_scores.mean():.4f}')

# 9. Прогноз для тестовых данных
test_pred = model.predict(X_test_scaled)
test_pred = np.expm1(test_pred)  # Обратное преобразование из логарифма

# 10. Сохранение результатов
submission = pd.DataFrame({
    'Id': test['Id'], 
    'SalePrice': test_pred
})
submission.to_csv('/Users/Skippi/Desktop/ML/keggle/submission.csv', index=False)
print("Файл submission.csv успешно создан!")
