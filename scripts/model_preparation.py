import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import joblib

#Загрузка предобработанных данных
df_train_scaled = pd.read_csv('train/df_train_scaled.csv')

#Выделяем признаки и целевую метку
X_train = df_train_scaled.drop('Precipitation', axis=1)
y_train = df_train_scaled['Precipitation']

#Создание и обучению модели
model = LinearRegression()
model.fit(X_train,y_train)

#Проверка модели на тренировочных данных
y_train_pred = model.predict(X_train)
mse_train = mean_squared_error(y_train,y_train_pred)
print(f'Среднеквадратичная ошибка на тренировочных данных:{mse_train}')

#Сохранение модели
if not os.path.exists('models'):
    os.mkdir('models')
joblib.dump(model, 'models/linear_regression_model.pkl')
print('Модель сохранена в соответствующую директорию')