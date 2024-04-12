import pandas as pd 
import numpy as np
import os

#Проверка наличия и создания папок
if not os.path.exists('train'):
    os.mkdir('train')
if not os.path.exists('test'):
    os.mkdir('test')

#Задания сида
np.random.seed(42)

#Генерация значения температуры и шума
noise = np.random.uniform(low=-2, high=2, size = 1000)
loc_temp, scale_temp = 15,7
temperature = np.random.normal(loc=loc_temp, scale=scale_temp, size= 1000)
temperature = temperature+noise
temperature = np.round(np.clip(temperature, -5,35), 1)

#Генерация давления в мм.рт.ст
pressure = np.random.normal(loc= 760, scale = 15, size = 1000).astype(int)

#Генерация даты
data = pd.date_range(start='2021-01-01', periods= 1000)

#Создание целевой переменной: количество осадков в мм
precipitation = np.random.normal(loc= 5, scale = 3, size = 1000)
precipitation_cliped = np.round(np.clip(precipitation, 0, 30), 2)

#Создание DataFrame
df = pd.DataFrame({
    'Date': data,
    'Temperature': temperature,
    'Pressure': pressure,
    'Precipitation': precipitation_cliped
})

#Разделение данных на тренировочную и тестовую
train_size = int(0.7*len(df))
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

#Сохранение в различных директориях
df_train.to_csv('train/df_train.csv', index=False)
df_test.to_csv('test/df_test.csv', index=False)
print('Тренировочная и тестовая выборка сгенерированна и хранится в соотвествующих папках в формате .csv')
