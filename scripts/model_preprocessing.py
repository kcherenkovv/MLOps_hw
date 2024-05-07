import pandas as pd 
from sklearn.preprocessing import StandardScaler
import os

#Загрузка данных из файла
df_train = pd.read_csv('train/df_train.csv')
df_test = pd.read_csv('test/df_test.csv')

#Выделяем признаки, целевые метки и удялем лишние
X_train = df_train.drop(['Date', 'Precipitation'], axis=1)
X_test = df_test.drop(['Date', 'Precipitation'], axis=1)
y_train = df_train['Precipitation']
y_test = df_test['Precipitation']

#Выполним стандартизацию данных
standard = StandardScaler()
X_train_scaled = standard.fit_transform(X_train)
X_test_scaled = standard.transform(X_test)

# Сохранение предобработанных данных
df_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
df_train_scaled['Precipitation'] = df_train['Precipitation']
df_train_scaled.to_csv('train/df_train_scaled.csv')

df_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
df_test_scaled['Precipitation'] = df_test['Precipitation']
df_test_scaled.to_csv('test/df_test_scaled.csv')

print('Предобработанные данные сохранены в соответствующих папках')