import pandas as pd

#Загружаем датасет
titanic_train = pd.read_csv('titanic_train.csv', sep = ',')

#Работаем с пропусками. Заменяем их на средние значение в признаке "Age"
mean_age = titanic_train['Age'].mean()

titanic_train['Age'] = titanic_train['Age'].apply(lambda x:mean_age if pd.isna(x) else x)

titanic_train.to_csv('titanic_train.csv', index=False)
