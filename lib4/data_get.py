from catboost.datasets import titanic

#Загружает датасет
titanic_train, titanic_test = titanic()


titanic_train = titanic_train['Pclass','Sex','Age']

#Сохраняем датасет в формате csv
titanic_train.to_csv('titanic_train.csv', index=False)
