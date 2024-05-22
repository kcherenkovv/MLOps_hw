Ссылка на [Google drive](https://drive.google.com/drive/folders/1mshc98OEjB9_lGSBtbdzVHKMIibFXhv-?usp=sharing)
**Последовательность команд:**

-pip install pandas numpy catboost - установка пакетных модулей
-cd lib4
-python|python3 data_get.py - модификация данных #0 - получение данных датасета titanic
-python|python3 data_null.py - модификация данных #1 - заполнение колонки Age средним значением
-python|python3 data_onecod.py - модификация данных #2 - применение one-hot-encoding для колонки Sex
-cd ..
-dvc add lib4/titanic_train.csv
-git commit -a -m 'Your commit'
-git push
-dvc push -r your_text - после каждой модификации данных - отправка на google drive через dvc
Пример файлов в Google drive в папке titanic: