Ссылка на google disc

Ссылка на основной репозиторий

Все правки выполнялись из директории scripts

Последовательность команд:

git clone - клонирование репозитория
pip install pandas numpy catboost - установка зависимостей
cd scripts - переход в директорию с python-скриптами
python|python3 get_data.py - модификация данных #0 - получение данных датасета titanic
python|python3 delete_data.py - если данные нужно удалить (опционально)
python|python3 modificator_1.py - модификация данных #1 - заполнение колонки Age средним значением
python|python3 modificator_2.py - модификация данных #2 - применение one-hot-encoding для колонки Sex
dvc push - после каждой модификации данных - отправка на google disk через dvc
Пример файлов в хранилище google disk: