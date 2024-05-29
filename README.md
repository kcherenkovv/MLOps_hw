**MLOPS**
**Практика 1**
Для запуска кода: 0. выполните команду `git clone` для данного проекта и перейдите в директорию проекта `cd MLOps_hw`
1. Установите зависимости из файла requirements.txt, который находится в корне проекта - pip install -r requirements.txt
2. Перейдите в директорию lab1 - cd lab1
3. Далее запустите скрипт pipeline.sh командой `./pipeline.sh предварительно убедившись, что файл является исполняемым (ls -la && chmod u+x job.sh)

Запуск отдельных скриптов из первоначальной директории до шага 0:
`python3 scripts/data_creation.py` - генерирует данные и записывает в соответствующие файлы (тренировочные и тестовые/валидационные)
`python3 scripts/model_preprocessing.py` - подготавливает данные для дальнейшей обработки
`python3 scripts/model_preparation.py` - создается и обучается на тренировочных данных модель, после чего записывается в отдельный файл
`python3 scripts/model_testing.py` - проверяет метрики модели на тестовых данных

**Практика №3**
# Распознавание автомобильных номеров
### Описание работы
- Находит на изображении фрагменты с автомобильными номерами и показывает их
### Используемые библиотеки и модели
- Для поиска номера использует модель из Hugging Face [keremberke/yolov5m-license-plate]
- Интерфейс пользователя - библиотека [streamlit]

### Как запустить WEB streamlit
-Перейти в директорию `lib3` где находится файл `docker-compose`. Выполнить в данной диреткории команду в терминале `docker-compose up`

Пример фотографий автомобилей можно взять в папке `templates`
### Как это выглядит
![screen1](https://github.com/kcherenkovv/MLOps_hw/blob/main/lib3_1/example/scr1.jpg)

# **Практика4**
# Ссылка на [Google drive](https://drive.google.com/drive/folders/1mshc98OEjB9_lGSBtbdzVHKMIibFXhv-?usp=sharing)
# **Последовательность команд:**

- pip install pandas numpy catboost - установка пакетных модулей
- cd lib4
- python|python3 data_get.py - модификация данных #0 - получение данных датасета titanic
- python|python3 data_null.py - модификация данных #1 - заполнение колонки Age средним значением
- python|python3 data_onecod.py - модификация данных #2 - применение one-hot-encoding для колонки Sex
- cd ..
- dvc add lib4/titanic_train.csv
- git commit -a -m 'Your commit'
- git push
- dvc push -r your_text - после каждой модификации данных - отправка на google drive через dvc
# Пример файлов в Google drive в папке titanic:
![screen1](https://github.com/kcherenkovv/MLOps_hw/blob/main/lib4/screen/screendrive.png)

# **Практика 5**
Применяем средства автоматизации тестирования python для автоматического тестирования качества работы модели машинного обучения на различных датасетах.
Для применения средств автоматизации необходимо запустить все ячейки ноутбука `script.ipynb`
![screen1](https://github.com/kcherenkovv/MLOps_hw/blob/main/lib5/image/Снимок%20экрана%202024-05-29%20152836.png)
