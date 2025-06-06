# FunGuy

## Описание

`FunGuy` - это программа на Python, предназначенная для автоматического анализа размеров спор на изображениях. Она обрабатывает изображения из указанной директории, определяет контуры объектов, соответствующих спорам, измеряет их размеры (длину большой и малой осей эллипса, аппроксимирующего спору) и сохраняет результаты анализа в Excel-файл. Программа использует библиотеки `cv2`, `numpy`, `pandas`, `tqdm` и `concurrent.futures` для эффективной обработки изображений и параллелизации вычислений.

## Ключевые функции
•   **`SporeAnalyzer(input_dir, output_file, scale=None)`**: Конструктор класса. Инициализирует анализатор с указанием директории входных изображений, имени выходного файла и масштаба (пиксели на микрометр) для преобразования размеров.

•   **`smart_imread(image_path)`**: Функция для чтения изображений, поддерживающая чтение файлов как с использованием `cv2.imread`, так и `cv2.imdecode`, что позволяет обрабатывать изображения с русскими символами в пути.

•   **`enhance_image(img)`**: Функция для улучшения качества изображения. Использует CLAHE, гауссово размытие, адаптивную пороговую обработку и морфологические операции для выделения контуров спор.

•   **`analyze_contour(contour)`**: Функция для анализа контура. Измеряет площадь и аппроксимирует контур эллипсом, вычисляя длины большой и малой осей. Если задан масштаб, размеры переводятся в микрометры.

•   **`process_image(image_path, filename)`**: Функция для обработки одного изображения. Читает изображение, улучшает его качество, находит контуры и анализирует их. Результаты измерений добавляются в списки `self.results` (статистика по изображению) и `self.individual_measurements` (данные по каждой споре).

•   **`run()`**: Основная функция, выполняющая анализ всех изображений в указанной директории. Использует многопоточность для ускорения обработки. Результаты сохраняются в Excel-файл в двух листах: `Individual_measurements` (данные по каждой споре) и `Summary_statistics` (сводная статистика по каждому изображению).

•   **`main()`**: Функция, обеспечивающая взаимодействие с пользователем. Запрашивает путь к папке с изображениями, имя выходного файла и масштаб, затем запускает анализ.

## Установка
  git clone https://github.com/annnkkorr/Python-project.git
  
  cd Python-project
  
   **Версия google colab**:
https://github.com/annnkkorr/Python-project/blob/main/funguy_colab.py
## Ввод
Программа принимает следующие данные:

•   **Входная директория**: Путь к папке, содержащей изображения для анализа (форматы: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`).

•   **Выходной файл**: Путь и имя Excel-файла (`.xlsx`), в который будут сохранены результаты анализа.

•   **Масштаб (необязательно)**: Значение, представляющее количество пикселей на микрометр. Если указано, размеры спор будут переведены в микрометры. Если не указано, размеры будут представлены в пикселях.
Все данные вводятся через командную строку при запуске программы.


## Вывод
Программа создает Excel-файл, содержащий два листа:

•   **`Individual_measurements`**: Таблица с данными по каждой обнаруженной споре.

Содержит следующие столбцы:

    •   `Image`: Имя файла изображения, из которого была выделена спора.
    
    •   `Max_Size`: Длина большой оси эллипса, аппроксимирующего спору (в пикселях или микрометрах, в зависимости от указанного масштаба).
    
    •   `Min_Size`: Длина малой оси эллипса, аппроксимирующего спору (в пикселях или микрометрах, в зависимости от указанного масштаба).

    
•   **`Summary_statistics`**: Таблица со сводной статистикой по каждому изображению. Содержит следующие столбцы:

    •   `Image`: Имя файла изображения.
    
    •   `Mean_Max_Size`: Среднее значение длины большой оси эллипса для спор на данном изображении с указанием стандартного отклонения (например, "10.50 ± 1.20").
    
    •   `Median_Max_Size`: Медианное значение длины большой оси эллипса для спор на данном изображении.
    
    •   `Mean_Min_Size`: Среднее значение длины малой оси эллипса для спор на данном изображении с указанием стандартного отклонения (например, "5.25 ± 0.60").
    
    •   `Median_Min_Size`: Медианное значение длины малой оси эллипса для спор на данном изображении.
    
    •   `Spore_Count`: Количество спор, обнаруженных на изображении.
    
## Авторы 
Учебный проект по предмету "Программирование на python". 
Коробейникова Анна, 
Уразматова Полина, 
Юхина Варвара. 
