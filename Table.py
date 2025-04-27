import cv2
import numpy as np
import pandas as pd
import os
from skimage import measure, morphology
from scipy import ndimage
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Tuple, Optional


class SporeAnalyzer:
    def __init__(self, input_dir: str, output_file: str):
        self.input_dir = input_dir
        self.output_file = output_file
        self.scale = None  # пиксели на микрометр
        self.min_spore_size = 5  # минимальный размер споры в пикселях
        self.max_spore_size = 100  # максимальный размер споры в пикселях
        self.setup_logging()

    def setup_logging(self):
        """Настройка логирования для отслеживания процесса"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='spore_analysis.log',
            filemode='w'
        )
        self.logger = logging.getLogger(__name__)

    def detect_scale_from_image(self, image: np.ndarray) -> Optional[float]:
        """Автоматическое определение масштаба по масштабной линейке на изображении"""
        # Здесь можно реализовать автоматическое определение масштаба
        # Например, поиск масштабной линейки и подсчет пикселей на микрометр
        # Пока возвращаем None, требуется ручная калибровка
        return None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Предварительная обработка изображения"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        return cleaned

    def filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Фильтрация контуров по размеру и форме"""
        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            # Фильтр по размеру
            if not (self.min_spore_size**2 < area < self.max_spore_size**2):
                continue

            # Фильтр по округлости (исключаем сильно вытянутые объекты)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.7:
                continue

            filtered.append(cnt)
        return filtered

    def measure_spore(self, contour: np.ndarray) -> Tuple[float, float]:
        """Измерение размеров споры"""
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]

        # Альтернативный метод - использование эллипса
        ellipse = cv2.fitEllipse(contour)
        major_axis = max(ellipse[1])
        minor_axis = min(ellipse[1])

        # Используем оба метода и выбираем лучший результат
        max_size = max(width, height, major_axis)
        min_size = min(width, height, minor_axis)

        if self.scale:
            max_size /= self.scale
            min_size /= self.scale

        return max_size, min_size

    def process_single_image(self, image_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Обработка одного изображения"""
        image_path = os.path.join(self.input_dir, image_file)
        image = cv2.imread(image_path)

        if image is None:
            self.logger.warning(f"Не удалось загрузить изображение: {image_file}")
            return pd.DataFrame(), pd.DataFrame()

        # Автоматическое определение масштаба при первом изображении
        if self.scale is None:
            self.scale = self.detect_scale_from_image(image)

        processed = self.preprocess_image(image)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = self.filter_contours(contours)

        individual_data = pd.DataFrame(columns=['Image', 'Max_Size', 'Min_Size'])
        max_sizes, min_sizes = [], []

        for cnt in filtered_contours:
            max_size, min_size = self.measure_spore(cnt)
            max_sizes.append(max_size)
            min_sizes.append(min_size)
            individual_data.loc[len(individual_data)] = {
                'Image': image_file,
                'Max_Size': max_size,
                'Min_Size': min_size
            }
