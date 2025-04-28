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
