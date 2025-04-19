#запуск программы ,до этапа улучшения изображения
import os
import sys
import cv2
import numpy as np
import pandas as pd
from skimage import morphology
import argparse
from tqdm import tqdm

class SporeAnalyzer:
    def __init__(self):
        self.results = []
        self.individual_measurements = []
    
    def fix_path(self, path):
        try:
            return os.path.abspath(os.path.expanduser(path))
        except:
            return path

    def smart_imread(self, image_path):
        try:
            with open(image_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                return cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"Ошибка чтения файла {image_path}: {str(e)}")
            return None

    def enhance_image(self, img):
        if img is None:
            return None
            
        try:
            if img.dtype == np.uint8:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                img = clahe.apply(img)

            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            binary = binary.astype(bool)
            cleaned = morphology.remove_small_objects(binary, min_size=50)
            cleaned = morphology.remove_small_holes(cleaned, area_threshold=50)
            return cleaned.astype(np.uint8) * 255
        except Exception as e:
            print(f"Ошибка улучшения изображения: {str(e)}")
            return None
