import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from google.colab import files
import logging


class SporeAnalyzerColab:
    def __init__(self, input_dir: str, output_file: str, scale: Optional[float] = None):
        self.input_dir = os.path.abspath(input_dir)
        self.output_file = os.path.abspath(output_file)
        self.scale = scale
        self.min_spore_area = 10
        self.results = []
        self.individual_measurements = []
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def smart_imread(self, image_path: str) -> Optional[np.ndarray]:
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
        except:
            pass
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            return img
        except:
            return None

    def enhance_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            return cleaned
        except Exception as e:
            self.logger.warning(f"Ошибка при обработке изображения: {e}")
            return None

    def analyze_contour(self, contour: np.ndarray) -> Optional[Tuple[float, float]]:
        if len(contour) < 5:
            return None
        area = cv2.contourArea(contour)
        if area < self.min_spore_area:
            return None
        try:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            if self.scale:
                major_axis /= self.scale
                minor_axis /= self.scale
            return major_axis, minor_axis
        except:
            return None

    def process_image(self, image_path: str, filename: str):
        img = self.smart_imread(image_path)
        if img is None:
            return
        processed = self.enhance_image(img)
        if processed is None:
            return
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_sizes, min_sizes = [], []
        for contour in contours:
            result = self.analyze_contour(contour)
            if result:
                max_size, min_size = result
                self.individual_measurements.append({
                    'Image': filename,
                    'Max_Size': max_size,
                    'Min_Size': min_size
                })
                max_sizes.append(max_size)
                min_sizes.append(min_size)
        if max_sizes and min_sizes:
            self.results.append({
                'Image': filename,
                'Mean_Max_Size': f"{np.mean(max_sizes):.2f} ± {np.std(max_sizes):.2f}",
                'Median_Max_Size': np.median(max_sizes),
                'Mean_Min_Size': f"{np.mean(min_sizes):.2f} ± {np.std(min_sizes):.2f}",
                'Median_Min_Size': np.median(min_sizes),
                'Spore_Count': len(max_sizes)
            })

    def run(self):
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda f: self.process_image(os.path.join(self.input_dir, f), f), image_files),
                      total=len(image_files), desc="Processing"))

        if not self.individual_measurements:
            print("❌ Не удалось обработать ни одной споры. Проверьте качество изображений.")
            return

        individual_df = pd.DataFrame(self.individual_measurements)
        summary_df = pd.DataFrame(self.results)

        with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
            individual_df.to_excel(writer, sheet_name='Individual_measurements', index=False)
            summary_df.to_excel(writer, sheet_name='Summary_statistics', index=False)

        print("✅ Готово. Скачайте файл ниже:")
        files.download(self.output_file)


# COLAB UI
uploader = widgets.FileUpload(
    multiple=True,
    description="✔️ Загрузить изображения",
    style={'description_width': 'initial'},
    button_style='info'
)
scale_input = widgets.FloatText(
    description='🔬 Масштаб (пикс/мкм):',
    value=None,
    style={'description_width': 'initial'}
)
run_button = widgets.Button(
    description="📈 Запустить анализ",
    button_style='success',
    layout=widgets.Layout(width='200px')
)
status_output = widgets.Output()

ui_box = widgets.VBox([
    widgets.HTML("<h2 style='color:#4CAF50'>Анализ размеров спор грибов</h2>"),
    widgets.HTML("<p><b>Шаг 1:</b> Загрузите изображения</p>"),
    uploader,
    widgets.HTML("<p><b>Шаг 2:</b> Укажите масштаб (опционально)</p>"),
    scale_input,
    run_button,
    status_output
])

display(ui_box)


def on_run_clicked(b):
    with status_output:
        clear_output()
        print("⏳ Обработка файлов...")
        os.makedirs("/content/spore_images", exist_ok=True)
        for name, file_info in uploader.value.items():
            with open(f"/content/spore_images/{name}", 'wb') as f:
                f.write(file_info['content'])
        analyzer = SporeAnalyzerColab(
            input_dir="/content/spore_images",
            output_file="/content/spore_results.xlsx",
            scale=scale_input.value if scale_input.value else None
        )
        analyzer.run()

run_button.on_click(on_run_clicked)
