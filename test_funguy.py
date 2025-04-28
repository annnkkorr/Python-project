import pytest
import numpy as np
import os
import shutil
import tempfile
import cv2

from funguy import SporeAnalyzer 

@pytest.fixture
def temp_image_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def create_test_image(temp_image_dir):
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (255), -1)
    img_path = os.path.join(temp_image_dir, "test_spore.png")
    cv2.imwrite(img_path, img)
    return img_path

def test_smart_imread(create_test_image):
    analyzer = SporeAnalyzer(input_dir=os.path.dirname(create_test_image), output_file="dummy.xlsx")
    img = analyzer.smart_imread(create_test_image)
    assert img is not None
    assert img.shape == (100, 100)

def test_enhance_image(create_test_image):
    analyzer = SporeAnalyzer(input_dir=os.path.dirname(create_test_image), output_file="dummy.xlsx")
    img = analyzer.smart_imread(create_test_image)
    enhanced = analyzer.enhance_image(img)
    assert enhanced is not None
    assert enhanced.shape == (100, 100)

def test_analyze_contour(create_test_image):
    analyzer = SporeAnalyzer(input_dir=os.path.dirname(create_test_image), output_file="dummy.xlsx")
    img = analyzer.smart_imread(create_test_image)
    processed = analyzer.enhance_image(img)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        result = analyzer.analyze_contour(contour)
        if result:
            major, minor = result
            assert major > 0
            assert minor > 0

def test_process_image(create_test_image):
    analyzer = SporeAnalyzer(input_dir=os.path.dirname(create_test_image), output_file="dummy.xlsx")
    analyzer.process_image(create_test_image, "test_spore.png")
    assert len(analyzer.individual_measurements) > 0
    assert len(analyzer.results) > 0

def test_run(temp_image_dir, create_test_image):
    analyzer = SporeAnalyzer(input_dir=temp_image_dir, output_file=os.path.join(temp_image_dir, "results.xlsx"))
    analyzer.run()
    assert os.path.exists(analyzer.output_file)
