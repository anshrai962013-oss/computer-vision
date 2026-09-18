import cv2
import numpy as np
from src.detector import FaceDetector

def test_detector_initialization():
    detector = FaceDetector()
    assert detector.face_cascade is not None
    assert not detector.face_cascade.empty()

def test_detect_faces_blank_image():
    detector = FaceDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    gray, faces = detector.detect_faces(dummy_frame)
    assert gray.shape == (480, 640)
    assert len(faces) == 0
  
