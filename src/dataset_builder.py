import os
import cv2
from src.camera_stream import CameraStream
from src.detector import FaceDetector

class DatasetBuilder:
    def __init__(self, dataset_dir="data"):
        self.dataset_dir = dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)

    def capture_user_samples(self, user_id, sample_count=30):
        cam = CameraStream()
        detector = FaceDetector()
        cam.start()

        print(f"[INFO] Collecting {sample_count} face samples for User ID {user_id}. Look at the camera...")
        count = 0

        while count < sample_count:
            ret, frame = cam.read_frame()
            if not ret:
                break

            gray, faces = detector.detect_faces(frame)
            for (x, y, w, h) in faces:
                count += 1
                roi_gray = gray[y:y+h, x:x+w]
                file_path = os.path.join(self.dataset_dir, f"User.{user_id}.{count}.jpg")
                cv2.imwrite(file_path, roi_gray)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Captured: {count}/{sample_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                break

            cv2.imshow("Dataset Capture", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        print(f"[SUCCESS] Saved {count} samples for User ID {user_id}.")
        cam.release()
