import os
import cv2
import numpy as np
from PIL import Image

class FaceTrainer:
    def __init__(self, dataset_dir="data", model_dir="models"):
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def train_model(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        image_paths = [os.path.join(self.dataset_dir, f) for f in os.listdir(self.dataset_dir) if f.endswith(".jpg")]

        if not image_paths:
            print("[ERROR] No image data found in 'data/' folder to train. Enroll users first.")
            return

        face_samples = []
        ids = []

        print("[INFO] Training faces. This might take a few seconds...")
        for path in image_paths:
            pil_img = Image.open(path).convert('L')
            img_numpy = np.array(pil_img, 'uint8')
            user_id = int(os.path.split(path)[-1].split(".")[1])

            face_samples.append(img_numpy)
            ids.append(user_id)

        recognizer.train(face_samples, np.array(ids))
        save_path = os.path.join(self.model_dir, "face_trainer.yml")
        recognizer.write(save_path)
        print(f"[SUCCESS] Model saved to '{save_path}' with {len(np.unique(ids))} registered users.")
