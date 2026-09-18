import sys
import cv2
from src.camera_stream import CameraStream
from src.detector import FaceDetector
from src.dataset_builder import DatasetBuilder
from src.face_trainer import FaceTrainer
from src.attendance_logger import AttendanceLogger

USERS = {1: "Alice Johnson", 2: "Bob Smith", 3: "Charlie Brown"}

def run_recognition():
    cam = CameraStream()
    detector = FaceDetector()
    logger = AttendanceLogger()
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    try:
        recognizer.read("models/face_trainer.yml")
    except Exception:
        print("[ERROR] Trained model file not found! Please run '--train' first.")
        return

    cam.start()
    print("[INFO] Scanner active. Looking for registered faces. Press 'q' to exit.")

    while True:
        ret, frame = cam.read_frame()
        if not ret:
            break

        gray, faces = detector.detect_faces(frame)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            user_id, confidence = recognizer.predict(roi_gray)

            # LBPH distance: lower score means higher match accuracy (0 is exact match)
            if confidence < 65:
                name = USERS.get(user_id, f"User_{user_id}")
                conf_str = f"{round(100 - confidence)}%"
                color = (0, 255, 0)
                logger.mark_attendance(user_id, name)
            else:
                name = "Unknown"
                conf_str = f"{round(100 - confidence)}%"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({conf_str})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Real-Time Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--enroll":
            uid = int(input("Enter Numeric User ID: "))
            DatasetBuilder().capture_user_samples(uid)
        elif mode == "--train":
            FaceTrainer().train_model()
        elif mode == "--scan":
            run_recognition()
        else:
            print("Unknown command. Usage: python main.py [--enroll | --train | --scan]")
    else:
        run_recognition()
