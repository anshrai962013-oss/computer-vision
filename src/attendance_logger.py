import os
from datetime import datetime
import pandas as pd

class AttendanceLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def mark_attendance(self, user_id, name):
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(self.log_dir, f"attendance_{today}.csv")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            df = pd.DataFrame(columns=["User_ID", "Name", "Timestamp"])

        if user_id not in df["User_ID"].values:
            now_time = datetime.now().strftime("%H:%M:%S")
            new_record = pd.DataFrame([{"User_ID": user_id, "Name": name, "Timestamp": now_time}])
            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(file_path, index=False)
            print(f"[ATTENDANCE] Checked in: {name} (ID: {user_id}) at {now_time}")
