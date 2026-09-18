# VisionAttend: Real-Time Face Recognition Attendance System

## 1. Overview
VisionAttend is an end-to-end computer vision application that replaces manual roll calls and physical fingerprint scanners. Using Haar Cascades for face localization and the Local Binary Patterns Histograms (LBPH) algorithm for facial texture identification, the system captures real-time video frames, authenticates individuals against stored biometric descriptors, and records structured attendance logs.

## 2. Features
- **Autonomous Face Detection:** Robust Haar Cascade detection handling various indoor lighting environments.
- **Local Texture Feature Representation:** LBPH-based facial classification allowing rapid, low-footprint local inference.
- **Automated Anti-Proxy Logging:** Single daily registration logic per user prevented by date-partitioned CSV tracking.
- **On-Screen Biometric Visuals:** Visual bounding indicators: Green for recognized entities, Red for unknown personnel.

## 3. Technologies & Tools Used
- **Language:** Python 3.9+
- **Vision Engine:** OpenCV (`opencv-python`, `opencv-contrib-python`)
- **Scientific Computing:** NumPy, Pillow
- **Data Persistence:** Pandas (Structured CSV formatting)
- **Quality Assurance:** Pytest

## 4. Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/anshrai962013-oss/computer-vision.git](https://github.com/anshrai962013-oss/computer-vision.git)
   cd computer-vision
