👁️ Monocular Depth Estimation using MiDaS

This project implements real-time monocular depth estimation using the MiDaS (Monocular Depth Estimation for Single Image) neural network architecture. It takes a single image stream (from a webcam) and generates a relative depth map, where brightness is inversely proportional to distance (brighter colors indicate closer objects).

This serves as the crucial depth perception component for subsequent 3D vision and robotics projects (e.g., Depth + Segmentation Fusion).

---

## 🚀 Key Features

* **Real-Time Processing:** Processes live video feed from a webcam (or video file).
* **MiDaS Integration:** Utilizes pre-trained MiDaS models (specifically `MiDaS_small`) from PyTorch Hub for efficient inference.
* **Relative Depth Map:** Outputs a continuous, high-resolution depth map visualized in real-time.
* **GPU Acceleration (Optional):** Supports CUDA acceleration for faster inference on compatible hardware.

---

## 🛠️ Prerequisites

* Python 3.8+
* NVIDIA GPU (Recommended for real-time performance)

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/depth-estimation-midas.git](https://github.com/YourUsername/depth-estimation-midas.git)
    cd depth-estimation-midas
    ```

2.  **Install Required Libraries:**
    It is highly recommended to use a virtual environment.
    ```bash
    pip install torch torchvision
    pip install numpy opencv-python tqdm
    ```

3.  **MiDaS Weights:** The model weights are automatically downloaded by PyTorch Hub upon the first run.

---

## 🏃 Getting Started

### Run the Depth Estimation Script

Ensure your webcam is connected and not in use by another application.

``bash
python depth_estimation.py

Outputs The script will open two windows:
Original Frame: The raw feed from your camera. 
Depth Map: A grayscale (or colormapped) image representing the relative depth. 

🧠 MiDaS Model Details This project uses the MiDaS architecture developed by Intel/ISL. 

Model Description Primary Use MiDaS_small A highly optimized model for speed and lightweight deployment.Real-time video processing.

The relative depth map $\mathbf{D}_{\text{rel}}$ is calculated using the raw prediction $P$ and is normalized to the range [0, 1] for visualization.$$\mathbf{D}_{\text{rel}} = \frac{P - \min(P)}{\max(P) - \min(P)}$$

📜 License
This project is licensed under the MIT License.
