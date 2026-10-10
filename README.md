# Agricultural Enterprise MLOps: Tomato Disease Detection (YOLOv8)

This is an object detection project developed using the YOLOv8 architecture. Its primary goal is to detect diseases in tomato leaves or plants in the agricultural sector with high accuracy and in real-time. It supports the ONNX model format to run on edge devices like Raspberry Pi, making it suitable for production (Edge AI).

## Project Structure

- `train.py`: Trains the YOLOv8 model using the dataset. Data augmentation and device detection (GPU/CPU) are handled automatically during training.
- `export_rsp.py`: Exports and simplifies the best trained PyTorch model (`best.pt`) to the ONNX format (`.onnx`) for faster inference on edge devices.
- `test_onnx_ytb.py`: Performs real-time tomato disease detection testing directly from a YouTube video stream using the exported ONNX model.

## Installation

Follow these steps to run the project on your local machine:

1. Clone this repository to your computer.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

*(Note: If you are using a virtual environment, make sure to activate it before running `pip install`.)*

## Usage

### 1. Model Training
To train the model with your own dataset or to restart the existing training, run the `train.py` file:
```bash
python train.py
```
*This script also performs validation after training and saves the best model.*

### 2. Exporting for Edge Devices (Raspberry Pi)
To convert your model to ONNX format for use on edge devices (Raspberry Pi, etc.):
```bash
python export_rsp.py
```

### 3. Live Testing via YouTube
To perform real-time inference via a YouTube stream with the trained or converted model:
```bash
python test_onnx_ytb.py
```
*(You can press the `q` key on the test screen to stop the stream.)*

## Model Performance Scores (Metrics)

The performance metrics obtained from the test dataset after training are as follows:

| Class | Images | Instances | Box(P) | R | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **all** | 778 | 2788 | **0.947** | **0.911** | **0.969** | **0.863** |

* **Precision (P):** 94.7%
* **Recall (R):** 91.1%
* **mAP50:** 96.9%
* **mAP50-95:** 86.3%

The model operates with high precision, reaching an overall success rate of **96.9% mAP50**.
