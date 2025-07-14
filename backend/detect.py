import os
import platform
import pathlib
import sys
import torch
import uuid
from PIL import Image

# ✅ Fix PosixPath issue on Windows
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# ✅ Define static path
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ✅ Add local yolov5 path to sys.path
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), "yolov5")
sys.path.append(YOLOV5_PATH)

# ✅ Import necessary modules from local yolov5
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# ✅ Initialize model
device = select_device('')
model = DetectMultiBackend(weights='model/best.pt', device=device)
model.eval()

# ✅ Treatment recommendations
TREATMENTS = {
    "Rust": "Use Myclobutanil fungicide. Remove infected leaves.",
    "Scab": "Use Captan or Mancozeb fungicides.",
    "Healthy": "No disease detected."
}

def detect_disease(file):
    # Clean static/
    for f in os.listdir(STATIC_DIR):
        try:
            os.remove(os.path.join(STATIC_DIR, f))
        except:
            pass

    # Save image
    img = Image.open(file.file).convert("RGB")
    image_name = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(STATIC_DIR, image_name)
    img.save(image_path)

    # Load image using LoadImages
    dataset = LoadImages(image_path, img_size=640)

    disease = "Healthy"

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device).float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                disease = model.names[int(det[0][-1])]
                break

    treatment = TREATMENTS.get(disease.capitalize(), "No treatment found.")
    return image_name, disease.capitalize(), treatment
