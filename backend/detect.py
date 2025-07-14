import platform
import pathlib
import torch
from PIL import Image
import uuid
import os

# ✅ Fix for Windows path issue
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# ✅ GitHub token (optional but safe for Render)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # if using .env in future
if GITHUB_TOKEN:
    torch.hub._DEFAULT_GITHUB_TOKEN = GITHUB_TOKEN
    torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: None

# ✅ Static folder
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ✅ Load YOLOv5 model
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='model/best.pt',
    source='github',
    force_reload=True
)

# ✅ Treatments
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

    # Save input image
    img = Image.open(file.file).convert("RGB")
    image_name = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(STATIC_DIR, image_name)
    img.save(image_path)

    # Predict
    results = model(image_path)
    rendered = results.render()[0]

    # Save rendered image
    Image.fromarray(rendered).save(image_path)

    # Read result
    df = results.pandas().xyxy[0]
    disease = df["name"][0].capitalize() if not df.empty else "Healthy"
    treatment = TREATMENTS.get(disease, "No treatment found.")

    return image_name, disease, treatment
