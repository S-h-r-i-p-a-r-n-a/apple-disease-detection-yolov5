import platform
import pathlib
import torch
from PIL import Image
import uuid
import os
from dotenv import load_dotenv

# ✅ Load environment variables from .env (only works locally)
load_dotenv()

# ✅ Fix for Windows path compatibility
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# ✅ GitHub token from environment (for Render or local)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if GITHUB_TOKEN:
    torch.hub._DEFAULT_GITHUB_TOKEN = GITHUB_TOKEN
    torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: None

# ✅ Define static path
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

# ✅ Disease-treatment mapping
TREATMENTS = {
    "Rust": "Use Myclobutanil fungicide. Remove infected leaves.",
    "Scab": "Use Captan or Mancozeb fungicides.",
    "Healthy": "No disease detected."
}

def detect_disease(file):
    # ✅ Clean static/ folder
    for f in os.listdir(STATIC_DIR):
        try:
            os.remove(os.path.join(STATIC_DIR, f))
        except:
            pass

    # ✅ Save uploaded image
    img = Image.open(file.file).convert("RGB")
    image_name = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(STATIC_DIR, image_name)
    img.save(image_path)

    # ✅ Run detection
    results = model(image_path)

    # ✅ Render result with bounding boxes
    rendered = results.render()[0]
    Image.fromarray(rendered).save(image_path)

    # ✅ Parse results
    df = results.pandas().xyxy[0]
    disease = df["name"][0].capitalize() if not df.empty else "Healthy"
    treatment = TREATMENTS.get(disease, "No treatment found.")

    return image_name, disease, treatment
