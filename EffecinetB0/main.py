
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import time

# ── 1. لود الموديل ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint  = torch.load("best_model.pth", map_location=device)
CLASS_NAMES = checkpoint['classes']
NUM_CLASSES = len(CLASS_NAMES)

# ── 2. بناء الهيكل ──
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(512, NUM_CLASSES)
)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()
print(f"✓ Model loaded — {NUM_CLASSES} classes")

# ── 3. Transform ──
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── 4. FastAPI ──
app = FastAPI(title="Palm Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status"  : "running",
        "model"   : "EfficientNet-B0",
        "accuracy": "98.57%"
    }

@app.post("/predict_img")
async def predict(file: UploadFile = File(...)):
    # تحقق إن الملف صورة
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="ارفع صورة بس")

    # اقرأ الصورة
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # شغّل الموديل
    start = time.time()
    with torch.no_grad():
        tensor = transform(image).unsqueeze(0).to(device)
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
    ms = (time.time() - start) * 1000

    # النتيجة
    top_idx = probs.argmax().item()

    return {
        "disease"       : CLASS_NAMES[top_idx],
        "confidence_pct": f"{probs[top_idx].item()*100:.2f}%",
        "is_healthy"    : "healthy" in CLASS_NAMES[top_idx].lower(),
        "inference_ms"  : round(ms, 2),
        "all_scores"    : sorted([
            {"class": CLASS_NAMES[i],
             "confidence": round(probs[i].item()*100, 2)}
            for i in range(NUM_CLASSES)
        ], key=lambda x: x["confidence"], reverse=True)
    }