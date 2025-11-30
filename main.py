from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------- TEXT MODEL -----------------
text_classifier = pipeline("text-classification", model="roberta-base-openai-detector")

@app.post("/check_text")
async def check_text(text: str = Form(...)):
    result = text_classifier(text)
    label = result[0]["label"]
    confidence = result[0]["score"]

    return {
        "is_ai": True if label == "LABEL_1" else False,
        "confidence": confidence
    }


# ------------- IMAGE MODEL -----------------
extractor = AutoFeatureExtractor.from_pretrained("nateraw/ai-generated-image-detector")
model_img = AutoModelForImageClassification.from_pretrained("nateraw/ai-generated-image-detector")

@app.post("/check_image")
async def check_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")
    outputs = model_img(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    label_id = probs.argmax(-1).item()
    confidence = probs.max().item()

    label = "AI" if label_id == 0 else "REAL"

    return {
        "label": label,
        "confidence": confidence
    }
