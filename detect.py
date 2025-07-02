import torch
from torchvision import transforms
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification, SiglipForImageClassification
import cv2
from PIL import Image

# --- MODELO 1: pipeline para letra J ---
pipe_model = pipeline("image-classification", model="prithivMLmods/Alphabet-Sign-Language-Detection")

# --- MODELO 2: ResNet para letras A-Y (excepto J y Z) ---
model_resnet = AutoModelForImageClassification.from_pretrained("achedguerra/resnet-50-signal_language")
model_resnet.eval()

# --- MODELO 3: SigLIP para letra Z ---
siglip_name = "prithivMLmods/Alphabet-Sign-Language-Detection"
processor_siglip = AutoImageProcessor.from_pretrained(siglip_name)
model_siglip = SiglipForImageClassification.from_pretrained(siglip_name)
model_siglip.eval()

# --- TRANSFORMACIONES ---
transform_resnet = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_siglip = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((processor_siglip.size["height"], processor_siglip.size["width"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor_siglip.image_mean, std=processor_siglip.image_std)
])

# --- FUNCIÓN PRINCIPAL ---
def detectar_letra(frame, letra_objetivo):
    letra_objetivo = letra_objetivo.upper()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    if letra_objetivo == 'J':
        result = pipe_model(img_pil)[0]
        label = result['label'].upper()
        score = result['score']

    elif letra_objetivo == 'Z':
        tensor = transform_siglip(img_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = model_siglip(pixel_values=tensor)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            idx = probs.argmax().item()
            label = model_siglip.config.id2label[idx].upper()
            score = probs[0, idx].item()

    else:  # A-Y sin J, Z
        tensor = transform_resnet(img_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = model_resnet(tensor)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            idx = probs.argmax().item()
            label = model_resnet.config.id2label[idx].upper()
            score = probs[0, idx].item()

    return {
        "letra_detectada": label,
        "score": round(score, 3),
        "es_correcto": label == letra_objetivo
    }
