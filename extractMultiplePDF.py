import os
import json
import torch
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import random as rng

# === Load Models ===
text_model = SentenceTransformer("models/all-MiniLM-L6-v2")
clip_model = CLIPModel.from_pretrained("models/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-large-patch14")
clip_model.eval()
device = torch.device("cpu")

# === Utility Functions ===
def save_plot_image(img, title, save_path, cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def is_duplicate(img, existing_imgs, ssim_threshold=0.90):
    img_resized = cv.cvtColor(cv.resize(img, (100, 100)), cv.COLOR_BGR2GRAY)
    for other in existing_imgs:
        other_resized = cv.cvtColor(cv.resize(other, (100, 100)), cv.COLOR_BGR2GRAY)
        score = ssim(img_resized, other_resized)
        if score > ssim_threshold:
            return True
    return False

# === Main PDF Processor ===
def process_pdf(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    image_dir = f"assets/images/{pdf_name}"
    output_dir = f"assets/embeddedJSONs/{pdf_name}"
    seg_dir = f"assets/segmentedImages/{pdf_name}"

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[→] Processing: {pdf_name}")

    # === Text & Image Extraction ===
    reader = PdfReader(pdf_path)
    texts = [page.extract_text() or "" for page in reader.pages]
    images = convert_from_path(pdf_path, dpi=200)

    for i, text in enumerate(texts):
        image_path = os.path.join(image_dir, f"page_{i+1}.jpg")
        images[i].save(image_path)

        data = {
            "file_name": os.path.basename(pdf_path),
            "page_number": i + 1,
            "text": text.strip(),
            "image_file": image_path
        }

        json_path = os.path.join(output_dir, f"page_{i+1}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print("[✓] Saved text and images.")

    # === Embeddings + Image Blocks ===
    for i in range(len(texts)):
        json_path = os.path.join(output_dir, f"page_{i+1}.json")
        with open(json_path, "r", encoding="utf-8") as f:
            page_data = json.load(f)

        text = page_data.get("text", "")
        text_embedding = text_model.encode(text).tolist()
        page_data["text_embedding"] = text_embedding

        # === Segment Image Blocks ===
        IMAGE_PATH = page_data["image_file"]
        SAVE_DIR = os.path.join(seg_dir, f"page_{i+1}")
        BLOCK_DIR = os.path.join(SAVE_DIR, "blocks")
        os.makedirs(SAVE_DIR, exist_ok=True)
        os.makedirs(BLOCK_DIR, exist_ok=True)

        im = cv.imread(IMAGE_PATH)
        if im is None:
            print(f"[!] Skipping unreadable image: {IMAGE_PATH}")
            continue

        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        canny = cv.Canny(blurred, 100, 150, L2gradient=True)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dilated = cv.dilate(canny, kernel, iterations=1)

        contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cropped_images, block_paths, image_embeddings = [], [], []

        for contour in contours:
            area = cv.contourArea(contour)
            if area < 8000:
                continue
            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = w / float(h)
            if not (0.2 <= aspect_ratio <= 5.0):
                continue

            x1, y1 = max(x - 10, 0), max(y - 10, 0)
            x2, y2 = min(x + w + 10, im.shape[1]), min(y + h + 10, im.shape[0])
            crop = im[y1:y2, x1:x2]

            if is_duplicate(crop, cropped_images, 0.75):
                continue

            cropped_images.append(crop)
            out_path = os.path.join(BLOCK_DIR, f"block_{len(cropped_images)}.png")
            cv.imwrite(out_path, crop)
            block_paths.append(out_path)

            try:
                img_pil = Image.open(out_path).convert("RGB")
                inputs = clip_processor(images=img_pil, return_tensors="pt").to(device)
                with torch.no_grad():
                    features = clip_model.get_image_features(**inputs)
                image_embeddings.append(features[0].cpu().numpy().tolist())
            except Exception as e:
                print(f"[!] Embedding error on {out_path}: {e}")
                image_embeddings.append([])

        page_data["block_images"] = block_paths
        page_data["image_embedding_for_all_block_images"] = image_embeddings

        # === Save updated JSON ===
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(page_data, f, indent=2, ensure_ascii=False)

        print(f"[✓] Page {i+1}: Processed {len(cropped_images)} blocks.")

# === 🔁 Process All PDFs in temp/ Directory ===
pdf_dir = "assets\LLM Dataset"
for file in os.listdir(pdf_dir):
    if not file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(pdf_dir, file)
    pdf_name = os.path.splitext(file)[0]
    image_dir = os.path.join("assets/images", pdf_name)
    output_dir = os.path.join("assets/embeddedJSONs", pdf_name)

    # Skip if already processed
    if os.path.exists(image_dir) and os.path.exists(output_dir) and \
       len(os.listdir(image_dir)) > 0 and len(os.listdir(output_dir)) > 0:
        print(f"[⏩] Skipping {file} — already processed.")
        continue

    # Process if not done
    process_pdf(pdf_path)

print("\n[✓✓] All PDFs processed successfully.")
