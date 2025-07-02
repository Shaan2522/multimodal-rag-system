import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

# === Load Models ===
text_model = SentenceTransformer("models/all-MiniLM-L6-v2")
clip_model = CLIPModel.from_pretrained("models/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-large-patch14", use_fast=False)
clip_model.eval()

# === Root Input Folder ===
root_input_folder = "assets/extractedText"
root_output_folder = "assets/embeddedJSONs"

# === Loop over all folders in root_input_folder ===
for folder in os.listdir(root_input_folder):
    input_folder = os.path.join(root_input_folder, folder)
    if not os.path.isdir(input_folder):
        continue  # Skip files

    output_folder = os.path.join(root_output_folder, folder)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n[→] Processing folder: {folder}")

    for filename in os.listdir(input_folder):
        if not filename.endswith(".json"):
            continue

        input_path = os.path.join(input_folder, filename)
        with open(input_path, "r") as f:
            data = json.load(f)

        file_name = data.get("file_name", "")
        page_number = data.get("page_number", 0)
        text = data.get("text", "")
        image_file = data.get("image_file", "")
        block_images = data.get("block_images", [])

        # --- 1. Text Embedding ---
        text_embedding = text_model.encode(text).tolist()

        # --- 2. Image Embeddings for block images ---
        image_embeddings = []
        for image_path in block_images:
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    features = clip_model.get_image_features(**inputs)
                image_embeddings.append(features[0].cpu().numpy().tolist())
            except Exception as e:
                print(f"[!] Error processing {image_path}: {e}")
                image_embeddings.append([])

        # --- 3. Final output JSON structure ---
        output_data = {
            "file_name": file_name,
            "page_number": page_number,
            "text": text,
            "text_embedding": text_embedding,
            "image_file": image_file,
            "block_images": block_images,
            "image_embedding_for_all_block_images": image_embeddings
        }

        # --- 4. Save output JSON ---
        output_path = os.path.join(output_folder, filename)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"[✓] Processed: {filename}")

print("\n[✓✓] All folders processed successfully.")
