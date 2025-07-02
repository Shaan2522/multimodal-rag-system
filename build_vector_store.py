import os
import json
import numpy as np
import faiss

# Path to the base directory containing folders of PDF JSONs
BASE_DIR = "embeddedJSONs"
OUTPUT_DIR = "vector_store"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize indices
text_index = faiss.IndexFlatL2(384)   # all-MiniLM-L6-v2
image_index = faiss.IndexFlatL2(768)  # CLIP-ViT-L/14

text_metadata = []
image_metadata = []

# Walk through each PDF folder
for pdf_folder in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, pdf_folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        # Text embedding
        if "text_embedding" in data:
            text_index.add(np.array([data["text_embedding"]], dtype="float32"))
            text_metadata.append({
                "file_name": data["file_name"],
                "page_number": data["page_number"],
                "text": data["text"]
            })

        # Image block embeddings
        for emb, path in zip(data.get("image_embedding_for_all_block_images", []),
                             data.get("block_images", [])):
            if emb:
                image_index.add(np.array([emb], dtype="float32"))
                image_metadata.append({
                    "file_name": data["file_name"],
                    "page_number": data["page_number"],
                    "image_path": path,
                    "text": data["text"]
                })

# Save FAISS indices
faiss.write_index(text_index, os.path.join(OUTPUT_DIR, "text_index.faiss"))
faiss.write_index(image_index, os.path.join(OUTPUT_DIR, "image_index.faiss"))

# Save metadata
with open(os.path.join(OUTPUT_DIR, "text_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(text_metadata, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "image_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(image_metadata, f, indent=2)

print("✅ Vector store built and saved.")
