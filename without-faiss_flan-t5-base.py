import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers.utils import logging

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load embedding models ===
text_model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)
clip_model = CLIPModel.from_pretrained("models/clip-vit-large-patch14", local_files_only=True).to(device)
clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-large-patch14", use_fast=False)

# ✅ Load FLAN-T5 model
flan_tokenizer = AutoTokenizer.from_pretrained("models/google-flan-t5-base", local_files_only=True)
flan_model = AutoModelForSeq2SeqLM.from_pretrained("models/google-flan-t5-base", local_files_only=True).to(device)
flan_model.eval()

# === Load JSON documents ===
kb_path = "assets/embeddedJSONs"
documents = []
for root, _, files in os.walk(kb_path):
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                documents.append(json.load(f))

# === Embedding helpers ===
def get_text_embedding(text):
    return text_model.encode(text)

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features[0].cpu().numpy()

# === Find best matching page ===
def find_best_matching_page(query_text=None, query_image_path=None):
    query_text_emb = get_text_embedding(query_text) if query_text else None
    query_image_emb = get_image_embedding(query_image_path) if query_image_path else None

    best_score = 0
    best_doc = None

    for doc in documents:
        score = 0
        if query_text_emb is not None and doc.get("text_embedding"):
            score = max(score, cosine_similarity([query_text_emb], [np.array(doc["text_embedding"])]).item())
        if query_image_emb is not None:
            for emb in doc.get("image_embedding_for_all_block_images", []):
                if emb:
                    score = max(score, cosine_similarity([query_image_emb], [np.array(emb)]).item())
        if score > best_score:
            best_score = score
            best_doc = doc

    return best_doc if best_score >= 0.5 else None

# === Run FLAN QA ===
def run_flan_t5_qa(context, question, max_tokens=300):
    prompt = (
        f"You are an expert technical assistant. Use the following PDF content to answer the question as accurately as possible, "
        f"but also add helpful information if you can infer something useful:\n\n"
        f"{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = flan_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,          # Enable sampling for variety
            top_p=0.95,              # Top-p sampling
            temperature=0.7          # Softer, more creative
        )
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# === Final Orchestrator ===
def query_pdf_rag_flan(query_text=None, query_image_path=None):
    matched_doc = find_best_matching_page(query_text, query_image_path)

    if not matched_doc:
        print("❌ No matching PDF found.")
        return

    file_name = matched_doc["file_name"]

    combined_text = ""
    combined_embeddings = []

    for doc in documents:
        if doc["file_name"] == file_name:
            text = doc.get("text", "").strip()
            emb = doc.get("text_embedding", [])
            if text and emb:
                combined_text += text + "\n"
                combined_embeddings.append(np.array(emb))

    if not combined_text.strip() or not combined_embeddings:
        print("⚠️ No usable content found in matched PDF.")
        return

    combined_embedding = np.mean(combined_embeddings, axis=0)
    query_embedding = get_text_embedding(query_text)
    similarity = cosine_similarity([query_embedding], [combined_embedding]).item()

    with open("temp.json", "w", encoding="utf-8") as f:
        json.dump({
            "query_text": query_text,
            "matched_pdf": file_name,
            "combined_text": combined_text,
            "query_embedding": query_embedding.tolist(),
            "combined_text_embedding": combined_embedding.tolist(),
            "similarity_score": similarity
        }, f, indent=2)

    print(f"[✓] Temp saved to 'temp.json'")
    print(f"[📄] Matched PDF     : {file_name}")
    print(f"[📊] Similarity Score: {similarity:.4f}")

    if similarity < 0.5:
        print("⚠️ Similarity too low — answer may not be relevant.")
        return

    # === Generate Answer with FLAN-T5 ===
    answer = run_flan_t5_qa(combined_text, query_text)

    print("\n--- User Query ---")
    print(query_text)
    print("\n--- 📘 Answer from FLAN-T5 ---")
    print(answer)

# === Example Usage ===
if __name__ == "__main__":
    query_pdf_rag_flan(
        query_text="Battery Monitoring System Computer Technical Data",
        query_image_path=""
    )
