# using minilm-uncased-squad2 llm (high speed - low accuracy)

import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers.utils import logging

logging.set_verbosity_error()  # Hides all logs except critical errors
# logging.set_verbosity_info()  # re-enable logging in terminal

# === Load models ===
device = torch.device("cpu")

text_model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)
clip_model = CLIPModel.from_pretrained("models/clip-vit-large-patch14", local_files_only=True).to(device)
clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-large-patch14", use_fast=False)

llm_tokenizer = AutoTokenizer.from_pretrained("models/minilm-uncased-squad2", local_files_only=True)
llm_model = AutoModelForQuestionAnswering.from_pretrained("models/minilm-uncased-squad2", local_files_only=True).to(device)
llm_model.eval()

# === Knowledge base directory ===
kb_path = "assets/embeddedJSONs"

# === Load all embedded JSON files ===
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

# === Step 1: Find best matching PDF page ===
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

# === Step 2: QA with sliding window ===
def generate_answer_sliding_window(context, question, window_size=400, stride=100):
    question_tokens = llm_tokenizer(question, return_tensors="pt")["input_ids"][0]
    context_tokens = llm_tokenizer(context, return_tensors="pt")["input_ids"][0]

    best_score = float("-inf")
    best_answer = "No answer found"

    for start in range(0, len(context_tokens), stride):
        end = min(start + window_size, len(context_tokens))
        chunk_tokens = context_tokens[start:end]
        chunk_text = llm_tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        inputs = llm_tokenizer(question, chunk_text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = llm_model(**inputs)

        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)

        if start_idx <= end_idx:
            score = outputs.start_logits[0][start_idx] + outputs.end_logits[0][end_idx]
            if score > best_score:
                best_score = score
                answer_ids = inputs["input_ids"][0][start_idx:end_idx + 1]
                best_answer = llm_tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    return best_answer if best_answer else "No answer found"

# === Final Orchestrator ===
def query_pdf_rag(query_text=None, query_image_path=None):
    matched_doc = find_best_matching_page(query_text, query_image_path)

    if not matched_doc:
        print("❌ No matching PDF found.")
        return

    file_name = matched_doc["file_name"]
    page_number = matched_doc["page_number"]
    page_image = matched_doc.get("image_file", "N/A")
    context = matched_doc.get("text", "")

    print(f"[✓] Matched PDF   : {file_name}")
    print(f"[📄] Matched Page : {page_number}")
    print(f"[📁] Page Image   : {page_image}")

    if not context.strip():
        print("⚠️ No readable text found on this page.")
        return

    answer = generate_answer_sliding_window(context, query_text)
    print("\n--- User Query ---")
    print(query_text+"\n")
    print("\n--- 📘 Answer from page text ---")
    print(answer)

# === Example Usage ===
if __name__ == "__main__":
    query_pdf_rag(
        query_text="what are the energy characteristics of epsilor rechargeable LI-Ion battery?",
        query_image_path=""
    )
