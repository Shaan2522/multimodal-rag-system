# using google-bertbert-large-uncased-whole-word-masking-finetuned-squad (high accuracy)

import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers.utils import logging

logging.set_verbosity_error()  # Suppress all logs except critical

# === Load models ===
device = torch.device("cpu")

text_model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)
clip_model = CLIPModel.from_pretrained("models/clip-vit-large-patch14", local_files_only=True).to(device)
clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-large-patch14", use_fast=False)

# Load high-accuracy QA model (BERT-Large fine-tuned on SQuAD)
reader_tokenizer = AutoTokenizer.from_pretrained(
    "models\google-bert-large-uncased-whole-word-masking-finetuned-squad", local_files_only=True
)
reader_model = AutoModelForQuestionAnswering.from_pretrained(
    "models\google-bert-large-uncased-whole-word-masking-finetuned-squad", local_files_only=True
).to(device)
reader_model.eval()

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

# === Step 2: QA with sliding window using BERT-Large ===
def generate_answer_sliding_window(context, question, window_size=400, stride=100, top_k=3):
    question_tokens = reader_tokenizer(question, return_tensors="pt")["input_ids"][0]
    context_tokens = reader_tokenizer(context, return_tensors="pt")["input_ids"][0]

    spans = []

    for start in range(0, len(context_tokens), stride):
        end = min(start + window_size, len(context_tokens))
        chunk_tokens = context_tokens[start:end]
        chunk_text = reader_tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        inputs = reader_tokenizer(question, chunk_text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = reader_model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        for _ in range(top_k):
            start_idx = torch.argmax(start_logits)
            end_idx = torch.argmax(end_logits)

            if start_idx <= end_idx:
                score = start_logits[start_idx] + end_logits[end_idx]
                answer_ids = inputs["input_ids"][0][start_idx:end_idx + 1]
                answer_text = reader_tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

                # Save answer span and its score
                spans.append((score.item(), answer_text))

                # Mask this span so it’s not picked again
                start_logits[start_idx] = -float("inf")
                end_logits[end_idx] = -float("inf")

    # Sort spans by score
    spans = sorted(spans, key=lambda x: x[0], reverse=True)

    # Filter unique, non-empty answers
    seen = set()
    answers = []
    for _, ans in spans:
        if ans not in seen and len(ans.split()) > 2:
            seen.add(ans)
            answers.append(ans)
        if len(answers) >= top_k:
            break

    return "\n\n".join(answers) if answers else "No answer found."

# === Final Orchestrator ===
from sklearn.metrics.pairwise import cosine_similarity

def query_pdf_rag(query_text=None, query_image_path=None):
    matched_doc = find_best_matching_page(query_text, query_image_path)

    if not matched_doc:
        print("❌ No matching PDF found.")
        return

    file_name = matched_doc["file_name"]

    combined_text = ""
    combined_embeddings = []

    # === Step 1: Combine text & collect stored embeddings from matched PDF ===
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

    # === Step 2: Compute average of existing text embeddings ===
    combined_embedding = np.mean(combined_embeddings, axis=0)
    query_embedding = get_text_embedding(query_text)
    similarity = cosine_similarity([query_embedding], [combined_embedding]).item()

    # === Step 3: Save all into temp.json for reference ===
    with open("temp.json", "w", encoding="utf-8") as f:
        json.dump({
            "query_text": query_text,
            "matched_pdf": file_name,
            "combined_text": combined_text,
            "query_embedding": query_embedding.tolist(),
            "combined_text_embedding": combined_embedding.tolist(),
            "similarity_score": similarity
        }, f, indent=2)

    print(f"[✓] Temp embedding + context saved to 'temp.json'")
    print(f"[📄] Matched PDF     : {file_name}")
    print(f"[📊] Similarity Score: {similarity:.4f}")

    if similarity < 0.5:
        print("⚠️ Similarity too low — answer may not be relevant.")
        return

    # === Step 4: QA ===
    answer = generate_answer_sliding_window(combined_text, query_text)

    print("\n--- User Query ---")
    print(query_text)
    print("\n--- 📘 Answer from combined document ---")
    print(answer)

# === Example Usage ===
if __name__ == "__main__":
    query_pdf_rag(
        query_text="What is Battery monitoring system?",
        query_image_path=""
    )
