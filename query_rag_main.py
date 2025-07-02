import json, os
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, CLIPProcessor, CLIPModel
import torch
from transformers.utils import logging

logging.set_verbosity_error()

VECTOR_STORE = "vector_store"
NUM_RESULTS = 3

# Load FAISS indices
text_index = faiss.read_index(f"{VECTOR_STORE}/text_index.faiss")
image_index = faiss.read_index(f"{VECTOR_STORE}/image_index.faiss")

# Load metadata
with open(f"{VECTOR_STORE}/text_metadata.json", "r", encoding="utf-8") as f:
    text_metadata = json.load(f)

with open(f"{VECTOR_STORE}/image_metadata.json", "r", encoding="utf-8") as f:
    image_metadata = json.load(f)

# Set device
device = torch.device("cpu")

# === Load Embedding Models ===
text_model = SentenceTransformer(
    "D:/proj-ltpes-main/proj-ltpes-main/models/all-MiniLM-L6-v2",
    local_files_only=True
)

clip_model = CLIPModel.from_pretrained(
    "D:/proj-ltpes-main/proj-ltpes-main/models/clip-vit-large-patch14",
    local_files_only=True
).to(device)

clip_processor = CLIPProcessor.from_pretrained(
    "D:/proj-ltpes-main/proj-ltpes-main/models/clip-vit-large-patch14",
    use_fast=False
)

# === Load FLAN-T5 Model ===
flan_tokenizer = AutoTokenizer.from_pretrained(
    "D:/proj-ltpes-main/proj-ltpes-main/models/google-flan-t5-base",
    local_files_only=True
)

flan_model = AutoModelForSeq2SeqLM.from_pretrained(
    "D:/proj-ltpes-main/proj-ltpes-main/models/google-flan-t5-base",
    local_files_only=True
).to(device)

flan_model.eval()

# === Embedding and Search Functions ===

def embed_text(query):
    return text_model.encode([query])[0].astype("float32")

def embed_text_for_clip(text):
    inputs = clip_processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
    return features[0].cpu().numpy().astype("float32")

def embed_image(path):
    try:
        # Optimize image loading
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        
        # Resize large images before processing to speed up embedding
        max_size = (1024, 1024)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.LANCZOS)
        
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        return features[0].cpu().numpy().astype("float32")
    except Exception as e:
        print(f"Error embedding image {path}: {e}")
        return None

def search_text(query):
    emb = embed_text(query)
    D, I = text_index.search(np.expand_dims(emb, axis=0), NUM_RESULTS)
    results = []
    for i, dist in zip(I[0], D[0]):
        meta = text_metadata[i]
        meta["similarity_score"] = float(1 / (1 + dist))  # Convert L2 to pseudo-confidence
        results.append(meta)
    return results

def search_image(image_path):
    emb = embed_image(image_path)
    if emb is None:
        return []
    D, I = image_index.search(np.expand_dims(emb, axis=0), NUM_RESULTS)
    results = []
    for i, dist in zip(I[0], D[0]):
        meta = image_metadata[i]
        meta["similarity_score"] = float(1 / (1 + dist))
        results.append(meta)
    return results

def flan_answer(context, question, max_len=512, stride=400):
    from math import ceil

    # Break context into sliding chunks of tokens
    context_tokens = flan_tokenizer(
        context,
        return_tensors="pt",
        truncation=False
    )["input_ids"][0]

    total_tokens = context_tokens.shape[0]
    answers = []

    for start in range(0, total_tokens, stride):
        end = min(start + max_len - 100, total_tokens)  # leave room for question
        chunk_tokens = context_tokens[start:end]

        # Decode chunk back into text
        chunk_text = flan_tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        prompt = f"Context:\n{chunk_text}\n\nQuestion: {question}\n\nAnswer:"
        inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)

        with torch.no_grad():
            output = flan_model.generate(**inputs, max_new_tokens=150)
        decoded = flan_tokenizer.decode(output[0], skip_special_tokens=True)
        answers.append(decoded)

        if end == total_tokens:
            break

    return "\n---\n".join(answers)

def validate_image_paths(image_paths):
    """Filter out non-existent image paths"""
    valid_paths = []
    for path in image_paths:
        if os.path.exists(path) and os.path.isfile(path):
            try:
                # Quick validation by attempting to open the image
                with Image.open(path) as img:
                    img.verify()
                valid_paths.append(path)
            except Exception as e:
                print(f"Invalid image file {path}: {e}")
        else:
            print(f"Image file not found: {path}")
    return valid_paths

def run_query(text_input=None, image_path=None, input_type="text", progress_callback=None):
    results = []
    query = ""
    matched_image = None
    matched_images = []
    matched_pdf = None
    page_images = []  # Store page images to return to GUI

    try:
        if progress_callback:
            progress_callback("Processing query...")

        if input_type == "text":
            if progress_callback:
                progress_callback("Searching text...")
            
            top_text_results = search_text(text_input)
            query = text_input

            matched_pdf = top_text_results[0]["file_name"]
            results = [entry for entry in text_metadata if entry["file_name"] == matched_pdf]
            results = sorted(results, key=lambda r: r["page_number"])[:3]  # Only top 3 pages

            print(f"\n📌 Text matched to PDF: {matched_pdf} — using all pages as context.")

            if progress_callback:
                progress_callback("Finding matching images...")

            text_emb_clip = embed_text_for_clip(text_input)
            pdf_image_blocks = [r for r in image_metadata if r["file_name"] == matched_pdf]
            embeddings, index_map = [], []

            for item in pdf_image_blocks:
                for i, emb in enumerate(item.get("image_embedding_for_all_block_images", [])):
                    if emb:
                        pdf_name = item["file_name"]
                        page_number = item["page_number"]
                        full_page_path = f"D:/proj-ltpes-main/proj-ltpes-main/assets/images/{pdf_name}/page_{page_number}.jpg"
                        embeddings.append(emb)
                        index_map.append(full_page_path)

            if embeddings:
                index = faiss.IndexFlatL2(768)
                index.add(np.array(embeddings, dtype=np.float32))
                _, I = index.search(np.expand_dims(text_emb_clip, axis=0), 3)
                matched_images = [index_map[idx] for idx in I[0]]

        elif input_type == "image":
            if progress_callback:
                progress_callback("Processing image...")

            image_results = search_image(image_path)
            query = "What does this image represent?"

            if not image_results:
                return "❌ No image matches found.", []

            matched_pdf = image_results[0]["file_name"]
            results = [entry for entry in text_metadata if entry["file_name"] == matched_pdf]
            results = sorted(results, key=lambda r: r["page_number"])[:3]  # Only top 3 pages

            if progress_callback:
                progress_callback("Finding matching images...")

            image_emb = embed_image(image_path)
            if image_emb is None:
                return "❌ Failed to process input image.", []

            pdf_image_blocks = [r for r in image_metadata if r["file_name"] == matched_pdf]
            embeddings, index_map = [], []

            for item in pdf_image_blocks:
                for i, emb in enumerate(item.get("image_embedding_for_all_block_images", [])):
                    if emb:
                        pdf_name = item["file_name"]
                        page_number = item["page_number"]
                        full_page_path = f"D:/proj-ltpes-main/proj-ltpes-main/assets/images/{pdf_name}/page_{page_number}.jpg"
                        embeddings.append(emb)
                        index_map.append(full_page_path)

            if embeddings:
                index = faiss.IndexFlatL2(768)
                index.add(np.array(embeddings, dtype=np.float32))
                _, I = index.search(np.expand_dims(image_emb, axis=0), 1)
                matched_image = index_map[I[0][0]]

        elif input_type == "both":
            if progress_callback:
                progress_callback("Processing image and text...")

            image_results = search_image(image_path)
            query = text_input

            if not image_results:
                return "❌ No image matches found.", []

            matched_pdf = image_results[0]["file_name"]
            results = [entry for entry in text_metadata if entry["file_name"] == matched_pdf]
            results = sorted(results, key=lambda r: r["page_number"])[:3]  # Only top 3 pages

            print(f"\n📌 Image matched to PDF: {matched_pdf} — using all pages as context.")

            if progress_callback:
                progress_callback("Finding matching images...")

            text_emb_clip = embed_text_for_clip(text_input)
            pdf_image_blocks = [r for r in image_metadata if r["file_name"] == matched_pdf]
            embeddings, index_map = [], []

            for item in pdf_image_blocks:
                for i, emb in enumerate(item.get("image_embedding_for_all_block_images", [])):
                    if emb:
                        pdf_name = item["file_name"]
                        page_number = item["page_number"]
                        full_page_path = f"D:/proj-ltpes-main/proj-ltpes-main/assets/images/{pdf_name}/page_{page_number}.jpg"
                        embeddings.append(emb)
                        index_map.append(full_page_path)

            if embeddings:
                index = faiss.IndexFlatL2(768)
                index.add(np.array(embeddings, dtype=np.float32))
                _, I = index.search(np.expand_dims(text_emb_clip, axis=0), 1)
                matched_image = index_map[I[0][0]]

        else:
            raise ValueError("Invalid input_type. Choose 'text', 'image', or 'both'.")

        if progress_callback:
            progress_callback("Generating answer...")

        # Build context and generate answer
        context = "\n---\n".join([r["text"] for r in results])
        answer = flan_answer(context, query)

        # Generate page images paths for GUI
        print("\n📄 Matched Page Image(s):")
        for r in results:
            pdf_name = os.path.splitext(r["file_name"])[0]  # removes .pdf
            page_number = r["page_number"]
            full_path = f"D:/proj-ltpes-main/proj-ltpes-main/assets/images/{pdf_name}/page_{page_number}.jpg"
            page_images.append(full_path)
            print(f"- {full_path}")

        # Output formatting
        output = f"\n🧠 Answer:\n{answer}"

        # Combine all image paths for GUI display
        all_matched_paths = []
        
        # Add page images first (these are the main result pages)
        all_matched_paths.extend(page_images)
        
        # Add specific matched images
        if matched_images:
            all_matched_paths.extend(matched_images)
        elif matched_image:
            all_matched_paths.append(matched_image)

        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in all_matched_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)

        # Validate image paths before returning
        valid_paths = validate_image_paths(unique_paths)
        
        if progress_callback:
            progress_callback(f"Found {len(valid_paths)} valid images")

        print(f"Returning {len(valid_paths)} valid image paths to GUI")
        return output, valid_paths

    except Exception as e:
        error_msg = f"❌ Error processing query: {str(e)}"
        print(error_msg)
        return error_msg, []

# === Main ===
# Text-only
# print(run_query("what is epsilor rechargable battery?", input_type="text"))

# Image-only
# print(run_query(image_path="image2.png", input_type="image"))

# Multimodal - both: text + image together
# print(run_query(
#         text_input="what is this image?",
#         image_path="image.png",
#         input_type="both"
#     ))