# multimodal-rag-system
Offline Multimodal RAG System for Text and Image-Based Document QA using Google's Flan-T5-Base LLM.

### 📝 Project Overview
This project implements an offline Retrieval-Augmented Generation (RAG) pipeline capable of answering questions using text and/or image queries against a collection of PDF documents. 
It supports:

1. PDF-to-text and PDF-to-image conversion

2. Visual block detection in images using OpenCV

3. Embeddings generation for both text (MiniLM) and images (CLIP)

4. FAISS-based similarity search (cosine similarity)

5. Natural language answer generation via FLAN-T5

6. Tkinter-based GUI to support user queries with previewed answers

### 🚀 Features

🧠 Multimodal QA: Accepts text, image, or both as user queries.

🖼️ Image Matching: Locates and shows the most relevant image blocks.

🔍 Semantic Retrieval: Uses powerful sentence and vision transformers.

📄 PDF Support: Handles multiple PDFs offline, including scanned/visual ones.

💬 Answer Generation: Natural answers generated using FLAN-T5.

🖥️ GUI Frontend: User-friendly interface using Python’s Tkinter.


### 📂 Project Structure
```bash
offline-rag/
├── assets/
│   ├── images/                    # Full-page rendered images (PDF → JPG)
│   │   └── <pdf_name>/page_X.jpg
│   └── segmentedImages/           # Block images cropped from full pages
│       └── <pdf_name>/<page_number>/blocks/block_X.png
│
├── models/                        # Local huggingface models
│   ├── all-MiniLM-L6-v2/
│   ├── clip-vit-large-patch14/
│   └── google-flan-t5-base/
│
├── vector_store/                  # FAISS indices and metadata
│   ├── text_index.faiss
│   ├── image_index.faiss
│   ├── text_metadata.json
│   └── image_metadata.json
│
├── scripts/
│   ├── pdf_to_images.py           # Converts PDFs to images
│   ├── extract_text.py            # Extracts text per PDF page
│   ├── detect_blocks.py           # Detects object blocks using OpenCV
│   ├── generate_embeddings.py     # Generates and stores embeddings
│   └── build_faiss_index.py       # Builds FAISS index
│
├── query_rag.py                   # Main backend inference logic
├── rag_gui.py                     # Tkinter GUI application
├── requirements.txt               # Python dependencies
└── README.md                      # You are here
```

### 🛠️ Installation
✅ 1. Clone the Repository

```bash
git clone https://github.com/yourusername/offline-rag.git
cd offline-rag
```

✅ 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

✅ 3. Download Models Locally
Download and place the following models into the ```models/``` folder:

1. all-MiniLM-L6-v2

2. clip-vit-large-patch14

3. google/flan-t5-base

Important: All models can be used completely offline (no internet required).

### ⚙️ Preprocessing Pipeline

#### 📄 Step 1: PDF to Images
```bash
python scripts/pdf_to_images.py
```
Converts every page of all PDFs into .jpg images at 200 DPI.

#### 🔍 Step 2: Text Extraction
```bash
python scripts/extract_text.py
```
Extracts per-page text using PyPDF2 (for scanned documents).

#### 🧩 Step 3: Visual Block Detection
```bash
python scripts/detect_blocks.py
```
Uses OpenCV to segment images into diagrams/tables/logos and stores block images.

#### 📊 Step 4: Embedding Generation
```bash
python scripts/generate_embeddings.py
```
Creates embeddings for:

1. Text (using MiniLM)

2. Images + blocks (using CLIP)

#### 🧠 Step 5: Build FAISS Index
```bash
python scripts/build_faiss_index.py
```
Creates ```text_index.faiss``` and ```image_index.faiss``` in the ```vector_store/``` folder.

🖼️ Using the GUI
Run the GUI with:

```bash
python rag_gui.py
```
Functionality:

1. Enter text query and/or select an image

2. Choose input type: text, image, or both

3. Click Run Query

4. View:

- Generated answer

- Matching page previews (click to open full size)

### 📡 Offline Compatibility

- ✅ All models and operations are fully offline.

- No API calls or external dependencies at runtime.

### 💡 Technologies Used

| Task	| Tool/Model Used |
|:-------:|:------------------:|
| PDF → Image | pdf2image |
| Text Extraction | PyPDF2 |
| Object Detection | OpenCV + SSIM |
| Text Embedding | all-MiniLM-L6-v2 |
| Image Embedding	| CLIP ViT-Large-Patch14 |
| Retrieval	| FAISS |
| Generation (QA)	| FLAN-T5 Base |
| GUI	| Tkinter, PIL (ImageTk) |

### ✅ Supported Inputs
| Input Type | Description |
|:------------:|:-------------:|
| Text Only |	Search by query text (e.g., “what is BMS?”) |
| Image Only | Match similar pages using image |
| Text + Image | Combined context from both modalities (image takes priority) |

📌 Example Output
```txt
📌 Text matched to PDF: Battery_Specs_Brochure.pdf

📄 Matched Page Image(s):
- D:/assets/images/Battery_Specs_Brochure/page_3.jpg
- D:/assets/images/Battery_Specs_Brochure/page_4.jpg
- D:/assets/images/Battery_Specs_Brochure/page_5.jpg

🧠 Answer:
The Battery Monitoring System consists of independent modules...
```

### 🏁 Future Improvements
1. OCR fallback for scanned text regions (Tesseract)

2. Custom-trained object detection (YOLO for diagrams)

3. Summarization on matched page text
