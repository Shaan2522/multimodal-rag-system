# multimodal-rag-system
multimodal rag system for text and image querying from PDFs

💡 Offline Multimodal RAG System for Text and Image-Based Document QA
📝 Project Overview
This project implements an offline Retrieval-Augmented Generation (RAG) pipeline capable of answering questions using text and/or image queries against a collection of PDF documents. It supports:

PDF-to-text and image conversion

Visual block detection in images using OpenCV

Embeddings generation for both text (MiniLM) and images (CLIP)

FAISS-based similarity search

Natural language answer generation via FLAN-T5

Tkinter-based GUI to support user queries with previewed answers

🚀 Features
🧠 Multimodal QA: Accepts text, image, or both as user queries.

🖼️ Image Matching: Locates and shows the most relevant image blocks.

🔍 Semantic Retrieval: Uses powerful sentence and vision transformers.

📄 PDF Support: Handles multiple PDFs offline, including scanned/visual ones.

💬 Answer Generation: Natural answers generated using FLAN-T5.

🖥️ GUI Frontend: User-friendly interface using Python’s Tkinter.

📂 Project Structure
bash
Copy
Edit
offline-rag/
├── assets/
│   ├── images/                      # Full-page rendered images (PDF → JPG)
│   │   └── <pdf_name>/page_X.jpg
│   └── segmentedImages/            # Block images cropped from full pages
│       └── <pdf_name>/<page_number>/blocks/block_X.png
│
├── models/                          # Local huggingface models
│   ├── all-MiniLM-L6-v2/
│   ├── clip-vit-large-patch14/
│   └── google-flan-t5-base/
│
├── vector_store/                   # FAISS indices and metadata
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
🛠️ Installation
✅ 1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/offline-rag.git
cd offline-rag
✅ 2. Install Python Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Make sure you have Python 3.8+ and poppler installed (required by pdf2image).

✅ 3. Download Models Locally
Download and place the following models into the models/ folder:

all-MiniLM-L6-v2

clip-vit-large-patch14

google/flan-t5-base

Important: All models must be used offline (no internet required).

⚙️ Preprocessing Pipeline
📄 Step 1: PDF to Images
bash
Copy
Edit
python scripts/pdf_to_images.py
Converts every page of all PDFs into .jpg images at 200 DPI.

🔍 Step 2: Text Extraction
bash
Copy
Edit
python scripts/extract_text.py
Extracts per-page text using PyPDF2 or Tesseract (for scanned documents).

🧩 Step 3: Visual Block Detection
bash
Copy
Edit
python scripts/detect_blocks.py
Uses OpenCV to segment images into diagrams/tables/logos and stores block images.

📊 Step 4: Embedding Generation
bash
Copy
Edit
python scripts/generate_embeddings.py
Creates embeddings for:

Text (using MiniLM)

Images + blocks (using CLIP)

🧠 Step 5: Build FAISS Index
bash
Copy
Edit
python scripts/build_faiss_index.py
Creates text_index.faiss and image_index.faiss in the vector_store/ folder.

🖼️ Using the GUI
Run the GUI with:

bash
Copy
Edit
python rag_gui.py
Functionality:

Enter text query and/or select an image

Choose input type: text, image, or both

Click Run Query

View:

Generated answer

Matching page previews (click to open full size)

📡 Offline Compatibility
✅ All models and operations are fully offline.
No API calls or external dependencies at runtime.

💡 Technologies Used
Task	Tool/Model Used
PDF → Image	pdf2image
Text Extraction	PyPDF2 / Tesseract
Object Detection	OpenCV + SSIM
Text Embedding	all-MiniLM-L6-v2
Image Embedding	CLIP ViT-Large-Patch14
Retrieval	FAISS
Generation (QA)	FLAN-T5 Base
GUI	Tkinter, PIL (ImageTk)

✅ Supported Inputs
Input Type	Description
Text Only	Search by query text (e.g., “what is BMS?”)
Image Only	Match similar pages using image
Text + Image	Combined context from both modalities

📌 Example Output
txt
Copy
Edit
📌 Text matched to PDF: Battery_Specs_Brochure.pdf

📄 Matched Page Image(s):
- D:/assets/images/Battery_Specs_Brochure/page_3.jpg
- D:/assets/images/Battery_Specs_Brochure/page_4.jpg
- D:/assets/images/Battery_Specs_Brochure/page_5.jpg

🧠 Answer:
The Battery Monitoring System consists of independent modules...
🏁 Future Improvements
OCR fallback for scanned text regions (Tesseract)

Custom-trained object detection (YOLO for diagrams)

Summarization on matched page text

Dockerized deployment

🧾 License
This project is licensed for educational and non-commercial use. Contact the author for enterprise licensing.
