from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import os
import json
from sentence_transformers import SentenceTransformer
import cv2 as cv
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import random as rng

# === Utility functions ===
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

# === ✅ NEW: Process all PDFs ===
pdf_folder = "assets/LLM Dataset"
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    pdf_name = os.path.splitext(pdf_file)[0]

    # === ✅ NEW: Output folder per PDF ===
    image_dir = f"assets/images/{pdf_name}"
    os.makedirs(image_dir, exist_ok=True)

    # === ✅ CHANGED: Convert current PDF to images
    images = convert_from_path(pdf_path, dpi=200)

    # === Save each page as image
    for i, page in enumerate(images):
        image_path = os.path.join(image_dir, f"page_{i+1}.jpg")
        page.save(image_path, "JPEG")

    num_files = len(images)  # One image per page

    # === Process each page
    for i in range(num_files):
        IMAGE_PATH = f'assets/images/{pdf_name}/page_{i+1}.jpg'
        SAVE_DIR = f'assets/segmentedImages/{pdf_name}/page_{i+1}'
        BLOCK_DIR = os.path.join(SAVE_DIR, 'blocks')

        MIN_AREA = 8000
        SSIM_THRESHOLD = 0.75
        ASPECT_RATIO_LIMITS = (0.2, 5.0)

        os.makedirs(SAVE_DIR, exist_ok=True)
        os.makedirs(BLOCK_DIR, exist_ok=True)

        im = cv.imread(IMAGE_PATH)
        if im is None:
            print(f"[WARNING] Could not load image: {IMAGE_PATH}")
            continue

        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        save_plot_image(gray, "Original Image", os.path.join(SAVE_DIR, "original_image.png"))

        # Edge detection
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        canny = cv.Canny(blurred, 100, 150, L2gradient=True)

        # Dilation
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dilated = cv.dilate(canny, kernel, iterations=1)
        save_plot_image(dilated, "Canny + Dilation", os.path.join(SAVE_DIR, "canny_output.png"))

        contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        drawing = np.zeros((canny.shape[0], canny.shape[1], 3), dtype=np.uint8)
        cropped_images = []
        block_paths = []

        for j, contour in enumerate(contours):
            area = cv.contourArea(contour)
            if area < MIN_AREA:
                continue

            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = w / float(h)
            if not (ASPECT_RATIO_LIMITS[0] <= aspect_ratio <= ASPECT_RATIO_LIMITS[1]):
                continue

            margin = 10
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, im.shape[1])
            y2 = min(y + h + margin, im.shape[0])
            crop = im[y1:y2, x1:x2]

            if is_duplicate(crop, cropped_images, SSIM_THRESHOLD):
                continue

            cropped_images.append(crop)

            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv.drawContours(drawing, [contour], -1, color, 2)
            cv.rectangle(drawing, (x, y), (x + w, y + h), color, 2)

            out_path = os.path.join(BLOCK_DIR, f"block_{len(cropped_images)}.png")
            cv.imwrite(out_path, crop)
            block_paths.append(out_path)

        save_plot_image(cv.cvtColor(drawing, cv.COLOR_BGR2RGB), "Detected Regions",
                        os.path.join(SAVE_DIR, "annotated_drawing.png"), cmap=None)

        # Save block grid image
        if cropped_images:
            cols = 5
            rows = math.ceil(len(cropped_images) / cols)
            fig = plt.figure(figsize=(cols * 3, rows * 3))

            for k, img in enumerate(cropped_images):
                fig.add_subplot(rows, cols, k + 1)
                img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                plt.imshow(img_rgb)
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, "cropped_blocks_grid.png"))
            plt.close()

            print(f"[INFO] [{pdf_name} pg {i+1}] Saved {len(cropped_images)} unique large blocks.")
        else:
            print(f"[INFO] [{pdf_name} pg {i+1}] No large non-text blocks found.")
