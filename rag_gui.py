import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import subprocess
import threading
from queue import Queue
from query_rag_main import run_query

# Maintain global image references to avoid garbage collection
image_refs = []

def browse_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
    if path:
        image_path_var.set(path)

def open_image(path):
    try:
        if os.name == 'nt':
            os.startfile(path)
        else:
            subprocess.call(['xdg-open', path])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open image:\n{e}")

def create_image_preview(img_path, max_size=(200, 200)):
    """Create a thumbnail preview of an image with optimization"""
    try:
        # Check if file exists
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            return None
            
        # Open and process image
        with Image.open(img_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create thumbnail (preserves aspect ratio)
            img.thumbnail(max_size, Image.LANCZOS)
            
            # Create PhotoImage
            tk_img = ImageTk.PhotoImage(img)
            return tk_img
            
    except Exception as e:
        print(f"Error creating preview for {img_path}: {e}")
        return None

def update_progress(message):
    """Update progress label"""
    progress_label.config(text=message)
    root.update_idletasks()

def clear_images():
    """Clear all image widgets"""
    global image_refs
    image_refs.clear()
    
    # Clear image frame
    for widget in image_frame.winfo_children():
        widget.destroy()

def display_images(matched_images):
    """Display images in a separate thread to avoid blocking"""
    def load_images():
        if not matched_images:
            root.after(0, lambda: show_no_images_message())
            return
        
        print(f"Loading {len(matched_images)} images...")
        loaded_images = []
        
        for i, img_path in enumerate(matched_images):
            try:
                # Update progress
                root.after(0, lambda p=f"Loading image {i+1}/{len(matched_images)}...": update_progress(p))
                
                tk_img = create_image_preview(img_path)
                if tk_img:
                    loaded_images.append((img_path, tk_img))
                    print(f"Successfully loaded: {os.path.basename(img_path)}")
                else:
                    print(f"Failed to load: {img_path}")
                    
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        # Display loaded images on main thread
        root.after(0, lambda: show_loaded_images(loaded_images))
    
    # Start loading in background thread
    threading.Thread(target=load_images, daemon=True).start()

def show_no_images_message():
    """Show message when no images are available"""
    update_progress("Ready")
    no_img_label = tk.Label(
        image_frame, 
        text="No matching images found", 
        bg="white",
        fg="gray",
        font=("Arial", 12)
    )
    no_img_label.pack(pady=30)

def show_loaded_images(loaded_images):
    """Display the loaded images in the GUI"""
    global image_refs
    
    update_progress("Ready")
    
    if not loaded_images:
        show_no_images_message()
        return
    
    # Create scrollable frame for images with increased height
    canvas = tk.Canvas(image_frame, bg="white", height=280)
    scrollbar = ttk.Scrollbar(image_frame, orient="horizontal", command=canvas.xview)
    scrollable_frame = tk.Frame(canvas, bg="white")
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(xscrollcommand=scrollbar.set)
    
    # Add images to scrollable frame
    for img_path, tk_img in loaded_images:
        # Create container for each image with increased padding
        img_container = tk.Frame(scrollable_frame, bg="white", relief=tk.RAISED, borderwidth=1)
        img_container.pack(side=tk.LEFT, padx=12, pady=12)
        
        # Image button
        btn = tk.Button(
            img_container, 
            image=tk_img, 
            command=lambda p=img_path: open_image(p),
            cursor="hand2",
            relief=tk.FLAT,
            borderwidth=0
        )
        btn.pack(pady=(8, 4))
        
        # Filename label with larger font
        filename = os.path.basename(img_path)
        display_name = filename[:20] + "..." if len(filename) > 20 else filename
        label = tk.Label(
            img_container, 
            text=display_name,
            bg="white",
            font=("Arial", 9),
            fg="darkblue"
        )
        label.pack(pady=(0, 8))
        
        # Keep reference to prevent garbage collection
        image_refs.append(tk_img)
    
    # Pack canvas and scrollbar
    canvas.pack(side="top", fill="x", expand=False)
    if len(loaded_images) > 3:  # Show scrollbar only if needed (adjusted for larger images)
        scrollbar.pack(side="bottom", fill="x")
        
        # Bind mousewheel to canvas
        def on_mousewheel(event):
            canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)
    
    print(f"Successfully displayed {len(loaded_images)} images")

def run_query_and_display():
    """Run query in background thread to prevent GUI freezing"""
    
    # Get input values
    text_input = text_box.get("1.0", tk.END).strip()
    image_path = image_path_var.get()
    input_type = input_type_var.get()

    # Validation
    if input_type == "text" and not text_input:
        messagebox.showwarning("Input Required", "Please enter text input.")
        return
    if input_type == "image" and not image_path:
        messagebox.showwarning("Input Required", "Please select an image.")
        return
    if input_type == "both" and (not text_input or not image_path):
        messagebox.showwarning("Input Required", "Please provide both text and image.")
        return

    # Disable run button and show progress
    run_button.config(state="disabled", text="Processing...")
    update_progress("Initializing query...")
    
    # Clear previous results
    clear_images()
    answer_display.config(state=tk.NORMAL)
    answer_display.delete("1.0", tk.END)
    answer_display.insert(tk.END, "Processing query, please wait...")
    answer_display.config(state=tk.DISABLED)
    
    def query_worker():
        """Worker function to run query in background"""
        try:
            # Run query with progress callback
            result = run_query(
                text_input=text_input if text_input else None,
                image_path=image_path if image_path else None,
                input_type=input_type,
                progress_callback=lambda msg: root.after(0, lambda: update_progress(msg))
            )

            if isinstance(result, str):
                answer = result
                matched_images = []
            else:
                answer, matched_images = result

            # Update GUI on main thread
            root.after(0, lambda: display_results(answer, matched_images))
            
        except Exception as e:
            error_msg = f"Error running query: {str(e)}"
            print(error_msg)
            root.after(0, lambda: display_error(error_msg))

    # Start query in background thread
    threading.Thread(target=query_worker, daemon=True).start()

def display_results(answer, matched_images):
    """Display query results on main thread"""
    # Display answer
    answer_display.config(state=tk.NORMAL)
    answer_display.delete("1.0", tk.END)
    answer_display.insert(tk.END, answer)
    answer_display.config(state=tk.DISABLED)
    
    # Display images
    display_images(matched_images)
    
    # Re-enable run button
    run_button.config(state="normal", text="🔍 Run Query")
    update_progress("Query completed")

def display_error(error_msg):
    """Display error message"""
    answer_display.config(state=tk.NORMAL)
    answer_display.delete("1.0", tk.END)
    answer_display.insert(tk.END, error_msg)
    answer_display.config(state=tk.DISABLED)
    
    run_button.config(state="normal", text="🔍 Run Query")
    update_progress("Error occurred")

# === GUI Setup ===
root = tk.Tk()
root.title("Multimodal RAG Viewer")
root.geometry("1200x900")  # Increased window size
root.configure(bg="white")

# Configure style
style = ttk.Style()
style.theme_use('clam')

# Title
title_label = tk.Label(
    root, 
    text="Multimodal RAG Viewer", 
    font=("Arial", 18, "bold"), 
    bg="white", 
    fg="darkblue"
)
title_label.pack(pady=(15, 10))

# Main frame
main_frame = tk.Frame(root, bg="white")
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

# Input Text Section
text_frame = tk.Frame(main_frame, bg="white")
text_frame.pack(fill=tk.X, pady=(0, 10))

tk.Label(text_frame, text="Enter Text Query:", font=("Arial", 10, "bold"), bg="white").pack(anchor="w")
text_box = tk.Text(text_frame, height=4, wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=2, font=("Arial", 9))
text_box.pack(fill=tk.X, pady=(5, 0))

# Image Browse Section
image_frame_input = tk.Frame(main_frame, bg="white")
image_frame_input.pack(fill=tk.X, pady=(0, 10))

tk.Label(image_frame_input, text="Select Image:", font=("Arial", 10, "bold"), bg="white").pack(anchor="w")
browse_frame = tk.Frame(image_frame_input, bg="white")
browse_frame.pack(fill=tk.X, pady=(5, 0))

browse_btn = tk.Button(
    browse_frame, 
    text="Browse Image", 
    command=browse_image, 
    bg="lightblue", 
    relief=tk.RAISED,
    font=("Arial", 9)
)
browse_btn.pack(side=tk.LEFT, padx=(0, 10))

image_path_var = tk.StringVar()
path_entry = tk.Entry(browse_frame, textvariable=image_path_var, relief=tk.SUNKEN, borderwidth=2, font=("Arial", 9))
path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

# Input Type Options
options_frame = tk.Frame(main_frame, bg="white")
options_frame.pack(fill=tk.X, pady=(0, 15))

tk.Label(options_frame, text="Select Input Type:", font=("Arial", 10, "bold"), bg="white").pack(anchor="w")
radio_frame = tk.Frame(options_frame, bg="white")
radio_frame.pack(anchor="w", pady=(5, 0))

input_type_var = tk.StringVar(value="text")
tk.Radiobutton(radio_frame, text="Text Only", variable=input_type_var, value="text", bg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 20))
tk.Radiobutton(radio_frame, text="Image Only", variable=input_type_var, value="image", bg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 20))
tk.Radiobutton(radio_frame, text="Text + Image", variable=input_type_var, value="both", bg="white", font=("Arial", 9)).pack(side=tk.LEFT)

# Run Button
run_button = tk.Button(
    main_frame, 
    text="🔍 Run Query", 
    command=run_query_and_display, 
    bg="green", 
    fg="white", 
    font=("Arial", 12, "bold"),
    height=2, 
    width=20,
    relief=tk.RAISED,
    cursor="hand2"
)
run_button.pack(pady=10)

# Progress Label
progress_label = tk.Label(main_frame, text="Ready", bg="white", fg="gray", font=("Arial", 9))
progress_label.pack(pady=(0, 5))

# Answer Display Section
answer_frame = tk.Frame(main_frame, bg="white")
answer_frame.pack(fill=tk.BOTH, expand=True)

tk.Label(answer_frame, text="Answer:", font=("Arial", 10, "bold"), bg="white").pack(anchor="w")
answer_display = tk.Text(
    answer_frame, 
    height=6,  # Reduced height to make room for larger images
    wrap=tk.WORD, 
    relief=tk.SUNKEN, 
    borderwidth=2,
    font=("Arial", 9),
    bg="#f8f8f8"
)
answer_display.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
answer_display.config(state=tk.DISABLED)

# Matched Images Display Section
tk.Label(main_frame, text="Matched Images:", font=("Arial", 10, "bold"), bg="white").pack(anchor="w", pady=(5, 5))
image_frame = tk.Frame(main_frame, bg="white", relief=tk.SUNKEN, borderwidth=1)
image_frame.pack(fill=tk.X, pady=(10, 100))

# Status bar
status_frame = tk.Frame(root, bg="lightgray", relief=tk.SUNKEN, borderwidth=1)
status_frame.pack(side=tk.BOTTOM, fill=tk.X)
status_label = tk.Label(status_frame, text="Multimodal RAG Viewer Ready", bg="lightgray", anchor="w", font=("Arial", 8))
status_label.pack(side=tk.LEFT, padx=10, pady=2)

# Handle window closing
def on_closing():
    root.quit()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the GUI
print("Starting Multimodal RAG Viewer...")
root.mainloop()
