import tkinter as tk
from tkinter import filedialog
import os
from pathlib import Path
from mask_to_yolo_bbox import mask_to_yolo_bbox

root = tk.Tk()
root.withdraw()

def select_folder(title: str):
    result = filedialog.askdirectory(
        parent=root,
        initialdir=os.getcwd(),
        title=title
    )
    directory_path = Path(result) if result else None
    return directory_path

def process_label(mask_dir: Path, image_dir: Path, class_id: int, output_dir: Path):
    for mask_path in sorted(mask_dir.glob("*_mask.png")):
        image_name = mask_path.name.replace("_mask", "")
        image_path = image_dir / image_name
        label_path = output_dir /mask_path.name.replace("_mask.png", ".txt")
        
        mask_to_yolo_bbox(mask_path, image_path, class_id, label_path)
        print(f"Processed: {mask_path.name}")

def process_good_images(image_dir: Path, output_dir: Path):
    for img in image_dir.glob("*.png"):
        (output_dir / img.name.replace(".png", ".txt")).touch()
        print(f"processed good image: {img}")
    
if __name__ == "__main__":
    class_id = int(input("Enter class ID for this folder (only numbers): "))
    mask_dir: Path = select_folder("Select Mask Path")
    image_dir: Path = select_folder("Select Image Path")
    output_dir: Path = select_folder("Select Output Path")
    process_label(mask_dir, image_dir, class_id, output_dir)
    good_image_dir = select_folder("Select Good Images Path")
    good_image_output = select_folder("Select Output Folder")
    process_good_images(good_image_dir, good_image_output)
    
    
