import tkinter as tk
from tkinter import filedialog

def ask_yaml_file() -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select Dataset YAML",
        filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
    )
    root.destroy()
    return path or None

def ask_model_file() -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select Custom Model (.pt)",
        filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")]
    )
    root.destroy()
    return path or None