import torch
import questionary
from questionary import Choice
from rich.console import Console
from pathlib import Path
from src.utils.styles import q_style
from src.utils.models import DETECTION, INSTANCE_SEGMENTATION, OBB, CLASSIFICATION, POSE_ESTIMATION
from src.utils.file_dialog import ask_yaml_file

console = Console()

def main_menu():
    console.clear()
    
    print("Yolo ToolKit")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print(f"Using CPU\n")
    
    return questionary.select("Main Menu", choices = ["Train", "Validate", "Predict"], style=q_style).ask()

def train_option():
    return questionary.select("Select Training Task", choices = ["Object Detection", "Instance Segmentation",
                                                                 "Image Classification", "Pose Estimation", 
                                                                 "Oriented Bounding Boxes Object Detection"],style=q_style).ask()

def task_menu(selected):
    dataset = None
    model = None
    
    while(True):
        dataset_label = f"Dataset yaml path: {dataset}" if dataset else "Dataset yaml path"
        model_label = f"Model: {model}" if model else "Select Model"
        ready = dataset and model
        
        choices = [
            Choice(title=dataset_label, value="dataset"),
            Choice(title=model_label, value="model"),
            Choice(
                title="Start Train",
                value="start",
                disabled="Select dataset and model first" if not ready else None
            ),
        ]
        
        choice = questionary.select("Training Setup", choices=choices, style=q_style).ask()
        
        if choice is None:
            return None
        elif choice == "dataset":
            result = ask_yaml_file()
            if result:
                dataset = result
        elif choice == "model":
            result = model_selection(selected)
            if result:
                model = result
        elif choice == "start":
            return {
                "model": model,
                "data": dataset
            }
            
            
def model_selection(selected):
    if selected == "Object Detection":
        return questionary.select("Select Model", choices=DETECTION, style=q_style).ask()
    elif selected == "Instance Segmentation":
        return questionary.select("Select Model", choices=INSTANCE_SEGMENTATION, style=q_style).ask()
    elif selected == "Image Classification":
        return questionary.select("Select Model", choices=CLASSIFICATION, style=q_style).ask()
    elif selected == "Pose Estimation":
        return questionary.select("Select Model", choices=POSE_ESTIMATION, style=q_style).ask()
    elif selected == "Oriented Bounding Boxes Object Detection":
        return questionary.select("Select Model", choices=OBB, style=q_style).ask()
    


        