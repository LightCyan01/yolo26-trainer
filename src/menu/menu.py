import torch
import questionary
from rich.console import Console
from src.utils.styles import q_style
from src.menu.train_menu import train_menu
from src.menu.val_menu import val_menu

console = Console()

def main_menu():
    console.clear()
    
    print("Yolo ToolKit")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print(f"Using CPU\n")
    
    return questionary.select("Main Menu", choices = ["Train", "Validate", "Predict", "Exit"], style=q_style).ask()

def train_option():
    return questionary.select("Select Training Task", choices = ["Object Detection", "Instance Segmentation",
                                                                 "Image Classification", "Pose Estimation", 
                                                                 "Oriented Bounding Boxes Object Detection"],style=q_style).ask()

def val_option():
    return questionary.select("Select Validation Task", choices = ["Object Detection", "Instance Segmentation",
                                                                   "Image Classification", "Pose Estimation",
                                                                   "Oriented Bounding Boxes Object Detection"], style=q_style).ask()

def predict_option():
    return questionary.select("Select Prediction Task", choices = ["Object Detection", "Instance Segmentation",
                                                                    "Image Classification", "Pose Estimation",
                                                                    "Oriented Bounding Boxes Object Detection"], style=q_style).ask()
    


        