import torch
import questionary
from questionary import Choice
from rich.console import Console
from pathlib import Path
from src.utils.styles import q_style
from src.utils.models import DETECTION, INSTANCE_SEGMENTATION, OBB, CLASSIFICATION, POSE_ESTIMATION
from src.utils.file_dialog import ask_yaml_file
from src.utils.train_settings import TrainSettings

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
    settings = TrainSettings()
    
    while(True):
        dataset_label = f"Dataset yaml path: {dataset}" if dataset else "Dataset yaml path"
        model_label = f"Model: {model}" if model else "Select Model"
        ready = dataset and model
        
        choices = [
            Choice(title=dataset_label, value="dataset"),
            Choice(title=model_label, value="model"),
            Choice(
                title="Train Settings",
                value="train"
            ),
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
        elif choice == "train":
            settings = settings_menu(settings)
        elif choice == "start":
            settings.model = model
            settings.data = dataset
            return settings
            
            
def settings_menu(settings: TrainSettings):
    BOOL_FIELDS = {"cache", "amp", "cos_lr"}

    while True:
        choices = [
            Choice(title=f"Epochs:        {settings.epochs}",      value="epochs"),
            Choice(title=f"Batch:         {settings.batch}",       value="batch"),
            Choice(title=f"Image Size:    {settings.imgsz}",       value="imgsz"),
            Choice(title=f"Device:        {settings.device}",      value="device"),
            Choice(title=f"Cache:         {settings.cache}",       value="cache"),
            Choice(title=f"AMP:           {settings.amp}",         value="amp"),
            Choice(title=f"Cosine LR:     {settings.cos_lr}",      value="cos_lr"),
            Choice(title=f"Patience:      {settings.patience}",    value="patience"),
            Choice(title=f"Close Mosaic:  {settings.close_mosiac}", value="close_mosiac"),
            Choice(title="Back",                                    value="back"),
        ]

        choice = questionary.select("Train Settings", choices=choices, style=q_style).ask()

        if choice is None or choice == "back":
            return settings

        if choice in BOOL_FIELDS:
            val = questionary.confirm(
                f"{choice}", default=getattr(settings, choice), style=q_style
            ).ask()
            if val is not None:
                setattr(settings, choice, val)
        else:
            val = questionary.text(
                f"{choice}",
                default=str(getattr(settings, choice)),
                validate=lambda v: v.isdigit() or "Enter a valid integer",
                style=q_style
            ).ask()
            if val is not None:
                setattr(settings, choice, int(val))


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
    


        