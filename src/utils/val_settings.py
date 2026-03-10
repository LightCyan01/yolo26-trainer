from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValSettings:
    use_custom_settings: bool = False
    # Core Validation Arguments
    data: str = ""                  # Path to dataset file (e.g., coco8.yaml)
    imgsz: int = 640                # Input image size for resizing
    batch: int = 16                 # Images per batch; -1 for AutoBatch
    save_json: bool = False         # Save results to JSON for COCO evaluation
    save_txt: bool = False          # Save detection results in text files
    save_conf: bool = False         # Include confidence scores in saved text files
    conf: float = 0.001             # Confidence threshold for detections
    iou: float = 0.7                # IoU threshold for NMS
    max_det: int = 300              # Max detections per image
    half: bool = False              # Use FP16 half-precision
    device: str = ""                # Device selection (e.g., cpu, cuda:0)
    dnn: bool = False               # Use OpenCV DNN for ONNX inference
    plots: bool = True              # Save PR curves and confusion matrices
    workers: int = 8                # Dataloader worker threads
    verbose: bool = True            # Display detailed per-class metrics
    project: str = ""               # Project directory to save validation outputs
    name: str = ""                  # Name of the validation run

    # Advanced Settings
    rect: bool = True               # Uses rectangular inference to reduce padding
    split: str = "val"              # Dataset split to use (train, val, or test)
    augment: bool = False           # Enables Test-Time Augmentation (TTA)
    agnostic_nms: bool = False      # Class-agnostic NMS for merging overlapping boxes
    single_cls: bool = False        # Treat all classes as a single class
    visualize: bool = False         # Visualize TPs, FPs, FNs per image
    compile: bool = False           # Enable torch.compile graph compilation
    classes: Optional[list] = None  # Filter validation to specific class IDs
    end2end: bool = False           # Overrides NMS-free mode for Ultralytics YOLO26
