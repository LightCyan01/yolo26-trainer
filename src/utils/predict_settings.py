from dataclasses import dataclass
from typing import Optional


@dataclass
class PredictSettings:
    use_custom_settings: bool = False   # Use custom settings; if False, uses defaults
    source: str = ""                    # Input source (0 for webcam, URL for YouTube)
    # Core Inference Arguments
    imgsz: int = 640                    # Input image size
    conf: float = 0.25                  # Confidence threshold
    iou: float = 0.7                    # IoU threshold for NMS
    max_det: int = 300                  # Maximum detections per image
    batch: int = 1                      # Batch size (directory/video only)
    device: str = ""                    # Device to run on (e.g., cpu, cuda:0)
    half: bool = False                  # Use FP16 half-precision
    rect: bool = True                   # Rectangular inference for faster processing
    vid_stride: int = 1                 # Frame stride for video sources
    stream_buffer: bool = False         # Queue incoming frames for video streams
    visualize: bool = False             # Visualize model features during inference
    augment: bool = False               # Test-Time Augmentation
    agnostic_nms: bool = False          # Class-agnostic NMS
    retina_masks: bool = False          # Return high-resolution segmentation masks
    classes: Optional[list] = None      # Filter by class IDs
    stream: bool = False                # Return a generator to save memory on long videos
    verbose: bool = True                # Print results to console
    compile: bool = False               # Enable torch.compile for PyTorch 2.x
    end2end: bool = False               # Override NMS-free mode for YOLO26
    # Visualization & Saving Arguments
    show: bool = False                  # Display results in a window
    save: bool = False                  # Save results to disk
    save_frames: bool = False           # Save individual video frames as images
    save_txt: bool = False              # Save results as text files
    save_conf: bool = False             # Include confidence in saved text files
    save_crop: bool = False             # Save cropped detection images
    show_labels: bool = True            # Show class labels on results
    show_conf: bool = True              # Show confidence scores on results
    show_boxes: bool = True             # Show bounding boxes on results
    line_width: Optional[int] = None    # Line width of boxes (None = auto)
    project: str = ""                   # Project directory for saved results
    name: str = ""                      # Name of the prediction run
