from dataclasses import dataclass, field


@dataclass
class HyperparamSettings:
    enabled: bool = False           # Enable custom hyperparameters for training
    lr0: float = 0.01               # Initial learning rate
    lrf: float = 0.01               # Final learning rate as a fraction of lr0
    momentum: float = 0.937         # SGD momentum / Adam beta1
    weight_decay: float = 0.0005    # L2 regularization penalty
    warmup_epochs: float = 3.0      # Number of warmup epochs
    warmup_momentum: float = 0.8    # Initial momentum during warmup
    warmup_bias_lr: float = 0.1     # Initial bias learning rate during warmup
    box: float = 7.5                # Weight of the box regression loss
    cls: float = 0.5                # Weight of the classification loss
    nbs: int = 64                   # Nominal batch size for loss normalization
    dropout: float = 0.0            # Dropout rate for classification head


@dataclass
class AugmentationSettings:
    hsv_h: float = 0.015            # Random hue shift (fraction of 360 degrees)
    hsv_s: float = 0.7              # Random saturation shift
    hsv_v: float = 0.4              # Random value/brightness shift
    degrees: float = 0.0            # Random rotation range in degrees
    translate: float = 0.1          # Random translation as fraction of image size
    scale: float = 0.5              # Random scale gain
    shear: float = 0.0              # Random shear angle in degrees
    perspective: float = 0.0        # Random perspective transformation
    flipud: float = 0.0             # Probability of vertical flip
    fliplr: float = 0.5             # Probability of horizontal flip
    bgr: float = 0.0                # Probability of BGR channel swap
    mosaic: float = 1.0             # Probability of mosaic augmentation
    mixup: float = 0.0              # Probability of mixup augmentation
    cutmix: float = 0.0             # Probability of cutmix augmentation
    copy_paste: float = 0.0         # Probability of segment copy-paste
    copy_paste_mode: str = "flip"   # Copy-paste mode (flip or mixup)
    auto_augment: str = "randaugment"  # Auto augment policy
    erasing: float = 0.4            # Probability of random erasing in classification
    close_mosaic: int = 10          # Disable mosaic for last N epochs


@dataclass
class TrainSettings:
    model: str = ""                 # Path to model file (e.g., yolo11n.pt)
    data: str = ""                  # Path to dataset YAML file
    epochs: int = 100               # Total training epochs
    batch: int = 16                 # Images per batch; -1 for AutoBatch
    imgsz: int = 640                # Input image size
    device: int = 0                 # Device to train on (e.g., 0 for GPU, cpu)
    optimizer: str = "auto"         # Optimizer to use (auto, SGD, Adam, etc.)
    patience: int = 100             # Early stopping patience epochs
    workers: int = 8                # Dataloader worker threads
    seed: int = 0                   # Random seed for reproducibility
    fraction: float = 1.0           # Fraction of dataset to train on
    cache: bool = False             # Cache images for faster training
    amp: bool = True                # Automatic Mixed Precision training
    cos_lr: bool = False            # Use cosine learning rate scheduler
    pretrained: bool = True         # Use pretrained weights
    rect: bool = False              # Rectangular training for less padding
    single_cls: bool = False        # Train as single-class dataset
    deterministic: bool = True      # Force deterministic algorithms
    multi_scale: float = 0.0        # Multi-scale training image size variation
    plots: bool = True              # Save training plots and metrics
    save_period: int = -1           # Save checkpoint every N epochs (-1 to disable)
    hyperparam: HyperparamSettings = field(default_factory=HyperparamSettings)
