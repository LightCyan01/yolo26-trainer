from dataclasses import dataclass

@dataclass
class TrainSettings:
    model: str = ""
    data: str = ""
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    device: int = 0
    cache: bool = True
    amp: bool = True
    cos_lr: bool = True
    patience: int = 50
    close_mosiac: int = 10
