import logging
from ultralytics import YOLO
from src.utils.train_settings import TrainSettings

logger = logging.getLogger(__name__)

def run_train(settings: TrainSettings):
    model = YOLO(f"models/{settings.model}")
    model.train(
        data=settings.data,
        epochs=settings.epochs,
        batch=settings.batch,
        imgsz=settings.imgsz,
        device=settings.device,
        cache=settings.cache,
        amp=settings.amp,
        cos_lr=settings.cos_lr,
        patience=settings.patience,
        close_mosaic=settings.close_mosiac,
    )