import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def run_train(model: str, data: str) -> Path:
    logger.info("Starting training with\nModel: %s\nData: %s", model, data)
    
    model = YOLO(f"models/{model}")
    results = model.train(data=data, epochs=100)
    save_dir = Path(results.save_dir)
    return save_dir