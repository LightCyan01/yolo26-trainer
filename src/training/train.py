from ultralytics import YOLO
from src.utils.train_settings import TrainSettings

def run_train(settings: TrainSettings):
    model = YOLO(f"models/{settings.model}")

    if settings.hyperparam.tune:
        model.tune(data=settings.data)
        return

    model.train(
        data=settings.data,
        epochs=settings.epochs,
        batch=settings.batch,
        imgsz=settings.imgsz,
        device=settings.device,
        optimizer=settings.optimizer,
        patience=settings.patience,
        workers=settings.workers,
        seed=settings.seed,
        fraction=settings.fraction,
        cache=settings.cache,
        amp=settings.amp,
        cos_lr=settings.cos_lr,
        pretrained=settings.pretrained,
        rect=settings.rect,
        single_cls=settings.single_cls,
        deterministic=settings.deterministic,
        multi_scale=settings.multi_scale,
        plots=settings.plots,
        save_period=settings.save_period,
        **({
            "lr0":             settings.hyperparam.lr0,
            "lrf":             settings.hyperparam.lrf,
            "momentum":        settings.hyperparam.momentum,
            "weight_decay":    settings.hyperparam.weight_decay,
            "warmup_epochs":   settings.hyperparam.warmup_epochs,
            "warmup_momentum": settings.hyperparam.warmup_momentum,
            "warmup_bias_lr":  settings.hyperparam.warmup_bias_lr,
            "box":             settings.hyperparam.box,
            "cls":             settings.hyperparam.cls,
            "nbs":             settings.hyperparam.nbs,
            "dropout":         settings.hyperparam.dropout,
        } if settings.hyperparam.enabled else {}),
    )