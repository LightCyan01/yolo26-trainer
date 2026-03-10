import os
from ultralytics import YOLO
from src.utils.val_settings import ValSettings


def run_val(settings: ValSettings, model: str):
    model_path = (f"models/{model}")
    m = YOLO(model_path)

    if not settings.use_custom_settings:
        m.val()
        return

    m.val(
        data=settings.data,
        imgsz=settings.imgsz,
        batch=settings.batch,
        save_json=settings.save_json,
        save_txt=settings.save_txt,
        save_conf=settings.save_conf,
        conf=settings.conf,
        iou=settings.iou,
        max_det=settings.max_det,
        half=settings.half,
        device=settings.device,
        dnn=settings.dnn,
        plots=settings.plots,
        workers=settings.workers,
        verbose=settings.verbose,
        project=settings.project,
        name=settings.name,
        rect=settings.rect,
        split=settings.split,
        augment=settings.augment,
        agnostic_nms=settings.agnostic_nms,
        single_cls=settings.single_cls,
        visualize=settings.visualize,
        compile=settings.compile,
        classes=settings.classes,
        end2end=settings.end2end,
    )