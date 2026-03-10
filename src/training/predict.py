import os
from ultralytics import YOLO
from src.utils.predict_settings import PredictSettings

def run_predict(settings: PredictSettings, model: str):
    model_path = model if os.path.isabs(model) else f"models/{model}"
    m = YOLO(model_path)

    if not settings.use_custom_settings:
        results = m.predict(source=settings.source, show=True, stream=True)
        for _ in results:
            pass
        return

    results = m.predict(
        source=settings.source,
        imgsz=settings.imgsz,
        conf=settings.conf,
        iou=settings.iou,
        max_det=settings.max_det,
        batch=settings.batch,
        device=settings.device or None,
        half=settings.half,
        rect=settings.rect,
        vid_stride=settings.vid_stride,
        stream_buffer=settings.stream_buffer,
        visualize=settings.visualize,
        augment=settings.augment,
        agnostic_nms=settings.agnostic_nms,
        retina_masks=settings.retina_masks,
        classes=settings.classes,
        stream=settings.stream,
        verbose=settings.verbose,
        compile=settings.compile,
        end2end=settings.end2end,
        show=settings.show,
        save=settings.save,
        save_frames=settings.save_frames,
        save_txt=settings.save_txt,
        save_conf=settings.save_conf,
        save_crop=settings.save_crop,
        show_labels=settings.show_labels,
        show_conf=settings.show_conf,
        show_boxes=settings.show_boxes,
        line_width=settings.line_width,
        project=settings.project or None,
        name=settings.name or None,
    )
    if settings.stream:
        for _ in results:
            pass
