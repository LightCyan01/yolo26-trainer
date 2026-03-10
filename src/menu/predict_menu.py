import questionary
from questionary import Choice
from src.utils.predict_settings import PredictSettings
from src.utils.styles import q_style
from src.utils.models import DETECTION, INSTANCE_SEGMENTATION, OBB, CLASSIFICATION, POSE_ESTIMATION
from src.utils.file_dialog import ask_model_file, ask_video_file
from src.utils.validators import validate_int, validate_float


def predict_menu(selected):
    model = None
    settings = PredictSettings()

    while True:
        model_label    = f"Model: {model}" if model else "Select Model"
        source_label   = f"Source: {settings.source}" if settings.source else "Select Source"
        settings_label = f"Custom Predict Settings"
        ready = model and settings.source

        choices = [
            Choice(title=model_label,    value="model"),
            Choice(title=source_label,   value="source"),
            Choice(title=settings_label, value="settings"),
            Choice(
                title="Start Predict",
                value="start",
                disabled="Select model and source first" if not ready else None
            ),
        ]

        choice = questionary.select("Prediction Setup", choices=choices, style=q_style).ask()

        if choice is None:
            return None
        elif choice == "model":
            result = predict_model_selection(selected)
            if result:
                model = result
        elif choice == "source":
            result = predict_source_selection()
            if result is not None:
                settings.source = result
        elif choice == "settings":
            settings = predict_settings_menu(settings)
        elif choice == "start":
            return settings, model


def predict_model_selection(selected):
    source = questionary.select(
        "Model source",
        choices=[
            Choice(title="Official Model",       value="official"),
            Choice(title="Custom Trained Model", value="custom"),
        ],
        style=q_style
    ).ask()

    if source is None:
        return None

    if source == "custom":
        return ask_model_file()

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


def predict_source_selection():
    source_type = questionary.select(
        "Select source type",
        choices=[
            Choice(title="Webcam",      value="webcam"),
            Choice(title="Video File",  value="video"),
        ],
        style=q_style
    ).ask()

    if source_type is None:
        return None

    if source_type == "webcam":
        return "0"

    return ask_video_file()


def predict_settings_menu(settings: PredictSettings) -> PredictSettings:
    BOOL_FIELDS  = {"half", "rect", "stream_buffer", "visualize", "augment", "agnostic_nms",
                    "retina_masks", "stream", "verbose", "compile", "end2end",
                    "show", "save", "save_frames", "save_txt", "save_conf", "save_crop",
                    "show_labels", "show_conf", "show_boxes"}
    FLOAT_FIELDS = {"conf", "iou"}

    while True:
        enabled_label = f"[{'x' if settings.use_custom_settings else ' '}] Enable Custom Settings"
        choices = [
            Choice(title=enabled_label,                                         value="use_custom_settings"),
            Choice(title=f"Image Size:       {settings.imgsz}",                value="imgsz"),
            Choice(title=f"Confidence:       {settings.conf}",                 value="conf"),
            Choice(title=f"IoU:              {settings.iou}",                  value="iou"),
            Choice(title=f"Max Det:          {settings.max_det}",              value="max_det"),
            Choice(title=f"Batch:            {settings.batch}",                value="batch"),
            Choice(title=f"Device:           {settings.device or 'auto'}",     value="device"),
            Choice(title=f"Half Precision:   {settings.half}",                 value="half"),
            Choice(title=f"Rect:             {settings.rect}",                 value="rect"),
            Choice(title=f"Frame Stride:     {settings.vid_stride}",           value="vid_stride"),
            Choice(title=f"Stream Buffer:    {settings.stream_buffer}",        value="stream_buffer"),
            Choice(title=f"Visualize:        {settings.visualize}",            value="visualize"),
            Choice(title=f"Augment (TTA):    {settings.augment}",              value="augment"),
            Choice(title=f"Agnostic NMS:     {settings.agnostic_nms}",         value="agnostic_nms"),
            Choice(title=f"Retina Masks:     {settings.retina_masks}",         value="retina_masks"),
            Choice(title=f"Stream:           {settings.stream}",               value="stream"),
            Choice(title=f"Verbose:          {settings.verbose}",              value="verbose"),
            Choice(title=f"Compile:          {settings.compile}",              value="compile"),
            Choice(title=f"End2End:          {settings.end2end}",              value="end2end"),
            Choice(title=f"Show:             {settings.show}",                 value="show"),
            Choice(title=f"Save:             {settings.save}",                 value="save"),
            Choice(title=f"Save Frames:      {settings.save_frames}",          value="save_frames"),
            Choice(title=f"Save TXT:         {settings.save_txt}",             value="save_txt"),
            Choice(title=f"Save Conf:        {settings.save_conf}",            value="save_conf"),
            Choice(title=f"Save Crop:        {settings.save_crop}",            value="save_crop"),
            Choice(title=f"Show Labels:      {settings.show_labels}",          value="show_labels"),
            Choice(title=f"Show Conf:        {settings.show_conf}",            value="show_conf"),
            Choice(title=f"Show Boxes:       {settings.show_boxes}",           value="show_boxes"),
            Choice(title=f"Line Width:       {settings.line_width or 'auto'}", value="line_width"),
            Choice(title=f"Project:          {settings.project or 'default'}", value="project"),
            Choice(title=f"Name:             {settings.name or 'default'}",    value="name"),
            Choice(title="Back",                                                value="back"),
        ]

        choice = questionary.select("Predict Settings", choices=choices, style=q_style).ask()

        if choice is None or choice == "back":
            return settings

        if choice == "use_custom_settings":
            settings.use_custom_settings = not settings.use_custom_settings
        elif choice in BOOL_FIELDS:
            val = questionary.confirm(
                f"{choice}", default=getattr(settings, choice), style=q_style
            ).ask()
            if val is not None:
                setattr(settings, choice, val)
        elif choice in FLOAT_FIELDS:
            val = questionary.text(
                f"{choice}",
                default=str(getattr(settings, choice)),
                validate=validate_float,
                style=q_style
            ).ask()
            if val is not None:
                setattr(settings, choice, float(val))
        elif choice in {"device", "project", "name"}:
            val = questionary.text(
                f"{choice}", default=getattr(settings, choice), style=q_style
            ).ask()
            if val is not None:
                setattr(settings, choice, val)
        elif choice == "line_width":
            val = questionary.text(
                "line_width (leave blank for auto)",
                default=str(settings.line_width) if settings.line_width is not None else "",
                style=q_style
            ).ask()
            if val is not None:
                settings.line_width = int(val) if val.strip() else None
        else:
            val = questionary.text(
                f"{choice}",
                default=str(getattr(settings, choice)),
                validate=validate_int,
                style=q_style
            ).ask()
            if val is not None:
                setattr(settings, choice, int(val))
