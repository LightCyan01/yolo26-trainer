import questionary
from questionary import Choice
from src.utils.val_settings import ValSettings
from src.utils.styles import q_style
from src.utils.models import DETECTION, INSTANCE_SEGMENTATION, OBB, CLASSIFICATION, POSE_ESTIMATION
from src.utils.file_dialog import ask_yaml_file, ask_model_file
from src.utils.validators import validate_int, validate_float


def val_menu(selected):
    dataset = None
    model = None
    settings = ValSettings()

    while True:
        dataset_label  = f"Dataset yaml: {dataset}" if dataset else "Dataset yaml (optional)"
        model_label    = f"Model: {model}" if model else "Select Model"
        settings_label = f"[{'x' if settings.use_custom_settings else ' '}] Custom Val Settings"
        ready = model is not None

        choices = [
            Choice(title=model_label,    value="model"),
            Choice(title=dataset_label,  value="dataset"),
            Choice(title=settings_label, value="settings"),
            Choice(
                title="Start Validate",
                value="start",
                disabled="Select a model first" if not ready else None
            ),
        ]

        choice = questionary.select("Validation Setup", choices=choices, style=q_style).ask()

        if choice is None:
            return None
        elif choice == "dataset":
            result = ask_yaml_file()
            if result:
                dataset = result
        elif choice == "model":
            result = val_model_selection(selected)
            if result:
                model = result
        elif choice == "settings":
            settings = val_settings_menu(settings)
        elif choice == "start":
            if dataset:
                settings.data = dataset
            return settings, model


def val_model_selection(selected):
    source = questionary.select(
        "Model source",
        choices=[
            Choice(title="Official Model",        value="official"),
            Choice(title="Custom Trained Model",   value="custom"),
        ],
        style=q_style
    ).ask()

    if source is None:
        return None

    if source == "custom":
        return ask_model_file()

    # Official model — pick from task list
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


def val_settings_menu(settings: ValSettings) -> ValSettings:
    BOOL_FIELDS  = {"save_json", "save_txt", "save_conf", "half", "dnn", "plots",
                    "rect", "augment", "agnostic_nms", "single_cls", "visualize",
                    "compile", "end2end", "verbose"}
    FLOAT_FIELDS = {"conf", "iou"}

    while True:
        enabled_label = f"[{'x' if settings.use_custom_settings else ' '}] Enable Custom Settings"
        choices = [
            Choice(title=enabled_label,                                    value="use_custom_settings"),
            Choice(title=f"Image Size:      {settings.imgsz}",             value="imgsz"),
            Choice(title=f"Batch:           {settings.batch}",             value="batch"),
            Choice(title=f"Confidence:      {settings.conf}",              value="conf"),
            Choice(title=f"IoU:             {settings.iou}",               value="iou"),
            Choice(title=f"Max Det:         {settings.max_det}",           value="max_det"),
            Choice(title=f"Workers:         {settings.workers}",           value="workers"),
            Choice(title=f"Device:          {settings.device or 'auto'}",  value="device"),
            Choice(title=f"Split:           {settings.split}",             value="split"),
            Choice(title=f"Project:         {settings.project or 'default'}", value="project"),
            Choice(title=f"Name:            {settings.name or 'default'}", value="name"),
            Choice(title=f"Save JSON:       {settings.save_json}",         value="save_json"),
            Choice(title=f"Save TXT:        {settings.save_txt}",          value="save_txt"),
            Choice(title=f"Save Conf:       {settings.save_conf}",         value="save_conf"),
            Choice(title=f"Half Precision:  {settings.half}",              value="half"),
            Choice(title=f"DNN:             {settings.dnn}",               value="dnn"),
            Choice(title=f"Plots:           {settings.plots}",             value="plots"),
            Choice(title=f"Verbose:         {settings.verbose}",           value="verbose"),
            Choice(title=f"Rect:            {settings.rect}",              value="rect"),
            Choice(title=f"Augment (TTA):   {settings.augment}",           value="augment"),
            Choice(title=f"Agnostic NMS:    {settings.agnostic_nms}",      value="agnostic_nms"),
            Choice(title=f"Single Class:    {settings.single_cls}",        value="single_cls"),
            Choice(title=f"Visualize:       {settings.visualize}",         value="visualize"),
            Choice(title=f"Compile:         {settings.compile}",           value="compile"),
            Choice(title=f"End2End:         {settings.end2end}",           value="end2end"),
            Choice(title="Back",                                            value="back"),
        ]

        choice = questionary.select("Val Settings", choices=choices, style=q_style).ask()

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
        elif choice == "split":
            val = questionary.select(
                "split", choices=["train", "val", "test"], style=q_style
            ).ask()
            if val is not None:
                settings.split = val
        elif choice in {"project", "name", "device"}:
            val = questionary.text(
                f"{choice}", default=getattr(settings, choice), style=q_style
            ).ask()
            if val is not None:
                setattr(settings, choice, val)
        else:
            val = questionary.text(
                f"{choice}",
                default=str(getattr(settings, choice)),
                validate=validate_int,
                style=q_style
            ).ask()
            if val is not None:
                setattr(settings, choice, int(val))
