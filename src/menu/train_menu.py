import questionary
from questionary import Choice
from src.utils.train_settings import TrainSettings, HyperparamSettings
from src.utils.styles import q_style
from src.utils.models import DETECTION, INSTANCE_SEGMENTATION, OBB, CLASSIFICATION, POSE_ESTIMATION
from src.utils.file_dialog import ask_yaml_file
from src.utils.validators import validate_int, validate_float


def train_menu(selected):
    dataset = None
    model = None
    settings = TrainSettings()

    while True:
        dataset_label = f"Dataset yaml path: {dataset}" if dataset else "Dataset yaml path"
        model_label = f"Model: {model}" if model else "Select Model"
        ready = dataset and model

        choices = [
            Choice(title=dataset_label, value="dataset"),
            Choice(title=model_label,   value="model"),
            Choice(title="Train Settings",          value="train"),
            Choice(title="Hyperparameter Settings", value="hyperparam"),
            Choice(
                title="Start Train",
                value="start",
                disabled="Select dataset and model first" if not ready else None
            ),
        ]

        choice = questionary.select("Training Setup", choices=choices, style=q_style).ask()

        if choice is None:
            return None
        elif choice == "dataset":
            result = ask_yaml_file()
            if result:
                dataset = result
        elif choice == "model":
            result = model_selection(selected)
            if result:
                model = result
        elif choice == "train":
            settings = settings_menu(settings)
        elif choice == "hyperparam":
            settings.hyperparam = hyperparam_menu(settings.hyperparam)
        elif choice == "start":
            settings.model = model
            settings.data = dataset
            return settings

def model_selection(selected):
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

def settings_menu(settings: TrainSettings):
    BOOL_FIELDS = {"cache", "amp", "cos_lr", "pretrained", "rect", "single_cls", "deterministic", "plots"}
    FLOAT_FIELDS = {"fraction", "multi_scale"}
    STR_FIELDS = {"optimizer"}

    while True:
        choices = [
            Choice(title=f"Epochs:          {settings.epochs}",       value="epochs"),
            Choice(title=f"Batch:           {settings.batch}",        value="batch"),
            Choice(title=f"Image Size:      {settings.imgsz}",        value="imgsz"),
            Choice(title=f"Device:          {settings.device}",       value="device"),
            Choice(title=f"Optimizer:       {settings.optimizer}",    value="optimizer"),
            Choice(title=f"Patience:        {settings.patience}",     value="patience"),
            Choice(title=f"Workers:         {settings.workers}",      value="workers"),
            Choice(title=f"Seed:            {settings.seed}",         value="seed"),
            Choice(title=f"Fraction:        {settings.fraction}",     value="fraction"),
            Choice(title=f"Cache:           {settings.cache}",        value="cache"),
            Choice(title=f"AMP:             {settings.amp}",          value="amp"),
            Choice(title=f"Cosine LR:       {settings.cos_lr}",       value="cos_lr"),
            Choice(title=f"Pretrained:      {settings.pretrained}",   value="pretrained"),
            Choice(title=f"Rect:            {settings.rect}",         value="rect"),
            Choice(title=f"Single Class:    {settings.single_cls}",   value="single_cls"),
            Choice(title=f"Deterministic:   {settings.deterministic}", value="deterministic"),
            Choice(title=f"Multi Scale:     {settings.multi_scale}",  value="multi_scale"),
            Choice(title=f"Plots:           {settings.plots}",        value="plots"),
            Choice(title=f"Save Period:     {settings.save_period}",  value="save_period"),
            Choice(title="Back",                                       value="back"),
        ]

        choice = questionary.select("Train Settings", choices=choices, style=q_style).ask()

        if choice is None or choice == "back":
            return settings

        if choice in BOOL_FIELDS:
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
        elif choice in STR_FIELDS:
            val = questionary.text(
                f"{choice}",
                default=str(getattr(settings, choice)),
                style=q_style
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


def hyperparam_menu(hp: HyperparamSettings):
    FLOAT_FIELDS = {"lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
                    "warmup_momentum", "warmup_bias_lr", "box", "cls", "dropout"}

    while True:
        enabled_label = f"[{'x' if hp.enabled else ' '}] Manual Hyperparameters"
        tune_label    = f"[{'x' if hp.tune else ' '}] Auto-Tune (model.tune())"
        choices = [
            Choice(title=enabled_label,                                    value="enabled"),
            Choice(title=tune_label,                                       value="tune"),
            Choice(title=f"--- Manual Settings ---",                       value="_divider1", disabled=" "),
            Choice(title=f"LR0:              {hp.lr0}",                    value="lr0"),
            Choice(title=f"LRF:              {hp.lrf}",                    value="lrf"),
            Choice(title=f"Momentum:         {hp.momentum}",               value="momentum"),
            Choice(title=f"Weight Decay:     {hp.weight_decay}",           value="weight_decay"),
            Choice(title=f"Warmup Epochs:    {hp.warmup_epochs}",          value="warmup_epochs"),
            Choice(title=f"Warmup Momentum:  {hp.warmup_momentum}",        value="warmup_momentum"),
            Choice(title=f"Warmup Bias LR:   {hp.warmup_bias_lr}",         value="warmup_bias_lr"),
            Choice(title=f"Box:              {hp.box}",                    value="box"),
            Choice(title=f"CLS:              {hp.cls}",                    value="cls"),
            Choice(title=f"NBS:              {hp.nbs}",                    value="nbs"),
            Choice(title=f"Dropout:          {hp.dropout}",                value="dropout"),
            Choice(title="Back",                                            value="back"),
        ]

        choice = questionary.select("Hyperparameter Settings", choices=choices, style=q_style).ask()

        if choice is None or choice == "back" or choice.startswith("_"):
            return hp

        if choice == "enabled":
            hp.enabled = not hp.enabled
            if hp.enabled:
                hp.tune = False
        elif choice == "tune":
            hp.tune = not hp.tune
            if hp.tune:
                hp.enabled = False
        elif choice in FLOAT_FIELDS:
            val = questionary.text(
                f"{choice}",
                default=str(getattr(hp, choice)),
                validate=validate_float,
                style=q_style
            ).ask()
            if val is not None:
                setattr(hp, choice, float(val))
        else:
            val = questionary.text(
                f"{choice}",
                default=str(getattr(hp, choice)),
                validate=validate_int,
                style=q_style
            ).ask()
            if val is not None:
                setattr(hp, choice, int(val))