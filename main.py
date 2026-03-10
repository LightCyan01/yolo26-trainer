import sys
from src.training.train import run_train
from src.training.val import run_val
from src.training.predict import run_predict
from src.menu.menu import main_menu, train_option, val_option, predict_option
from src.menu.train_menu import train_menu
from src.menu.val_menu import val_menu
from src.menu.predict_menu import predict_menu

def main():
    while True:
        task = main_menu()

        if task is None or task == "Exit":
            sys.exit(0)

        if task == "Train":
            selected = train_option()
            if selected is None:
                continue
            result = train_menu(selected)
            if result:
                run_train(result)

        elif task == "Validate":
            selected = val_option()
            if selected is None:
                continue
            result = val_menu(selected)
            if result:
                settings, model = result
                run_val(settings, model)

        elif task == "Predict":
            selected = predict_option()
            if selected is None:
                continue
            result = predict_menu(selected)
            if result:
                settings, model = result
                run_predict(settings, model)

if __name__ == "__main__":
    main()