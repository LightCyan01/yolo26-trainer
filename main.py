from src.training.train import run_train
from src.training.val import run_val
from src.menu.menu import main_menu, train_option, val_option
from src.menu.train_menu import train_menu
from src.menu.val_menu import val_menu

def main():
    task = main_menu()
    
    if task == "Train":
        selected = train_option()
        result = train_menu(selected)
        if result:
            run_train(result)

    elif task == "Validate":
        selected = val_option()
        result = val_menu(selected)
        if result:
            settings, model = result
            run_val(settings, model)

if __name__ == "__main__":
    main()