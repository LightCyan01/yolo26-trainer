from src.training import run_train
from src.menu import main_menu, task_menu, train_option

def main():
    task = main_menu()
    
    if task == "Train":
        selected = train_option()
        result = task_menu(selected)
        
        if result:
            run_train(
                model=result["model"],
                data=result["data"]
            )

if __name__ == "__main__":
    main()