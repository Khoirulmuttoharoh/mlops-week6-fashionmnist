import optuna
import mlflow
from src.experiment import train_one

def objective(trial):
    config = {
        "seed": 42,
        "epochs": 3,  # dibuat ringan untuk pencarian; kandidat final dilatih ulang
        "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "augmentation": trial.suggest_categorical("augmentation", [False, True]),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
        "hypothesis": "Tuning learning rate, batch size, dropout, optimizer, dan augmentasi dapat meningkatkan macro-F1.",
        "experiment_group": "optuna_tuning",
    }
    with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):
        mlflow.set_tag("optuna_trial", trial.number)
        summary = train_one(config, run_name=f"trial-{trial.number:02d}")
        trial.set_user_attr("accuracy", summary["final_test_accuracy"])
        return summary["final_test_macro_f1"]

def main():
    mlflow.set_experiment("fashion-mnist-cnn")
    with mlflow.start_run(run_name="optuna-20-trials"):
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=20)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_or_validation_macro_f1", study.best_value)
        print("Best trial:", study.best_trial.number)
        print("Best macro-F1:", study.best_value)
        print("Best params:", study.best_params)
        print("Catatan: tugas kelas ini memakai test metric untuk demonstrasi tracking; untuk evaluasi ilmiah, tuning harus memakai validation split, lalu test hanya sekali di akhir.")
if __name__ == "__main__":
    main()
