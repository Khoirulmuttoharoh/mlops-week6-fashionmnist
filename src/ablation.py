import mlflow
from src.experiment import train_one

def main():
    mlflow.set_experiment("fashion-mnist-cnn")
    # Ubah best_config setelah melihat hasil tuning.
    best_config = {"seed": 42, "epochs": 5, "batch_size": 128,
                   "learning_rate": 0.001, "dropout": 0.25,
                   "augmentation": True, "optimizer": "adam"}
    experiments = [
        ("ablation-full", best_config, "Konfigurasi lengkap menjadi pembanding."),
        ("ablation-no-augmentation", {**best_config, "augmentation": False},
         "Menghapus augmentasi akan menunjukkan kontribusi augmentasi."),
        ("ablation-no-dropout", {**best_config, "dropout": 0.0},
         "Menghapus dropout akan menunjukkan kontribusi regularisasi dropout."),
    ]
    with mlflow.start_run(run_name="ablation-study"):
        for name, cfg, hypothesis in experiments:
            train_one({**cfg, "hypothesis": hypothesis, "experiment_group": "ablation"},
                      name)
if __name__ == "__main__":
    main()
