import argparse, json
import numpy as np
import mlflow
from src.experiment import train_one

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    # Ganti nilai berikut dengan parameter terbaik yang dicetak oleh tune.py.
    best = {"batch_size": 128, "learning_rate": 0.001, "dropout": 0.25,
            "augmentation": True, "optimizer": "adam"}
    rows = []
    mlflow.set_experiment("fashion-mnist-cnn")
    with mlflow.start_run(run_name="best-config-3-seeds"):
        for seed in [17, 42, 2026]:
            config = {**best, "seed": seed, "epochs": args.epochs,
                      "hypothesis": "Konfigurasi terbaik tetap memberikan performa relatif stabil pada seed berbeda.",
                      "experiment_group": "three_seed_evaluation"}
            rows.append(train_one(config, f"best-seed-{seed}"))
        for metric in ["final_test_accuracy", "final_test_macro_f1"]:
            vals = [r[metric] for r in rows]
            mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1))
            mlflow.log_metric(f"{metric}_mean", mean)
            mlflow.log_metric(f"{metric}_std", std)
            print(f"{metric}: {mean:.4f} ± {std:.4f} (sample std, n=3)")
        with open("outputs/three_seed_results.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        mlflow.log_artifact("outputs/three_seed_results.json")
if __name__ == "__main__":
    main()
