import argparse
import json
from pyexpat import model
import time
from pathlib import Path

import mlflow
from mlflow.dspy import log_model
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.common import DATA_DIR, dataset_hash, get_datasets, set_seed
from src.model import SmallCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_one(config, run_name, parent_run_id=None, log_model=True):
    seed = int(config.get("seed", 42))
    set_seed(seed)
    augmentation = bool(config.get("augmentation", False))
    train_ds, test_ds = get_datasets(augmentation=augmentation)
    data_hash = dataset_hash()
    batch_size = int(config.get("batch_size", 128))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    model = SmallCNN(float(config.get("dropout", 0.25))).to(DEVICE)
    optimizer_name = str(config.get("optimizer", "adam")).lower()
    lr = float(config.get("learning_rate", 0.001))
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    epochs = int(config.get("epochs", 5))

    with mlflow.start_run(run_name=run_name, nested=(mlflow.active_run() is not None)):
        mlflow.set_tags({
            "dataset": "Fashion-MNIST",
            "dataset_sha256": data_hash,
            "device": DEVICE,
            "hypothesis": config.get("hypothesis", "Baseline: CNN kecil menjadi titik pembanding."),
            "experiment_group": config.get("experiment_group", "baseline"),
        })
        mlflow.log_params({k: v for k, v in config.items() if k not in ("hypothesis",)})
        mlflow.log_param("device", DEVICE)
        mlflow.log_param("dataset_hash", data_hash)
        mlflow.log_param("train_samples", len(train_ds))
        mlflow.log_param("test_samples", len(test_ds))
        mlflow.log_param("model_architecture", "SmallCNN: Conv32-Conv64-FC128")
        start = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            model.eval()
            all_y, all_pred = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    logits = model(x.to(DEVICE))
                    pred = logits.argmax(dim=1).cpu().numpy()
                    all_pred.extend(pred.tolist())
                    all_y.extend(y.numpy().tolist())
            test_acc = accuracy_score(all_y, all_pred)
            test_f1 = f1_score(all_y, all_pred, average="macro")
            mlflow.log_metric("train_loss", float(np.mean(losses)), step=epoch)
            mlflow.log_metric("test_accuracy", float(test_acc), step=epoch)
            mlflow.log_metric("test_macro_f1", float(test_f1), step=epoch)
            print(f"{run_name} | epoch {epoch}/{epochs} "
                  f"loss={np.mean(losses):.4f} acc={test_acc:.4f} macro_f1={test_f1:.4f}")

        elapsed = time.time() - start
        mlflow.log_metric("training_seconds", elapsed)
        mlflow.log_metric("final_test_accuracy", float(test_acc))
        mlflow.log_metric("final_test_macro_f1", float(test_f1))
        Path("outputs").mkdir(exist_ok=True)
        summary = {
            "run_name": run_name, "seed": seed, "config": config,
            "dataset_sha256": data_hash, "device": DEVICE,
            "final_test_accuracy": float(test_acc),
            "final_test_macro_f1": float(test_f1),
            "training_seconds": elapsed,
        }
        summary_path = Path("outputs") / f"{run_name.replace('/', '_').replace(' ', '_')}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(summary_path))
        
        if log_model:
            mlflow.pytorch.log_model(
                model,
                name="model",
                serialization_format="pickle",
        )
    return summary

def run_single(args):
    mlflow.set_experiment("fashion-mnist-cnn")
    config = {
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "dropout": args.dropout,
        "augmentation": args.augmentation, "optimizer": args.optimizer,
        "hypothesis": args.hypothesis, "experiment_group": args.group,
    }
    train_one(config, args.name)

def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST CNN experiment tracking")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="Jalankan satu eksperimen")
    p.add_argument("--name", default="baseline-cnn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--augmentation", action="store_true")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--hypothesis", default="CNN sederhana mampu menjadi baseline yang terukur.")
    p.add_argument("--group", default="baseline")
    p.set_defaults(func=run_single)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
