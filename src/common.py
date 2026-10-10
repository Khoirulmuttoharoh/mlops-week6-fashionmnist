from pathlib import Path
import hashlib
import json
import random
import numpy as np
import torch
from torchvision import datasets, transforms

DATA_DIR = Path("data/raw/FashionMNIST")
MODEL_NAME = "fashion-mnist-cnn"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Mengutamakan hasil yang reproducible; beberapa operasi GPU masih dapat
    # berbeda antarversi perangkat keras/library.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dataset_hash(root=DATA_DIR):
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        return "not-downloaded"
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()

def get_datasets(augmentation=False, download=True):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Augmentasi hanya diterapkan pada data training.
    train_ops = []
    if augmentation:
        train_ops.append(transforms.RandomAffine(degrees=10, translate=(0.08, 0.08)))
    train_ops += [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    test_ops = [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    train = datasets.FashionMNIST(DATA_DIR, train=True, download=download,
                                  transform=transforms.Compose(train_ops))
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=download,
                                 transform=transforms.Compose(test_ops))
    return train, test

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
