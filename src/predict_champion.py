import torch
import mlflow.pytorch
from src.common import get_datasets
from torch.utils.data import DataLoader

def main():
    model = mlflow.pytorch.load_model("models:/fashion-mnist-cnn@champion")
    model.eval()
    _, test_ds = get_datasets(augmentation=False)
    loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    x, y = next(iter(loader))
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    print("Prediksi kelas:", pred.tolist())
    print("Label aktual :", y.tolist())
if __name__ == "__main__":
    main()
