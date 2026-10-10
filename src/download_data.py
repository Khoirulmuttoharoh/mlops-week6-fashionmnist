
from pathlib import Path
import hashlib
import json

from torchvision.datasets import FashionMNIST


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
MANIFEST_PATH = PROJECT_DIR / "data" / "dataset_manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Folder proyek:", PROJECT_DIR)
    print("Folder data:", DATA_DIR)
    print("\nMengunduh Fashion-MNIST...")

    train = FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=True
    )

    test = FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True
    )

    print("\nDataset berhasil dimuat!")
    print("Jumlah data training:", len(train))
    print("Jumlah data testing:", len(test))
    print("Ukuran gambar:", tuple(train.data.shape[1:]))
    print("Jumlah kelas:", len(train.classes))
    print("Nama kelas:", train.classes)

    files = sorted(
        path for path in DATA_DIR.rglob("*")
        if path.is_file()
    )

    manifest = {
        str(path.relative_to(PROJECT_DIR)): sha256_file(path)
        for path in files
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8"
    )

    combined = "".join(
        f"{name}:{manifest[name]}\n"
        for name in sorted(manifest)
    )

    dataset_hash = hashlib.sha256(
        combined.encode("utf-8")
    ).hexdigest()

    print("\nJumlah file:", len(files))
    print("SHA-256 dataset:", dataset_hash)
    print("Manifest:", MANIFEST_PATH)
    print("\nPengunduhan dan pencatatan hash selesai!")


if __name__ == "__main__":
    main()
