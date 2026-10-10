import argparse
import mlflow
from mlflow import MlflowClient
from mlflow.tracking import MlflowClient as Client

MODEL_NAME = "fashion-mnist-cnn"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True,
                        help="Run ID dari run terbaik yang memiliki model artifact")
    args = parser.parse_args()
    client = MlflowClient()
    model_uri = f"runs:/{args.run_id}/model"
    registered = mlflow.register_model(model_uri, MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", registered.version)
    client.set_model_version_tag(MODEL_NAME, registered.version, "validation_status", "reviewed")
    print(f"Registered {MODEL_NAME} version {registered.version}; alias @champion telah dipasang.")
    print(f"URI untuk memuat model: models:/{MODEL_NAME}@champion")

if __name__ == "__main__":
    main()
