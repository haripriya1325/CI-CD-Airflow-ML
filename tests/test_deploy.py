from pathlib import Path
from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.deploy import deploy_model

def test_deploy_model():
    preprocess_data()
    train_model()
    evaluate_model()
    deploy_model()
    assert (Path("production_model") / "model.joblib").exists()
    assert (Path("production_model") / "scaler.joblib").exists()
    assert (Path("production_model") / "model_manifest.json").exists()
