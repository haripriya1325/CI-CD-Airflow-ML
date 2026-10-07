from pathlib import Path
from src.data_preprocessing import preprocess_data
from src.train import train_model

def test_train_model():
    preprocess_data()
    model = train_model()
    assert model is not None
    assert (Path("models") / "model.joblib").exists()
