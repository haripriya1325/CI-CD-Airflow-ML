from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]


def train_model():
    data = joblib.load(
        ROOT / "data" / "processed" / "processed_data.joblib"
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        data["X_train"],
        data["y_train"]
    )

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    joblib.dump(
        model,
        models_dir / "model.joblib"
    )

    print("Model trained successfully and saved to models/model.joblib")

    # Return model for pytest
    return model


if __name__ == "__main__":
    train_model()

    