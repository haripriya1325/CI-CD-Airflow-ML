from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_data():
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data
    joblib.dump(
        {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
        },
        DATA_DIR / "processed_data.joblib",
    )

    # Save scaler
    joblib.dump(
        scaler,
        MODELS_DIR / "scaler.joblib",
    )

    print(
        f"Data preprocessed successfully. "
        f"Train shape: {X_train_scaled.shape}, "
        f"Test shape: {X_test_scaled.shape}"
    )

    # Return values required by pytest
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    preprocess_data()