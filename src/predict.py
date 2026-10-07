from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]

def predict(sample=None):
    if sample is None:
        sample = [[5.1, 3.5, 1.4, 0.2]]
    scaler = joblib.load(ROOT / "production_model" / "scaler.joblib")
    model = joblib.load(ROOT / "production_model" / "model.joblib")
    X = scaler.transform(sample)
    pred = model.predict(X)
    probs = model.predict_proba(X)
    output = [{"class_id": int(p), "probabilities": [round(float(x), 4) for x in row]} for p, row in zip(pred, probs)]
    print(f"Sample Input: {sample}")
    print(f"Prediction Output: {output}")
    return output

if __name__ == "__main__":
    predict()
