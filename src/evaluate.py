from pathlib import Path
import json
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.85

def evaluate_model():
    data = joblib.load(ROOT / "data" / "processed" / "processed_data.joblib")
    model = joblib.load(ROOT / "models" / "model.joblib")
    pred = model.predict(data["X_test"])

    metrics = {
        "accuracy": round(accuracy_score(data["y_test"], pred), 4),
        "precision": round(precision_score(data["y_test"], pred, average="weighted"), 4),
        "recall": round(recall_score(data["y_test"], pred, average="weighted"), 4),
        "f1_score": round(f1_score(data["y_test"], pred, average="weighted"), 4),
        "accuracy_threshold": THRESHOLD,
    }
    metrics["passed_quality_gate"] = metrics["accuracy"] >= THRESHOLD

    (ROOT / "models").mkdir(exist_ok=True)
    with open(ROOT / "models" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Model Evaluation Metrics:")
    print(f"- Accuracy: {metrics['accuracy']:.4f} (Threshold: {THRESHOLD})")
    print(f"- Quality Gate Status: {'PASSED' if metrics['passed_quality_gate'] else 'FAILED'}")

    if not metrics["passed_quality_gate"]:
        raise RuntimeError("Quality gate failed: accuracy is below 0.85.")
    return metrics

if __name__ == "__main__":
    evaluate_model()
