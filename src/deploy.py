from pathlib import Path
import json
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "production_model"

def deploy_model():
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    if not metrics["passed_quality_gate"]:
        raise RuntimeError("Deployment blocked because quality gate failed.")

    PROD.mkdir(exist_ok=True)
    for name in ["model.joblib", "scaler.joblib", "metrics.json"]:
        shutil.copy2(ROOT / "models" / name, PROD / name)

    manifest = {
        "status": "DEPLOYED",
        "deployment_timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "v1.0.0",
        "deployed_files": ["model.joblib", "scaler.joblib", "metrics.json"],
        "metrics": {
            "accuracy": metrics["accuracy"],
            "passed_quality_gate": metrics["passed_quality_gate"],
        },
    }
    (PROD / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Model successfully deployed to 'production_model'")

if __name__ == "__main__":
    deploy_model()
