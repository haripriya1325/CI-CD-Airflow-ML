# CI-CD-Airflow-ML

Automated ML pipeline using Pytest, GitHub Actions CI/CD and Apache Airflow.

## Pipeline
preprocess -> train -> evaluate (accuracy >= 0.85) -> deploy -> predict

## Run locally
```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
python src/predict.py
```

## Test
```powershell
pytest tests/ -v
```

## GitHub
```powershell
git init
git branch -M main
git add .
git commit -m "Initial ML CI/CD Airflow project"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```
