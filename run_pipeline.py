import subprocess
import sys

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)

print(">>> STAGE 1: Running Unit Tests (CI)...")
run(f"{sys.executable} -m pytest tests/ -v")

print(">>> STAGE 2: Preprocessing Data...")
run(f"{sys.executable} src/data_preprocessing.py")

print(">>> STAGE 3: Training Model...")
run(f"{sys.executable} src/train.py")

print(">>> STAGE 4: Evaluating Model & Validating Quality Gate...")
run(f"{sys.executable} src/evaluate.py")

print(">>> STAGE 5: Deploying Validated Model...")
run(f"{sys.executable} src/deploy.py")

print(">>> STAGE 6: Running Production Inference Smoke Test...")
run(f"{sys.executable} src/predict.py")

print("\nSUCCESS: All pipeline stages completed successfully!")
