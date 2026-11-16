"""Create bangla-ocr-assignment.zip with the source code only.

Leaves out the dataset, trained model, MLflow runs, screenshots, caches and
virtual environments, as the assignment requires.
Run:  python make_submission_zip.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ["train.py", "app.py", "ocr_utils.py", "words.txt", "labels.json", "requirements.txt",
         "Dockerfile", ".dockerignore", ".gitignore", "README.md", "github_link.txt",
         "colab_train.ipynb", "make_submission_zip.py"]

out = ROOT / "bangla-ocr-assignment.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for name in FILES:
        path = ROOT / name
        if path.exists():
            z.write(path, f"bangla-ocr-assignment/{name}")
        else:
            print(f"(skipped, not found: {name})")
print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
