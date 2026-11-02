import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("maskan3d-lab/notebooks/maskan3d_kaggle_test.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in maskan3d_kaggle_test: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))[:70].replace("\n", " ")
    print(f"{i:2d} | {c['cell_type']:8s} | {src}")
