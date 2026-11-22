"""Read-only structural validation; does not execute any notebook code."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "classification/gaussian-naive-bayes-cifar10.ipynb",
    "classification/logistic-regression-iris.ipynb",
    "deep-learning/lenet-vs-mini-vgg-cifar10.ipynb",
    "regression/ford-custom-loss.ipynb",
)


def main() -> int:
    failed = False
    for relative_path in NOTEBOOKS:
        path = ROOT / relative_path
        if not path.is_file():
            print(f"MISSING: {relative_path}")
            failed = True
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cells = notebook["cells"]
            assert isinstance(cells, list)
            assert notebook.get("nbformat") == 4
            assert all(cell.get("cell_type") in ("markdown", "code", "raw") for cell in cells)
            saved_errors = sum(
                output.get("output_type") == "error"
                for cell in cells
                for output in cell.get("outputs", [])
            )
            code_cells = sum(cell.get("cell_type") == "code" for cell in cells)
            print(f"OK: {relative_path} ({len(cells)} cells, {code_cells} code, {saved_errors} saved errors)")
            if saved_errors:
                failed = True
        except (OSError, ValueError, KeyError, AssertionError, TypeError) as exc:
            print(f"INVALID: {relative_path} ({exc})")
            failed = True
    if failed:
        print("Notebook structure check failed. No code was executed.")
        return 1
    print("All four notebook structures passed. No model code was executed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
