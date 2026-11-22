# Reproducibility Notes

**Status:** The uploaded notebooks are preserved byte-for-byte. They have not been reexecuted in this packaging step, and no exact environment lockfile was included in the original materials.

## Environment

The included `requirements.txt` is a **dependency inventory, not a lockfile**. A virtual environment is recommended. Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/validate_notebooks.py
jupyter lab
```

Choose a Python version supported by your installed TensorFlow and PyTorch builds. macOS (including Apple Silicon), Linux and CUDA GPU environments may need different installs for ML frameworks; check their official installation guides if installation fails. No specific Python/framework version combination has been fully tested for this bundle.

## Run order and data dependencies

| Notebook | Network / dataset requirement | Notes |
| --- | --- | --- |
| `classification/gaussian-naive-bayes-cifar10.ipynb` | `tensorflow.keras.datasets.cifar10.load_data()` downloads CIFAR-10 on first use | Runs manual Gaussian classifier over 15,000 training and 300 selected test images |
| `classification/logistic-regression-iris.ipynb` | `pd.read_csv` downloads Iris CSV from a UCI URL | Uses NumPy gradient descent, not `sklearn.LogisticRegression`; `display()` expects an interactive notebook environment |
| `deep-learning/lenet-vs-mini-vgg-cifar10.ipynb` | `torchvision.datasets.CIFAR10(..., download=True)` saves under `./data/` | Multiple CNN runs; GPU suggested. `num_workers=2` may need adjustment in some notebook runtimes; additionally writes `part1_folds.csv`, `part1_representatives.csv`, `part2_hypotheses.csv` in the working directory |
| `regression/ford-custom-loss.ipynb` | Input price values embedded directly in notebook | No data download needed. Run top-to-bottom because functions and variables are reused/redefined |

**Working directory:** Open the notebook from the project root or check the current directory before running. The CNN notebook writes a local `data/` directory, and relative outputs depend on the working directory.

## Verification levels

1. **Structure check:** `python scripts/validate_notebooks.py` confirms valid JSON, expected notebook paths, basic cell structure and saved error-output status. It **does not** train or evaluate any model.
2. **Interactive execution:** Open notebooks in JupyterLab and run cells sequentially; this was **not performed** during package creation.
3. **Exact result reproduction:** Compare model metrics to saved outputs on a fresh environment and record Python, NumPy, framework versions, dataset checksums, CPU/GPU type and seed. This remains **to be done**.

## Evaluation improvements to consider

- Gaussian NB: Compare to a majority-class baseline and a stronger image baseline using a documented common test protocol.
- Iris: Report uncertainty or repeated stratified splits, since 19/20 correct is a small test sample.
- CNN: Choose architecture and hyperparameters from validation only; use the untouched test set once, after final model selection.
- Ford: Separate train/test years or use time-based validation; compare to a simple price baseline; document and validate data source and units.

## Publication checks

Confirm any course restrictions on posting graded work, verify that all dataset and third-party materials are properly attributed, and confirm the collaborator-specific wording of the Ford experiment before switching the repository to public visibility.
