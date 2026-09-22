# ME/CFS Data Explorer

A small end-to-end practice project: public ME/CFS omics data → sklearn baseline →
PyTorch classifier → FastAPI service.

**This is a learning project, not a diagnostic tool.** Sample sizes in this field are
small (dozens to low hundreds) and "case"/"control" labels are criteria-based, not
lab-confirmed. Every model here should be treated as "does this study's dataset separate
these two labeled groups," never as "does this model diagnose ME/CFS."

See `PROJECT_PLAN.md` for the full write-up, dataset table, and research questions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Get data

```bash
# List what's available / see download instructions for each dataset
python data/download_data.py --list

# Download a specific GEO series (example: DNA methylation dataset)
python data/download_data.py --geo GSE59489

# Pull Metabolomics Workbench study summary + factors (metadata only — see notes in script
# for why full data tables usually need the paper's supplementary files)
python data/download_data.py --mw ST001849
```

Some datasets (the Che et al. 106-vs-91 metabolomics cohort, the fMRI dataset) aren't
available as a single clean download — you'll pull them from the paper's supplementary
materials by hand. That's normal for this field; `data/download_data.py --list` prints
where to find each one.

## 2. Baseline

Open `notebooks/02_baseline_sklearn.ipynb`. It loads whatever you dropped in
`data/processed/`, runs an EDA pass, and fits a regularized logistic regression +
random forest with stratified k-fold cross-validation as your benchmark. Beat this
before you trust any deep learning result.

## 3. Train the PyTorch model

```bash
python src/train.py --data data/processed/dataset.csv --target label --epochs 100
```

This trains a small MLP (`src/models/mlp.py`) with k-fold CV, early stopping, and
saves the best fold's weights + a `metrics.json` to `data/processed/model/`.

## 4. Serve it

```bash
uvicorn api.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for the interactive API docs.

## 5. Test

```bash
pytest tests/
```

## Repo layout

```
mecfs-explorer/
├── data/
│   ├── raw/               # downloaded files (gitignored)
│   ├── processed/         # cleaned data + trained model artifacts
│   └── download_data.py
├── notebooks/
│   └── 02_baseline_sklearn.ipynb
├── src/
│   ├── data/loaders.py     # cleaning / feature prep, shared by notebook + train.py
│   ├── models/mlp.py        # PyTorch model definition
│   ├── train.py              # k-fold training loop
│   └── evaluate.py            # metrics, SHAP feature importance
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routers/
│       ├── predict.py
│       └── eda.py
├── tests/test_api.py
├── requirements.txt
├── Dockerfile
└── PROJECT_PLAN.md
```
