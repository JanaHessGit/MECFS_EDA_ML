# ME/CFS Data Explorer: A Python + Deep Learning + FastAPI Practice Project

## Why this is a good practice project

ME/CFS is scientifically messy on purpose here: no validated biomarker, small cohorts, heterogeneous data types (metabolomics, transcriptomics, questionnaires, wearables). That messiness is exactly what makes it good practice — you'll deal with real-world problems like small-N statistics, missing data, class imbalance, and the temptation to over-claim from a model, rather than a clean Kaggle CSV. It also means every research question below is *genuinely open*, not textbook.

**The shape of the project:** an end-to-end pipeline — public dataset → cleaning/EDA notebook → PyTorch (or TensorFlow) model → FastAPI service that serves predictions and EDA summaries → optional lightweight dashboard. Small enough to finish, layered enough to touch every skill you listed.

---

## 1. Datasets (real, public, verified)

| Dataset | Type | Size | Access | Good for |
|---|---|---|---|---|
| **GSE128078** (PRJNA526259) — whole-blood RNA-seq, ME/CFS patients before/after exercise challenge | Transcriptomics (time series) | 14 cases, 11 controls, longitudinal | Free via [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128078) | Time-series deep learning practice, differential expression |
| **GSE59489** — DNA methylation (Illumina 450K) in PBMCs | Epigenomics | Case/control | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE59489) | High-dimensional feature reduction (PCA/autoencoders) |
| **GSE93266** — DNA methylome, larger ME/CFS cohort | Epigenomics | Larger cohort | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE93266) | Same as above, bigger N |
| **GSE14577** — PBMC gene expression, post-viral CFS vs. healthy | Transcriptomics | 8 cases, 7 controls | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14577) | Small-N classification, careful cross-validation practice |
| **GSE16059** — monozygotic twins discordant for chronic fatigue | Transcriptomics | 44 twin pairs | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE16059) | Paired-sample statistics, negative-result case study (this study found *no* significant difference — good for teaching that null results matter) |
| **Metabolomics Workbench, Lipkin/Columbia study** — plasma metabolomics + fecal metagenomics + clinical data | Metabolomics/microbiome | 50 cases, 50 controls | [Metabolomics Workbench](https://www.metabolomicsworkbench.org/) (search "ME/CFS Lipkin") | Multi-omics fusion, tabular deep learning |
| **Che et al. 2022 plasma metabolomics** ("Metabolomic evidence for peroxisomal dysfunction") | Metabolomics | 106 cases, 91 controls, 888 analytes, 5 US sites | Supplementary data via [PMC9320121](https://pmc.ncbi.nlm.nih.gov/articles/PMC9320121/) | The best-powered public ME/CFS omics dataset — good primary dataset for your classifier |
| **fMRI + n-back memory task dataset** (Provenzano et al. 2020) | Neuroimaging | 38 CFS, 31 controls | Described in [PMC7000378](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7000378/); request via Georgetown/Baraniuk Lab | Stretch goal — imaging pipelines, transfer learning |
| **Synthetic ME/CFS questionnaire generator** (Lacasa et al. 2023, Vall d'Hebron) | Synthetic tabular (SF-36, HAD, SCL-90R, FIS, PSQI) | Trained on 2,522 real patients | Paper: [Scientific Reports](https://doi.org/10.1038/s41598-023-40364-6) — check supplementary materials/corresponding author for the generator itself | Because it's synthetic, it's a safe first dataset for building your pipeline before touching real patient data |
| **CDC NHANES** (general population, has fatigue/sleep/functioning survey items) | Tabular survey | Tens of thousands | [NHANES data portal](https://wwwn.cdc.gov/nchs/nhanes/) — no ME/CFS diagnosis field, but has self-reported fatigue and functional-limitation items | Practicing FastAPI + real government tabular data at scale before moving to the small clinical datasets |

**A practical note on access:** GEO datasets download as raw expression matrices (`.txt`/`.soft`/`.CEL` files) via `GEOparse` or direct FTP — that's real practice in itself. The omics datasets are small in *sample count* (rows) but huge in *feature count* (columns: hundreds to thousands of genes/metabolites), which is a distinct ML problem from "big data" — you'll need dimensionality reduction, regularization, and honest small-N cross-validation (leave-one-out or repeated stratified k-fold, not a single train/test split).

---

## 2. Research questions to frame your work around

Pick one as your spine; treat the rest as extensions.

1. **Classification under small N:** Can plasma metabolomic/lipidomic profiles (Che et al. dataset) separate ME/CFS cases from controls better than a simple logistic regression baseline — and does a neural net actually help here, or does it overfit? (This is a genuinely useful question — the original paper got AUC ~0.87 with classical ML; beating or replicating that with a small MLP in PyTorch is a real, calibrated goal.)
2. **Subgroup structure:** Within ME/CFS cases, do unsupervised methods (clustering, autoencoders + k-means) recover meaningful subtypes (e.g., by IBS comorbidity, sex, symptom severity) rather than a single homogeneous "ME/CFS" class?
3. **Exercise-challenge signal:** In the longitudinal RNA-seq data (GSE128078), does gene expression actually shift after exertion in a way a model can detect — and how does that compare to the original paper's negative finding? (Reproducing a null result honestly is a great exercise in resisting p-hacking.)
4. **Twin discordance:** With the monozygotic twin dataset, what does a paired/siamese-network approach do differently from a standard classifier, given the original study found no significant transcriptomic difference?
5. **Synthetic-to-real transfer:** Train a model on the synthetic questionnaire data, then discuss (you likely won't have real patient questionnaire data to test on) what would need to hold for that model to generalize — a good essay/writeup exercise in dataset shift.
6. **Feature importance vs. biological plausibility:** Whatever model you build, use SHAP or integrated gradients to extract top features, then check them against the literature (e.g., do your top metabolites match plasmalogens/sphingomyelins/phosphatidylcholines, which multiple independent studies flag?). This teaches you to sanity-check a model against domain knowledge instead of trusting it blindly.

---

## 3. Suggested architecture

```
mecfs-explorer/
├── data/
│   ├── raw/              # downloaded GEO/Metabolomics Workbench files (gitignored)
│   ├── processed/        # cleaned parquet/csv
│   └── download_data.py  # scripted fetch (GEOparse, requests)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_sklearn.ipynb   # ALWAYS build the boring baseline first
│   └── 03_deep_model.ipynb
├── src/
│   ├── data/              # loaders, cleaning, feature engineering
│   ├── models/             # PyTorch nn.Module or TF model definitions
│   ├── train.py
│   └── evaluate.py          # cross-validation, calibration, SHAP
├── api/
│   ├── main.py              # FastAPI app
│   ├── schemas.py           # Pydantic request/response models
│   └── routers/
│       ├── predict.py        # POST endpoint: features in -> class + confidence out
│       └── eda.py            # GET endpoints serving summary stats/plots as JSON
├── tests/
├── Dockerfile
└── requirements.txt
```

**API design worth practicing (this is where FastAPI actually earns its keep, not just as a wrapper):**
- `POST /predict` — takes a feature vector (Pydantic model with validation), returns predicted class + probability + top contributing features (SHAP values) — teaches you to *not* return a bare label, since a health-adjacent model should always surface uncertainty and caveats
- `GET /dataset/summary` — serves descriptive stats (means, missingness, group sizes) computed at startup
- `GET /model/metadata` — returns training date, dataset version, performance metrics, and a hard-coded disclaimer that this is not diagnostic
- Background task or `/retrain` endpoint (optional, stretch) — practice with FastAPI's async and background task support

---

## 4. Phased roadmap

**Phase 1 — Python + data refresh (few days)**
Pick the Che et al. metabolomics dataset or NHANES as your warm-up. Do EDA in pandas: missingness, class balance, distribution plots. Build a scikit-learn baseline (logistic regression, random forest) with proper stratified cross-validation. This baseline is your benchmark for everything after — if your neural net can't beat it, that's a real and useful finding, not a failure.

**Phase 2 — Deep learning (1–2 weeks)**
Build a small MLP in PyTorch (or TensorFlow/Keras) for the same tabular task. Practice: custom `Dataset`/`DataLoader`, handling class imbalance (weighted loss or SMOTE), regularization (dropout, weight decay) since N is small, early stopping, and k-fold cross-validation loops (not just one split — small-N deep learning without this is close to meaningless). Add SHAP or captum for explainability.

**Phase 3 — FastAPI service (3–5 days)**
Wrap the trained model behind the API design above. Practice Pydantic validation, dependency injection (e.g., loading the model once at startup), async endpoints, error handling, and OpenAPI docs (`/docs` comes free — inspect it). Write a few `pytest` tests with `TestClient`.

**Phase 4 — Stretch goals (pick what excites you)**
- Dockerize the API and deploy it (Fly.io / Render free tier) so it's a live, shareable artifact
- Add the RNA-seq time-series dataset and try a small RNN/1D-CNN for the exercise-challenge question
- Add a lightweight frontend (Streamlit, or a small HTML page hitting your API) for interactive EDA
- Write up your findings against the published literature — did your feature importances match?

---

## 5. Honest caveats to build in from day one

- **N is small everywhere.** A model reporting 95% accuracy on 90 samples is almost always overfit — treat any single high number with suspicion and always report cross-validated ranges, not point estimates.
- **No dataset here is a diagnostic ground truth.** ME/CFS diagnosis itself is criteria-based (e.g., Institute of Medicine / Canadian Consensus Criteria), not lab-confirmed, so "case" vs. "control" labels carry real uncertainty. Frame your model's job as "distinguishing this study's labeled cohorts," not "diagnosing ME/CFS."
- **Some data needs a data use agreement or institutional request** (fMRI dataset, some Metabolomics Workbench raw files) — GEO and NHANES are the easiest fully-open starting points.
- If you ever write this up publicly, it's worth a line of respect for the patient community: this disease has a long history of being dismissed, and "a hobby project found X" headlines can do real harm if overstated.

---

Want me to scaffold the actual repo (starter `download_data.py` for GEO/Metabolomics Workbench, the FastAPI skeleton, and a baseline sklearn notebook) as real files you can run?
