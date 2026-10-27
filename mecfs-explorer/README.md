# ME/CFS cfRNA Expression Analysis

I have been stunned by ME/CFS and the impact it has on lifes and how little treatment is yet available. I wanted to use my scientific interest in this disease to do a training project with a ME/CFS dataset. This project explores publicly available ME/CFS gene-expression data, with the
main analysis focused on GEO series [GSE293840](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293840), published in Gardella AE, et al. Circulating cell-free RNA signatures for the characterization and diagnosis of myalgic encephalomyelitis/chronic fatigue syndrome. Proc Natl Acad Sci U S A. 2025 Aug 19;122(33):e2507345122. doi: 10.1073/pnas.2507345122.

The scripts prepare raw counts and sample metadata, perform exploratory analysis,
test for differential gene expression, and investigate sample structure with PCA
and hierarchical clustering.

**This is exploratory research code**

## Data

The main workflow expects these files under `data/raw/GSE293840/`:

- `GSE293840_raw_counts_all.csv` — raw gene-count matrix.
- `sample_metadata_from_github.tsv` — sample phenotypes and clinical,
  demographic, and technical metadata. The metadata source is the
  [cfRNA_MECFS study repository](https://github.com/aegardella/cfRNA_MECFS/). And the metadata file is here: https://github.com/aegardella/cfRNA_MECFS/blob/main/Scripts/Manuscript_files/sample_metadata.ts
- alternatvely the GEO family.soft.gz file can be used with small adjustments in prepare_data.py prepare().

The repository also contains supplementary materials for another study under
`data/raw/che2022/`; those files are not part of the current GSE293840 analysis
pipeline. Keep original input files unchanged when reproducing an analysis.

## Analysis workflow

1. **Prepare the data** — `code/prepare_data.py` transposes the count matrix so
   samples are rows, standardizes columns, creates clinical
   variable bins as defined in the Supplementary Materials Dataset S01, and joins metadata to counts by patient ID.
2. **Explore the data** — `code/explore_data.py` checks missing values and group assignments, then plots cohort characteristics such as age, weight, height, etc. Calculates correlation between self-reported scores for general health and MFI-20 (multidimensional fatigue inventory).
3. **Differential expression** — `code/run_deseq2_de_analysis.py` filters genes
   with fewer than 10 counts in at least 3 samples, applies a variance
   stabilizing transformation, and runs PyDESeq2. The current default design is
   `~sex+test_site+phenotype`, comparing cases with controls. It also generates
   PCA, volcano, MA, and heatmap visualizations, and can save normalized counts
   and result tables.
4. **Explore sample clusters** — `code/clustering.py` uses highly variable
   transformed genes, PCA, and hierarchical clustering to explore whether
   sample groupings may relate to phenotype or other cohort metadata.
5. **Compare inputs and results** — `code/compare_data.py` contains checks for
   differences between GEO and study-repository metadata and between analysis
   outputs.

`code/differential_abundance_analysis.py` is an old exploratory
analysis script. `code/tmp.py` is scratch work.

## Setup

Create and activate a virtual environment from the project directory:

```bash
python3 -m venv venv_MECFS
source venv_MECFS/bin/activate   # Windows: venv_MECFS\Scripts\activate
python -m pip install --upgrade pip
python -m pip install pandas GEOparse numpy scipy matplotlib seaborn pydeseq2 scikit-learn openpyxl
```

`requirements.txt` is currently a local environment snapshot rather than a
curated list of this project's dependencies; the install command above names
the packages used by the analysis scripts.

## Running the scripts

Prepare the merged sample-by-gene table from the project directory:

```bash
python code/prepare_data.py
```

Run the exploratory analysis or main differential-expression analysis from
`code/`, because these scripts use paths relative to that directory:

```bash
cd code
python explore_data.py
python run_deseq2_de_analysis.py
```

The main analysis currently sets `save_data = False`. Change that setting to
`True` in `run_deseq2_de_analysis.py` to save outputs under
`data/processed/GSE293840/phenotype_sex_and_testsite/`. Outputs include
normalized and variance-stabilized counts, differential-expression tables,
PCA plots, a volcano plot, MA plots, dispersion estimates, and a heatmap.
The analysis displays plots interactively as well; use a graphical session or
configure a non-interactive Matplotlib backend when running headlessly.

The clustering script also calls the analysis with saving disabled by default.
Its parameters, including the number of variable genes and clusters, are
exploratory choices in the script and should be assessed for sensitivity.

## Project layout

```text
mecfs-explorer/
├── code/
│   ├── prepare_data.py
│   ├── explore_data.py
│   ├── run_deseq2_de_analysis.py
│   ├── differential_abundance_analysis.py
│   ├── clustering.py
│   ├── compare_data.py
│   └── tmp.py
├── data/
│   ├── raw/       # GEO counts, metadata, and study supplements
│   └── processed/ # prepared tables, analysis results, and plots
├── requirements.txt
└── setup_virt_environment.txt
```
