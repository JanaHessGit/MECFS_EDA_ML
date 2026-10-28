# ME/CFS cfRNA Expression Analysis

I have been stunned by ME/CFS and the impact it has on lifes and how little 
treatment is yet available. I wanted to use my scientific interest in this 
disease to do a training project with a ME/CFS dataset. This project 
explores publicly available gene-expression data from people with
myalgic encephalomyelitis/chronic fatigue syndrome (ME/CFS). The primary
analysis uses GEO series [GSE293840](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293840),
described in:

> Gardella AE, et al. Circulating cell-free RNA signatures for the
> characterization and diagnosis of myalgic encephalomyelitis/chronic
> fatigue syndrome. *Proceedings of the National Academy of Sciences*.
> 2025;122(33):e2507345122. [doi:10.1073/pnas.2507345122](https://doi.org/10.1073/pnas.2507345122)

The scripts prepare counts and sample metadata, explore cohort characteristics,
test for differential gene expression, and visualize sample structure.

> **Scope:** This is exploratory research code, not a clinical or diagnostic
> tool. Analysis choices and findings require scientific review and
> independent validation.

## Data

The primary workflow expects these files in `data/raw/GSE293840/`:

- `GSE293840_raw_counts_all.csv` — raw gene-count matrix, with genes as rows
  and samples as columns.
- `sample_metadata_from_github.tsv` — sample phenotypes and clinical,
  demographic, and technical metadata from the
  [cfRNA_MECFS study repository](https://github.com/aegardella/cfRNA_MECFS).

The directory may also contain GEO archives (`.gz` files), the GEO family and
series-matrix files, and the paper's supplementary materials. The primary
scripts use the uncompressed count matrix and TSV listed above; they do not
automatically extract the compressed files. `compare_data.py` can also read
the GEO family file when comparing metadata sources.

Supplementary materials for the separate Che et al. 2022 study are in
`data/raw/che2022/`; they are not inputs to the GSE293840 workflow. Keep raw
input files unchanged when reproducing an analysis.

## Analysis workflow

Run `prepare_data.py` from the project root, and run the other main analyses
from the `code/` directory:

1. **Prepare data** — `code/prepare_data.py` transposes the count matrix so
   samples are rows, standardizes metadata categories, creates clinical
   variable bins, and joins metadata to counts by patient ID. It writes the
   prepared tables to `data/processed/GSE293840/`.
2. **Explore cohort data** — `code/explore_data.py` checks missing values and
   group assignments, summarizes cohort characteristics, and explores the
   relationship between self-reported general-health scores and MFI-20 scores.
3. **Differential expression** — `code/run_deseq2_de_analysis.py` removes genes
   with fewer than 10 counts in at least 3 samples, applies a variance
   stabilizing transformation, and runs PyDESeq2. The default design is
   `~sex+test_site+phenotype`, comparing cases with controls. The analysis also
   creates PCA, volcano, MA, and heatmap visualizations.
4. **Explore sample clusters** — `code/clustering.py` uses the n most
   variable transformed genes, PCA, and hierarchical clustering to explore
   sample groupings. The number of genes, components, and clusters are
   exploratory choices in the script and should be assessed for sensitivity.
5. **Compare data and results** — `code/compare_data.py` contains checks for
   differences between GEO and study-repository metadata and between analysis
   outputs. Some comparisons expect existing processed results.

`code/differential_abundance_analysis.py` is an older exploratory analysis.

## Setup

From the project directory, create and activate a virtual environment, then
install the packages used by the analysis scripts:

```bash
python3 -m venv venv_MECFS
source venv_MECFS/bin/activate  # Windows: venv_MECFS\Scripts\activate
python -m pip install --upgrade pip
python -m pip install pandas GEOparse numpy scipy matplotlib seaborn pydeseq2 scikit-learn openpyxl
```

`requirements.txt` is an environment snapshot, not a curated project
dependency list. The install command above is the project-specific setup.

## Running the scripts

Prepare the merged sample-by-gene table from the project root:

```bash
python code/prepare_data.py
```

Run the exploratory or differential-expression analysis from `code/`, since
those scripts use paths relative to that directory:

```bash
cd code
python explore_data.py
python run_deseq2_de_analysis.py
```

The differential-expression script sets `save_data = False` by default.
Change it to `True` in `run_deseq2_de_analysis.py` to save tables and plots
under `data/processed/GSE293840/phenotype_sex_and_testsite/`. The script
displays plots interactively; use a graphical session or configure a
non-interactive Matplotlib backend when running headlessly.

The clustering script also runs the differential-expression workflow with
saving disabled by default, then displays clustering plots and writes its
cluster visualizations under `output/`.

## Project layout

```text
mecfs-explorer/
├── code/
│   ├── prepare_data.py
│   ├── explore_data.py
│   ├── run_deseq2_de_analysis.py
│   ├── clustering.py
│   ├── compare_data.py
│   └── differential_abundance_analysis.py
├── data/
│   ├── raw/       # GEO counts, metadata, and study supplements
│   └── processed/ # prepared tables, analysis results, and plots
├── tests/
├── requirements.txt
└── setup_virt_environment.txt
```
