"""Prepare GSE293840 data: transpose counts and merge sample metadata."""


from pathlib import Path
from datetime import date
import pandas as pd
import logging
import GEOparse


#TODO test functions

logging.basicConfig(level=logging.INFO)

# Read expression counts, rows are genes and columns are samples
def read_counts(counts_path: Path) -> pd.DataFrame:
    logging.info("Reading counts from %s", counts_path)
    counts = pd.read_csv(counts_path, index_col=0)
    logging.info("Counts shape: %s", counts.shape)
    return counts


# Read the tabular matrix from the GEO .soft file and isolate patient ids in separate column
def read_metadata(metadata_path: Path) -> pd.DataFrame:
    logging.info("Reading sample characteristics from %s", metadata_path)
    gse = GEOparse.get_GEO(filepath=str(metadata_path), silent=True)
    metadata = gse.phenotype_data
    metadata["patient_id"] = (metadata["title"].str.extract(r"(cfs_cfrna_\d+)", expand=False).str.replace(r"(cfs_cfrna_)0+(\d+)$", r"\1\2", regex=True))
    return metadata




def rename_columns_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "characteristics_ch1.0.tissue": "tissue",
        "characteristics_ch1.1.batch": "batch",
        "characteristics_ch1.3.Sex": "sex",
        "characteristics_ch1.2.phenotype": "phenotype",
        "characteristics_ch1.5.age bin": "age_bin"
        })
    return df


# Merge count data with metadata based on patient id. And remove shortly added patient id column after merging.
def merge_counts_and_metadata(xls: pd.DataFrame, counts_t: pd.DataFrame) -> pd.DataFrame:
    counts_t.insert(0, "patient_id", counts_t.index.astype(str))
    df_merged = xls.merge(counts_t, on="patient_id", how="right")
    return df_merged


def prepare(counts_path: Path, metadata_path: Path, out_dir: Path, save: bool, geo_id: str) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = read_counts(counts_path)
    # transpose so rows are samples and columns are genes
    counts_t = counts.T
    logging.info("Transposed counts shape: %s", counts_t.shape)

    xls = read_metadata(metadata_path)
    xls = rename_columns_metadata(xls)
    counts_meta_merged = merge_counts_and_metadata(xls, counts_t)


    if save:
        run_date = date.today().isoformat()
        counts_t_path = out_dir / f"{run_date}_{geo_id}_counts_transposed.csv"
        merged_path = out_dir / f"{run_date}_{geo_id}_counts_with_meta.csv"
        logging.info("Writing transposed counts to %s", counts_t_path)
        counts_t.to_csv(counts_t_path)
        logging.info("Writing counts+metadata to %s", merged_path)
        counts_meta_merged.to_csv(merged_path)
    logging.info("Done")
    return counts_meta_merged


if __name__ == "__main__":
    geo_id = "GSE293840"
    base = Path(str("../data/raw/") + geo_id)
    counts_path = base / str(geo_id + "_raw_counts_all.csv")
    metadata_path = base / str(geo_id + "_family.soft.gz")
    out_dir = Path(str("../data/processed/") + geo_id)
    prepare(counts_path, metadata_path, out_dir, save=True, geo_id=geo_id)
