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


# Read the tabular matrix from the GEO .soft file and isolate patient ids in separate column
def read_metadata_from_github(metadata_path_gh: Path) -> pd.DataFrame:
    logging.info("Reading sample characteristics from %s", metadata_path_gh)
    metadata_gh = pd.read_csv(metadata_path_gh, sep="\t")
    metadata_gh.rename(columns={"sample_id": "patient_id"}, inplace=True)
    metadata_gh["phenotype"] = metadata_gh["phenotype"].str.replace("HC", "control")
    metadata_gh["phenotype"] = metadata_gh["phenotype"].str.replace("MECFS", "case")
    metadata_gh["sex"] = metadata_gh["sex"].str.replace("M", "male")
    metadata_gh["sex"] = metadata_gh["sex"].str.replace("F", "female")
    metadata_gh["test_site"] = metadata_gh["test_site"].str.replace("LA", "Los Angeles, CA")
    metadata_gh["test_site"] = metadata_gh["test_site"].str.replace("ITH", "Ithaca, NY")
    metadata_gh["test_site"] = metadata_gh["test_site"].str.replace("NYC", "New York City, NY")
    logging.info("Metadata shape: %s", metadata_gh.shape)
    return metadata_gh


def rename_columns_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "characteristics_ch1.0.tissue": "tissue",
        "characteristics_ch1.1.batch": "batch",
        "characteristics_ch1.2.phenotype": "phenotype",
        "characteristics_ch1.3.Sex": "sex",
        "characteristics_ch1.4.test site": "test_site",
        "characteristics_ch1.5.age bin": "age_bin"
        })
    return df


# Compare the two metadata dataframes
def shared_metadata_equal(metadata_gh: pd.DataFrame, metadata_gse: pd.DataFrame) -> pd.DataFrame:
    metadata_gh_filtered = metadata_gh.loc[:,["patient_id", "phenotype", "sex", "test_site"]]
    metadata_gse_filtered = metadata_gse.loc[:,["patient_id", "phenotype", "sex", "test_site"]]
    equal_bool = metadata_gh_filtered.equals(metadata_gse_filtered)
    if not equal_bool:
        comparison = metadata_gh_filtered.merge(metadata_gse_filtered, on="patient_id", how="outer", suffixes=("_gh", "_gse"), indicator=True)
        diff_mask = (
        (comparison["phenotype_gh"] != comparison["phenotype_gse"]) |
        (comparison["sex_gh"] != comparison["sex_gse"]) |
        (comparison["test_site_gh"] != comparison["test_site_gse"])
        )
        diff = comparison.loc[diff_mask]
        print(diff)
    return equal_bool



# Merge count data with metadata based on patient id. And remove shortly added patient id column after merging.
def merge_counts_and_metadata(xls: pd.DataFrame, counts_t: pd.DataFrame) -> pd.DataFrame:
    counts_t.insert(0, "patient_id", counts_t.index.astype(str))
    df_merged = xls.merge(counts_t, on="patient_id", how="right")
    return df_merged

#Age Bin 1: >18 to <=35; Bin 2: >35 to <=45;Bin 3: >45 to <=55; Bin 4: >55 to <=70. 
#BMI Bins: Bin 1: >0 to <=18.5; Bin 2: >18.5 to <=25; Bin 3: >25 to <=27; Bin 4: >27 to <=30; Bin 5: >30.
#VO2 peak Bins: Bin 1: >0 to <=15; Bin 2: >15 to <=20; Bin 3: >20 to <=25; Bin 4: >25 to <=30; Bin 5: >30.
#GH Bins: Bin 1: >0 to <=20; Bin 2: >20 to <=40; Bin 3: >40 to <=60; Bin 4: >60 to <=80; Bin 5: >80. 
#MFI20_Total Bins: Bin 1: >0 to <=20;Bin 2: >20 to <=40; Bin 3: >40 to <=60; Bin 4: >60 to <=80; Bin 5: >80.
#Duration of ME/CFS Bins:Bin 1: >0 to <=5; Bin 2: >5 to <=100.
def add_bins(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata["age_bin"] = pd.cut(metadata["age"], bins=[0, 35, 45, 55, 70], labels=["18-35", "36-45", "46-55", "56-70"])
    metadata["bmi_bin"] = pd.cut(metadata["bmi"], bins=[0, 18.5, 25, 27, 30, float('inf')], labels=["0-18.5", "18.5-25", "25-27", "27-30", "30+"])
    metadata["vo2_peak_bin"] = pd.cut(metadata["vo2peak1"], bins=[0, 15, 20, 25, 30, float('inf')], labels=["0-15", "15-20", "20-25", "25-30", "30+"])
    metadata["gh_bin"] = pd.cut(metadata["GH"], bins=[0, 20, 40, 60, 80, float('inf')], labels=["0-20", "20-40", "40-60", "60-80", "80+"])
    metadata["mfi20_total_bin"] = pd.cut(metadata["mfi20_total"], bins=[0, 20, 40, 60, 80, float('inf')], labels=["0-20", "20-40", "40-60", "60-80", "80+"])
    metadata["duration_of_mecfs_bin"] = pd.cut(metadata["mecfs_duration"], bins=[0, 5, 100], labels=["0-5", "5-100"])
    return metadata


def prepare(counts_path: Path, metadata_path_gh: Path, out_dir: Path, save: bool, geo_id: str) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = read_counts(counts_path)
    # transpose so rows are samples and columns are genes
    counts_t = counts.T
    logging.info("Transposed counts shape: %s", counts_t.shape)

    metadata_gh = read_metadata_from_github(metadata_path_gh)
    metadata_gh = add_bins(metadata_gh)
    #metadata_gse = read_metadata(metadata_path)
    #metadata_gse = rename_columns_metadata(metadata_gse)
    #if not shared_metadata_equal(metadata_gh, metadata_gse):
    #    logging.warning("Metadata from GEO and GitHub are not equal. Using metadata from GitHub.")
    counts_meta_merged = merge_counts_and_metadata(metadata_gh, counts_t)


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
    base = Path(str("./data/raw/") + geo_id)
    counts_path = base / str(geo_id + "_raw_counts_all.csv")
    metadata_path = base / str(geo_id + "_family.soft.gz")
    metadata_path_gh = base / "sample_metadata_from_github.tsv"
    out_dir = Path(str("./data/processed/") + geo_id)
    prepare(counts_path, metadata_path_gh, out_dir, save=True, geo_id=geo_id)
