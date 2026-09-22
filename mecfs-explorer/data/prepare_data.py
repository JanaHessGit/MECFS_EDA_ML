"""Prepare GSE293840 data: transpose counts and merge sample metadata.

Writes:
- data/processed/GSE293840/GSE293840_counts_transposed.csv
- data/processed/GSE293840/GSE293840_counts_with_meta.csv
"""
from pathlib import Path
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)


def read_counts(counts_path: Path) -> pd.DataFrame:
    logging.info("Reading counts from %s", counts_path)
    # gene ids are in the first column, use it as index
    counts = pd.read_csv(counts_path, index_col=0)
    logging.info("Counts shape: %s", counts.shape)
    return counts


def read_metadata(metadata_path: Path) -> dict:
    logging.info("Reading metadata from %s", metadata_path)
    # read all sheets and try to find the sheet/column that matches sample IDs
    xls = pd.read_excel(metadata_path, sheet_name=None, engine="openpyxl")
    logging.info("Found sheets: %s", list(xls.keys()))
    return xls


def find_sample_column_in_sheet(df: pd.DataFrame, sample_ids: set):
    # Try to find a column that contains many of the sample IDs
    best_col = None
    best_match = 0
    for col in df.columns:
        try:
            values = df[col].astype(str).str.strip().str.lower()
        except Exception:
            continue
        if values.empty:
            continue
        # compute intersection
        intersect = int(values.isin(sample_ids).sum())
        if intersect > best_match:
            best_match = intersect
            best_col = col
    return best_col, best_match


def select_metadata_sheet(xls: dict, sample_ids: set, min_overlap_ratio: float = 0.1):
    # For each sheet, try to detect a column matching sample ids
    best_info = (None, 0, None)  # sheet, match_count, col
    for sheet_name, df in xls.items():
        col, match = find_sample_column_in_sheet(df, sample_ids)
        logging.info("Sheet '%s' best col '%s' matches %d", sheet_name, col, match)
        if match > best_info[1]:
            best_info = (sheet_name, match, col)
    sheet_name, match, col = best_info
    if sheet_name is None or match == 0:
        raise ValueError("Could not find a metadata sheet with matching sample IDs")
    df = xls[sheet_name].copy()
    sample_count = len(sample_ids)
    if match / max(1, sample_count) < min_overlap_ratio:
        logging.warning("Only %d/%d (%.2f%%) metadata entries matched sample IDs", match, sample_count, 100 * match / max(1, sample_count))
    # standardize sample id column name
    df = df.rename(columns={col: "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str).str.strip().str.lower()
    return df


def standardize_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Try to find common metadata columns and rename them to standard names
    colmap = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for key in ("age", "age_years", "age_yrs"):
        if key in lower_cols:
            colmap[lower_cols[key]] = "age"
            break
    for key in ("sex", "gender"):
        if key in lower_cols:
            colmap[lower_cols[key]] = "sex"
            break
    for key in ("case_control", "group", "phenotype", "disease", "diagnosis", "case"):
        if key in lower_cols:
            colmap[lower_cols[key]] = "case_control"
            break
    if colmap:
        df = df.rename(columns=colmap)
    return df


def prepare(counts_path: Path, metadata_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = read_counts(counts_path)
    # transpose so rows are samples and columns are genes
    # %%
    counts_t = counts.T
    logging.info("Transposed counts shape: %s", counts_t.shape)

    # normalize sample ids in counts to lowercase strings
    counts_t.index = counts_t.index.astype(str).str.strip().str.lower()

    xls = read_metadata(metadata_path)
    sample_ids = set(counts_t.index.astype(str).str.lower())
    meta_df = select_metadata_sheet(xls, sample_ids)
    meta_df = standardize_metadata_columns(meta_df)

    # make metadata index on sample_id
    meta_df = meta_df.set_index("sample_id")

    # align metadata to counts rows
    meta_aligned = meta_df.reindex(counts_t.index)
    missing_meta = meta_aligned.index[meta_aligned.isnull().all(axis=1)]
    if len(missing_meta) > 0:
        logging.warning("%d samples have no metadata rows after reindexing", len(missing_meta))

    # concat counts and metadata side-by-side
    counts_with_meta = pd.concat([counts_t, meta_aligned], axis=1)

    # write outputs
    counts_t_path = out_dir / "GSE293840_counts_transposed.csv"
    merged_path = out_dir / "GSE293840_counts_with_meta.csv"
    logging.info("Writing transposed counts to %s", counts_t_path)
    counts_t.to_csv(counts_t_path)
    logging.info("Writing counts+metadata to %s", merged_path)
    counts_with_meta.to_csv(merged_path)
    logging.info("Done")


if __name__ == "__main__":
    base = Path("data/raw/GSE293840")
    counts_path = base / "GSE293840_raw_counts_all.csv"
    metadata_path = base / "pnas.2507345122.sd01_fom_suppMat.xlsx"
    out_dir = Path("data/processed/GSE293840")
    prepare(counts_path, metadata_path, out_dir)

# %%
