from pathlib import Path
import pandas as pd
from pathlib import Path
from prepare_data import read_metadata, rename_columns_metadata


new = pd.read_csv(
    "../data/processed/GSE293840/test/deseq2_de_results_0.05_0.5.csv", index_col=0
)
old = pd.read_csv(
    "../data/processed/GSE293840/deseq2_de_results_0.05_0.5.csv", index_col=0
)
new.equals(old)


new2 = pd.read_csv(
    "../data/processed/GSE293840/test/counts_vst_output.csv", index_col=0
)
old2 = pd.read_csv("../data/processed/GSE293840/counts_vst_output.csv", index_col=0)
new2.equals(old2)


base = Path("../data/raw/GSE293840")

""" ###NOT THE SAME - GITHUB ONLY HAS VST COUNT DATA (VARIANCE STABILIZATION FROM DESEG2)
# compare_data from GEO and github https://github.com/aegardella/cfRNA_MECFS/blob/main/Scripts/Manuscript_files/counts_vst_output.csv

dt_geo_counts = counts = pd.read_csv(base / "GSE293840_raw_counts_all.csv", index_col=0)

dt_github_counts = pd.read_csv(base / "counts_vst_output.csv", index_col=0)
dt_github_counts.columns = dt_github_counts.columns.str.replace(r"^cfs_cfrna_0+(\d+)$", r"cfs_cfrna_\1", regex=True)

# Gene IDs present in one dataframe but not the other
missing_in_geo = sorted(set(dt_github_counts.index) - set(dt_geo_counts.index))
missing_in_github = sorted(set(dt_geo_counts.index) - set(dt_github_counts.index))

print("Gene IDs only in GitHub counts:", missing_in_geo)
print("Gene IDs only in GEO counts:", missing_in_github)

# show counts of differences
print(f"n genes only in GitHub: {len(missing_in_geo)}")
print(f"n genes only in GEO: {len(missing_in_github)}")
 """

# compare metadata from GEO and github
metadata_github = pd.read_csv(base / "sample_metadata_from_github.tsv", sep="\t")

metadata_geo = read_metadata(Path("../data/raw/GSE293840/GSE293840_family.soft.gz"))
metadata_geo = rename_columns_metadata(metadata_geo)
metadata_geo = metadata_geo.rename(columns={"patient_id": "sample_id"})

# check if sample_id contains the same entries in both dfs
github_sample_ids = set(metadata_github["sample_id"])
geo_sample_ids = set(metadata_geo["sample_id"])
print("Same sample IDs:", github_sample_ids == geo_sample_ids)
# print("Sample IDs only in GitHub:", sorted(github_sample_ids - geo_sample_ids))
# print("Sample IDs only in GEO:", sorted(geo_sample_ids - github_sample_ids))


# check if entries for phenotype, sex and age are same in both dfs
metadata_geo_selected = metadata_geo[
    ["sample_id", "phenotype", "sex", "age_bin"]
].sort_values("sample_id")
metadata_geo_selected = metadata_geo_selected.reset_index(drop=True)
metadata_geo_selected = metadata_geo_selected.rename(columns={"age_bin": "age"})

metadata_github_selected = metadata_github[
    ["sample_id", "phenotype", "sex", "age"]
].sort_values("sample_id")
metadata_github_selected["sex"] = metadata_github_selected["sex"].replace(
    {"F": "female", "M": "male"}
)
metadata_github_selected["phenotype"] = metadata_github_selected["phenotype"].replace(
    {"MECFS": "case", "HC": "control"}
)
metadata_github_selected = metadata_github_selected.reset_index(drop=True)


print("Same shape:", metadata_geo_selected.shape == metadata_github_selected.shape)
print("Same values:", metadata_geo_selected.equals(metadata_github_selected))
differences = metadata_geo_selected.compare(
    metadata_github_selected, result_names=("geo", "github")
)
print(differences)
