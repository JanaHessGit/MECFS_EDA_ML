from pathlib import Path
from prepare_data import prepare
import matplotlib.pyplot as plt
import seaborn as sns

base = Path("../data/raw/GSE293840")

# Load prepared data: counts and metadata merged 
df = prepare(counts_path=base / "GSE293840_raw_counts_all.csv", metadata_path=base / "GSE293840_family.soft.gz", out_dir=Path("../data/processed/GSE293840"), save=False, geo_id="GSE293840")


# EDA
print(df.shape)
print(df.head())
#df.describe().T
#TODO
print(df.isnull().sum().where(lambda x: x > 0).dropna())


# Make sure all phenotypes are accounted for
df["phenotype"].unique()
n_mecfs = df.loc[df["phenotype"] == "case", "patient_id"].nunique()
n_hc = df.loc[df["phenotype"] == "control", "patient_id"].nunique()
n_na = df.loc[df["phenotype"].isna(), "patient_id"].nunique()


if df.shape[0] != n_mecfs + n_hc:
        raise ValueError("Not all patients have a phenotype assigned")


# Check out metadata
df.columns[0:41]


#TODO