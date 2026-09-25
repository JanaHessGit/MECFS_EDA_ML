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

#check for missing values across all columns
missing = df.isna().sum()
print(missing[missing > 0])



# Make sure no phenotypes are missing
df["phenotype"].unique()
n_mecfs = df.loc[df["phenotype"] == "case", "patient_id"].nunique()
n_hc = df.loc[df["phenotype"] == "control", "patient_id"].nunique()
n_na_phen = df.loc[df["phenotype"].isna(), "patient_id"].nunique()


if df.shape[0] != n_mecfs + n_hc or n_na_phen > 0:
        raise ValueError("Not all patients have a phenotype assigned")


# Make sure no gender info is missing
df["sex"].unique()
n_f = df.loc[df["sex"] == "female", "patient_id"].nunique()
n_m = df.loc[df["sex"] == "male", "patient_id"].nunique()
n_na_sex = df.loc[df["sex"].isna(), "patient_id"].nunique()

if df.shape[0] != n_f + n_m or n_na_sex > 0:
        raise ValueError("Not all patients have a sex assigned")


# Tissue type
df["tissue"].unique()
if len(df["tissue"].unique()) != 1:
		raise ValueError("More than one tissue type found in the dataset")



# Age bin
df_metadata = df.iloc[:, :41]
df_metadata["age_bin"].unique()
# age annotation not consistent, some are in years, some in bins
#TODO age verstehen


#list metadata columns
df_metadata.columns[0:41]

# Plot case/control counts
sex_by_phenotype = (
    df.loc[:, ["patient_id", "sex", "phenotype"]]
    .dropna(subset=["patient_id", "sex", "phenotype"])
    .drop_duplicates()
    .assign(sex=lambda d: d["sex"].astype(str).str.strip().str.lower())
)

counts_phenotype = (
    sex_by_phenotype.groupby(["phenotype"])
    .size()
)

p_phen = counts_phenotype.plot(kind="bar", figsize=(8, 5), color=["#d67ac7", "#5b9bd5"])
p_phen.set_ylabel("Number of individuals")
for container in p_phen.containers:
    p_phen.bar_label(container, padding=3)
plt.tight_layout()
plt.show()




# Plot how many female and male subjects are in each phenotype group
sex_by_phenotype = sex_by_phenotype[
    sex_by_phenotype["sex"].isin(["female", "male"])
]
counts = (
    sex_by_phenotype.groupby(["phenotype", "sex"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=["case", "control"], columns=["female", "male"], fill_value=0)
)

p_sex = counts.plot(kind="bar", figsize=(8, 5), color=["green", "blue"])
p_sex.set_xlabel("Phenotype")
p_sex.set_ylabel("Number of individuals")
p_sex.set_title("Female/Male by phenotype")
p_sex.legend(title="Sex")
for container in p_sex.containers:
    p_sex.bar_label(container, padding=3)
plt.tight_layout()
plt.show()
# A lot more females than males in both groups, but especially in the case group. 

#explore counts
# Count zero entries for each column
zero_counts = (df == 0).sum()
print("Zero counts per column:")
print(zero_counts)


#TODO ausführbar machen
#TODO vllt metadata von github verwenden??? da ist mehr enthalten... and also counts??? https://media.githubusercontent.com/media/aegardella/cfRNA_MECFS/refs/heads/main/Scripts/Manuscript_files/counts_vst_output.csv
#GH and MFI-20 scores?
#disease duration info???
#differential abundance analysis


