from pathlib import Path
from prepare_data import prepare
import numpy as np
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


# Check tissue type
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




### Explore counts
# Count zero entries for each column
zero_counts = (df == 0).sum()
print("Zero counts per column:")
print(zero_counts)



# Histogram of gene-count values across all samples and genes
gene_counts = df.iloc[:, 41:].to_numpy().ravel()

plt.hist(gene_counts, color="steelblue", edgecolor="white", bins=200)
plt.ticklabel_format(axis="x", style="plain", useOffset=False)
plt.ticklabel_format(axis="y", style="plain", useOffset=False)
plt.xlabel("Raw expression count")
plt.ylabel("Number of genes")
plt.title("Distribution of gene counts")
plt.tight_layout()
plt.show()



# Mean vs. variance plot

# Cases
df_cases = df.loc[df["phenotype"] == "case", :]
if df_cases.phenotype.unique() != "case":
    raise ValueError("Phenotype column does not contain only 'case' values")

df_cases = df_cases.iloc[:, 41:]  # select only count columns
#per gene calculate mean and variance across samples to see if the data is overdispersed (variance > mean) or underdispersed (variance < mean)
cases_mean_per_gene = df_cases.mean(axis=0)
cases_var_per_gene = df_cases.var(axis=0)


plt.figure(figsize=(8, 6))
plt.scatter(cases_mean_per_gene, cases_var_per_gene, color="black")
#for gene, mean, variance in zip(cases_mean_per_gene.index, cases_mean_per_gene, cases_var_per_gene):
#    plt.annotate(str(gene), (mean, variance), fontsize=6)
plt.axline((0, 0), slope=1, color="red") 
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Mean counts")
plt.ylabel("Variance counts")
plt.title("Mean vs variance across samples")
plt.tight_layout()
plt.show()

#For the genes with high mean expression, the variance across samples tends to be greater than the mean.
#For the genes with low mean expression the variance is lower
#The variance of the genes with high mean expression is less dispersed than the variance of the genes with low mean expression. But also less genes.




# Controls
df_controls = df.loc[df["phenotype"] == "control", :]
if df_controls.phenotype.unique() != "control":
    raise ValueError("Phenotype column does not contain only 'control' values")

df_controls = df_controls.iloc[:, 41:]  # select only count columns
#per gene calculate mean and variance across samples to see if the data is overdispersed (variance > mean) or underdispersed (variance < mean)
controls_mean_per_gene = df_controls.mean(axis=0)
controls_var_per_gene = df_controls.var(axis=0)


plt.figure(figsize=(8, 6))
plt.scatter(controls_mean_per_gene, controls_var_per_gene, color="black")
#for gene, mean, variance in zip(controls_mean_per_gene.index, controls_mean_per_gene, controls_var_per_gene):
#    plt.annotate(str(gene), (mean, variance), fontsize=6)
plt.axline((0, 0), slope=1, color="red") 
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Mean counts")
plt.ylabel("Variance counts")
plt.title("Mean vs variance across samples")
plt.tight_layout()
plt.show()

#plot looks very similar to the case group.



df_counts = df.iloc[:, 41:]

genes_with_less_10counts = df_counts.columns[df_counts.sum(axis=0) < 10]



#gene = df_counts.columns[0]  # choose the gene to compare
gene = "ENSG00000210082.2" # choose the gene to compare

plot_data = df[["phenotype"]].copy()
plot_data["count"] = df_counts[gene].to_numpy()
plot_data["log_count"] = np.log1p(plot_data["count"])

sns.boxplot(
    data=plot_data,
    x="phenotype",
    y="count",
    order=["case", "control"],
    showfliers=False,
)
sns.stripplot(
    data=plot_data,
    x="phenotype",
    y="count",
    order=["case", "control"],
    jitter=0.2,
    alpha=0.6,
    size=4,
    color="black",
)

plt.xlabel("Phenotype")
plt.ylabel(f"Count for {gene}")
plt.tight_layout()
plt.show()


#TODO ausführbar machen
#TODO vllt metadata von github verwenden??? da ist mehr enthalten... and also counts??? https://media.githubusercontent.com/media/aegardella/cfRNA_MECFS/refs/heads/main/Scripts/Manuscript_files/counts_vst_output.csv
#GH and MFI-20 scores?
#disease duration info???
#differential abundance analysis


