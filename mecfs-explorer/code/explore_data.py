from pathlib import Path
from scipy import stats
from prepare_data import prepare
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


base = Path("../data/raw/GSE293840")

# Load prepared data: counts and metadata merged 
df = prepare(counts_path=base / "GSE293840_raw_counts_all.csv", metadata_path_gh=base / "sample_metadata_from_github.tsv", out_dir=Path("../data/processed/GSE293840"), save=False, geo_id="GSE293840")


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



df_metadata = df.loc[:, ~df.columns.str.startswith("ENSG")]

#Age
#show dstribution of age in the dataset
plt.figure(figsize=(8, 5))
#sns.histplot(df_metadata["age"], bins=20, kde=True, color="steelblue")
sns.violinplot(data=df_metadata, y="age")
plt.xlabel("Age")
plt.ylabel("Number of individuals")
plt.title("Distribution of age in the dataset")
plt.tight_layout()
plt.show()



#show dstribution of age in the dataset per group
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df_metadata,
    x="phenotype",
    y="age",
    hue="sex",
    split=True,
    order=["case", "control"],
    palette={"female": "red", "male": "green"},
    inner="quartile"
)
sns.stripplot(
    data=df_metadata,
    x="phenotype",
    y="age",
    hue="sex",
    dodge=True,
    order=["case", "control"],
    color= "black",
    size=4,
    jitter=0.15
)
plt.title("Age by group")
plt.xlabel("Age")
plt.ylabel("Number of individuals")
plt.tight_layout()
plt.show()


#Weight
#show dstribution of weight in the dataset per group
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df_metadata,
    x="phenotype",
    y="weight",
    hue="sex",
    split=True,
    order=["case", "control"],
    palette={"female": "red", "male": "green"},
    inner="quartile"
)
sns.stripplot(
    data=df_metadata,
    x="phenotype",
    y="weight",
    hue="sex",
    dodge=True,
    order=["case", "control"],
    color= "black",
    size=4,
    jitter=0.15
)
plt.title("Weight by group")
plt.xlabel("Phenotype")
plt.ylabel("Weight")
plt.tight_layout()
plt.show()




#Height
#show dstribution of height in the dataset per group
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df_metadata,
    x="phenotype",
    y="height",
    hue="sex",
    split=True,
    order=["case", "control"],
    palette={"female": "red", "male": "green"},
    inner="quartile"
)
sns.stripplot(
    data=df_metadata,
    x="phenotype",
    y="height",
    hue="sex",
    dodge=True,
    order=["case", "control"],
    color= "black",
    size=4,
    jitter=0.15
)
plt.title("Height by group")
plt.xlabel("Phenotype")
plt.ylabel("Height")
plt.tight_layout()
plt.show()


#BMI
#show dstribution of BMI in the dataset per group
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df_metadata,
    x="phenotype",
    y="bmi",
    hue="sex",
    split=True,
    order=["case", "control"],
    palette={"female": "red", "male": "green"},
    inner="quartile"
)
sns.stripplot(
    data=df_metadata,
    x="phenotype",
    y="bmi",
    hue="sex",
    dodge=True,
    order=["case", "control"],
    color= "black",
    size=4,
    jitter=0.15
)
plt.title("BMI by group")
plt.xlabel("Phenotype")
plt.ylabel("BMI")
plt.tight_layout()
plt.show()



#VO2Peak
#show dstribution of VO2Peak in the dataset per group
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df_metadata,
    x="phenotype",
    y="vo2peak1",
    hue="sex",
    split=True,
    order=["case", "control"],
    palette={"female": "red", "male": "green"},
    inner="quartile"
)
sns.stripplot(
    data=df_metadata,
    x="phenotype",
    y="vo2peak1",
    hue="sex",
    dodge=True,
    order=["case", "control"],
    color= "black",
    size=4,
    jitter=0.15
)
plt.title("VO2Peak by group")
plt.xlabel("Phenotype")
plt.ylabel("VO2Peak")
plt.tight_layout()
plt.show()




#mecfs_duration has some missing values
tmp = df_metadata[df_metadata["phenotype"] == "case"]
tmp["mecfs_duration"].isna().sum()
plt.figure(figsize=(8, 5))
sns.histplot(
    data=tmp,
    x="mecfs_duration",
    bins=25,
    kde=True,
    color="steelblue"
)
plt.xlabel("ME/CFS duration in years")
plt.ylabel("Number of individuals")
plt.title("Distribution of ME/CFS duration")
plt.tight_layout()
plt.show()

#collection_date
df_metadata["collection_date"].unique()
#GH (general health) survey scores ranged from 0 (worst health) to 100 (perfect health)
#MFI-20 (multidimensional fatigue inventory) ranged from 100 (highest fatigue) to 0 (no fatigue)
#both scores are self reported


#show dstribution of GH in the dataset per group
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df_metadata,
    x="phenotype",
    y="GH",
    hue="sex",
    split=True,
    order=["case", "control"],
    palette={"female": "red", "male": "green"},
    inner="quartile"
)
sns.stripplot(
    data=df_metadata,
    x="phenotype",
    y="GH",
    hue="sex",
    dodge=True,
    order=["case", "control"],
    color= "black",
    size=4,
    jitter=0.15
)
plt.title("GH by group (range 0 (worst health) to 100 (perfect health))")
plt.xlabel("Phenotype")
plt.ylabel("GH")
plt.tight_layout()
plt.show()




#show dstribution of MFI-20 in the dataset per group
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df_metadata,
    x="phenotype",
    y="mfi20_total",
    hue="sex",
    split=True,
    order=["case", "control"],
    palette={"female": "red", "male": "green"},
    inner="quartile"
)
sns.stripplot(
    data=df_metadata,
    x="phenotype",
    y="mfi20_total",
    hue="sex",
    dodge=True,
    order=["case", "control"],
    color= "black",
    size=4,
    jitter=0.15
)
plt.title("MFI-20 total by group")
plt.xlabel("Phenotype")
plt.ylabel("MFI-20 total (from 100 (highest fatigue) to 0 (no fatigue))")
#both scores are self reported)")
plt.tight_layout()
plt.show()




# keep only complete cases for the two variables
corr_df = df_metadata[["GH", "mfi20_total","phenotype"]].dropna()

# Pearson correlation
r = corr_df["GH"].corr(corr_df["mfi20_total"])
print(f"Pearson correlation GH vs MFI-20: {r:.3f}")
rho, p_value = stats.spearmanr(corr_df["GH"], corr_df["mfi20_total"])
print(f"Spearman correlation GH vs MFI-20: {rho:.3f}")

# scatter plot with regression line
plt.figure(figsize=(7, 6))
sns.regplot(
    data=corr_df,
    x="GH",
    y="mfi20_total",
    scatter_kws={"alpha": 0.7},
    line_kws={"color": "crimson"}
)

plt.title(f"GH vs MFI-20_total (cases only) (rho = {rho:.3f})")
plt.xlabel("GH (0 = worst health, 100 = perfect health)")
plt.ylabel("MFI-20 total (100 = highest fatigue, 0 = no fatigue)")
plt.tight_layout()
plt.show()




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





#test_site
#plot case control by test site
vars = ["phenotype", "sex"]
for v in vars:
    phenotype_by_testsite_counts = (
        df.loc[:, ["patient_id", "test_site", v]].groupby([v, "test_site"])
        .size()
    )

    p_phen = phenotype_by_testsite_counts.plot(kind="bar", figsize=(8, 5), color=["#d67ac7", "#5b9bd5", "green"])
    p_phen.set_ylabel("Number of individuals")
    for container in p_phen.containers:
        p_phen.bar_label(container, padding=3)
    plt.tight_layout()
    plt.show()

# -> especially for LA are phenotypes not balanced. For NYC and Ithaca, the case/control ratio is more balanced.



### Explore counts
df_counts = df.loc[:, df.columns.str.startswith("ENSG")]

# Count zero entries for each column
zero_counts = (df_counts == 0).sum()
print("Zero counts per column:")
print(zero_counts)


# Histogram of gene-count values across all samples and genes
gene_counts = df_counts.to_numpy().ravel()

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

df_cases = df_cases.loc[:, df_cases.columns.str.startswith("ENSG")] # select only count columns
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

df_controls = df_controls.loc[:, df_controls.columns.str.startswith("ENSG")] # select only count columns
#per gene calculate mean and variance across samples to see if the data is overdispersed (variance > mean) or underdispersed (variance < mean)
controls_mean_per_gene = df_controls.mean(axis=0)
controls_var_per_gene = df_controls.var(axis=0)


plt.figure(figsize=(8, 6))
plt.scatter(controls_mean_per_gene, controls_var_per_gene, color="black")
plt.axline((0, 0), slope=1, color="red") 
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Mean counts")
plt.ylabel("Variance counts")
plt.title("Mean vs variance across samples")
plt.tight_layout()
plt.show()

#plot looks very similar to the case group.






genes_with_less_10counts = df_counts.columns[df_counts.sum(axis=0) < 10]


genes_with_less_10counts
#gene = df_counts.columns[0]  # choose the gene to compare
gene = "ENSG00000005981.13" # choose the gene to compare

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


