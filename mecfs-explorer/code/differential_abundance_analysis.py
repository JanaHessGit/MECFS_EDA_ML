from pathlib import Path
from prepare_data import prepare
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data
from sklearn.decomposition import PCA


base = Path("../data/raw/GSE293840")
SAVE = True
OUTPUT_PATH = Path("../data/processed/GSE293840")

# Load prepared data: counts and metadata merged 
df = prepare(counts_path=base / "GSE293840_raw_counts_all.csv", metadata_path=base / "GSE293840_family.soft.gz", out_dir=Path("../data/processed/GSE293840"), save=False, geo_id="GSE293840")
df.index = df["patient_id"]

df_metadata = df.iloc[:, :41]
df_metadata["phenotype"] = pd.Categorical(df_metadata["phenotype"], categories=["control", "case"])
df_counts = df.iloc[:, 41:]


#filter out lowly expressed genes
min_counts = 10
min_occurrence = 3
genes_with_min_x_counts = ((df_counts >= min_counts).sum(axis=0) >= min_occurrence)
genes_with_min_x_counts = genes_with_min_x_counts[genes_with_min_x_counts]
df_counts = df_counts.loc[:, genes_with_min_x_counts.index]

#set up a deseq2 dataset
inference = DefaultInference(n_cpus=8) #TODO ????
dds = DeseqDataSet(
    counts=df_counts,
    metadata=df_metadata,
    design="~phenotype",
    refit_cooks=True,
    inference=inference,
)


# Variance-stabilizing transformation (VST) to get values that are now approximately homoskedastic: variance is constant across the range of mean values, especially for low mean.
# genes with higher counts often have larger variance because we just are in higher numbers (=counts)
dds.vst()

if SAVE:
    dds.to_df(layer="vst_counts").to_csv(OUTPUT_PATH / "counts_vst_output.csv")


# Do PCA and check if I need to consider any other confounders in the deseq2 diff abundance analysis
if not dds.to_df(layer="vst_counts").index.equals(df_metadata.index):
    raise ValueError("Not the same patient id order.")

vst = dds.layers["vst_counts"]

#TODO add age
X = np.asarray(vst)
pca = PCA(n_components=2).fit_transform(X)
selected_phenotype = ["phenotype","sex","characteristics_ch1.4.test site","batch"]
for p in selected_phenotype:
    cond = df_metadata[p].values
    for phenotype in np.unique(cond):
        mask = cond == phenotype
        plt.scatter(pca[mask, 0], pca[mask, 1], label=phenotype)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("VST-PCA")
    plt.legend(title=p)
    plt.show()

#does not seem like the case/control, hospital location, sex are separator in the PCA plot




#count normalization by median of ratios method
dds.deseq2()
print(dds)
tmp=dds.layers["normed_counts"]
tmp.shape
print(dds.varm["LFC"])


#examine if data is a good fit for the DESeq2 model
#data generally scatters around the curve, with the dispersion decreasing with increasing mean levels.
#Red dots: Individual genes with very high dispersion outliers. These genes have a raw variance that is much higher than what the fitted red/blue trend line predicts for their expression level.
dds.plot_dispersions()
plt.show()

if SAVE:
    dds.to_df(layer="normed_counts").to_csv(OUTPUT_PATH / "normalized_counts.csv")

#DESeq2 estimates fold changes from a negative-binomial generalized linear model, rather than simply calculating ratios of group means. The resulting coefficient represents the log2 fold change between the conditions.
#pvalue is calculated using Wald test
#padj is determined by Benjamini Hochberg procedure
ds = DeseqStats(dds, contrast=["phenotype", "case", "control"], inference=inference, alpha=0.05)
ds.summary()
ds.results_df.head()

if SAVE:
    ds.results_df.to_csv(OUTPUT_PATH / "deseq2_results.csv")



# LFC shrinkage using Bayesian inference
# goal is to make log2 fold-change estimates more reliable, 
# particularly for genes with low counts or high variation. 
# It pulls noisy, extreme estimates toward zero while leaving well-supported estimates largely unchanged. 
# Stabilizes the estimated effect sizes, conservative estimate given the uncertainty in the data.
ds_unshrunken = ds
ds_unshrunken_fc = ds.results_df["log2FoldChange"]
print(ds_unshrunken.shrunk_LFCs)

ds_unshrunken.plot_MA()
plt.show()

dds.obsm["design_matrix"].columns
ds.lfc_shrink(coeff="phenotype[T.case]")
print(ds.shrunk_LFCs)

# Scatter plot of mean expression of normalized counts across all samples vs LFC between phenotypes
# y axis shows up- or downregulated genes and x axis shows if gene is in mean highly/lowly expressed across all samplea and phenotypes
ds.plot_MA() 
plt.show()

ds_unshrunken_fc.equals(ds.results_df["log2FoldChange"])


if SAVE:
    ds.results_df.to_csv(OUTPUT_PATH / "deseq2_results_with_shrunk_lfcs.csv")


#filter diff abundance results based on log2fc and padj
padj_cutoff = 0.05
shrunk_lfc_cutoff = 1
filtered_results_padj = ds.results_df[ds.results_df["padj"].notna() & (ds.results_df["padj"] < padj_cutoff)]
filtered_results_lfc = ds.results_df[ds.results_df["padj"].notna() & (ds.results_df["log2FoldChange"].abs() > shrunk_lfc_cutoff)]
filtered_results = ds.results_df[
    ds.results_df["padj"].notna() & 
    (ds.results_df["padj"] < padj_cutoff) & 
    (ds.results_df["log2FoldChange"].abs() > shrunk_lfc_cutoff)
    ]

filtered_results.index




#TODO plotCounts
#TODO volcano plot
#TODO google common mistakes in deseq2 analysis and check
#TODO perform the analysis and compare with the papers github & results https://github.com/aegardella/cfRNA_MECFS/blob/main/Scripts/Manuscript_files/deseq2_res.csv
#TODO add additional metadta from their github??
