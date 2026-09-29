#I can perform the analysis and compare with the papers results https://github.com/aegardella/cfRNA_MECFS/blob/main/Scripts/Manuscript_files/deseq2_res.csv
#TODO check this out https://github.com/hbctraining/Intro-to-DGE/blob/master/lessons/01b_DGE_setup_and_overview.md



from pathlib import Path
from prepare_data import prepare
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data
from sklearn.decomposition import PCA


base = Path("../data/raw/GSE293840")
SAVE = False
OUTPUT_PATH = Path("../data/processed/GSE293840")

# Load prepared data: counts and metadata merged 
df = prepare(counts_path=base / "GSE293840_raw_counts_all.csv", metadata_path=base / "GSE293840_family.soft.gz", out_dir=Path("../data/processed/GSE293840"), save=False, geo_id="GSE293840")
df.index = df["patient_id"]

df_metadata = df.iloc[:, :41]
df_counts = df.iloc[:, 41:]


#filter out lowly expressed genes
min_counts = 10
genes_with_less_x_counts = df_counts.sum(axis=0) < min_counts
genes_with_min_x_counts = df_counts.sum(axis=0) >= min_counts
df_counts = df_counts.loc[:, genes_with_min_x_counts]

#set up a deseq2 dataset
inference = DefaultInference(n_cpus=8)
#TODO account for the hospital location?
dds = DeseqDataSet(
    counts=df_counts,
    metadata=df_metadata,
    design="~phenotype",
    refit_cooks=True,
    inference=inference,
)

#count normalization by median of ratios method
dds.deseq2()
print(dds)
tmp=dds.layers["normed_counts"]
tmp.shape
print(dds.varm["LFC"])

if SAVE:
    dds.to_df(layer="normed_counts").to_csv(OUTPUT_PATH / "normalized_counts.csv")

#DESeq2 estimates fold changes from a negative-binomial generalized linear model, rather than simply calculating ratios of group means. The resulting coefficient represents the log2 fold change between the conditions.
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
dds.obsm["design_matrix"].columns
ds.lfc_shrink(coeff="phenotype[T.control]")
print(ds.shrunk_LFCs)
# Scatter plot of mean expression of normalized counts across all samples vs LFC between phenotypes
# y axis shows up- or downregulated genes and x axis shows if gene is in mean highly/lowly expressed across all samplea and phenotypes
ds.plot_MA() 
plt.show()


# Variance-stabilizing transformation (VST) to get values that are now approximately homoskedastic: variance is constant across the range of mean values, especially for low mean.
# genes with higher counts often have larger variance because we just are in higher numbers (=counts)
dds.vst()
vst = dds.layers["vst_counts"]



# Do PCA
X = np.asarray(vst)
pca = PCA(n_components=2).fit_transform(X)
 
# Plot, color by condition
cond = df_metadata["phenotype"].values
plt.scatter(pca[:, 0], pca[:, 1], c=(cond == cond[0]))
for i, sid in enumerate(df_metadata.index):
    plt.text(pca[i, 0], pca[i, 1], sid, fontsize=8)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("VST-PCA by phenotype")
plt.show()

#TODO figure out if data has been normalized already
#TODO I can use the normalized counts from the paper and compare with my results.