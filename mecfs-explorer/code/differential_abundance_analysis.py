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
from matplotlib.patches import Patch


base = Path("../data/raw/GSE293840")
SAVE = True
OUTPUT_PATH = Path("../data/processed/GSE293840")


######### Preparation of the data for DESeq2 analysis
# Load prepared data: counts and metadata merged 
df = prepare(counts_path=base / "GSE293840_raw_counts_all.csv", metadata_path_gh=base / "sample_metadata_from_github.tsv", out_dir=Path("../data/processed/GSE293840"), save=False, geo_id="GSE293840")
df.index = df["patient_id"]

#find columns that do not start with ENSG
df_metadata = df.loc[:, ~df.columns.str.startswith("ENSG")]
#set order of phenotype categories to control first, case second, so that the log2 fold change is calculated as case vs control
df_metadata["phenotype"] = pd.Categorical(df_metadata["phenotype"], categories=["control", "case"])
#find count columns that start with ENSG
df_counts = df.loc[:, df.columns.str.startswith("ENSG")]



#filter out lowly expressed genes
min_counts = 10
min_occurrence = 3
#at least 10 counts in at least 3 samples
check_min_x_count = ((df_counts >= min_counts).sum(axis=0) >= min_occurrence)
#at least 10 counts across all samples
#check_min_x_count = df_counts.sum(axis=0) >= min_counts
genes_keep = check_min_x_count[check_min_x_count]
genes_remove = check_min_x_count[~check_min_x_count]
tmp = df_counts.copy()
df_counts = df_counts.loc[:, genes_keep.index]
df_counts_removed = tmp.loc[:, genes_remove.index]
print("Number of genes removed due to low counts:", df_counts_removed.shape[1])

#ensure that the order of samples in counts and metadata is the same
if not df_counts.index.equals(df_metadata.index):
    raise ValueError("Counts and metadata sample order differs after preparation step.")

######### Set up a deseq2 object with the counts and metadata, and specify the design formula for the differential abundance analysis
inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(
    counts=df_counts,
    metadata=df_metadata,
    design="~sex+phenotype",
    refit_cooks=True,
    inference=inference,
)



######### VST transformation
# Variance-stabilizing transformation (VST) to transform values approximately homoskedastic: variance is constant across the range of mean values, especially for low mean.
# genes with higher counts often have larger variance because we just are in higher numbers (=counts)
# needed for visualization and clustering, e.g. PCA
dds.vst()
vst = dds.to_df(layer="vst_counts")

if SAVE:
    vst.to_csv(OUTPUT_PATH / "counts_vst_output.csv")



######### PCA 
if not vst.index.equals(df_metadata.index):
    raise ValueError("Not the same patient id order after vst transformation.")

#check if I need to consider any other confounders in the deseq2 diff abundance analysis
X = np.asarray(vst)
pca = PCA(n_components=2).fit_transform(X)
selected_phenotype = ["phenotype","sex","test_site","age_bin","bmi_bin","vo2_peak_bin","gh_bin","mfi20_total_bin","duration_of_mecfs_bin"]#,"batch"]
for p in selected_phenotype:
    cond = df_metadata[p].values.astype("string").fillna("NA").to_numpy()
    for phenotype in np.unique(cond):
        mask = cond == phenotype
        plt.scatter(pca[mask, 0], pca[mask, 1], label=phenotype)
    #add some labels
    label_ids = ["cfs_cfrna_58"]
    for patient_id in label_ids:
        idx = df_metadata.index.get_loc(patient_id)
        plt.text(pca[idx, 0], pca[idx, 1], patient_id, fontsize=8)

    """# add labels in the original sample order
    for i, patient_id in enumerate(df_metadata.index):
        plt.text(
            pca[i, 0] + 0.02,
            pca[i, 1] + 0.02,
            str(patient_id),
            fontsize=7,
            alpha=0.7,
        ) 
    """
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("VST-PCA")
    plt.legend(title=p)
    plt.show()

    if SAVE:
        plt.savefig(OUTPUT_PATH / "pca_plots" / f"vst_pca_colored_by_{p}.png", dpi=300, bbox_inches="tight")



# -> does not seem like the batch, case/control, hospital location, sex are clear separators in the PCA plot
# -> cfs_cfrna_58 seems to be an outlier in the PCA plot

#explore the metadata for this patient
df_metadata.iloc[df_metadata.index.get_loc("cfs_cfrna_58")]





######### Count normalization by median of ratios method with DESeq2
dds.deseq2()
normalized_counts=dds.to_df(layer="normed_counts")
normalized_counts.shape
print(dds.varm["LFC"])


#Cooks distance
#The deseq2() function calculates, for every gene and for every sample, a diagnostic test for outliers called Cook’s distance. 
#Cook’s distance is a measure of how much a single sample is influencing the fitted coefficients for a gene, and a large value of Cook’s distance is intended to indicate an outlier count.
#it might also replace outlier counts and refit affected genes when the replicate requirements are met.
cooks = pd.DataFrame(dds.layers["cooks"], index=dds.obs_names, columns=dds.var_names)
max_cooks_by_gene = cooks.max(axis=0).sort_values(ascending=False)
print(max_cooks_by_gene.head(20))
refitted = dds.var["refitted"].fillna(False).astype(bool)
print("Number of refitted genes:", refitted.sum())
print("Refitted genes:")
print(dds.var.index[refitted].tolist())
replaced = dds.var["replaced"].fillna(False).astype(bool)
print("Number of replaced genes:", replaced.sum())
print("Replaced genes:")
print(dds.var.index[replaced].tolist())




#Dispersion
#examine if data is a good fit for the DESeq2 model
#data generally scatters around the curve, with the dispersion decreasing with increasing mean levels.
#Red dots: Individual genes with very high dispersion outliers. These genes have a raw variance that is much higher than what the fitted red/blue trend line predicts for their expression level.
dds.plot_dispersions()

if SAVE:
    dds.to_df(layer="normed_counts").to_csv(OUTPUT_PATH / "normalized_counts.csv")
    dds.plot_dispersions(save_path = OUTPUT_PATH / "dispersion_estimates.png")

"""
#If I do not include sex in the design formula, sex is producing a large amount of systematic variation in the data.
#If i include sex in the design formula, Y/XIST dispersion outliers largely disappear

#look at the genes with the highest dispersion relative to the fitted trend line
df = pd.DataFrame({
    "mean": dds.var["_normed_means"],
    "disp": dds.var["genewise_dispersions"],
    "trend": dds.var["fitted_dispersions"],
}, index=dds.var_names)
df["log_ratio"] = np.log2(df["disp"] / df["trend"])
high = df[(df["mean"] > 10) & (df["log_ratio"] > 2)].sort_values("log_ratio", ascending=False)
print(len(high))
print(high.head(40))
print(high.head(40).index.tolist())
#-> many Y-linked genes among the top dispersion deviations


genes_of_interest = high.head(20).index.tolist()


for g in genes_of_interest:
    if g not in normalized_counts.columns:
        raise KeyError(f"{gY/XIST dispersion outliers largely disappear} not found in normalized counts.")

    if not df_metadata.index.equals(normalized_counts.index):
        raise ValueError("Metadata and normalized-count sample order differs.")

    plot_data = pd.DataFrame({
        "sex": df_metadata.loc[normalized_counts.index, "sex"].astype(str),
        "expression": normalized_counts[g]
    })

    plt.figure(figsize=(6, 5))
    sns.boxplot(
        data=plot_data,
        x="sex",
        y="expression",
        order=["female", "male"],
        width=0.5,
        boxprops={"facecolor": "none", "zorder": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        medianprops={"color": "black", "linewidth": 2},
        showfliers=False,
    )
    sns.stripplot(
        data=plot_data,
        x="sex",
        y="expression",
        order=["female", "male"],
        jitter=0.2,
        size=7,
    )
    plt.xlabel("Sex")
    plt.ylabel(f"Normalized counts")
    plt.title(f"{g}")
    plt.tight_layout()
    plt.show()


"""


#Plot the unfiltered count matrix as heatmap for 20 genes with highest count across samples
heatmap_gene_count = min(20, normalized_counts.shape[1])
top_abundant_genes = normalized_counts.sum(axis=0).nlargest(heatmap_gene_count).index
heatmap_data = normalized_counts.loc[:, top_abundant_genes].T

phenotype_palette = {"control": "#4C78A8", "case": "#E45756"}
sample_phenotype = df_metadata.loc[heatmap_data.columns, "phenotype"].astype(str)
column_colors = sample_phenotype.map(phenotype_palette).rename("Phenotype")

heatmap = sns.clustermap(
    heatmap_data,
    col_colors=column_colors,
    cmap="viridis",
    figsize=(max(10, normalized_counts.shape[0] * 0.18), 12),
    xticklabels=False,
    yticklabels=True,
    cbar_kws={"label": "Raw counts"},
)
heatmap.fig.suptitle("Top 20 Genes by Total Normalized Counts", y=1.02)
heatmap.ax_heatmap.set_xlabel("Samples")
heatmap.ax_heatmap.set_ylabel("Top variable genes")
heatmap.ax_heatmap.legend(
    handles=[
        Patch(facecolor=color, label=phenotype)
        for phenotype, color in phenotype_palette.items()
    ],
    title="Phenotype",
    loc="upper left",
    bbox_to_anchor=(1.15, 1),
)
plt.show()

if SAVE:
    heatmap.savefig(OUTPUT_PATH / "original_counts_heatmap.png", dpi=300)





######### Differential abundance analysis with DESeq2
#DESeq2 estimates fold changes from a negative-binomial generalized linear model, rather than simply calculating ratios of group means. The resulting coefficient represents the log2 fold change between the conditions.
#pvalue is calculated using Wald test
#padj is determined by Benjamini Hochberg procedure
#genes which contain a Cook’s distance above a cutoff for samples which have 3 or more replicates are automatically flagged. The pval and padj for these genes are set to NA. When there are 7 or more replicates for a given sample, the DESeq function will automatically replace counts with large Cook’s distance with the trimmed mean over all samples, scaled up by the size factor or normalization factor for that sample. 
ds = DeseqStats(dds, 
                contrast=["phenotype", "case", "control"], 
                inference=inference, 
                alpha=0.05,    
                )
ds.summary()
ds.results_df.head()

if SAVE:
    ds.results_df.to_csv(OUTPUT_PATH / "deseq2_results.csv")









######### LFC shrinkage using Bayesian inference
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






######### Filter diff. abundance results based on log2fc and padj
padj_cutoff = 0.05
shrunk_lfc_cutoff = 0.5
print("Number of genes with pvalue missing:", ds.results_df["pvalue"].isna().sum()," out of", ds.results_df.shape[0])
print("Number of genes with padj missing:", ds.results_df["padj"].isna().sum()," out of", ds.results_df.shape[0])
print("Number of genes with log2FoldChange missing:", ds.results_df["log2FoldChange"].isna().sum()," out of", ds.results_df.shape[0])


#DESeq2 outputs NA values for adjusted p values based on independent filtering of genes which have low counts. 
#If a row is filtered by automatic independent filtering, for having a low mean normalized count, then only the adjusted p value will be set to NA.
#A threshold on the filter statistic is found which optimizes the number of adjusted p values lower than a significance level alpha
#The goal of independent filtering is to filter out those tests from the procedure that have no, or little chance of showing significant evidence, without even looking at their test statistic. Typically, this results in increased detection power at the same experiment-wide type I error. Here, we measure experiment-wide type I error in terms of the false discovery rate.
#deseq2 ython does not expose the used treshold
res = ds.results_df
res_padj_na = res[res["padj"].isna()]
res_padj_and_pval_na = res[res["padj"].isna() & res["pvalue"].isna()]


filtered_results_padj = ds.results_df[ds.results_df["padj"].notna() & (ds.results_df["padj"] < padj_cutoff)]
filtered_results_lfc = ds.results_df[ds.results_df["padj"].notna() & (ds.results_df["log2FoldChange"].abs() >= shrunk_lfc_cutoff)]
filtered_results = ds.results_df[
    ds.results_df["padj"].notna() & 
    (ds.results_df["padj"] < padj_cutoff) & 
    (ds.results_df["log2FoldChange"].abs() >= shrunk_lfc_cutoff)
    ]

filtered_results.index
if SAVE:
    filtered_results.to_csv(OUTPUT_PATH / f"deseq2_de_results_{padj_cutoff}_{shrunk_lfc_cutoff}.csv")







######### Visualize results

# Scatterplot of normalized expression for one gene by phenotype
de_genes = list(filtered_results.index)

if SAVE:
    boxplot_output_path = OUTPUT_PATH / "normalized_count_boxplots"
    boxplot_output_path.mkdir(parents=True, exist_ok=True)

for g in de_genes:
    if g not in normalized_counts.columns:
        raise KeyError(f"{g} not found in normalized counts.")

    if not df_metadata.index.equals(normalized_counts.index):
        raise ValueError("Metadata and normalized-count sample order differs.")

    plot_data = pd.DataFrame({
        "phenotype": df_metadata.loc[normalized_counts.index, "phenotype"].astype(str),
        "expression": normalized_counts[g]
    })

    plt.figure(figsize=(6, 5))
    sns.boxplot(
        data=plot_data,
        x="phenotype",
        y="expression",
        order=["control", "case"],
        width=0.5,
        boxprops={"facecolor": "none", "zorder": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        medianprops={"color": "black", "linewidth": 2},
        showfliers=False,
    )
    sns.stripplot(
        data=plot_data,
        x="phenotype",
        y="expression",
        order=["control", "case"],
        jitter=0.2,
        size=7,
    )
    plt.xlabel("Phenotype")
    plt.ylabel(f"Normalized counts")
    plt.title(f"{g}")
    plt.tight_layout()
    if SAVE:
        plt.savefig(
            boxplot_output_path / f"{g}_normalized_counts_by_phenotype.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()




# Volcano plot for all genes tested by DESeq2
volcano_data = ds.results_df.copy()
volcano_data = volcano_data.dropna(subset=["log2FoldChange", "padj"])
volcano_data["significant"] = ((volcano_data["padj"] < padj_cutoff) & (volcano_data["log2FoldChange"].abs() >= shrunk_lfc_cutoff))

# Avoid infinite values when p-values are exactly zero
volcano_data["neg_log10_pvalue"] = -np.log10(volcano_data["padj"].clip(lower=np.finfo(float).tiny))



plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=volcano_data,
    x="log2FoldChange",
    y="neg_log10_pvalue",
    hue="significant",
    palette={True: "red", False: "lightgray"},
    s=25,
    linewidth=0,
)

plt.axvline(shrunk_lfc_cutoff, color="black", linestyle="--", linewidth=1)
plt.axvline(-shrunk_lfc_cutoff, color="black", linestyle="--", linewidth=1)
plt.axhline(-np.log10(padj_cutoff), color="black", linestyle="--", linewidth=1)

plt.xlabel("Log2 fold change")
plt.ylabel(f"-Log10(adjusted p-value)")
plt.title("DESeq2 volcano plot")
plt.legend(title="", labels=[f"pval < {padj_cutoff} and log2FC >= {shrunk_lfc_cutoff}", "Rest"])
plt.tight_layout()
if SAVE:
    plt.savefig(
        OUTPUT_PATH / f"volcano_plot.png",
        dpi=300,
        bbox_inches="tight",
    )
plt.show()






# Heatmap of top differentially expressed genes
if not df_metadata.index.equals(normalized_counts.index):
    raise ValueError("Metadata and normalized-count sample order differs.")

de_genes = filtered_results.index.intersection(normalized_counts.columns)

if len(de_genes) == 0:
    print("No filtered genes to plot.")
else:
    heatmap_data = np.log2(normalized_counts.loc[:, de_genes] + 1).T

    phenotype_palette = {"control": "#4C78A8", "case": "#E45756"}
    sample_phenotype = df_metadata.loc[normalized_counts.index, "phenotype"].astype(str)
    column_colors = sample_phenotype.map(phenotype_palette).rename("Phenotype")

    heatmap = sns.clustermap(
        heatmap_data,
        col_colors=column_colors,
        cmap="viridis",
        figsize=(
            max(10, normalized_counts.shape[0] * 0.18),
            max(6, len(de_genes) * 0.22),
        ),
        xticklabels=False,
        yticklabels=True,
        cbar_kws={"label": "log2(normalized counts + 1)"},
    )
    heatmap.fig.suptitle("Filtered Differential Genes", y=1.02)
    heatmap.ax_heatmap.set_xlabel("Samples")
    heatmap.ax_heatmap.set_ylabel("Genes")
    heatmap.ax_heatmap.legend(
        handles=[
            Patch(facecolor=color, label=phenotype)
            for phenotype, color in phenotype_palette.items()
        ],
        title="Phenotype",
        loc="upper left",
        bbox_to_anchor=(1.15, 1),
    )

    if SAVE:
        heatmap.savefig(
            OUTPUT_PATH / "differential_genes_normalized_counts_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()



#-> paper had different results for DESEq2 analysis but challenging to compare since they did not publish their exact code

#TODO sex in die design formula aufnehmen???
#TODO add additional metadta from their github??
#TODO make markdown out of it
#TODO functionen draus machen, die ich testen kann?
