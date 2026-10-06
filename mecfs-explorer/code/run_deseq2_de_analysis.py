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


# Load the data for DESeq2 analysis
def prepare_data_GSE293840(counts_path, metadata_path_gh, out_dir, save, geo_id):

    # Load prepared data: counts and metadata merged 
    df = prepare(counts_path=counts_path, metadata_path_gh=metadata_path_gh, out_dir=out_dir, save=save, geo_id=geo_id)
    df.index = df["patient_id"]

    #find columns that do not start with ENSG
    df_metadata = df.loc[:, ~df.columns.str.startswith("ENSG")]
    #set order of phenotype categories to control first, case second, so that the log2 fold change is calculated as case vs control
    df_metadata["phenotype"] = pd.Categorical(df_metadata["phenotype"], categories=["control", "case"])
    #find count columns that start with ENSG
    df_counts = df.loc[:, df.columns.str.startswith("ENSG")]

    return df_counts, df_metadata


# Filter out lowly expressed genes: at least min_count counts in at least min_samples samples
def filter_low_counts_min_samples(df_counts, min_count=10, min_samples=3):
    check_min_x_count = ((df_counts >= min_count).sum(axis=0) >= min_samples)
    #other version: at least 10 counts across all samples
    #check_min_x_count = df_counts.sum(axis=0) >= min_counts

    genes_keep = check_min_x_count[check_min_x_count]
    genes_remove = check_min_x_count[~check_min_x_count]
    tmp = df_counts.copy()
    df_counts = df_counts.loc[:, genes_keep.index]
    df_counts_removed = tmp.loc[:, genes_remove.index]
    print("Number of genes removed due to low counts:", df_counts_removed.shape[1])

    return df_counts


# Filter out lowly expressed genes: at least min_count counts across samples
def filter_low_counts(df_counts, min_count=10):
    check_min_x_count = df_counts.sum(axis=0) >= min_count

    genes_keep = check_min_x_count[check_min_x_count]
    genes_remove = check_min_x_count[~check_min_x_count]
    tmp = df_counts.copy()
    df_counts = df_counts.loc[:, genes_keep.index]
    df_counts_removed = tmp.loc[:, genes_remove.index]
    print("Number of genes removed due to low counts:", df_counts_removed.shape[1])

    return df_counts



######### Set up a deseq2 object with the counts and metadata, and specify the design formula for the differential abundance analysis
def set_up_deseq2_object(df_counts, df_metadata, design_formula, save, out_path):
    inference = DefaultInference(n_cpus=8)
    dds = DeseqDataSet(
        counts=df_counts,
        metadata=df_metadata,
        design=design_formula,
        refit_cooks=True,
        inference=inference,
    )

    ######### VST transformation
    # Variance-stabilizing transformation (VST) to transform values approximately homoskedastic: variance is constant across the range of mean values, especially for low mean.
    # genes with higher counts often have larger variance because we just are in higher numbers (=counts)
    # needed for visualization and clustering, e.g. PCA

    dds.vst()
    vst = dds.to_df(layer="vst_counts")

    if save:
        vst.to_csv(out_path / "counts_vst_output.csv")

    return dds, inference, vst




def count_normalization(dds, save, out_path):

    ######### Count normalization by median of ratios method with DESeq2
    dds.deseq2()
    normalized_counts=dds.to_df(layer="normed_counts")
    normalized_counts.shape
    print(dds.varm["LFC"])

    #Dispersion
    #examine if data is a good fit for the DESeq2 model
    #data generally scatters around the curve, with the dispersion decreasing with increasing mean levels.
    #Red dots: Individual genes with very high dispersion outliers. These genes have a raw variance that is much higher than what the fitted red/blue trend line predicts for their expression level.
    dds.plot_dispersions()

    if save:
        dds.to_df(layer="normed_counts").to_csv(out_path / "normalized_counts.csv")
        dds.plot_dispersions(save_path = out_path / "dispersion_estimates.png")

    return dds, normalized_counts









def deseq2_de_analysis(dds, inference, contrast, alpha, coeff_lfc_shrinking, save, out_path):
    ######### Differential abundance analysis with DESeq2
    #DESeq2 estimates fold changes from a negative-binomial generalized linear model, rather than simply calculating ratios of group means. The resulting coefficient represents the log2 fold change between the conditions.
    #pvalue is calculated using Wald test
    #padj is determined by Benjamini Hochberg procedure
    #genes which contain a Cook’s distance above a cutoff for samples which have 3 or more replicates are automatically flagged. The pval and padj for these genes are set to NA. When there are 7 or more replicates for a given sample, the DESeq function will automatically replace counts with large Cook’s distance with the trimmed mean over all samples, scaled up by the size factor or normalization factor for that sample. 
    ds = DeseqStats(dds, 
                    contrast=contrast, 
                    inference=inference, 
                    alpha=alpha,    
                    )
    ds.summary()
    ds.results_df.head()

    if save:
        ds.results_df.to_csv(out_path / "deseq2_results.csv")



    ######### LFC shrinkage using Bayesian inference
    # goal is to make log2 fold-change estimates more reliable, 
    # particularly for genes with low counts or high variation. 
    # It pulls noisy, extreme estimates toward zero while leaving well-supported estimates largely unchanged. 
    # Stabilizes the estimated effect sizes, conservative estimate given the uncertainty in the data.
    ds_unshrunken = ds
    ds_unshrunken_fc = ds.results_df["log2FoldChange"]
    ds_unshrunken.plot_MA()
    plt.show()
    if save:
        ds_unshrunken.plot_MA(save_path= out_path / "deseq2_MA_plot_before_lfc_shrinkage.png") 


    ds.lfc_shrink(coeff=coeff_lfc_shrinking)
    print(ds.shrunk_LFCs)

    if ds_unshrunken_fc.equals(ds.results_df["log2FoldChange"]):
        raise

    # Scatter plot of mean expression of normalized counts across all samples vs LFC between phenotypes
    # y axis shows up- or downregulated genes and x axis shows if gene is in mean highly/lowly expressed across all samplea and phenotypes
    ds.plot_MA() 
    plt.show()

    if save:
        ds.plot_MA(save_path= out_path / "deseq2_MA_plot_after_lfc_shrinkage.png") 
        ds.results_df.to_csv(out_path / "deseq2_results_with_shrunk_lfcs.csv")

    return ds





# Filter diff. abundance results based on log2fc and padj
def filter_de_results(ds, padj_cutoff, shrunk_lfc_cutoff, save, out_path):

    print("Number of genes with pvalue missing:", ds.results_df["pvalue"].isna().sum()," out of", ds.results_df.shape[0])
    print("Number of genes with padj missing:", ds.results_df["padj"].isna().sum()," out of", ds.results_df.shape[0])
    print("Number of genes with log2FoldChange missing:", ds.results_df["log2FoldChange"].isna().sum()," out of", ds.results_df.shape[0])

    #DESeq2 outputs NA values for adjusted p values based on independent filtering of genes which have low counts. 
    #If a row is filtered by automatic independent filtering, for having a low mean normalized count, then only the adjusted p value will be set to NA.
    #A threshold on the filter statistic is found which optimizes the number of adjusted p values lower than a significance level alpha
    #The goal of independent filtering is to filter out those tests from the procedure that have no, or little chance of showing significant evidence, without even looking at their test statistic. Typically, this results in increased detection power at the same experiment-wide type I error. Here, we measure experiment-wide type I error in terms of the false discovery rate.
    #deseq2 ython does not expose the used treshold
    filtered_results = ds.results_df[
        ds.results_df["padj"].notna() & 
        (ds.results_df["padj"] < padj_cutoff) & 
        (ds.results_df["log2FoldChange"].abs() >= shrunk_lfc_cutoff)
        ]
    if save:
        filtered_results.to_csv(out_path / f"deseq2_de_results_{padj_cutoff}_{shrunk_lfc_cutoff}.csv")

    return filtered_results





def visualize_as_pca(vst, df_metadata, save, out_path, selected_phenotypes):
    if not vst.index.equals(df_metadata.index):
        raise ValueError("Not the same patient id order after vst transformation.")
    
    if not all(p in df_metadata.columns for p in selected_phenotypes):
        raise ValueError(f"One or more selected phenotypes not found in metadata columns.")

    print(selected_phenotypes)
    X = np.asarray(vst)
    pc_model = PCA(n_components=2)
    pca = pc_model.fit_transform(X)     
    load_pc1 = pd.Series(pc_model.components_[0], index=vst.columns)
    print(load_pc1.sort_values().head(20))
    load_pc2 = pd.Series(pc_model.components_[1], index=vst.columns)
    print(load_pc2.sort_values().head(20))

    for p in selected_phenotypes:
        cond = df_metadata[p].values.astype("string").fillna("NA").to_numpy()
        for phenotype in np.unique(cond):
            mask = cond == phenotype
            plt.scatter(pca[mask, 0], pca[mask, 1], label=phenotype)
        """
        #add some labels
        label_ids = ["cfs_cfrna_58"]
        for patient_id in label_ids:
            idx = df_metadata.index.get_loc(patient_id)
            plt.text(pca[idx, 0], pca[idx, 1], patient_id, fontsize=8)
        """

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("VST-PCA")
        plt.legend(title=p)

        if save:
            pca_output_path = out_path / "pca_plots"
            pca_output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(pca_output_path / f"vst_pca_colored_by_{p}.png", dpi=300, bbox_inches="tight")
        plt.show()



# Volcano plot for all genes tested by DESeq2
def visualize_results_as_volcanoplot(ds, padj_cutoff, shrunk_lfc_cutoff, save, out_path):
    
    volcano_data = ds.results_df.copy()
    volcano_data = volcano_data.dropna(subset=["log2FoldChange", "padj"])
    volcano_data["significant"] = ((volcano_data["padj"] < padj_cutoff) & (volcano_data["log2FoldChange"].abs() >= shrunk_lfc_cutoff))

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
    if save:
        plt.savefig(
            out_path / f"volcano_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()





# Heatmap of top differentially expressed genes
def visualize_results_as_heatmap(normalized_counts, df_metadata, filtered_results, condition, save, out_path):
    if not df_metadata.index.equals(normalized_counts.index):
        raise ValueError("Metadata and normalized-count sample order differs.")

    de_genes = filtered_results.index.intersection(normalized_counts.columns)

    if len(de_genes) == 0:
        print("No filtered genes to plot.")
    else:
        heatmap_data = np.log2(normalized_counts.loc[:, de_genes] + 1).T

        #phenotype_palette = {"control": "#4C78A8", "case": "#E45756"}
        sample_condition = df_metadata.loc[normalized_counts.index, condition].astype(str)
        #column_colors = sample_condition.map(phenotype_palette).rename("Phenotype")

        heatmap = sns.clustermap(
            heatmap_data,
            #col_colors=column_colors,
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
#            handles=[
#                Patch(facecolor=color, label=phenotype)
#                for phenotype, color in phenotype_palette.items()
#            ],
#            title="Phenotype",
            loc="upper left",
            bbox_to_anchor=(1.15, 1),
        )

        if save:
            heatmap.savefig(
                out_path / "differential_genes_normalized_counts_heatmap.png",
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

""" 
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

"""






def run_deseq2_de_analysis(counts_path, metadata_path_gh, design_f,padj_cutoff,shrunk_lfc_cutoff,con, coeff_lfc_shrinking, selected_phenotypes_pca, save, out_dir, geo_id):
    df_counts, df_metadata = prepare_data_GSE293840(counts_path, metadata_path_gh, out_dir, save=save, geo_id=geo_id)
    df_counts_filtered = filter_low_counts_min_samples(df_counts, min_count=10, min_samples=3)
    #ensure that the order of samples in counts and metadata is the same
    if not df_counts.index.equals(df_metadata.index):
        raise ValueError("Counts and metadata sample order differs after preparation step.")
    dds, inference, vst = set_up_deseq2_object(df_counts_filtered, df_metadata, design_formula=design_f, save=save, out_path=out_dir)
    visualize_as_pca(vst, df_metadata, save, out_path=out_dir, selected_phenotypes=selected_phenotypes_pca)
    dds, normalized_counts = count_normalization(dds, save=save, out_path=out_dir)
    ds = deseq2_de_analysis(dds, inference, contrast=con, alpha=0.05, coeff_lfc_shrinking=coeff_lfc_shrinking, save=save, out_path=out_dir)
    filtered_results = filter_de_results(ds, padj_cutoff=padj_cutoff, shrunk_lfc_cutoff=shrunk_lfc_cutoff, save=save, out_path=out_dir)
    visualize_results_as_volcanoplot(ds, padj_cutoff=padj_cutoff, shrunk_lfc_cutoff=shrunk_lfc_cutoff, save=save, out_path=out_dir)
    visualize_results_as_heatmap(normalized_counts, df_metadata, filtered_results, condition="phenotype", save=save, out_path=out_dir)
    
    return filtered_results, ds
    


if __name__ == "__main__":
    geo_id = "GSE293840"
    base = Path(str("../data/raw/") + "GSE293840")
    counts_path = base / str("GSE293840" + "_raw_counts_all.csv")
    metadata_path_gh = base / "sample_metadata_from_github.tsv"
    condition = "phenotype"
    con = ["phenotype", "case", "control"]
    padj_cutoff = 0.05
    shrunk_lfc_cutoff = 0.5
    coeff_lfc_shrinking = "phenotype[T.case]"
    selected_phenotypes_pca = ["phenotype","sex","test_site","age_bin","bmi_bin","vo2_peak_bin","gh_bin","mfi20_total_bin","duration_of_mecfs_bin", "project","lib_prep_date"]
    save_data = True
    out_dir = Path(str("../data/processed/") + "GSE293840" + "/phenotype_sex_and_testsite")
    design_f = "~sex+test_site+phenotype"
    #design_f = "~sex+phenotype"
    #out_dir = Path(str("../data/processed/") + "GSE293840" + "/phenotype_and_sex")


    run_deseq2_de_analysis(counts_path, metadata_path_gh, design_f = design_f, padj_cutoff = padj_cutoff, shrunk_lfc_cutoff = shrunk_lfc_cutoff, con = con, coeff_lfc_shrinking = coeff_lfc_shrinking, selected_phenotypes_pca = selected_phenotypes_pca, save = save_data, out_dir = out_dir, geo_id = geo_id)