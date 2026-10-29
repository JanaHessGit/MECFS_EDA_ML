from run_deseq2_de_analysis import run_deseq2_de_analysis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from matplotlib.lines import Line2D

# Clustering
# Before training the classifier, lets check if data builds any clusters that are not 
# related to case/phenotype e.g. test_site, lib_prep_date, or mtRNA_rate
# and if those are unevenly distributed between cases and controls, a classifier can 
# partly learn site or batch instead of disease.
# Use data after vst normalization and PCA components as input for clustering since
# PCA keeps the coordinated signal and discards much of the noise. Also, it helps
# with curse of dimensionality because less features are considered


##### Run_deseq2_de_analysis to have vst values and DE genes
geo_id = "GSE293840"
base = Path(str("../data/raw/") + "GSE293840")
counts_path = base / str("GSE293840" + "_raw_counts_all.csv")
metadata_path_gh = base / "sample_metadata_from_github.tsv"
design_f = "~sex+test_site+phenotype"
save_data = False
out_dir = Path(str(""))

# not important for here
selected_phenotypes_pca = [
    "phenotype",
]
padj_cutoff = 0.05
shrunk_lfc_cutoff = 0.5
condition = "phenotype"
con = ["phenotype", "case", "control"]
coeff_lfc_shrinking = "phenotype[T.case]"


df_metadata, filtered_results_deseq2, ds, vst, pc_model, pca = run_deseq2_de_analysis(
    counts_path,
    metadata_path_gh,
    design_f=design_f,
    padj_cutoff=padj_cutoff,
    shrunk_lfc_cutoff=shrunk_lfc_cutoff,
    con=con,
    coeff_lfc_shrinking=coeff_lfc_shrinking,
    selected_phenotypes_pca=selected_phenotypes_pca,
    save=save_data,
    out_dir=out_dir,
    geo_id=geo_id,
)




####### Cluster Analysis ######

def perform_pca(selected_data, output_dir):
    X = np.asarray(selected_data)
    pca = PCA(n_components=50, random_state=0)
    pca_fit = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_ * 100
    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(ev) + 1), ev, marker="o", linestyle="-", color="b")
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.xticks(range(1, len(ev) + 1))
    plt.grid()
    fig.savefig(
        output_dir / f"pca_screeplot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return pca_fit, pca


# Create a DataFrame from the first n PCA components
def df_from_pca_components(n_components, vst_dt, pca_fit, df_metadata):
    Z_n = pca_fit[:, :n_components]
    pca_columns = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(Z_n, columns=pca_columns)
    # pca keeps the row order so assign the patient_ids to pca_df
    pca_df.index = vst_dt.index
    # add metadata
    merged_df = pd.merge(pca_df, df_metadata, left_index=True, right_index=True)
    return pca_df, merged_df


def draw_dendogram(pca_df, method, output_dir):
    fig = plt.figure(figsize=(8, 5))
    linkage_matrix = linkage(pca_df, method = method)
    dendrogram(linkage_matrix)
    plt.title("Dendrogram")
    plt.xlabel("Samples")
    fig.savefig(
        output_dir / f"dendogram_hierarcical_clustering_method_{method}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    return linkage_matrix


def perform_hierachichal_clustering(pca_df, n_clusters, linkage_matrix, criterion_clust, output_dir):

    cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion=criterion_clust)
    fig =plt.figure(figsize=(8, 6))
    plt.scatter(pca_df["PC1"], pca_df["PC2"], c=cluster_labels, cmap="tab10", s=50)
    plt.title("Hierarchical Clustering Scatter Plot (fcluster)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Cluster")
    fig.savefig(
        output_dir / f"PC1_PC2_hierarcical_clustering_method_{criterion_clust}.png",
        dpi=300,
        bbox_inches="tight",
    )    
    plt.show()
    return cluster_labels



# Scatter plot showing clusters as number and other metadata as color
def scatter_plot_clusters_as_number_metadata_as_color(df, vars_to_plot, cluster_labels, pca_df, output_dir):
    cluster_by_sample = pd.Series(cluster_labels, index=pca_df.index)

    for name_var in vars_to_plot:
        values = df[name_var].dropna().unique()

        if len(values) == 0:
            raise ValueError(f"No non-missing values found for {name_var}.")

        cmap_name = "tab10" if len(values) <= 10 else "tab20"
        cmap = plt.get_cmap(cmap_name, len(values))
        category_colors = {
            value: cmap(i) for i, value in enumerate(values)
        }

        fig, ax = plt.subplots(figsize=(8, 6))

        for value in values:
            subset = df.loc[df[name_var] == value]
            subset_cluster_labels = cluster_by_sample.loc[subset.index]

            ax.scatter(
                subset["PC1"],
                subset["PC2"],
                color=category_colors[value],
                s=100,
                edgecolors="white",
                linewidths=0.5,
                label=str(value),
            )

            for x, y, cluster in zip(
                subset["PC1"],
                subset["PC2"],
                subset_cluster_labels,
            ):
                ax.annotate(
                    str(cluster),
                    (x, y),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    bbox={
                        "boxstyle": "circle,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0,
                    },
                )

        ax.set_title(f"PCA clusters by {name_var}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.text(
            0.02,
            0.02,
            "Number on each point = cluster",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.legend(title=name_var, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"clusters_by_{name_var}_color_numbered.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
    return 1



# Scatter plot showing clusters as color and other metadata as shape
def scatter_plot_clusters_as_color_metadata_as_shape(df, vars_to_plot, cluster_labels, pca_df, output_dir):
    marker_symbols = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    cluster_by_sample = pd.Series(cluster_labels, index=pca_df.index)

    for name_var in vars_to_plot:
        values = df[name_var].dropna().unique()

        if len(values) == 0:
            raise ValueError(f"No non-missing values found for {name_var}.")
        if len(values) > len(marker_symbols):
            raise ValueError(
                f"{name_var} has {len(values)} categories, "
                f"but only {len(marker_symbols)} marker shapes are defined."
            )

        var_markers = dict(zip(values, marker_symbols))

        fig, ax = plt.subplots(figsize=(8, 6))

        for value, marker in var_markers.items():
            subset = df.loc[df[name_var] == value]
            subset_cluster_labels = cluster_by_sample.loc[subset.index]

            scatter = ax.scatter(
                subset["PC1"],
                subset["PC2"],
                c=subset_cluster_labels,
                cmap="tab10",
                vmin=cluster_labels.min(),
                vmax=cluster_labels.max(),
                s=50,
                marker=marker,
            )

        legend_handles = [
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="None",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=8,
                label=str(value),
            )
            for value, marker in var_markers.items()
        ]

        ax.legend(handles=legend_handles, title=name_var)
        ax.set_title(f"PCA clusters by {name_var}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(scatter, ax=ax, label="Cluster")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"clusters_by_{name_var}_shape_colored.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show() 
    return 1


#### Run clustering analysis

"""
# Version 1: Use all patients (case & control), top_n variable genes in vst data, 
# independent from DE analysis because they would already incorporate the case/control info

 
top_n = 1000
gene_var = vst.var(axis=0)
top_genes = gene_var.sort_values(ascending=False).index[:top_n]
vst_top_genes = vst[top_genes]
# explore which genes were selected
m = vst.mean(axis=0)
sel = m.index.isin(top_genes)
plt.scatter(m[~sel], gene_var[~sel], s=2, c="lightgray", label="other")
plt.scatter(m[sel], gene_var[sel], s=2, c="tab:red", label=str("top "+ str(top_n)))
plt.xlabel("mean VST")
plt.ylabel("variance")
plt.legend()
plt.show()
# -> genes across all means were selected



n_components = 10
n_clusters = 6
output_plot_dir = Path(f"../output/top_n_variable_vst_genes_all_patients/clustering_plots_{n_components}pcs_{top_n}genes_{n_clusters}clusters")
output_plot_dir.mkdir(parents=True, exist_ok=True)


pca_fit1, pca1 = perform_pca(vst_top_genes, output_plot_dir)
pca_df1, merged_df1 = df_from_pca_components(n_components, vst_top_genes, pca_fit1, df_metadata)
linkage_matrix1 = draw_dendogram(pca_df1, "ward", output_plot_dir)
cluster_labels1 = perform_hierachichal_clustering(pca_df1, n_clusters, linkage_matrix1, "maxclust", output_plot_dir)


variables_to_plot = [
    "phenotype",
    "sex",
    "test_site",
    "age_bin",
    "bmi_bin",
    "vo2_peak_bin",
    "gh_bin",
    "project",
]

scatter_plot_clusters_as_number_metadata_as_color(merged_df1, variables_to_plot, cluster_labels1, pca_df1, output_plot_dir)
scatter_plot_clusters_as_color_metadata_as_shape(merged_df1, variables_to_plot, cluster_labels1, pca_df1, output_plot_dir)
"""

""" Findings: 
5,10 components similar findings:
1000 genes, 6 clusters:
project: all patients with high PC2 values are project 1,
but project 1 also has low/mid PC1 values, project 1 also 
has tendency for lower PC1 values

test_site: also test_site seems not to be evenly distributed, 
no high PC2 values are Ithaca but Ithaca seems to be dense 
in low PC1 values but also most patients came from Ithaca

Project 5 is only Ithaca, but project 5 seems to be evenly distributed across PC1

2000 genes, 7 clusters and 5000 genes, 7 clusters:
test_site: also here test_site seems to be dense at low
PC1 values and almost no other test sites here but here 
Ithaca has also high PC2 values 

"""



# Version 2: Cases only, n most DE genes from DESEQ2 analysis using vst data for clustering
# then compare with clinical variables like mecfs_duration, mecfs_sudden_gradual, mfi20 scores.

#Filter cases and top n DE genes
n_top_de_genes = 200
case_ids = df_metadata.loc[df_metadata["phenotype"] == "case"].index.tolist()
ds_top_de_genes = filtered_results_deseq2.sort_values(
                by="log2FoldChange",
                key=lambda column: column.abs(),
                ascending=False,
            ).index[:n_top_de_genes]

vst_top_de_genes = vst[ds_top_de_genes]
vst_top_de_genes_cases = vst_top_de_genes.loc[case_ids]



# Clstering on cases and n_top_de_genes DE genes in vst data independent (based on DESEQ2)
n_components = 10
n_clusters = 7
out_dir = Path(f"../output/top_de_genes_cases/clustering_plots_{n_components}pcs_{n_top_de_genes}genes_{n_clusters}clusters")
out_dir.mkdir(parents=True, exist_ok=True)


pca_fit_cases_topDE, pca_cases_topDE = perform_pca(vst_top_de_genes_cases, out_dir)
pca_df_cases_topDE, merged_df_cases_topDE = df_from_pca_components(n_components, vst_top_de_genes_cases, pca_fit_cases_topDE, df_metadata)
linkage_matrix_cases_topDE = draw_dendogram(pca_df_cases_topDE, "ward", out_dir)
cluster_labels = perform_hierachichal_clustering(pca_df_cases_topDE, n_clusters, linkage_matrix_cases_topDE, "maxclust", out_dir)

variables_to_plot = [
    "phenotype",
    "sex",
    "test_site",
    "age_bin",
    "bmi_bin",
    "vo2_peak_bin",
    "project",
    "gh_bin",
    "duration_of_mecfs_bin",
    "mfi20_total_bin",
    "mecfs_sudden_gradual"
]

scatter_plot_clusters_as_number_metadata_as_color(merged_df_cases_topDE, variables_to_plot, cluster_labels, pca_df_cases_topDE, out_dir)
scatter_plot_clusters_as_color_metadata_as_shape(merged_df_cases_topDE, variables_to_plot, cluster_labels, pca_df_cases_topDE, out_dir)


""" Findings: 
test_site clustering not visible here
no clusters can be clearly mapped to any metadata 
no clear clusters in general, except maybe the two patients with high PC2 values
"""


#TODO:
# 1. Cluster analysis: cases only, on top variable genes (not DE but based on vst), to compare with step 1. Check if both analyses find similar patient groups.
# 2. Test cluster–variable associations. For categorical variables, use contingency-table tests; for continuous variables, compare distributions across clusters. 
#    With a small cohort, prefer exact or permutation-based tests where appropriate. Report effect sizes and adjust for multiple comparisons, for example with Benjamini–Hochberg correction.
# 3. Case-only subgroup analysis to e.g. explore patient subgroups. Would have to use genes that vary among cases (new analysis).
# 4. Try other clustering method?
