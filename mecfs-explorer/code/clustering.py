from run_deseq2_de_analysis import run_deseq2_de_analysis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Clustering
# Before training the classifier, check if data builds any clusters that are not related to case/phenotype e.g. test_site, lib_prep_date, or mtRNA_rate
# and if those are unevenly distributed between cases and controls, a classifier can partly learn site or batch instead of disease.
# Use all samples and the top 2000 variable genes bafter vst normalization.
# do not use DE genes because they would already incorporate the case/control info
# do not use sex genes since they are expected to differ
# Use PCAs for clustering


# run PCA from run_deseq2_de_analysis
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
    "sex",
    "test_site",
    "age_bin",
    "bmi_bin",
    "vo2_peak_bin",
    "gh_bin",
    "mfi20_total_bin",
    "duration_of_mecfs_bin",
    "project",
    "lib_prep_date",
]
padj_cutoff = 0.05
shrunk_lfc_cutoff = 0.5
condition = "phenotype"
con = ["phenotype", "case", "control"]
coeff_lfc_shrinking = "phenotype[T.case]"


df_metadata, filtered_results, ds, vst, pc_model, pca = run_deseq2_de_analysis(
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

# TODO Rerun clustering with 1000,2000 and 5000 to see if choice does not influence too much
# TODO remove sex genes

# filter top_n variable genes in vst data
top_n = 1000
gene_var = vst.var(axis=0)
top_genes = gene_var.sort_values(ascending=False).index[:top_n]
vst_top_genes = vst[top_genes]
# explore which genes were selected
m = vst.mean(axis=0)
sel = m.index.isin(top_genes)
plt.scatter(m[~sel], gene_var[~sel], s=2, c="lightgray", label="other")
plt.scatter(m[sel], gene_var[sel], s=2, c="tab:red", label="top 2000")
plt.xlabel("mean VST")
plt.ylabel("variance")
plt.legend()
plt.show()
# ->genes across all means were selected

# Perform PCA and visualize scree plot
X = np.asarray(vst_top_genes)
pca = PCA(n_components=50, random_state=0)
pca_fit = pca.fit_transform(X)
ev = pca.explained_variance_ratio_ * 100
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ev) + 1), ev, marker="o", linestyle="-", color="b")
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")
plt.xticks(range(1, len(ev) + 1))
plt.grid()
plt.show()
# -> elbow method, use first 9 PCAs for clustering


# Create a DataFrame from the first 9 PCA components
n_components = 9
Z_9 = pca_fit[:, :n_components]
pca_columns = [f"PC{i + 1}" for i in range(n_components)]
pca_df = pd.DataFrame(Z_9, columns=pca_columns)
# pca keeps the row order so assign the patient_ids to pca_df
pca_df.index = vst_top_genes.index
# add metadata
merged_df = pd.merge(pca_df, df_metadata, left_index=True, right_index=True)


# Perform hierarchical clustering
plt.figure(figsize=(8, 5))
linkage_matrix = linkage(pca_df, method="ward")
dendrogram(linkage_matrix)
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Euclidean Distance")
plt.show()
# -> look at 4 clusters

n_clusters = 4
cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

plt.figure(figsize=(8, 6))
plt.scatter(pca_df["PC1"], pca_df["PC2"], c=cluster_labels, cmap="tab10", s=50)
plt.title("Hierarchical Clustering Scatter Plot (fcluster)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.show()


phenotype_markers = {"case": "o", "control": "s"}
# TODO also test other markers
# TODO save plots

# TODO understand
# Scatter plot showing clusters and phenotype as shapes
plt.figure(figsize=(8, 6))
for phenotype, marker in phenotype_markers.items():
    subset = merged_df[merged_df["phenotype"] == phenotype]
    subset_cluster_labels = cluster_labels[subset.index.map(merged_df.index.get_loc)]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        c=subset_cluster_labels,
        cmap="tab10",
        s=50,
        marker=marker,
        label=phenotype,
    )

plt.title("Hierarchical Clustering Scatter Plot with Phenotype Shapes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.legend(title="Phenotype")
plt.show()
# TODO other clustering method?


# 2. Secondary analysis: cases only, clustered on your ~200 DE genes (VST, centered), then compared with clinical variables like mecfs_duration, mecfs_sudden_gradual, SF-36 scores, and bas_score.
# 3. Optional: cases only, on top variable genes, to compare with step 2. If both find similar patient groups, that's a stronger result.
