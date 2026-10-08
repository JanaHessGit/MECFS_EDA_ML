na_genes = ds.results_df[ds.results_df["pvalue"].isna()].index
ds_nocook.results_df.loc[ds_nocook.results_df.index.isin(na_genes)]
na_genes=['ENSG00000132972.19', 'ENSG00000144820.8', 'ENSG00000187268.12',
       'ENSG00000224525.2', 'ENSG00000234710.2', 'ENSG00000252850.1',
       'ENSG00000256971.2', 'ENSG00000278791.1', 'ENSG00000286172.1']
#check the expression data for the na genes
#Check which samples cause the outliers.
# #TODO
df_counts_fil = df_counts.loc[:, df_counts.columns.isin(na_genes)]




