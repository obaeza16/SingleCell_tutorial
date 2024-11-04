# Python packages
import scanpy as sc
import scvi
import bbknn
import scib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# R interface
from rpy2.robjects import pandas2ri
from rpy2.robjects import r
import rpy2.rinterface_lib.callbacks
import anndata2ri

pandas2ri.activate()
anndata2ri.activate()

%load_ext rpy2.ipython

# We load the data from bash:
# wget https://figshare.com/ndownloader/files/45452260 -O openproblems_bmmc_multiome_genes_filtered.h5ad

adata_raw = sc.read_h5ad("openproblems_bmmc_multiome_genes_filtered.h5ad")
adata_raw.layers["logcounts"] = adata_raw.X
adata_raw
# The full dataset contains 69,249 cells and measurements for 129,921 features

# We define variables to hold these names so that it is clear how we are using them in the code
label_key = "cell_type"
batch_key = "batch"

# We look at the different batches we have:
adata_raw.obs[batch_key].value_counts()
# We have 13 different batches, we will select three samples to use:

keep_batches = ["s1d3", "s2d1", "s3d7"]
adata = adata_raw[adata_raw.obs[batch_key].isin(keep_batches)].copy()
adata

# We will focus on the geneexpression features (GEX)
adata.var["feature_types"].value_counts()

adata = adata[:, adata.var["feature_types"] == "GEX"].copy()
sc.pp.filter_genes(adata, min_cells=1)
adata

# Due to the subsetting, we need to renormalize the data
adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata.layers["logcounts"] = adata.X.copy()

# We will have a look at the raw data with highly variable gene (HVG) selection, PCA and UMAP
sc.pp.highly_variable_genes(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
adata

# We plot the UMAP colouring the points by cell identity and batch labels.
adata.uns[batch_key + "_colors"] = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
]  # Set custom colours for batches
sc.pl.umap(adata, color=[label_key, batch_key], wspace=1)

# Batch aware fature selection

# We can perform batch-aware highly variable gene selection by setting the batch_key 
# argument in the scanpy highly_variable_genes() function.
sc.pp.highly_variable_genes(
    adata, n_top_genes=2000, flavor="cell_ranger", batch_key=batch_key
)
adata
adata.var
# Check to see how many batches each gene was variable in:
n_batches = adata.var["highly_variable_nbatches"].value_counts()
ax = n_batches.plot(kind="bar")
n_batches
# We select the highly_variable genes
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
adata_hvg

# The first integration model we will use is scVI (single-cell Variational Inference)
# In benchmarking studies scVI has been shown to perform well across a range of 
# datasets with a good balance of batch correction while conserving biological 
# variability [Luecken et al., 2021]. 

adata_scvi = adata_hvg.copy()
# First step is to prepare our anndata object
scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts", batch_key=batch_key)
adata_scvi
# Now we construct an scVI model object
model_scvi = scvi.model.SCVI(adata_scvi)
model_scvi
model_scvi.view_anndata_setup()
# The model is not trained, so that is the next step; this training is a compute
# intensive process, about 20~40 minutes
max_epochs_scvi = np.min([round((20000 / adata.n_obs) * 400), 400])
max_epochs_scvi
model_scvi.train()

# The main result we want to extract from the trained model is the latent representation
# for each cell. This is a multi-dimensional embedding where the batch effects have been
# removed that can be used in a similar way to how we use PCA dimensions when analysing 
# a single dataset. We store this in obsm with the key X_scvi.
adata_scvi.obsm["X_scVI"] = model_scvi.get_latent_representation()

# Now we plot a UMAP based on the representation by scVI
sc.pp.neighbors(adata_scvi, use_rep="X_scVI")
sc.tl.umap(adata_scvi)
adata_scvi
sc.pl.umap(adata_scvi, color=[label_key, batch_key], wspace=1)

# This looks better! Before, the various batches were shifted apart from each other. 
# Now, the batches overlap more and we have a single blob for each cell identity label.
# In many cases, we would not already have identity labels so from this stage we would 
# continue with clustering, annotation and further analysis as described in other chapters.


# When we have labels for at least some of the cells we can use scANVI 
# (single-cell ANnotation using Variational Inference) [Xu et al., 2021].

# We start by creating a scANVI model object. Note that because scANVI refines 
# an already trained scVI model, we provide the scVI model rather than an AnnData
# object. If we had not already trained an scVI model we would need to do that first.
# Normally we would need to run scVI first but we have already done that here
# model_scvi = scvi.model.SCVI(adata_scvi) etc.
model_scanvi = scvi.model.SCANVI.from_scvi_model(
    model_scvi, labels_key=label_key, unlabeled_category="unlabelled"
)
print(model_scanvi)
model_scanvi.view_anndata_setup()
# As we are refining the model, we need less epochs
max_epochs_scanvi = int(np.min([10, np.max([2, round(max_epochs_scvi / 3.0)])]))
model_scanvi.train(max_epochs=max_epochs_scanvi)

adata_scanvi = adata_scvi.copy()
adata_scanvi.obsm["X_scANVI"] = model_scanvi.get_latent_representation()
sc.pp.neighbors(adata_scanvi, use_rep="X_scANVI")
sc.tl.umap(adata_scanvi)
sc.pl.umap(adata_scanvi, color=[label_key, batch_key], wspace=1)


# The next method we will visit is named BBKNN or “Batch Balanced KNN” 
# [Polański et al., 2019].
neighbors_within_batch = 25 if adata_hvg.n_obs > 100000 else 3
neighbors_within_batch

adata_bbknn = adata_hvg.copy()
adata_bbknn.X = adata_bbknn.layers["logcounts"].copy()
sc.pp.pca(adata_bbknn)

bbknn.bbknn(
    adata_bbknn, batch_key=batch_key, neighbors_within_batch=neighbors_within_batch
)
adata_bbknn
# We then plot the UMAP
sc.tl.umap(adata_bbknn)
sc.pl.umap(adata_bbknn, color=[label_key, batch_key], wspace=1)


# The SEURAT integration method
# This outputs a corrected expression matrix. As it is an R package, here we 
# adapt our AnnData object to trasfer to R
adata_seurat = adata_hvg.copy()
# Convert categorical columns to strings
adata_seurat.obs[batch_key] = adata_seurat.obs[batch_key].astype(str)
adata_seurat.obs[label_key] = adata_seurat.obs[label_key].astype(str)
# Delete uns as this can contain arbitrary objects which are difficult to convert
del adata_seurat.uns
adata_seurat
# The prepared AnnData is now available in R as a SingleCellExperiment object thanks
# to anndata2ri. Note that this is transposed compared to an AnnData object so our 
# observations (cells) are now the columns and our variables (genes) are now the rows.

# FROM HERE, R
%%R -i adata_seurat
# The prepared AnnData is now available in R as a SingleCellExperiment object thanks to anndata2ri
adata_seurat
# We just provide the SingleCellExperiment object and tell Seurat which assays 
# (layers in our AnnData object) contain raw counts and normalised expression 
# (which Seurat stores in a slot called “data”).
seurat <- as.Seurat(adata_seurat, counts = "counts", data = "logcounts")
seurat
# Seurat integration functions require a list of objects. We create this using the 
# SplitObject() function.
batch_list <- SplitObject(seurat, split.by = batch_key)
batch_list
# Usually, you would identify batch-aware highly variable genes first (using the 
# FindVariableFeatures() and SelectIntegrationFeatures() functions) but as we have 
# done that already we tell Seurat to use all the features in the object.
anchors <- FindIntegrationAnchors(batch_list, anchor.features = rownames(seurat))
anchors

integrated <- IntegrateData(anchors)
integrated
# The result is another Seurat object, but notice now that the active assay is 
# called “integrated”. This contains the corrected expression matrix which is the
# final output of the integration.
# Here we extract that matrix and prepare it for transfer back to Python.
# Extract the integrated expression matrix
integrated_expr <- GetAssayData(integrated)
# Make sure the rows and columns are in the same order as the original object
integrated_expr <- integrated_expr[rownames(seurat), colnames(seurat)]
# Transpose the matrix to AnnData format
integrated_expr <- t(integrated_expr)
print(integrated_expr[1:10, 1:10])

# From now on, PYTHON instructions

# We will now store the corrected expression matrix as a layer in our AnnData 
# object. We also set adata.X to use this matrix.
adata_seurat.X = integrated_expr
adata_seurat.layers["seurat"] = integrated_expr
print(adata_seurat)
adata.X
# Plot UMAP
# Reset the batch colours because we deleted them earlier
adata_seurat.uns[batch_key + "_colors"] = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
]
sc.tl.pca(adata_seurat)
sc.pp.neighbors(adata_seurat)
sc.tl.umap(adata_seurat)
sc.pl.umap(adata_seurat, color=[label_key, batch_key], wspace=1)

# It is tempting to select an integration based on the UMAPs but this does 
# not fully represent the quality of an integration. In the next section, 
# we present some approaches to more rigorously evaluate integration methods.

# We will use scib to benchmark our integration
# We run the metriccs for each integration
metrics_scvi = scib.metrics.metrics_fast(
    adata, adata_scvi, batch_key, label_key, embed="X_scVI"
)
metrics_scanvi = scib.metrics.metrics_fast(
    adata, adata_scanvi, batch_key, label_key, embed="X_scANVI"
)
metrics_bbknn = scib.metrics.metrics_fast(adata, adata_bbknn, batch_key, label_key)
metrics_seurat = scib.metrics.metrics_fast(adata, adata_seurat, batch_key, label_key)
metrics_hvg = scib.metrics.metrics_fast(adata, adata_hvg, batch_key, label_key)
# Scores are between 0 and 1, where 1 is a good performance and 0 is a poor performance
# Concatenate metrics results
metrics = pd.concat(
    [metrics_scvi, metrics_scanvi, metrics_bbknn, metrics_seurat, metrics_hvg],
    axis="columns",
)
# Set methods as column names
metrics = metrics.set_axis(
    ["scVI", "scANVI", "BBKNN", "Seurat", "Unintegrated"], axis="columns"
)
# Select only the fast metrics
metrics = metrics.loc[
    [
        "ASW_label",
        "ASW_label/batch",
        "PCR_batch",
        "isolated_label_silhouette",
        "graph_conn",
        "hvg_overlap",
    ],
    :,
]
# Transpose so that metrics are columns and methods are rows
metrics = metrics.T
# Remove the HVG overlap metric because it's not relevant to embedding outputs
metrics = metrics.drop(columns=["hvg_overlap"])
metrics
metrics.style.background_gradient(cmap="Blues")

# To see the difference between integrations more clearly
metrics_scaled = (metrics - metrics.min()) / (metrics.max() - metrics.min())
metrics_scaled.style.background_gradient(cmap="Blues")
# The values now better represent the differences between methods (and better match the colour scale). 
# If we wanted to add another method we would need to perform the scaling again

# The evaluation metrics can be grouped into two categories, those that measure
# the removal of batch effects and those that measure the conservation of biological variation. 
# We can calculate summary scores for each of these categories by taking the mean of the scaled
# values for each group.
metrics_scaled["Batch"] = metrics_scaled[
    ["ASW_label/batch", "PCR_batch", "graph_conn"]
].mean(axis=1)
metrics_scaled["Bio"] = metrics_scaled[["ASW_label", "isolated_label_silhouette"]].mean(
    axis=1
)
metrics_scaled.style.background_gradient(cmap="Blues")
# Plotting the two summary scores against each other gives an indication of the 
# priorities of each method. Some will be biased towards batch correction while
# others will favour retaining biological variation.

fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
metrics_scaled.plot.scatter(
    x="Batch",
    y="Bio",
    c=range(len(metrics_scaled)),
    ax=ax,
)

for k, v in metrics_scaled[["Batch", "Bio"]].iterrows():
    ax.annotate(
        k,
        v,
        xytext=(6, -3),
        textcoords="offset points",
        family="sans-serif",
        fontsize=12,
    )
# In our small example scenario BBKNN is clearly the worst performer, getting the
# lowest scores for both batch removal and biological conservation. The other three
# methods have similar batch correction scores with scANVI scoring highest for 
# biological conservation followed by scVI and Seurat.

# To get an overall score for each method we can combine the two summary scores. 
# The scIB paper suggests a weighting of 40% batch correction and 60% biological 
# conservation but you may prefer to weight things differently depending on the 
# priorities for your dataset.

metrics_scaled["Overall"] = 0.4 * metrics_scaled["Batch"] + 0.6 * metrics_scaled["Bio"]
metrics_scaled.style.background_gradient(cmap="Blues")
# Let’s make a quick bar chart to visualise the overall performance.
metrics_scaled.plot.bar(y="Overall")