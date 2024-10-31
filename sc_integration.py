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

adata_hvg = adata[:, adata.var["highly_variable"]].copy()
adata_hvg