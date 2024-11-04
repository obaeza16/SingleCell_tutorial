from pathlib import Path

import scanpy as sc
import scvelo as scv
scv.settings.set_figure_params("scvelo")
# Loading data
DATA_DIR = Path("./")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH = DATA_DIR/"pancreas.h5ad"
adata = scv.datasets.pancreas(file_path=FILE_PATH)
adata

# Since scRNA-seq data is noisy and sparse, the data must be preprocessed in order
# to infer RNA velocity with the steady-state or EM model. As a first step, we filter 
# out genes that are not sufficiently expressed both unspliced and spliced RNA 
# (here, at least 20). Following, the cell size is normalized for both unspliced and
# spliced RNA, and counts in adata.X log1p transformed to reduce the effect of outliers.
# Next, we also identify and filter for highly variable genes (2000 here)

scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
#  In the case of RNA velocity, we additionally smooth observations by the mean 
# expression in their neighborhood. This can be done using scVelo’s moments function.
sc.tl.pca(adata)
sc.pp.neighbors(adata)
scv.pp.moments(adata, n_pcs=None, n_neighbors=None)


# In a typical workflow, we would cluster the data, infer cell types, and visualize 
# the data in a two-dimensional embedding. Luckily, for the pancreas data, this 
# information has already been calculated a priori and directly be used.
scv.pl.scatter(adata, basis="umap", color="clusters")

# As a first step, we calculate RNA velocity under the steady state model. In this 
# case, we call scVelo’s velocity function with mode="deterministic".
scv.tl.velocity(adata, mode="deterministic")
scv.tl.velocity_graph(adata, n_jobs=20)
scv.pl.velocity_embedding_stream(adata, basis="umap", color="clusters")

# RNA velocity inference - EM model
scv.tl.recover_dynamics(adata, n_jobs=20)
top_genes = adata.var["fit_likelihood"].sort_values(ascending=False).index
scv.pl.scatter(adata, basis=top_genes[:5], color="clusters", frameon=False)

scv.tl.velocity(adata, mode="dynamical")
scv.tl.velocity_graph(adata, n_jobs=20)
scv.pl.velocity_embedding_stream(adata, basis="umap")