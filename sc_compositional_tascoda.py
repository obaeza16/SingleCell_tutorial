# With labeled clusters and hierarchical structure
# Typical single-cell dataset also contains information about the similarity of the
# different cells in the form of a tree-based hierarchical ordering.
# tascCODA is an extension of scCODA that integrates hierarchical information and 
# experimental covariate data into the generative modeling of compositional count 
# data[Ostner et al., 2021]. This is especially beneficial for cell atlassing efforts
# with increased resolution.
import warnings
import pandas as pd
import scanpy as sc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import altair as alt
import pertpy as pt
import schist

# schist is tricky to import
# https://github.com/dawe/schist/issues/72

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

adata = pt.dt.haber_2017_regions()
adata
adata.obs

# use logcounts to calculate PCA and neighbors
adata.layers["counts"] = adata.X.copy()
adata.layers["logcounts"] = sc.pp.log1p(adata.layers["counts"]).copy()
adata.X = adata.layers["logcounts"].copy()
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30, random_state=1234)
sc.tl.umap(adata)

# Then, we can run schist on the AnnData object, which results in a clustering
# that is defined through a set of columns “nsbm_level_{i}” in adata.obs:

schist.inference.nested_model(adata, random_seed=5678)
adata.obs

sc.pl.umap(
    adata, color=["nsbm_level_1", "nsbm_level_2", "cell_label"], ncols=3, wspace=0.5
)
plt.show()

# The load function of Tasccoda will prepare a MuData object and it converts our 
# tree representation into a ete tree structure and save it as tasccoda_data['coda'].uns["tree"].
# To get some clusters that are not too small, we cut the tree before the last 
# level by leaving out "nsbm_level_0".
tasccoda_model = pt.tl.Tasccoda()
tasccoda_data = tasccoda_model.load(
    adata,
    type="cell_level",
    cell_type_identifier="nsbm_level_1",
    sample_identifier="batch",
    covariate_obs=["condition"],
    levels_orig=["nsbm_level_4", "nsbm_level_3", "nsbm_level_2", "nsbm_level_1"],
    add_level_name=True,
)
tasccoda_data

tasccoda_model.plot_draw_tree(tasccoda_data)

tasccoda_model.prepare(
    tasccoda_data,
    modality_key="coda",
    reference_cell_type="18",
    formula="condition",
    pen_args={"phi": 0, "lambda_1": 3.5},
    tree_key="tree",
)

tasccoda_model.run_nuts(
    tasccoda_data, modality_key="coda", rng_key=1234, num_samples=10000, num_warmup=1000
)

tasccoda_model.summary(tasccoda_data, modality_key="coda")

tasccoda_model.plot_draw_effects(
    tasccoda_data,
    modality_key="coda",
    tree="tree",
    covariate="condition[T.Salmonella]",
    show_leaf_effects=False,
    show_legend=False,
)

tasccoda_model.plot_effects_barplot(
    tasccoda_data, modality_key="coda", covariates="condition"
)

# Another insightful representation can be gained by plotting the effect sizes for
# each condition on the UMAP embedding, and comparing it to the cell type assignments:

kwargs = {"ncols": 3, "wspace": 0.25, "vcenter": 0, "vmax": 1.5, "vmin": -1.5}
tasccoda_model.plot_effects_umap(
    tasccoda_data,
    effect_name=[
        "effect_df_condition[T.Salmonella]",
        "effect_df_condition[T.Hpoly.Day3]",
        "effect_df_condition[T.Hpoly.Day10]",
    ],
    cluster_key="nsbm_level_1",
    **kwargs
)

sc.pl.umap(
    tasccoda_data["rna"], color=["cell_label", "nsbm_level_1"], ncols=2, wspace=0.5
)

# Without labeled clusters

# We first use the standard scanpy workflow for dimensionality reduction to 
# qualitatively assess whether we see a batch effect in this dataset.
milo = pt.tl.Milo()
adata = pt.dt.haber_2017_regions()
mdata = milo.load(adata)
mdata

# use logcounts to calculate PCA and neighbors
adata.layers["counts"] = adata.X.copy()
adata.layers["logcounts"] = sc.pp.log1p(adata.layers["counts"]).copy()
adata.X = adata.layers["logcounts"].copy()

sc.pp.highly_variable_genes(
    adata, n_top_genes=3000, subset=False
)  # 3k genes as used by authors for clustering

sc.pp.pca(adata)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
sc.tl.umap(adata)

sc.pl.umap(adata, color=["condition", "batch", "cell_label"], ncols=3, wspace=0.25)

# To minimize these errors, we apply the scVI method to learn a batch-corrected
# latent space, as introduced in the integration chapter.
import scvi

adata_scvi = adata[:, adata.var["highly_variable"]].copy()
scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts", batch_key="batch")
model_scvi = scvi.model.SCVI(adata_scvi)
max_epochs_scvi = int(np.min([round((20000 / adata.n_obs) * 400), 400]))
model_scvi.train(max_epochs=max_epochs_scvi)
adata.obsm["X_scVI"] = model_scvi.get_latent_representation()

sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata)

sc.pl.umap(adata, color=["condition", "batch", "cell_label"], ncols=3, wspace=0.25)