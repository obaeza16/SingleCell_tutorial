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

# If no neighbors_key parameter is specified, Milo uses the neighbours from .obsp.
# Therefore, ensure that sc.pp.neighbors was run on the correct representation,
# i.e. an integrated latent space if batch correction was required.

milo.make_nhoods(mdata, prop=0.1)
# Now the binary assignment of cells to neighbourhood is stored in adata.obsm['nhoods']
adata.obsm["nhoods"]
# We check the mean number of cells in each neigh, to make sure we can detect 
# differences between samples
nhood_size = adata.obsm["nhoods"].toarray().sum(0)
plt.hist(nhood_size, bins=20)
plt.xlabel("# cells in neighbourhood")
plt.ylabel("# neighbouthoods")

np.median(nhood_size)
# Based on the plot above, we have a large number of neighbourhoods with less than 
# 30 cells, which could lead to an underpowered test. To solve this, we just need
# to recompute the KNN graph using n_neighbors=30. To distinguish this KNN graph
# used for neighbourhood-level DA analysis from the graph used for UMAP building,
# we will store this as a distinct graph in adata.obsp.

sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_scVI", key_added="milo")
milo.make_nhoods(mdata, neighbors_key="milo", prop=0.1)
# Lets check that the distribution has shifted:
nhood_size = adata.obsm["nhoods"].toarray().sum(0)
plt.hist(nhood_size, bins=20)
plt.xlabel("# cells in neighbourhood")
plt.ylabel("# neighbouthoods")

# In the next step, Milo counts cells belonging to each of the samples (here
# identified by the batch column in adata.obs).

milo.count_nhoods(mdata, sample_col="batch")
mdata["milo"]
# We can verify that the number of cells per sample times the number of samples
# roughly corresponds to the number of cells in a neighbourhood.
mean_n_cells = mdata["milo"].X.toarray().mean(0)
plt.plot(nhood_size, mean_n_cells, ".")
plt.xlabel("# cells in nhood")
plt.ylabel("Mean # cells per sample in nhood")

# Milo uses edgeR’s QLF test to test if there are statistically significant 
# differences between the number of cells from a condition of interest in 
# each neighborhood.

# Let’s first test for differences associated with Salmonella infection.

milo.da_nhoods(
    mdata, design="~condition", model_contrasts="conditionSalmonella-conditionControl"
)
milo_results_salmonella = mdata["milo"].obs.copy()
milo_results_salmonella

# For each neighbourhood, we calculate a set of statistics.
# Before any exploration and interpretation of the results, we can visualize these
# statistics with a set of diagnostics plots to sanity check our statistical test
def plot_milo_diagnostics(mdata):
    alpha = 0.1  ## significance threshold

    with matplotlib.rc_context({"figure.figsize": [12, 12]}):

        ## Check P-value histogram
        plt.subplot(2, 2, 1)
        plt.hist(mdata["milo"].var["PValue"], bins=20)
        plt.xlabel("Uncorrected P-value")

        ## Visualize extent of multiple-testing correction
        plt.subplot(2, 2, 2)
        plt.scatter(
            mdata["milo"].var["PValue"],
            mdata["milo"].var["SpatialFDR"],
            s=3,
        )
        plt.xlabel("Uncorrected P-value")
        plt.ylabel("SpatialFDR")

        ## Visualize volcano plot
        plt.subplot(2, 2, 3)
        plt.scatter(
            mdata["milo"].var["logFC"],
            -np.log10(mdata["milo"].var["SpatialFDR"]),
            s=3,
        )
        plt.axhline(
            y=-np.log10(alpha),
            color="red",
            linewidth=1,
            label=f"{int(alpha*100)} % SpatialFDR",
        )
        plt.legend()
        plt.xlabel("log-Fold Change")
        plt.ylabel("- log10(SpatialFDR)")
        plt.tight_layout()

        ## Visualize MA plot
        df = mdata["milo"].var
        emp_null = df[df["SpatialFDR"] >= alpha]["logFC"].mean()
        df["Sig"] = df["SpatialFDR"] < alpha

        plt.subplot(2, 2, 4)
        sns.scatterplot(data=df, x="logCPM", y="logFC", hue="Sig")
        plt.axhline(y=0, color="grey", linewidth=1)
        plt.axhline(y=emp_null, color="purple", linewidth=1)
        plt.legend(title=f"< {int(alpha*100)} % SpatialFDR")
        plt.xlabel("Mean log-counts")
        plt.ylabel("log-Fold Change")
        plt.show()


plot_milo_diagnostics(mdata)

# After sanity check, we can visualize the DA results for each neighbourhood
# by the position of the index cell on the UMAP embedding, to qualitatively 
# assess which cell types may be most affected by the infection.

milo.annotate_nhoods(mdata, anno_col="cell_label")
# Define as mixed if fraction of cells in nhood with same label is lower than 0.75

mdata["milo"].var.loc[
    mdata["milo"].var["nhood_annotation_frac"] < 0.75, "nhood_annotation"
] = "Mixed"

milo.plot_da_beeswarm(mdata)
plt.show()

## Turn into continuous variable
mdata["rna"].obs["Hpoly_timecourse"] = (
    mdata["rna"]
    .obs["condition"]
    .cat.reorder_categories(["Salmonella", "Control", "Hpoly.Day3", "Hpoly.Day10"])
)
mdata["rna"].obs["Hpoly_timecourse"] = mdata["rna"].obs["Hpoly_timecourse"].cat.codes

## Here we exclude salmonella samples
test_samples = (
    mdata["rna"]
    .obs.batch[mdata["rna"].obs.condition != "Salmonella"]
    .astype("str")
    .unique()
)
milo.da_nhoods(mdata, design="~ Hpoly_timecourse", subset_samples=test_samples)

plot_milo_diagnostics(mdata)
with matplotlib.rc_context({"figure.figsize": [10, 10]}):
    milo.plot_nhood_graph(mdata, alpha=0.1, min_size=5, plot_edges=False)
    
milo.plot_da_beeswarm(mdata)
plt.show()

# We can verify that the test captures a linear increase in cell numbers across 
# the time course by plotting the number of cells per sample by condition in 
# neighborhoods where significant enrichment or depletion was detected.

entero_ixs = mdata["milo"].var_names[
    (mdata["milo"].var["SpatialFDR"] < 0.1)
    & (mdata["milo"].var["logFC"] < 0)
    & (mdata["milo"].var["nhood_annotation"] == "Enterocyte")
]

plt.title("Enterocyte")
milo.plot_nhood_counts_by_cond(
    mdata, test_var="Hpoly_timecourse", subset_nhoods=entero_ixs
)
plt.show()


tuft_ixs = mdata["milo"].var_names[
    (mdata["milo"].var["SpatialFDR"] < 0.1)
    & (mdata["milo"].var["logFC"] > 0)
    & (mdata["milo"].var["nhood_annotation"] == "Tuft")
]
plt.title("Tuft cells")
milo.plot_nhood_counts_by_cond(
    mdata, test_var="Hpoly_timecourse", subset_nhoods=tuft_ixs
)
plt.show()

# For example, if we take the neighbourhoods of Goblet cells, we can see that
# neighbourhoods enriched upon infection display a higher expression of Retnlb,
# which is a gene implicated in anti-parasitic immunity [Haber et al., 2017].

## Compute average Retnlb expression per neighbourhood
# (you can add mean expression for all genes using milo.utils.add_nhood_expression)
mdata["rna"].obs["Retnlb_expression"] = (
    mdata["rna"][:, "Retnlb"].layers["logcounts"].toarray().ravel()
)
milo.annotate_nhoods_continuous(mdata, "Retnlb_expression")
# milo.annotate_nhoods(mdata, "Retnlb_expression")

## Subset to Goblet cell neighbourhoods
nhood_df = mdata["milo"].var.copy()
nhood_df = nhood_df[nhood_df["nhood_annotation"] == "Goblet"]

sns.scatterplot(data=nhood_df, x="logFC", y="nhood_Retnlb_expression")
plt.show()

# In Milo, we can express this type of test design using the syntax ~ 
# confounder + condition.
## Make dummy confounder for the sake of this example
np.random.seed(42)
nhood_adata = mdata["milo"].copy()
conf_dict = dict(
    zip(
        nhood_adata.obs_names,
        np.random.choice(["group1", "group2"], nhood_adata.n_obs),
    )
)
mdata["rna"].obs["dummy_confounder"] = [conf_dict[x] for x in mdata["rna"].obs["batch"]]

milo.da_nhoods(mdata, design="~ dummy_confounder+condition")
mdata["milo"].var

