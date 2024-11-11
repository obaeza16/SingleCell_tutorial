# Our goal here is to decipher which cell types were most affected by INF-β treatment.
# First, we import pertpy and scanpy.

import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# This is required to catch warnings when the multiprocessing module is used
import os

os.environ["PYTHONWARNINGS"] = "ignore"
import pertpy as pt
import scanpy as sc

adata = pt.dt.kang_2018()

adata.obs.rename({"label": "condition"}, axis=1, inplace=True)
adata.obs["condition"].replace({"ctrl": "control", "stim": "stimulated"}, inplace=True)
adata.obs.cell_type.value_counts()

# We now create an Augur object using pertpy based on our estimator of interest
# to measure how predictable the perturbation labels for each cell type in the dataset are.

ag_rfc = pt.tl.Augur("random_forest_classifier")


loaded_data = ag_rfc.load(adata, label_col="condition", cell_type_col="cell_type")
loaded_data

v_adata, v_results = ag_rfc.predict(
    loaded_data, subsample_size=20, n_threads=4, select_variance_features=True, span=1
)

v_results["summary_metrics"]

lollipop = ag_rfc.plot_lollipop(v_results)

sc.pp.neighbors(v_adata)
sc.tl.umap(v_adata)
sc.pl.umap(adata=v_adata, color=["augur_score", "cell_type", "label"])


important_features = ag_rfc.plot_important_features(v_results)

# We will now evaluate the effect of the withdraw_15d_Cocaine and withdraw_48h_Cocaine
# conditions compared to Maintenance_Cocaine

# differential prioritization is obtained through a permutation test of the difference
# in AUC between two sets of cell-type prioritizations, compared with the expected
# AUC difference between the same two prioritizations after random permutation of
# sample labels[Squair et al., 2021]. 

bhattacherjee_adata = pt.dt.bhattacherjee()
ag_rfc = pt.tl.Augur("random_forest_classifier")

sc.pp.log1p(bhattacherjee_adata)

# Default mode
bhattacherjee_15 = ag_rfc.load(
    bhattacherjee_adata,
    condition_label="Maintenance_Cocaine",
    treatment_label="withdraw_15d_Cocaine",
)

bhattacherjee_adata_15, bhattacherjee_results_15 = ag_rfc.predict(
    bhattacherjee_15, random_state=None, n_threads=4
)
bhattacherjee_results_15["summary_metrics"].loc["mean_augur_score"].sort_values(
    ascending=False
)

# Permute mode
bhattacherjee_adata_15_permute, bhattacherjee_results_15_permute = ag_rfc.predict(
    bhattacherjee_15,
    augur_mode="permute",
    n_subsamples=100,
    random_state=None,
    n_threads=4,
)

# Default mode
bhattacherjee_48 = ag_rfc.load(
    bhattacherjee_adata,
    condition_label="Maintenance_Cocaine",
    treatment_label="withdraw_48h_Cocaine",
)

bhattacherjee_adata_48, bhattacherjee_results_48 = ag_rfc.predict(
    bhattacherjee_48, random_state=None, n_threads=4
)

bhattacherjee_results_48["summary_metrics"].loc["mean_augur_score"].sort_values(
    ascending=False
)

# Permute mode
bhattacherjee_adata_48_permute, bhattacherjee_results_48_permute = ag_rfc.predict(
    bhattacherjee_48,
    augur_mode="permute",
    n_subsamples=100,
    random_state=None,
    n_threads=4,
)

scatter = ag_rfc.plot_scatterplot(bhattacherjee_results_15, bhattacherjee_results_48)

# To figure out which cell type was most affected when comparing withdraw_48h_Cocaine
# and withdraw_15d_Cocaine we can run differential prioritization.

pvals = ag_rfc.predict_differential_prioritization(
    augur_results1=bhattacherjee_results_15,
    augur_results2=bhattacherjee_results_48,
    permuted_results1=bhattacherjee_results_15_permute,
    permuted_results2=bhattacherjee_results_48_permute,
)
pvals

diff = ag_rfc.plot_dp_scatter(pvals)

# Predicting IFN-β response for CD4-T cells
import scanpy as sc
import pertpy as pt

adata = pt.dt.kang_2018()
# scGen works best with log tranformed data
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
# We rename label to condition and the conditions themselves for improved readability.
adata.obs.rename({"label": "condition"}, axis=1, inplace=True)
adata.obs["condition"].replace({"ctrl": "control", "stim": "stimulated"}, inplace=True)
# We have seven cell types
adata.obs.cell_type.value_counts()
# We remove CD4T to simulate a real-world scenario where we do not capture these 
# cells in our experiment

adata_t = adata[
    ~(
        (adata.obs["cell_type"] == "CD4 T cells")
        & (adata.obs["condition"] == "stimulated")
    )
].copy()

cd4t_stim = adata[
    (
        (adata.obs["cell_type"] == "CD4 T cells")
        & (adata.obs["condition"] == "stimulated")
    )
].copy()

pt.tl.SCGEN.setup_anndata(adata_t, batch_key="condition", labels_key="cell_type")

model = pt.tl.SCGEN(adata_t, n_hidden=800, n_latent=100, n_layers=2)

model.train(
    max_epochs=100, batch_size=32, early_stopping=True, early_stopping_patience=25
)
# We will now plot the model using a UMAP
adata_t.obsm["scgen"] = model.get_latent_representation()

sc.pp.neighbors(adata_t, use_rep="scgen")
sc.tl.umap(adata_t)

sc.pl.umap(adata_t, color=["condition", "cell_type"], wspace=0.4, frameon=False)

# Predicting CD4T responses to IFN-β stimulation
pred, delta = model.predict(
    ctrl_key="control", stim_key="stimulated", celltype_to_predict="CD4 T cells"
)

# we annotate the predicted cells to distinguish them later from ground truth cells.
pred.obs["condition"] = "predicted stimulated"

# Evaluating the predicted IFN-β response
ctrl_adata = adata[
    ((adata.obs["cell_type"] == "CD4 T cells") & (adata.obs["condition"] == "control"))
]
# concatenate pred, control and real CD4 T cells in to one object
eval_adata = ctrl_adata.concatenate(cd4t_stim, pred)
# First look at the PCA
sc.tl.pca(eval_adata)
sc.pl.pca(eval_adata, color="condition", frameon=False)

cd4t_adata = adata[adata.obs["cell_type"] == "CD4 T cells"]

# We estimate DEGs using scanpy’s implementation of the Wilcoxon test.
sc.tl.rank_genes_groups(cd4t_adata, groupby="condition", method="wilcoxon")
diff_genes = cd4t_adata.uns["rank_genes_groups"]["names"]["stimulated"]
diff_genes

from scvi import REGISTRY_KEYS

r2_value = model.plot_reg_mean_plot(
    eval_adata,
    condition_key="condition",
    axis_keys={"x": "predicted stimulated", "y": "stimulated"},
    gene_list=diff_genes[:10],
    top_100_genes=diff_genes,
    labels={"x": "predicted", "y": "ground truth"},
    show=True,
    legend=False,
)

sc.pl.violin(eval_adata, keys="ISG15", groupby="condition")

# Analysing single-pooled CRISPR screens
# For the purpose of this analysis we will be using the Papalexi 2021 dataset
# [Papalexi et al., 2021]. 

# For this specific analysis, we want to:
#     Remove confounding sources of variation such as cell cycle effects or batch effects.
#     Determine which cells were affected by the desired perturbations and which cells escaped.
#     Visualize perturbation responses.
import pertpy as pt
import muon as mu
import scanpy as sc

mdata = pt.dt.papalexi_2021()
mdata

sc.pp.normalize_total(mdata["rna"])
sc.pp.log1p(mdata["rna"])
sc.pp.highly_variable_genes(mdata["rna"], subset=True)

mu.prot.pp.clr(mdata["adt"])
# Data exploration
sc.pp.pca(mdata["rna"])
# We calculate neighbors with the cosine distance similarly to the original Seurat implementation
sc.pp.neighbors(mdata["rna"], metric="cosine")
sc.tl.umap(mdata["rna"])
sc.pl.umap(mdata["rna"], color=["replicate", "Phase", "perturbation"])

# When glancing at the UMAPs we identify two clearly visible issues:
#     Many cells are separated by replicate ID. This is a common sign of a batch effect.
#     The cell cycle phase is a confounder in the embedding.

# Calculating local perturbation signatures
# Following the recommendations of Papalexi et al., we recommend setting from the 
# range of 20 < k 30. A k that is too small or too large is unlikely to remove any
# technical variation from the dataset
ms = pt.tl.Mixscape()

ms.perturbation_signature(
    mdata["rna"],
    pert_key="perturbation",
    control="NT",
    split_by="replicate",
    n_neighbors=20,
)

# We create a copy of the object to recalculate the PCA.
# Alternatively we could replace the X of the RNA part of our MuData object with the `X_pert` layer.
adata_pert = mdata["rna"].copy()
adata_pert.X = adata_pert.layers["X_pert"]
sc.pp.pca(adata_pert)
sc.pp.neighbors(adata_pert, metric="cosine")
sc.tl.umap(adata_pert)
sc.pl.umap(adata_pert, color=["replicate", "Phase", "perturbation"])

# Identifying cells with no detectable perturbation
ms.mixscape(adata=mdata["rna"], control="NT", labels="gene_target", layer="X_pert")
ms.plot_barplot(mdata["rna"], guide_rna_column="guide_ID")
# Perturbance score of an example target gene
ms.plot_perturbscore(
    adata=mdata["rna"], labels="gene_target", target_gene="IFNGR2", color="orange"
)

sc.settings.set_figure_params(figsize=(10, 10))
ms.plot_violin(
    adata=mdata["rna"],
    target_gene_idents=["NT", "IFNGR2 NP", "IFNGR2 KO"],
    groupby="mixscape_class",
)

ms.plot_heatmap(
    adata=mdata["rna"],
    labels="gene_target",
    target_gene="IFNGR2",
    layer="X_pert",
    control="NT",
)

mdata["adt"].obs["mixscape_class_global"] = mdata["rna"].obs["mixscape_class_global"]
ms.plot_violin(
    adata=mdata["adt"],
    target_gene_idents=["NT", "JAK2", "STAT1", "IFNGR1", "IFNGR2", "IRF1"],
    keys="PDL1",
    groupby="gene_target",
    hue="mixscape_class_global",
)

ms.lda(adata=mdata["rna"], control="NT", labels="gene_target", layer="X_pert")
ms.plot_lda(adata=mdata["rna"], control="NT")


