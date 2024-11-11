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