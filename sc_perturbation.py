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
