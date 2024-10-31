# Taken from https://www.sc-best-practices.org/cellular_structure/annotation.html

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numba
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)

import scanpy as sc
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import celltypist
from celltypist import models
import scarches as sca
import urllib.request

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

sc.set_figure_params(figsize=(5, 5))

# Download of the data, weigths 2G
adata = sc.read(
    filename="s4d8_clustered.h5ad",
    backup_url="https://figshare.com/ndownloader/files/41436666",
)

# list a set of markers for cell types in the bone marrow here that is based on 
# literature: previous papers that study specific cell types and subtypes and 
# report marker genes for those cell types. 
marker_genes = {
    "CD14+ Mono": ["FCN1", "CD14"],
    "CD16+ Mono": ["TCF7L2", "FCGR3A", "LYN"],
    "ID2-hi myeloid prog": [
        "CD14",
        "ID2",
        "VCAN",
        "S100A9",
        "CLEC12A",
        "KLF4",
        "PLAUR",
    ],
    "cDC1": ["CLEC9A", "CADM1"],
    "cDC2": [
        "CST3",
        "COTL1",
        "LYZ",
        "DMXL2",
        "CLEC10A",
        "FCER1A",
    ],  # Note: DMXL2 should be negative
    "Normoblast": ["SLC4A1", "SLC25A37", "HBB", "HBA2", "HBA1", "TFRC"],
    "Erythroblast": ["MKI67", "HBA1", "HBB"],
    "Proerythroblast": [
        "CDK6",
        "SYNGR1",
        "HBM",
        "GYPA",
    ],  # Note HBM and GYPA are negative markers
    "NK": ["GNLY", "NKG7", "CD247", "GRIK4", "FCER1G", "TYROBP", "KLRG1", "FCGR3A"],
    "ILC": ["ID2", "PLCG2", "GNLY", "SYNE1"],
    "Lymph prog": [
        "VPREB1",
        "MME",
        "EBF1",
        "SSBP2",
        "BACH2",
        "CD79B",
        "IGHM",
        "PAX5",
        "PRKCE",
        "DNTT",
        "IGLL1",
    ],
    "Naive CD20+ B": ["MS4A1", "IL4R", "IGHD", "FCRL1", "IGHM"],
    "B1 B": [
        "MS4A1",
        "SSPN",
        "ITGB1",
        "EPHA4",
        "COL4A4",
        "PRDM1",
        "IRF4",
        "CD38",
        "XBP1",
        "PAX5",
        "BCL11A",
        "BLK",
        "IGHD",
        "IGHM",
        "ZNF215",
    ],  # Note IGHD and IGHM are negative markers
    "Transitional B": ["MME", "CD38", "CD24", "ACSM3", "MSI2"],
    "Plasma cells": ["MZB1", "HSP90B1", "FNDC3B", "PRDM1", "IGKC", "JCHAIN"],
    "Plasmablast": ["XBP1", "RF4", "PRDM1", "PAX5"],  # Note PAX5 is a negative marker
    "CD4+ T activated": ["CD4", "IL7R", "TRBC2", "ITGB1"],
    "CD4+ T naive": ["CD4", "IL7R", "TRBC2", "CCR7"],
    "CD8+ T": ["CD8A", "CD8B", "GZMK", "GZMA", "CCL5", "GZMB", "GZMH", "GZMA"],
    "T activation": ["CD69", "CD38"],  # CD69 much better marker!
    "T naive": ["LEF1", "CCR7", "TCF7"],
    "pDC": ["GZMB", "IL3RA", "COBLL1", "TCF4"],
    "G/M prog": ["MPO", "BCL2", "KCNQ5", "CSF3R"],
    "HSC": ["NRIP1", "MECOM", "PROM1", "NKAIN2", "CD34"],
    "MK/E prog": [
        "ZNF385D",
        "ITGA2B",
        "RYR3",
        "PLCB1",
    ],  # Note PLCB1 is a negative marker
}

# Subset to only the markers that were detected in our data.
marker_genes_in_data = dict()
for ct, markers in marker_genes.items():
    markers_found = list()
    for marker in markers:
        if marker in adata.var.index:
            markers_found.append(marker)
    marker_genes_in_data[ct] = markers_found

# To start we store our raw counts in .layers['counts'], so that we will still have access 
# to them later if needed. We then set our adata.X to the scran-normalized, log-transformed counts.
adata.layers["counts"] = adata.X
adata.X = adata.layers["scran_normalization"]

# We furthermore set our adata.var.highly_variable to the highly deviant genes.
adata.var["highly_variable"] = adata.var["highly_deviant"]

# Now perform PCA. We use the highly deviant genes (set as “highly variable” above) 
# to reduce noise and strengthen signal in our data and set number of components to the default n=50
sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# Let’s list the B cell subtypes we want to show the markers for:
B_plasma_cts = [
    "Naive CD20+ B",
    "B1 B",
    "Transitional B",
    "Plasma cells",
    "Plasmablast",
]

for ct in B_plasma_cts:
    print(f"{ct.upper()}:")  # print cell subtype name
    sc.pl.umap(
        adata,
        color=marker_genes_in_data[ct],
        vmin=0,
        vmax="p99",  # set vmax to the 99th percentile of the gene count instead of the maximum, to prevent outliers from making expression in other cells invisible. Note that this can cause problems for extremely lowly expressed genes.
        sort_order=False,  # do not plot highest expression on top, to not get a biased view of the mean expression among cells
        frameon=False,
        cmap="Blues",  # or choose another color map e.g. from here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    )
    print("\n\n\n")  # print white space for legibility
    
# We will now cluster our data, using the Leiden algorithm
sc.tl.leiden(adata, resolution=1, key_added="leiden_1")
sc.pl.umap(adata, color="leiden_1")

# We will try a higher resolution:
sc.tl.leiden(adata, resolution=2, key_added="leiden_2")
sc.pl.umap(adata, color="leiden_2")
# With numbers on the UMAP:
sc.pl.umap(adata, color="leiden_2", legend_loc="on data")

# We can see that clusters 4 and 6 are the clusters consistently expressing Naive CD20+ B cell markers.
# We will visualize using a dotplot:

B_plasma_markers = {
    ct: [m for m in ct_markers if m in adata.var.index]
    for ct, ct_markers in marker_genes.items()
    if ct in B_plasma_cts
}

sc.pl.dotplot(
    adata,
    groupby="leiden_2",
    var_names=B_plasma_markers,
    standard_scale="var",  # standard scale: normalize each gene to range from 0 to 1
)

# We will now start annotating the clusters:
cl_annotation = {
    "4": "Naive CD20+ B",
    "6": "Naive CD20+ B",
    "8": "Transitional B",
    "18": "B1 B",  # note that IGHD and IGHM are negative markers, in this case more lowly expressed than in the other B cell clusters
}
# And now we will visualize the annotations
adata.obs["manual_celltype_annotation"] = adata.obs.leiden_2.map(cl_annotation)
sc.pl.umap(adata, color=["manual_celltype_annotation"])

# Let’s calculate the differentially expressed genes for every cluster, compared to the rest of the cells in our adata:
sc.tl.rank_genes_groups(
    adata, groupby="leiden_2", method="wilcoxon", key_added="dea_leiden_2"
)
sc.pl.rank_genes_groups_dotplot(
    adata, groupby="leiden_2", standard_scale="var", n_genes=5, key="dea_leiden_2"
)

# We can filter the differentially expressed genes to select for more cluster-specific differentially expressed genes:
sc.tl.filter_rank_genes_groups(
    adata,
    min_in_group_fraction=0.2,
    max_out_group_fraction=0.2,
    key="dea_leiden_2",
    key_added="dea_leiden_2_filtered",
)
sc.pl.rank_genes_groups_dotplot(
    adata,
    groupby="leiden_2",
    standard_scale="var",
    n_genes=5,
    key="dea_leiden_2_filtered",
)

sc.pl.umap(
    adata,
    color=["CDK6", "ETV6", "NKAIN2", "GNAQ", "leiden_2"],
    vmax="p99",
    legend_loc="on data",
    frameon=False,
    cmap="Reds",
)
sc.pl.umap(
    adata,
    color=[
        "ZNF385D",
        "ITGA2B",
        "RYR3",
        "PLCB1",
    ],
    vmax="p99",
    legend_loc="on data",
    frameon=False,
    cmap="Reds",
)
# Manual Annotation is tricky
cl_annotation["12"] = "HSCs + MK/E prog (?)"
adata.obs["manual_celltype_annotation"] = adata.obs.leiden_2.map(cl_annotation)

## AUTOMATED ANNOTATION ##
# We will proceed with CellTypist and Clustifyr
# We need to prepare our data so that counts are normalized to 10,000 counts per cell, then log1p-transformed

adata_celltypist = adata.copy()  # make a copy of our adata
adata_celltypist.X = adata.layers["counts"]  # set adata.X to raw counts
sc.pp.normalize_per_cell(
    adata_celltypist, counts_per_cell_after=10**4
)  # normalize to 10,000 counts per cell
sc.pp.log1p(adata_celltypist)  # log-transform
# make .X dense instead of sparse, for compatibility with celltypist:
adata_celltypist.X = adata_celltypist.X.toarray()
# We now download the models for inmune cells:
models.download_models(
    force_update=True, model=["Immune_All_Low.pkl", "Immune_All_High.pkl"]
)

# Let’s try out both the Immune_All_Low and Immune_All_High models 
# (these annotate immune cell types finer annotation level (low) and coarser (high)):

model_low = models.Model.load(model="Immune_All_Low.pkl")
model_high = models.Model.load(model="Immune_All_High.pkl")

model_high.cell_types
model_low.cell_types

# Now we will run the models, first the coarse one:
predictions_high = celltypist.annotate(
    adata_celltypist, model=model_high, majority_voting=True
)
predictions_high_adata = predictions_high.to_adata()
# Now copy the results to the original AnnData object:
adata.obs["celltypist_cell_label_coarse"] = predictions_high_adata.obs.loc[
    adata.obs.index, "majority_voting"
]
adata.obs["celltypist_conf_score_coarse"] = predictions_high_adata.obs.loc[
    adata.obs.index, "conf_score"
]


# Now we do the same but for the fine model:
predictions_low = celltypist.annotate(
    adata_celltypist, model=model_low, majority_voting=True
)

predictions_low_adata = predictions_low.to_adata()
# Now copy the results to the original AnnData object:
adata.obs["celltypist_cell_label_fine"] = predictions_low_adata.obs.loc[
    adata.obs.index, "majority_voting"
]
adata.obs["celltypist_conf_score_fine"] = predictions_low_adata.obs.loc[
    adata.obs.index, "conf_score"
]

# Now we will plot both models:

sc.pl.umap(
    adata,
    color=["celltypist_cell_label_coarse", "celltypist_conf_score_coarse"],
    frameon=False,
    sort_order=False,
    wspace=1,
)

sc.pl.umap(
    adata,
    color=["celltypist_cell_label_fine", "celltypist_conf_score_fine"],
    frameon=False,
    sort_order=False,
    wspace=1,
)

# One way of getting a feeling for the quality of these annotations is by 
# looking if the observed cell type similarities correspond to our expectations:
sc.pl.dendrogram(adata, groupby="celltypist_cell_label_fine")

# We see that our data is not that well annotated by the automatic method

## MAPPING TO A REFERENCE ANNOTATION ##

# We will use scArches, that requires raw, non-normalized counts:
adata_to_map = adata.copy()
for layer in list(adata_to_map.layers.keys()):
    if layer != "counts":
        del adata_to_map.layers[layer]
adata_to_map.X = adata_to_map.layers["counts"]

# It is important that we use the same input features (i.e. genes) as were used
# for training our reference model and that we put those features in the same 
# order. The reference model’s feature information is stored together with the 
# model. Let’s load the feature table.

reference_model_features = pd.read_csv(
    "https://figshare.com/ndownloader/files/41436645", index_col=0
)

# We will therefore set our row names for both our adata and the reference model
# features to gene_ids. Importantly, we have to make sure to also store the gene 
# names for later use: these are much easier to understand than the gene ids.
adata_to_map.var["gene_names"] = adata_to_map.var.index
adata_to_map.var.set_index("gene_ids", inplace=True)

reference_model_features["gene_names"] = reference_model_features.index
reference_model_features.set_index("gene_ids", inplace=True)

print("Total number of genes needed for mapping:", reference_model_features.shape[0])

print(
    "Number of genes found in query dataset:",
    adata_to_map.var.index.isin(reference_model_features.index).sum(),
)

# We are missing a few genes. We will manually add those and set their counts to 
# 0, as it seems like these genes were not detected in our data.

missing_genes = [
    gene_id
    for gene_id in reference_model_features.index
    if gene_id not in adata_to_map.var.index
]

missing_gene_adata = sc.AnnData(
    X=csr_matrix(np.zeros(shape=(adata.n_obs, len(missing_genes))), dtype="float32"),
    obs=adata.obs.iloc[:, :1],
    var=reference_model_features.loc[missing_genes, :],
)
missing_gene_adata.layers["counts"] = missing_gene_adata.X

# We now concatenate our original data to our missing genes data, but first we will 
# remove the PCs matrix

if "PCs" in adata_to_map.varm.keys():
    del adata_to_map.varm["PCs"]

adata_to_map_augmented = sc.concat(
    [adata_to_map, missing_gene_adata],
    axis=1,
    join="outer",
    index_unique=None,
    merge="unique",
)
# Now subset to the genes used in the model and order correctly:
adata_to_map_augmented = adata_to_map_augmented[
    :, reference_model_features.index
].copy()

(adata_to_map_augmented.var.index == reference_model_features.index).all()

# We now set the genes to gene names back, to easier interpretation:
adata_to_map_augmented.var["gene_ids"] = adata_to_map_augmented.var.index
adata_to_map_augmented.var.set_index("gene_names", inplace=True)

# We now check for the batch variable, and to see if it's unique:
adata_to_map_augmented.obs.batch.unique()


# We now load the model and pass it the data we want to map:
# loading model.pt from figshare
if not os.path.exists("./reference_model"):
    os.mkdir("./reference_model")
elif not os.path.exists("./reference_model/model.pt"):
    urllib.request.urlretrieve(
        "https://figshare.com/ndownloader/files/41436648",
        filename="reference_model/model.pt",
    )

scarches_model = sca.models.SCVI.load_query_data(
    adata=adata_to_map_augmented,
    reference_model="./reference_model",
    freeze_dropout=True,
)

# We will update the reference model so we can embed our new data:
scarches_model.train(max_epochs=500, plan_kwargs=dict(weight_decay=0.0))

# With the model trained, we can now calculate the representation of our query
adata.obsm["X_scVI"] = scarches_model.get_latent_representation()

# Lets caluclate the UMAP now:
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata)
# Lets look at a few markers if their expression is localized in the UMAP:
sc.pl.umap(
    adata,
    color=["IGHD", "IGHM", "PRDM1"],
    vmin=0,
    vmax="p99",  # set vmax to the 99th percentile of the gene count instead of the maximum, to prevent outliers from making expression in other cells invisible. Note that this can cause problems for extremely lowly expressed genes.
    sort_order=False,  # do not plot highest expression on top, to not get a biased view of the mean expression among cells
    frameon=False,
    cmap="Reds",  # or choose another color map e.g. from here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
)

# We combine the inferred space of our query data with the existing reference embedding.
# Using this joint embedding, we will not only be able to e.g. visualize and cluster 
# the two together, but we can also do label transfer from the query to the reference.
# Let’s load the reference embedding: this is often made publicly available with existing atlases.

ref_emb = sc.read(
    filename="reference_embedding.h5ad",
    backup_url="https://figshare.com/ndownloader/files/41376264",
)

ref_emb.obs["reference_or_query"] = "reference"
# We will perform the label transfer now:
adata_emb = sc.AnnData(X=adata.obsm["X_scVI"], obs=adata.obs)
adata_emb.obs["reference_or_query"] = "query"
emb_ref_query = sc.concat(
    [ref_emb, adata_emb],
    axis=0,
    join="outer",
    index_unique=None,
    merge="unique",
)
 
# And we visualize the joint data with a UMAP:
sc.pp.neighbors(emb_ref_query)
sc.tl.umap(emb_ref_query)

sc.pl.umap(
    emb_ref_query,
    color=["reference_or_query"],
    sort_order=False,
    frameon=False,
)
# We can see a fairly well integrated join, if not we would see a total separation 
# between query and reference
# We will make the figure bigger

sc.set_figure_params(figsize=(8, 8))
sc.pl.umap(
    emb_ref_query,
    color=["cell_type"],
    sort_order=False,
    frameon=False,
    legend_loc="on data",
    legend_fontsize=10,
    na_color="black",
)

# We can guess the type of each of our cells by seeing at wich cell types from the
# reference are surrounded by

# Let’s perform the KNN-based label transfer.
# We setup the label transfer model:
knn_transformer = sca.utils.knn.weighted_knn_trainer(
    train_adata=ref_emb,
    train_adata_emb="X",  # location of our joint embedding
    n_neighbors=15,
)
# We perform the label transfer:
labels, uncert = sca.utils.knn.weighted_knn_transfer(
    query_adata=adata_emb,
    query_adata_emb="X",  # location of our embedding, query_adata.X in this case
    label_keys="cell_type",  # (start of) obs column name(s) for which to transfer labels
    knn_model=knn_transformer,
    ref_adata_obs=ref_emb.obs,
)
# And store the results in our adata object
adata_emb.obs["transf_cell_type"] = labels.loc[adata_emb.obs.index, "cell_type"]
adata_emb.obs["transf_cell_type_unc"] = uncert.loc[adata_emb.obs.index, "cell_type"]

# Let’s transfer the results to our query adata object which also has our UMAP and 
# gene counts, so that we can visualize all of those together.
adata.obs.loc[adata_emb.obs.index, "transf_cell_type"] = adata_emb.obs[
    "transf_cell_type"
]
adata.obs.loc[adata_emb.obs.index, "transf_cell_type_unc"] = adata_emb.obs[
    "transf_cell_type_unc"
]

# Now visualize the transferred labels in our data UMAP
sc.set_figure_params(figsize=(5, 5))
sc.pl.umap(adata, color="transf_cell_type", frameon=False)

# Based on the neighbors of each of our query cells we can not only guess the cell 
# type these cells belong to, but also generate a measure for certainty of that 
# label: if a cell has neighbors from several different cell types, our guess will 
# be highly uncertain. This is relevant to assess to what extent we can “trust” the 
# transferred labels! Let’s visualize the uncertainty scores:


sc.pl.umap(adata, color="transf_cell_type_unc", frameon=False)

# Let’s check for each cell type label how high the label transfer uncertainty 
# levels were. This gives us a first impression of which annotations are more 
# contentious/need more manual checks.
fig, ax = plt.subplots(figsize=(8, 3))
ct_order = (
    adata.obs.groupby("transf_cell_type")
    .agg({"transf_cell_type_unc": "median"})
    .sort_values(by="transf_cell_type_unc", ascending=False)
)
sns.boxplot(
    adata.obs,
    x="transf_cell_type",
    y="transf_cell_type_unc",
    color="grey",
    ax=ax,
    order=ct_order.index,
)
ax.tick_params(rotation=90, axis="x")
# we set cells with an uncertainty score above e.g. 0.2 to “unknown”:
adata.obs["transf_cell_type_certain"] = adata.obs.transf_cell_type.tolist()
adata.obs.loc[
    adata.obs.transf_cell_type_unc > 0.2, "transf_cell_type_certain"
] = "Unknown"

sc.pl.umap(adata, color="transf_cell_type_certain", frameon=False)
# We will color only the unknown cells:
sc.pl.umap(adata, color="transf_cell_type_certain", groups="Unknown")

# Now we will check to what extend does the automatic annotation match the
# manually annotated cells above