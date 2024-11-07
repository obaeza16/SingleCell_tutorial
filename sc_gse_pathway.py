# To determine the pathways enriched in a cell type-specific manner between two
# conditions, first a relevant collection of gene set signatures is selected, 
# where each gene set defines a biological process (e.g. epithelial to mesenchymal
# transition, metabolism etc) or pathway (e.g. MAPK signalling). For each gene set
# in the collection, DE genes present in the gene set are used to obtain a test 
# statistic that is then used to assess the enrichment of the gene set. Depending
# on the type of the enrichment test chosen, gene expression measurements may or 
# may not be used for the computation of the test statistic.

# Gene sets are a curated list of gene names (or gene ids) that are known to be 
# involved in a biological process through previous studies and/or experiments. 
# The Molecular Signatures Database (MSigDB) [Liberzon et al., 2011, Subramanian 
# et al., 2005] is the most comprehensive database consisting of 9 collections 
# of gene sets. 

# Case study: Pathway enrichment analysis and activity level scoring in human 
# PBMC single cells

from __future__ import annotations

import numpy as np
import pandas as pd

import scanpy as sc
import anndata as ad
import decoupler
import seaborn
import matplotlib as ptl
import seaborn.objects as so

import session_info

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))

# Filtering warnings from current version of matplotlib
import warnings

warnings.filterwarnings(
    "ignore", message=".*Parameters 'cmap' will be ignored.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message="Tight layout not applied.*", category=UserWarning
)

# Setting up R dependencies
# 
import os
os.environ['R_HOME'] = '/home/oscar/miniconda3/envs/sc_gse/lib/R'
# 
import rpy2
import rpy2.ipython
import rpy2.interactive
import anndata2ri
import rpy2.robjects as robjects
import random

%load_ext rpy2.ipython

anndata2ri.activate()
%%R
suppressPackageStartupMessages({
    library(SingleCellExperiment)
})

adata = sc.read(
    "kang_counts_25k.h5ad", backup_url="https://figshare.com/ndownloader/files/34464122"
)
adata

# Storing the counts for later use
adata.layers["counts"] = adata.X.copy()
# Renaming label to condition
adata.obs = adata.obs.rename({"label": "condition"}, axis=1)

# Normalizing
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Finding highly variable genes using count data
sc.pp.highly_variable_genes(
    adata, n_top_genes=4000, flavor="seurat_v3", subset=False, layer="counts"
)

# We will recompute UMAP and PCA embeddings
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

sc.pl.pca(adata,
    color=["condition", "cell_type"],
    frameon=False,
    ncols=2,
)

sc.pl.umap(
    adata,
    color=["condition", "cell_type"],
    frameon=False,
    ncols=2,
)

# We generally recommend determining the differentially expressed genes as outlined
# in the Differential gene expression chapter. For simplicity, here we run a 
# t-test using rank_genes_groups in scanpy to rank genes according to their test
# statistics for differential expression:

adata.obs["group"] = adata.obs.condition.astype("string") + "_" + adata.obs.cell_type
# find DE genes by t-test
sc.tl.rank_genes_groups(adata, "group", method="t-test", key_added="t-test")

# Let’s extract the ranks for genes differentially expressed in response to IFN 
# stimulation in the CD16 Monocyte (FCGR3A+ Monocytes) cluster. We use these ranks
# and the gene sets from REACTOME to find gene sets enriched in this cell population
# compared to all other populations using GSEA as implemented in decoupler.

celltype_condition = "stim_FCGR3A+ Monocytes"  # 'stimulated_B',  'stimulated_CD8 T', 'stimulated_CD14 Mono'
# extract scores
t_stats = (
    # Get dataframe of DE results for condition vs. rest
    sc.get.rank_genes_groups_df(adata, celltype_condition, key="t-test")
    # Subset to highly variable genes
    .set_index("names")
    .loc[adata.var["highly_variable"]]
    # Sort by absolute score
    .sort_values("scores", key=np.abs, ascending=False)
    # Format for decoupler
    [["scores"]]
    .rename_axis(["stim_FCGR3A+ Monocytes"], axis=1)
)
t_stats

# Downloading reactome pathways
from pathlib import Path

# download from bash:
# wget -O 'c2.cp.reactome.v7.5.1.symbols.gmt' https://figshare.com/ndownloader/files/35233771

def gmt_to_decoupler(pth: Path) -> pd.DataFrame:
    """
    Parse a gmt file to a decoupler pathway dataframe.
    """
    from itertools import chain, repeat

    pathways = {}

    with Path(pth).open("r") as f:
        for line in f:
            name, _, *genes = line.strip().split("\t")
            pathways[name] = genes

    return pd.DataFrame.from_records(
        chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
        columns=["geneset", "genesymbol"],
    )
    
reactome = gmt_to_decoupler("c2.cp.reactome.v7.5.1.symbols.gmt")

# For stability of this tutorial we are using a fixed version of the gene set collection.
# Retrieving via python
msigdb = decoupler.get_resource("MSigDB")

# Get reactome pathways
reactome = msigdb.query("collection == 'reactome_pathways'")
# Filter duplicates
reactome = reactome[~reactome.duplicated(("geneset", "genesymbol"))]
reactome

# Instead we will simply manually filter gene sets to have a minimum of 15 genes
# and a maximum of 500 genes.
# Filtering genesets to match behaviour of fgsea
geneset_size = reactome.groupby("geneset").size()
gsea_genesets = geneset_size.index[(geneset_size > 15) & (geneset_size < 500)]

scores, norm, pvals = decoupler.run_gsea(
    t_stats.T,
    reactome[reactome["geneset"].isin(gsea_genesets)],
    source="geneset",
    target="genesymbol",
)

gsea_results = (
    pd.concat({"score": scores.T, "norm": norm.T, "pval": pvals.T}, axis=1)
    .droplevel(level=1, axis=1)
    .sort_values("pval")
)

plot1=(
    so.Plot(
        data=(
            gsea_results.head(20).assign(
                **{"-log10(pval)": lambda x: -np.log10(x["pval"])}
            )
        ),
        x="-log10(pval)",
        y="source",
    ).add(so.Bar())
)

so.Plot.show(plot1)

gsea_results.head(10)

# Unlike the previous approach where we assessed gene set enrichment per cluster
# (or rather cell type), one can score the activity level of pathways and gene 
# sets in each individual cell, that is based on absolute gene expression in the
# cell, regardless of expression of genes in the other cells. This we can achieve
# by activity scoring tools such as AUCell.
decoupler.run_aucell(
    adata,
    reactome,
    source="geneset",
    target="genesymbol",
    use_raw=False,
)

ifn_pathways = [
    "REACTOME_INTERFERON_SIGNALING",
    "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING",
    "REACTOME_INTERFERON_GAMMA_SIGNALING",
]

adata.obs[ifn_pathways] = adata.obsm["aucell_estimate"][ifn_pathways]

sc.pl.umap(
    adata,
    color=["condition", "cell_type"] + ifn_pathways,
    frameon=False,
    ncols=2,
    wspace=0.3,
)


# limma-fry workflow that generalize to realistic data analysis routines, say,
# for single-cell case control studies. We first create pseudo-bulk replicates
# per cell type and condition (3 replicates per condition - cell type combination).
# We then find gene sets enriched in stimulated compared to control cells in a
# cell type. We also assess gene set enrichment between two stimulated cell 
# type populations to find differences in signalling pathways.

def subsampled_summation(
    adata: ad.AnnData,
    groupby: str | list[str],
    *,
    n_samples_per_group: int,
    n_cells: int,
    random_state: None | int | np.random.RandomState = None,
    layer: str = None,
) -> ad.AnnData:
    """
    Sum sample of X per condition.

    Drops conditions which don't have enough samples.

    Parameters
    ----------
    adata
        AnnData to sum expression of
    groupby
        Keys in obs to groupby
    n_samples_per_group
        Number of samples to take per group
    n_cells
        Number of cells to take per sample
    random_state
        Random state to use when sampling cells
    layer
        Which layer of adata to use

    Returns
    -------
    AnnData with same var as original, obs with columns from groupby, and X.
    """
    from scipy import sparse
    from sklearn.utils import check_random_state

    # Checks
    if isinstance(groupby, str):
        groupby = [groupby]
    random_state = check_random_state(random_state)

    indices = []
    labels = []

    grouped = adata.obs.groupby(groupby)
    for k, inds in grouped.indices.items():
        # Check size of group
        if len(inds) < (n_cells * n_samples_per_group):
            continue

        # Sample from group
        condition_inds = random_state.choice(
            inds, n_cells * n_samples_per_group, replace=False
        )
        for i, sample_condition_inds in enumerate(np.split(condition_inds, 3)):
            if isinstance(k, tuple):
                labels.append((*k, i))
            else:  # only grouping by one variable
                labels.append((k, i))
            indices.append(sample_condition_inds)

    # obs of output AnnData
    new_obs = pd.DataFrame.from_records(
        labels,
        columns=[*groupby, "sample"],
        index=["-".join(map(str, l)) for l in labels],
    )
    n_out = len(labels)

    # Make indicator matrix
    indptr = np.arange(0, (n_out + 1) * n_cells, n_cells)
    indicator = sparse.csr_matrix(
        (
            np.ones(n_out * n_cells, dtype=bool),
            np.concatenate(indices),
            indptr,
        ),
        shape=(len(labels), adata.n_obs),
    )

    return ad.AnnData(
        X=indicator @ sc.get._get_obs_rep(adata, layer=layer),
        obs=new_obs,
        var=adata.var.copy(),
    )


pb_data = subsampled_summation(
    adata, ["cell_type", "condition"], n_cells=75, n_samples_per_group=3, layer="counts"
)
pb_data
# Does PC1 captures a meaningful biological or technical fact?
pb_data.obs["lib_size"] = pb_data.X.sum(1)

# Let’s normalize this data and take a quick look at it. We won’t use a neighbor
# embedding here since the sample size is significantly reduced.

pb_data.layers["counts"] = pb_data.X.copy()
sc.pp.normalize_total(pb_data)
sc.pp.log1p(pb_data)
sc.pp.pca(pb_data)
sc.pl.pca(pb_data, color=["cell_type", "condition", "lib_size"], ncols=1, size=250)

# PC1 now captures difference between lymphoid (T, NK, B) and myeloid (Mono, DC)
# populations, while the second PC captures variation due to administration of 
# stimulus (i.e. difference between control and stimulated pseudo-replicates). 
# Ideally, the variation of interest has to be detectable in top few PCs of the 
# pseudo-bulk data.

# In this case, since we are indeed interested in stimulation effect per cell type,
# we proceed to gene set testing. We re-iterate that the purpose of plotting PCs 
# is to explore various axes of variability in the data and to spot unwanted 
# variabilities that can substantial influence the test results. Users may proceed
# with the rest of the analyses should they be satisfied with the the variations 
# in their data.

groups = pb_data.obs.condition.astype("string") + "_" + pb_data.obs.cell_type

%%R -i groups
group <-  as.factor(gsub(" |\\+","_", groups))
design <- model.matrix(~ 0 + group)
head(design)

%%R
colnames(design)

%%R 
kang_pbmc_con <- limma::makeContrasts(
    
    # the effect if stimulus in CD16 Monocyte cells
    groupstim_FCGR3A__Monocytes - groupctrl_FCGR3A__Monocytes,
    
    # the effect of stimulus in CD16 Monocytes compared to CD8 T Cells
    (groupstim_FCGR3A__Monocytes - groupctrl_FCGR3A__Monocytes) - (groupstim_CD8_T_cells - groupctrl_CD8_T_cells), 
    levels = design
)

log_norm_X = pb_data.to_df().T

%%R -i log_norm_X -i reactome
# Move pathway info from python to R
pathways = split(reactome$genesymbol, reactome$geneset)
# Map gene names to indices
idx = limma::ids2indices(pathways, rownames(log_norm_X))

# As done in the gsea method, let’s remove gene sets with less than 15 genes

%%R
keep_gs <- lapply(idx, FUN=function(x) length(x) >= 15)
idx <- idx[unlist(keep_gs)]

# fry test for Stimulated vs Control
%%R -o fry_results
fry_results <- limma::fry(log_norm_X, index = idx, design = design, contrast = kang_pbmc_con[,1])

fry_results.head()

(
    so.Plot(
        data=(
            fry_results.head(20)
            .assign(**{"-log10(FDR)": lambda x: -np.log10(x["FDR"])})
            .rename_axis(index="Pathway")
        ),
        x="-log10(FDR)",
        y="Pathway",
    ).add(so.Bar())
)

# fry test for the comparison between two stimulated cell types
%%R -o fry_results_negative_ctrl
fry_results_negative_ctrl <- limma::fry(log_norm_X, index = idx, design = design, contrast = kang_pbmc_con[,2])

(
    so.Plot(
        data=(
            fry_results_negative_ctrl.head(20)
            .assign(**{"-log10(FDR)": lambda x: -np.log10(x["FDR"])})
            .rename_axis(index="Pathway")
        ),
        x="-log10(FDR)",
        y="Pathway",
    ).add(so.Bar())
)

# Therefore, the results by fry might be of more interest biologically

# On the effect of filtering low-expression genes
# As mentioned before, Ideally, the variation of interest has to be detectable
# in top few PCs of the pseudo-bulk data. Let’s remove genes with low 
# expression in the data, apply CPM transformation and repeat the PCA plots
counts_df = pb_data.to_df(layer="counts").T

%%R -i counts_df
keep <- edgeR::filterByExpr(counts_df) # in real analysis, supply the desig matrix to the function to retain as more genes as possible
counts_df <- counts_df[keep,]
logCPM <- edgeR::cpm(counts_df, log=TRUE, prior.count = 2)

%%R -o logCPM
logCPM = data.frame(logCPM)

pb_data.uns["logCPM_FLE"] = logCPM.T  # FLE for filter low exprs

pb_data.obsm["logCPM_FLE_pca"] = sc.pp.pca(logCPM.T.to_numpy(), return_info=False)

sc.pl.embedding(pb_data, "logCPM_FLE_pca", color=pb_data.obs, ncols=1, size=250)
