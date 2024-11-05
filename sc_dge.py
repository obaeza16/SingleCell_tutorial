# Here, we focus on more advanced use-cases of differential gene expression testing
# on more complex experimental designs which involve one or more conditions such 
# as diseases, genetic knockouts or drugs
# This statistical test can be applied to arbitrary groups, but in the case of 
# single-cell RNA-Seq is commonly applied on the cell type level.


# A differential gene expression test usually returns the log2 fold-change and 
# the adjusted p-value per compared genes per compared conditions. This list 
# can then be sorted by p-value and investigated in more detail.

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import pandas as pd
import numpy as np
import random
import sc_toolbox
import pertpy

import rpy2.rinterface_lib.callbacks
import anndata2ri
import logging

from rpy2.robjects import pandas2ri
from rpy2.robjects import r

sc.settings.verbosity = 0
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

pandas2ri.activate()
anndata2ri.activate()

%load_ext rpy2.ipython
%%R
library(edgeR)
library(MAST)

# We will use the Kang dataset, which is a 10x droplet-based scRNA-seq peripheral
# blood mononuclear cell data from 8 Lupus patients before and after 6h-treatment 
# with INF-β (16 samples in total)
adata = pertpy.data.kang_2018()
adata

adata.obs[:5]
# We need to work with the raw counts; we will put them into the counts layer of
# our AnnData object
np.max(adata.X)
adata.layers["counts"] = adata.X.copy()
# We have 8 control and 8 disease patients
print(len(adata[adata.obs["label"] == "ctrl"].obs["replicate"].cat.categories))
print(len(adata[adata.obs["label"] == "stim"].obs["replicate"].cat.categories))

# We filter cells with less than 200 detected genes and genes found in less than
# 3 cells
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata

# We will use edgeR. Since edgeR was introduced as a method for DE analysis for
# bulk data, we first need to create pseudobulk samples from our single-cell dataset

# First, let’s prepare the data.
adata.obs["sample"] = [
    f"{rep}_{l}" for rep, l in zip(adata.obs["replicate"], adata.obs["label"])
]
# Prepare data to avoid conversion issues
adata.obs["cell_type"] = [ct.replace(" ", "_") for ct in adata.obs["cell_type"]]
adata.obs["cell_type"] = [ct.replace("+", "") for ct in adata.obs["cell_type"]]
# We need to set categorical metadata to be indeed categorical to create pseudobulks.
adata.obs["replicate"] = adata.obs["replicate"].astype("category")
adata.obs["label"] = adata.obs["label"].astype("category")
adata.obs["sample"] = adata.obs["sample"].astype("category")
adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

# aggregate_and_filter is a function that creates an AnnData object with one pseudo-replicate
# for each donor for a specified subpopulation from the original single-cell AnnData object.
# Here, we also filter out donors that have fewer than 30 cells for the specified population.
# By changing the replicates_per_patient parameter, several (n) pseudo-replicates can be created
# for each sample; cells are then split into n subsets of roughly equal sizes.

NUM_OF_CELL_PER_DONOR = 30


def aggregate_and_filter(
    adata,
    cell_identity,
    donor_key="sample",
    condition_key="label",
    cell_identity_key="cell_type",
    obs_to_keep=[],  # which additional metadata to keep, e.g. gender, age, etc.
    replicates_per_patient=1,
):
    # subset adata to the given cell identity
    adata_cell_pop = adata[adata.obs[cell_identity_key] == cell_identity].copy()
    # check which donors to keep according to the number of cells specified with NUM_OF_CELL_PER_DONOR
    size_by_donor = adata_cell_pop.obs.groupby([donor_key]).size()
    donors_to_drop = [
        donor
        for donor in size_by_donor.index
        if size_by_donor[donor] <= NUM_OF_CELL_PER_DONOR
    ]
    if len(donors_to_drop) > 0:
        print("Dropping the following samples:")
        print(donors_to_drop)
    df = pd.DataFrame(columns=[*adata_cell_pop.var_names, *obs_to_keep])

    adata_cell_pop.obs[donor_key] = adata_cell_pop.obs[donor_key].astype("category")
    for i, donor in enumerate(donors := adata_cell_pop.obs[donor_key].cat.categories):
        print(f"\tProcessing donor {i+1} out of {len(donors)}...", end="\r")
        if donor not in donors_to_drop:
            adata_donor = adata_cell_pop[adata_cell_pop.obs[donor_key] == donor]
            # create replicates for each donor
            indices = list(adata_donor.obs_names)
            random.shuffle(indices)
            indices = np.array_split(np.array(indices), replicates_per_patient)
            for i, rep_idx in enumerate(indices):
                adata_replicate = adata_donor[rep_idx]
                # specify how to aggregate: sum gene expression for each gene for each donor and also keep the condition information
                agg_dict = {gene: "sum" for gene in adata_replicate.var_names}
                for obs in obs_to_keep:
                    agg_dict[obs] = "first"
                # create a df with all genes, donor and condition info
                df_donor = pd.DataFrame(adata_replicate.X.A)
                df_donor.index = adata_replicate.obs_names
                df_donor.columns = adata_replicate.var_names
                df_donor = df_donor.join(adata_replicate.obs[obs_to_keep])
                # aggregate
                df_donor = df_donor.groupby(donor_key).agg(agg_dict)
                df_donor[donor_key] = donor
                df.loc[f"donor_{donor}_{i}"] = df_donor.loc[donor]
    print("\n")
    # create AnnData object from the df
    adata_cell_pop = sc.AnnData(
        df[adata_cell_pop.var_names], obs=df.drop(columns=adata_cell_pop.var_names)
    )
    return adata_cell_pop

# fit_model takes a SingleCellExperiment object as input, creates the design 
# matrix and outputs the fitted GLM. We also output the edgeR object of class 
# DGEList to do some exploratory data analysis (EDA).

# %%R
# fit_model <- function(adata_){
#     # create an edgeR object with counts and grouping factor
#     y <- DGEList(assay(adata_, "X"), group = colData(adata_)$label)
#     # filter out genes with low counts
#     print("Dimensions before subsetting:")
#     print(dim(y))
#     print("")
#     keep <- filterByExpr(y)
#     y <- y[keep, , keep.lib.sizes=FALSE]
#     print("Dimensions after subsetting:")
#     print(dim(y))
#     print("")
#     # normalize
#     y <- calcNormFactors(y)
#     # create a vector that is a concatenation of condition and cell type that we will later use with contrasts
#     group <- paste0(colData(adata_)$label, ".", colData(adata_)$cell_type)
#     replicate <- colData(adata_)$replicate
#     # create a design matrix: here we have multiple donors so also consider that in the design matrix
#     design <- model.matrix(~ 0 + group + replicate)
#     # estimate dispersion
#     y <- estimateDisp(y, design = design)
#     # fit the model
#     fit <- glmQLFit(y, design)
#     return(list("fit"=fit, "design"=design, "y"=y))
# }

# We have created all the functions we need, so we can proceed to create the pseudobulks
obs_to_keep = ["label", "cell_type", "replicate", "sample"]
adata.X = adata.layers["counts"].copy()

# Next, we create the AnnData object with pseudobulks.
# process first cell type separately...
cell_type = adata.obs["cell_type"].cat.categories[0]
print(
    f'Processing {cell_type} (1 out of {len(adata.obs["cell_type"].cat.categories)})...'
)
adata_pb = aggregate_and_filter(adata, cell_type, obs_to_keep=obs_to_keep)
for i, cell_type in enumerate(adata.obs["cell_type"].cat.categories[1:]):
    print(
        f'Processing {cell_type} ({i+2} out of {len(adata.obs["cell_type"].cat.categories)})...'
    )
    adata_cell_type = aggregate_and_filter(adata, cell_type, obs_to_keep=obs_to_keep)
    adata_pb = adata_pb.concatenate(adata_cell_type)
    
# We do some EDA on the pseudo-replicates; We save the raw counts in the 'counts' 
# layer, then normalize the counts and calculate the PCA coordinates for the 
# normalized pseudobulk counts.
adata_pb.layers['counts'] = adata_pb.X.copy()
sc.pp.normalize_total(adata_pb, target_sum=1e6)
sc.pp.log1p(adata_pb)
sc.pp.pca(adata_pb)

adata_pb.obs["lib_size"] = np.sum(adata_pb.layers["counts"], axis=1)
adata_pb.obs["log_lib_size"] = np.log(adata_pb.obs["lib_size"])

sc.pl.pca(adata_pb, color=adata_pb.obs, ncols=1, size=300)

adata_pb.X = adata_pb.layers['counts'].copy()
# We run the pipeline on CD14+ Monocytes subset of the data
adata_mono = adata_pb[adata_pb.obs["cell_type"] == "CD14_Monocytes"]
adata_mono
# Clean the sample names
adata_mono.obs_names = [
    name.split("_")[2] + "_" + name.split("_")[3] for name in adata_mono.obs_names
]

# %%time
# %%R -i adata_mono
# outs <-fit_model(adata_mono)
# %%R
# fit <- outs$fit
# y <- outs$y
# plotMDS(y, col=ifelse(y$samples$group == "stim", "red", "blue"))
# plotBCV(y)
# colnames(y$design)
# myContrast <- makeContrasts('groupstim.CD14_Monocytes-groupctrl.CD14_Monocytes', levels = y$design)
# qlf <- glmQLFTest(fit, contrast=myContrast)
# get all of the DE genes and calculate Benjamini-Hochberg adjusted FDR
# tt <- topTags(qlf, n = Inf)
# tt <- tt$table

tt.shape
tt[:5]

# %%R
# tr <- glmTreat(fit, contrast=myContrast, lfc=1.5)
# print(head(topTags(tr)))
# plotSmear(qlf, de.tags = rownames(tt)[which(tt$FDR<0.01)])


# MAST takes normalized counts as input, so we first take the ‘counts’ layer and
# then perform the normalization step.
adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.log1p(adata)

# Function to eliminate frictions between R and python
def prep_anndata(adata_):
    def fix_dtypes(adata_):
        df = pd.DataFrame(adata_.X.A, index=adata_.obs_names, columns=adata_.var_names)
        df = df.join(adata_.obs)
        return sc.AnnData(df[adata_.var_names], obs=df.drop(columns=adata_.var_names))

    adata_ = fix_dtypes(adata_)
    sc.pp.filter_genes(adata_, min_cells=3)
    return adata_

# we only show the MAST-RE pipeline for one cell type, namely CD14 Monocytes,
# to shorten the runtime.

adata_mono = adata[adata.obs["cell_type"] == "CD14_Monocytes"].copy()
adata_mono

sc.pp.filter_genes(adata_mono, min_cells=3)
adata_mono
# Next we filter both objects as mentioned above.
adata_mono = prep_anndata(adata_mono)
adata_mono

adata_mono.obs["cell_type"] = [
    ct.replace(" ", "_") for ct in adata_mono.obs["cell_type"]
]
adata_mono.obs["cell_type"] = [
    ct.replace("+", "") for ct in adata_mono.obs["cell_type"]
]

# %%R
# find_de_MAST_RE <- function(adata_){
#     # create a MAST object
#     sca <- SceToSingleCellAssay(adata_, class = "SingleCellAssay")
#     print("Dimensions before subsetting:")
#     print(dim(sca))
#     print("")
#     # keep genes that are expressed in more than 10% of all cells
#     sca <- sca[freq(sca)>0.1,]
#     print("Dimensions after subsetting:")
#     print(dim(sca))
#     print("")
#     # add a column to the data which contains scaled number of genes that are expressed in each cell
#     cdr2 <- colSums(assay(sca)>0)
#     colData(sca)$ngeneson <- scale(cdr2)
#     # store the columns that we are interested in as factors
#     label <- factor(colData(sca)$label)
#     # set the reference level
#     label <- relevel(label,"ctrl")
#     colData(sca)$label <- label
#     celltype <- factor(colData(sca)$cell_type)
#     colData(sca)$celltype <- celltype
#     # same for donors (which we need to model random effects)
#     replicate <- factor(colData(sca)$replicate)
#     colData(sca)$replicate <- replicate
#     # create a group per condition-celltype combination
#     colData(sca)$group <- paste0(colData(adata_)$label, ".", colData(adata_)$cell_type)
#     colData(sca)$group <- factor(colData(sca)$group)
#     # define and fit the model
#     zlmCond <- zlm(formula = ~ngeneson + group + (1 | replicate), 
#                    sca=sca, 
#                    method='glmer', 
#                    ebayes=F, 
#                    strictConvergence=F,
#                    fitArgsD=list(nAGQ = 0)) # to speed up calculations
    
#     # perform likelihood-ratio test for the condition that we are interested in    
#     summaryCond <- summary(zlmCond, doLRT='groupstim.CD14_Monocytes')
#     # get the table with log-fold changes and p-values
#     summaryDt <- summaryCond$datatable
#     result <- merge(summaryDt[contrast=='groupstim.CD14_Monocytes' & component=='H',.(primerid, `Pr(>Chisq)`)], # p-values
#                      summaryDt[contrast=='groupstim.CD14_Monocytes' & component=='logFC', .(primerid, coef)],
#                      by='primerid') # logFC coefficients
#     # MAST uses natural logarithm so we convert the coefficients to log2 base to be comparable to edgeR
#     result[,coef:=result[,coef]/log(2)]
#     # do multiple testing correction
#     result[,FDR:=p.adjust(`Pr(>Chisq)`, 'fdr')]
#     result = result[result$FDR<0.01,, drop=F]

#     result <- stats::na.omit(as.data.frame(result))
#     return(result)
# }

# %%time
# %%R -i adata_mono -o res
# res <-find_de_MAST_RE(adata_mono)
# Lets take a look at the results
# res[:5]

res["gene_symbol"] = res["primerid"]
res["cell_type"] = "CD14_Monocytes"
sc_toolbox.tools.de_res_to_anndata(
    adata,
    res,
    groupby="cell_type",
    score_col="coef",
    pval_col="Pr(>Chisq)",
    pval_adj_col="FDR",
    lfc_col="coef",
    key_added="MAST_CD14_Monocytes",
)

adata_copy = adata.copy()

# We normalize the data before plotting the heatmaps to see the differences in
# expression between two conditions better.
adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.log1p(adata)
# Next, we define a helper plotting function for the heatmaps.
def plot_heatmap(adata, group_key, group_name="cell_type", groupby="label"):
    cell_type = "_".join(group_key.split("_")[1:])
    res = sc.get.rank_genes_groups_df(adata, group=cell_type, key=group_key)
    res.index = res["names"].values
    res = res[
        (res["pvals_adj"] < FDR) & (abs(res["logfoldchanges"]) > LOG_FOLD_CHANGE)
    ].sort_values(by=["logfoldchanges"])
    print(f"Plotting {len(res)} genes...")
    markers = list(res.index)
    sc.pl.heatmap(
        adata[adata.obs[group_name] == cell_type].copy(),
        markers,
        groupby=groupby,
        swap_axes=True,
    )
    
plot_heatmap(adata, "edgeR_CD14_Monocytes")
plot_heatmap(adata, "MAST_CD14_Monocytes")

# MAST identified 436 DEG with our given cut-offs for adjusted p-values and
# logfold change, while edgeR identified 303 genes.
# Next, we define the helper plotting function for the volcano plots.

FDR = 0.01
LOG_FOLD_CHANGE = 1.5


def volcano_plot(adata, group_key, group_name="cell_type", groupby="label", title=None):
    cell_type = "_".join(group_key.split("_")[1:])
    result = sc.get.rank_genes_groups_df(adata, group=cell_type, key=group_key).copy()
    result["-logQ"] = -np.log(result["pvals"].astype("float"))
    lowqval_de = result.loc[abs(result["logfoldchanges"]) > LOG_FOLD_CHANGE]
    other_de = result.loc[abs(result["logfoldchanges"]) <= LOG_FOLD_CHANGE]

    fig, ax = plt.subplots()
    sns.regplot(
        x=other_de["logfoldchanges"],
        y=other_de["-logQ"],
        fit_reg=False,
        scatter_kws={"s": 6},
    )
    sns.regplot(
        x=lowqval_de["logfoldchanges"],
        y=lowqval_de["-logQ"],
        fit_reg=False,
        scatter_kws={"s": 6},
    )
    ax.set_xlabel("log2 FC")
    ax.set_ylabel("-log Q-value")

    if title is None:
        title = group_key.replace("_", " ")
    plt.title(title)
    plt.show()

volcano_plot(adata, "MAST_CD14_Monocytes")
volcano_plot(adata, "edgeR_CD14_Monocytes")

# From the heatmaps and especially from the volcano plots, one can see that edgeR
# identified more up-regulated than down-regulated genes (in stimulated vs control)
# in contrast to MAST which identified similar number of up- and down-regulated genes.