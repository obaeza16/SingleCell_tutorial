# Compositional analysis can be done on the level of cell identity clusters in
# the form of known cell types or cell states corresponding to, for example, 
# cells recently affected by perturbations.
# This chapter will introduce both approaches and apply them to the Haber
# dataset[Haber et al., 2017]. This dataset contains 53,193 individual 
# epithelial cells from the small intestine and organoids of mice.

# we load the dataset

import warnings

import pandas as pd

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import scanpy as sc
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import altair as alt
import pertpy as pt

adata = pt.dt.haber_2017_regions()
adata
adata.obs

# We will use scCODA
# As a first step, we instantiate a scCODA model.

# Then, we use load function to prepare a MuData object for subsequent processing, 
# and it creates a compositional analysis dataset from the input adata. And we 
# specify the cell_type_identifier as cell_label, sample_identifier as batch, 
# and covariate_obs as condition in our case.

sccoda_model = pt.tl.Sccoda()
sccoda_data = sccoda_model.load(
    adata,
    type="cell_level",
    generate_sample_level=True,
    cell_type_identifier="cell_label",
    sample_identifier="batch",
    covariate_obs=["condition"],
)
sccoda_data

# To get an overview of the cell type distributions across conditions we can use 
# scCODA’s boxplots

sccoda_model.plot_boxplots(
    sccoda_data,
    modality_key="coda",
    feature_name="condition",
    figsize=(12, 5),
    add_dots=True,
    palette= "Paired",
)
plt.show()

# stacked barplot as provided by scCODA
sccoda_model.plot_stacked_barplot(
    sccoda_data, modality_key="coda", feature_name="condition", figsize=(4, 2)
)
plt.show()

# In our case we specify the condition as the only covariate. 
# If we wanted to model multiple covariates at once, simply adding them in the
# formula (i.e. formula = "covariate_1 + covariate_2")
# scCODA can either automatically select an appropriate cell type as reference,
# which is a cell type that has nearly constant relative abundance over all 
# samples, or be run with a user specified reference cell type.
# An alternative to setting a reference cell type manually is to set the 
# reference_cell_type to "automatic"

sccoda_data = sccoda_model.prepare(
    sccoda_data,
    modality_key="coda",
    formula="condition",
    reference_cell_type="Endocrine",
)
sccoda_model.run_nuts(sccoda_data, modality_key="coda", rng_key=1234)
sccoda_data["coda"].varm["effect_df_condition[T.Salmonella]"]

# The acceptance rate describes the fraction of proposed samples that are accepted
# after the initial burn-in phase, and can be an ad-hoc indicator for a bad 
# optimization run. In the case of scCODA, the desired acceptance rate is between 
# 0.4 and 0.9. Acceptance rates that are way higher or too low indicate issues with
# the sampling process.

sccoda_data

# The desired FDR level can be easily set after inference via sim_results.set_fdr().
# Per default, the value is 0.05. Since, depending on the dataset, the FDR can have 
# a major influence on the result, we recommend to try out different FDRs up to 0.2 
# to get the most prominent effects.

# In our case, we use less strict FDR of 0.2.
sccoda_model.set_fdr(sccoda_data, 0.2)

# The fold-changes describe whether the cell type is more or less present. Hence, 
# we will plot them alongside the binary classification below.
sccoda_model.credible_effects(sccoda_data, modality_key="coda")

# To plot the fold changes together with the binary classification, we can easily
# use effects_bar_plot function.
sccoda_model.plot_effects_barplot(sccoda_data, "coda", "condition")
plt.show()