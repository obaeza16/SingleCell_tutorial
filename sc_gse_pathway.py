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
import seaborn.objects as so

import session_info

