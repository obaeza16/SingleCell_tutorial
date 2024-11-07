import anndata
import numpy
import scanpy
import mudata
import tempfile
import os

os.environ['R_HOME'] = '/home/oscar/miniconda3/envs/interoperability/lib/R'

import rpy2.robjects
import anndata2ri
from scipy.sparse import csr_matrix

anndata2ri.activate()
%load_ext rpy2.ipython