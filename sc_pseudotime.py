from pathlib import Path
import scanpy as sc
import matplotlib

DATA_DIR = Path("./")
DATA_DIR.mkdir(parents=True, exist_ok=True)
FILE_NAME = DATA_DIR/"bone_marrow.h5ad"
# We download the adata object
adata = sc.read(
    filename=FILE_NAME,
    backup_url="https://figshare.com/ndownloader/files/35826944",
)
adata

# To construct pseudotimes, the data must be preprocessed. Here, we filter out 
# genes expressed in only a few number of cells (here, at least 20). Notably, 
# the construction of the pseudotime later on is robust to the exact choice of 
# the threshold. Following to this first gene filtering, the cell size is 
# normalized, and counts log1p transformed to reduce the effect of outliers. 
# As usual, we also identify and annotate highly variable genes. Finally, a 
# nearest neighbor graph is constructed based on which we will define the 
# pseudotime. The number of principle components is chosen based on the 
# explained variance.
sc.pp.filter_genes(adata, min_counts=20)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)

sc.tl.pca(adata)
sc.pp.neighbors(adata, n_pcs=10)

sc.pl.scatter(adata, basis="tsne", color="clusters")
sc.tl.diffmap(adata)

# Setting root cell as described above
root_ixs = adata.obsm["X_diffmap"][:, 3].argmin()
sc.pl.scatter(
    adata,
    basis="diffmap",
    color=["clusters"],
    components=[2, 3],
)

adata.uns["iroot"] = root_ixs
sc.tl.dpt(adata)

sc.pl.scatter(
    adata,
    basis="tsne",
    color=["dpt_pseudotime", "palantir_pseudotime"],
    color_map="gnuplot2",
)

sc.pl.violin(
    adata,
    keys=["dpt_pseudotime", "palantir_pseudotime"],
    groupby="clusters",
    rotation=45,
    order=[
        "HSC_1",
        "HSC_2",
        "Precursors",
        "Ery_1",
        "Ery_2",
        "Mono_1",
        "Mono_2",
        "CLP",
        "DCs",
        "Mega",
    ],
)