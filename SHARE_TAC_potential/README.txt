Finetuning Attune for potential analysis on SHARE-seq TAC dataset:
step1. Data preparation
- Download the SHARE-seq TAC dataset from https://doi.org/10.6084/m9.figshare.22032437.v1
- Save adata_atac_SHARE_TAC.h5ad and adata_rna_SHARE_TAC.h5ad in ./data

step2. Run Attune pretraining model and fine-tune prediction model and transformer model
- Run infer_potential.ipynb

step3. Download share.rds from https://doi.org/10.6084/m9.figshare.22032371.v1
- Save share.rds in ./experiment_SHARE-seq

step4. Calcuate residual and visualize
- Run ./experiment_SHARE-seq/residual.ipynb

step5. Potential analysis
- Run ./experiment_SHARE-seq/regulation_shift.ipynb
