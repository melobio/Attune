Run Attune pretraining pipeline on greenleaf dataset:
step1. Data preparation
- Download the greenleaf dataset from https://doi.org/10.6084/m9.figshare.22032494.v1
- Then save greenleaf-final-ATAC_filter_rm_dc1r3_r1.h5ad and greenleaf-final-RNA_wox_filter_hvg_rm_dc1r3_r1.h5ad in ./data

step2. Run Attune pretraining model
- Run run_integration.ipynb

step3. Calculate cosine similarity between cell embeddings
- Run ./experiment_Cortex/cosine_similarity.ipynb

