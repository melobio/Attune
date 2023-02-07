Finetuning Attune for analysing regulatory network on 10X multiome dataset:
step1. Run transformer model
- Run run_regulatory.ipynb

step2. Download pbmc.rds from https://doi.org/10.6084/m9.figshare.22032209.v1
- Save pbmc.rds in ./experiment_pbmc

step3. Analyse regulatory network from cross-attention weight
- Run ./experiment_pbmc/cross attention analysis.ipynb
