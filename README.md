# Attune
This repository contains code and tutorials for the following tasks: multimodal pretraining, cross-modal prediction, inferring regulatory network and potential analysis.
## Getting started
In the first step, you will need to download datasets to run each notebook and reproduce the result. 
The download links for datasets are shown in a folder named "data" in the following tasks directories.

## Multimodal pretraining (Fig1 and )
- Notebook path: 10x_pretrain/run_integration.ipynb
- [Readme for the pretraining tutorial](10x_pretrain/README.txt)
- Description: 10X multiome dataset is used to pretrain Attune.
## Cross-modal prediction (Fig1)
- Notebook path: 10x_prediction/run_prediction.ipynb
- [Readme for the prediction tutorial](prediction/README.txt)
- Description: We use 10X multiome dataset to pretrain Attune and then finetune Attune via a prediction model to predict RNA counts.
## Inferring regulatory network (Fig2b,c,d)
- Notebook path: 10x_regulatory/run_regulatory.ipynb
- [Readme for the regulatory tutorial](10x_regulatory/README.txt)
- Description: To infer regulatory interactions on 10X multiome dataset, we pretrain Attune and then finetune Attune via a transformer model. 
## Potential analysis (Fig2e,f and Supplementary Fig7)
- Note book path: SHARE_TAC_potential/infer_potential.ipynb
- [Readme for the potential analysis tutorial](SHARE_TAC_potential/README.txt)
- Description: Attune reveals chromatin potential for hair follicle maturation at the gene level and illuminates the priming of lineage.
## Analysis of cell embeddings (Fig2j and Supplementary Fig8)
- Note book path: greenleaf_pretrain/run_integration.ipynb
- [Readme for the analysis of cell embeddings tutorial](greenleaf_pretrain/README.txt)
- Description: Cell embeddings learned by Attune preserve biological signals in the high-dimensional space on fetal human cortex dataset.

