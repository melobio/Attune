# Attune
This repository contains code and tutorials for the following tasks: multimodal pretraining, cross-modal prediction, inferring regulatory network and potential analysis.
### Requirements:
* tensorflow-gpu 2.5.0
* torch 1.10.1
* h5py 3.1.0
* einops 0.4.1
* scanpy 1.7.2
* scipy 1.5.4
* scikit-learn 0.24.2

## Getting started
In the first step, you will need to download datasets to run each notebook and reproduce the result. 
The download links for datasets are shown in a folder named "data" in the following tasks directories.

## Multimodal pretraining (Fig2a,b and Supplementary Fig2)
- Notebook path: 10x_pretrain/run_integration.ipynb
- [Readme for the pretraining tutorial](10x_pretrain/README.txt)
- Description: 10X multiome dataset is used to pretrain Attune.
## Cross-modal prediction (Fig2f)
- Notebook path: 10x_prediction/run_prediction.ipynb
- [Readme for the prediction tutorial](10x_prediction/README.txt)
- Description: We use 10X multiome dataset to pretrain Attune and then finetune Attune via a prediction model to predict RNA counts.
## Inferring regulatory network (Fig3a,d,e,f and Supplementary Fig4b)
- Notebook path: 10x_regulatory/run_regulatory.ipynb
- [Readme for the regulatory tutorial](10x_regulatory/README.txt)
- Description: To infer regulatory interactions on 10X multiome dataset, we pretrain Attune and then finetune Attune via a transformer model. 
## Potential analysis (Fig3g and Fig4a,c,d,e and Supplementary Fig6a)
- Note book path: SHARE_TAC_potential/infer_potential.ipynb
- [Readme for the potential analysis tutorial](SHARE_TAC_potential/README.txt)
- Description: Attune reveals chromatin potential for hair follicle maturation at the gene level and illuminates the priming of lineage.
## Analysis of cell embeddings (Fig5c and Supplementary Fig8a,b)
- Note book path: greenleaf_pretrain/run_integration.ipynb
- [Readme for the analysis of cell embeddings tutorial](greenleaf_pretrain/README.txt)
- Description: Cell embeddings learned by Attune preserve biological signals in the high-dimensional space on fetal human cortex dataset.

