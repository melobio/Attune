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

## Multimodal pretraining (Fig2a,b and Supplementary Fig2,9)
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
## Potential analysis (Fig3g and Fig4a,c,d,e and Supplementary Fig6a,7)
- Note book path: SHARE_TAC_potential/infer_potential.ipynb
- [Readme for the potential analysis tutorial](SHARE_TAC_potential/README.txt)
- Description: Attune reveals chromatin potential for hair follicle maturation at the gene level and illuminates the priming of lineage.
## Analysis of cell embeddings (Fig5b,c and Supplementary Fig8a,b)
- Note book path: greenleaf_pretrain/run_integration.ipynb
- [Readme for the analysis of cell embeddings tutorial](greenleaf_pretrain/README.txt)
- Description: Cell embeddings learned by Attune preserve biological signals in the high-dimensional space on fetal human cortex dataset.
## Data and manuscript figures
|fig|script|data|data path|
|---|------|----|----|
|fig2a,b<br>Supplementary fig2,9|10x_pretrain/run_integration.ipynb|ad_atac.h5ad(c60296d27e026b70c371a3b0e80a1fb2)<br>ad_rna.h5ad(c326a877c845b9582cf22e7f62206a78)<br>weight_decoder_embedding_epoch12.h5(67ebe1a315a1ced46e4b8698a5dc6884)<br>weight_decoder_epoch12.h5(787fa8f66844d53a28a52307582a44e6)<br>weight_encoder_embedding_epoch12.h5(fe06e48aa4157b62e93440fba04c257c)<br>weight_encoder_epoch12.h5(28b1488543675c4ecda48bf6f877e7f0)|https://doi.org/10.6084/m9.figshare.22032170.v1<br>10x_pretrain/weight/pretrain/weight_decoder_embedding_epoch12.h5<br>10x_pretrain/weight/pretrain/weight_decoder_epoch12.h5<br>10x_pretrain/weight/pretrain/weight_encoder_embedding_epoch12.h5<br>10x_pretrain/weight/pretrain/weight_encoder_epoch12.h5|
|fig2f|10x_prediction/run_prediction.ipynb|ad_atac.h5ad(c60296d27e026b70c371a3b0e80a1fb2)<br>ad_rna.h5ad(c326a877c845b9582cf22e7f62206a78)<br>weight_decoder_embedding_epoch7.h5(0231b735eef6f69dbf66a72c7235bc05)<br>weight_project_epoch7.h5(6645b08222266a6c75d27b8b0bd47a2b)|https://doi.org/10.6084/m9.figshare.22032170.v1<br>10x_prediction/weight/predict/weight_decoder_embedding_epoch7.h5<br>10x_prediction/weight/predict/weight_project_epoch7.h5|
|fig3a,d,e,f<br>Supplementary fig4b|10x_regulatory/run_regulatory.ipynb|ad_atac.h5ad(c60296d27e026b70c371a3b0e80a1fb2)<br>ad_rna.h5ad(c326a877c845b9582cf22e7f62206a78)<br>weight_transformer_epoch10.h5(27c9085f062dab3db98a18e1f8ba003b)|https://doi.org/10.6084/m9.figshare.22032170.v1<br>10x_regulatory/weight/regulatory/weight_transformer_epoch10.h5|


## License Statement
The codebase is licensed under ​GPL-3.0, requiring derivative works to remain open-source, while all data files in /data and /results are released under ​CC0 1.0, waiving copyright and permitting unrestricted use (including commercial) without attribution.

