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

## Multimodal pretraining (Supplementary Fig2,9)
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
## Data and Manuscript Figures
|fig|script|data(md5)|data path|
|---|------|----|----|
|Supplementary fig2,9|10x_pretrain/run_integration.ipynb|ad_atac.h5ad(c60296d27e026b70c371a3b0e80a1fb2)<br>ad_rna.h5ad(c326a877c845b9582cf22e7f62206a78)<br>weight_decoder_embedding_epoch12.h5(67ebe1a315a1ced46e4b8698a5dc6884)<br>weight_decoder_epoch12.h5(787fa8f66844d53a28a52307582a44e6)<br>weight_encoder_embedding_epoch12.h5(fe06e48aa4157b62e93440fba04c257c)<br>weight_encoder_epoch12.h5(28b1488543675c4ecda48bf6f877e7f0)|https://doi.org/10.6084/m9.figshare.22032170.v1<br>10x_pretrain/weight/pretrain/weight_decoder_embedding_epoch12.h5<br>10x_pretrain/weight/pretrain/weight_decoder_epoch12.h5<br>10x_pretrain/weight/pretrain/weight_encoder_embedding_epoch12.h5<br>10x_pretrain/weight/pretrain/weight_encoder_epoch12.h5|
|fig2f|10x_prediction/run_prediction.ipynb|ad_atac.h5ad(c60296d27e026b70c371a3b0e80a1fb2)<br>ad_rna.h5ad(c326a877c845b9582cf22e7f62206a78)<br>weight_decoder_embedding_epoch7.h5(0231b735eef6f69dbf66a72c7235bc05)<br>weight_project_epoch7.h5(6645b08222266a6c75d27b8b0bd47a2b)|https://doi.org/10.6084/m9.figshare.22032170.v1<br>10x_prediction/weight/predict/weight_decoder_embedding_epoch7.h5<br>10x_prediction/weight/predict/weight_project_epoch7.h5|
|fig3a,d,e,f<br>Supplementary fig4b|10x_regulatory/run_regulatory.ipynb|ad_atac.h5ad(c60296d27e026b70c371a3b0e80a1fb2)<br>ad_rna.h5ad(c326a877c845b9582cf22e7f62206a78)<br>weight_transformer_epoch10.h5(27c9085f062dab3db98a18e1f8ba003b)|https://doi.org/10.6084/m9.figshare.22032170.v1<br>10x_regulatory/weight/regulatory/weight_transformer_epoch10.h5|
|fig3g<br>fig4a,c,d,e<br>Supplementary fig6a,7|SHARE_TAC_potential/infer_potential.ipynb|adata_atac_SHARE_TAC.h5ad(0a70b5cfab579b99e6d21af1dfde706b)<br>adata_rna_SHARE_TAC.h5ad(7676fd5b0b9700279275be3c04405b98)<br>weight_transformer_epoch10.h5(a32b3537a449913d45d4ee5d40c12b45)|https://doi.org/10.6084/m9.figshare.22032437.v1<br>SHARE_TAC_potential/weight/regulatory/weight_transformer_epoch10.h5|
|fig5b,c<br>Supplementary fig8a,b|greenleaf_pretrain/run_integration.ipynb|greenleaf-final-ATAC_filter_rm_dc1r3_r1.h5ad(7869af440b67d9a4ea99ce83e105fa1e)<br>greenleaf-final-RNA_wox_filter_hvg_rm_dc1r3_r1.h5ad(a4b2db8d6fb11f5dfd75a1c6c3fccc57)<br>weight_decoder_embedding_epoch20.h5(498abc174d4b75bb3946620cd6780158)<br>weight_decoder_epoch20.h5(bd13adc5ad99364a924d51936b543e4e)<br>weight_encoder_embedding_epoch20.h5(423e5336ae21650b5f6fd9f8d1c4d3e4)<br>weight_encoder_epoch20.h5(d9e24691f20cfc9da69f329abf41deda)|https://doi.org/10.6084/m9.figshare.22032494.v1<br>greenleaf_pretrain/weight/pretrain/weight_decoder_embedding_epoch20.h5<br>greenleaf_pretrain/weight/pretrain/weight_decoder_epoch20.h5<br>greenleaf_pretrain/weight/pretrain/weight_encoder_embedding_epoch20.h5<br>greenleaf_pretrain/weight/pretrain/weight_encoder_epoch20.h5|
|fig2|ftp:<br>Figure2/benchmark.ipynb|10x_metrics.csv(46e81425516009aee8a16764ac61c994)<br>metrics_ablation.txt(5b0dfdb372e3002c4e4821527b60b2bc)<br>metrics_alignment.txt(8c2af9a62491a1722ea7dc4f5e54cd6c)<br>metrics_embeding.csv(8d1693a15c65e563ed38a1d279d02ff3)<br>metrics_gene_peak.csv(91a2955e40a45b27240b3c690a2cd567)<br>metrics_prediction.txt(f4b32d6b9f5ab7a3375e68a97a9b0c9e)<br>omics_mixing_biology_conservation.txt(4482e61a6fb68b7f1d7d881edec11fda)|ftp:<br>Figure2/10x_metrics.csv<br>Figure2/metrics_ablation.txt<br>Figure2/metrics_alignment.txt<br>Figure2/metrics_embeding.txt<br>Figure2/metrics_gene_peak.txt<br>Figure2/metrics_prediction.txt<br>Figure2/omics_mixing_biology_conservation.txt|
|Supplementary fig3|ftp:<br>Supplementary_Figure_S3/eval_SHARE-seq.ipynb|adata_ATAC_wox_filter.h5ad(9193f27d2504717c6cce29cbe726d988)<br>adata_rna_wox_filter_hvg2000.h5ad(4c52a4ee3ce8aae286f2741bdb66cb12)<br>result_ep40.npz(b5e079636520ff78399ad60b483b2662)|ftp:<br>Supplementary_Figure_S3/data/adata_ATAC_wox_filter.h5ad<br>Supplementary_Figure_S3/data/adata_rna_wox_filter_hvg2000.h5ad<br>Supplementary_Figure_S3/result/result_ep40.npz|
|Supplementary fig5|ftp:<br>Supplementary_Figure_S5/share-seq.ipynb|cross_attention.csv(e8493ea3d8c50254604186fa5fb65257)<br>gene.list(cb8e265e86eea2006e1facd7c80eabd3)<br>share.rds(4943abe26a9b66a2134a3c6349825c0e)|ftp:<br>Supplementary_Figure_S5/cross_attention.csv<br>Supplementary_Figure_S5/gene.list<br>Supplementary_Figure_S5/share.rds|


## License Statement
The codebase is licensed under ​GPL-3.0, requiring derivative works to remain open-source, while all data files in /data and /results are released under ​CC0 1.0, waiving copyright and permitting unrestricted use (including commercial) without attribution.

