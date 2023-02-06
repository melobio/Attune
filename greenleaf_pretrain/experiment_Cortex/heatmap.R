suppressPackageStartupMessages({
    library(dplyr)
    library(ComplexHeatmap)
    library(circlize)
})

ann_colors = list(celltype=c(`Cyc. Prog.`='#5470c6',`mGPC/OPC`='#3ba272',`RG/Astro`='#fac858',`nIPC/ExN`='#ee6666',`ExM`='#ea7ccc',`ExUp`='#91cc75',`ExDp`='#73c0de',`SP`='#9a60b4'))

df <- read.csv('pair_sim_matrix.csv', row.names=1, check.names=F)

ptime <- read.csv('celltype_time.tsv', sep="\t")

rna_od_time <- ptime %>% arrange(time) %>% pull(rna)
atac_od_time <- ptime %>% arrange(time) %>% pull(atac)

ct_rna <- ptime[,c('rna', 'celltype')]
ct_atac <- ptime[,c('atac', 'celltype')]

rownames(ct_rna) <- ct_rna$rna
ct_rna <- ct_rna[,-1,drop=F]

rownames(ct_atac) <- ct_atac$atac
ct_atac <- ct_atac[,-1,drop=F]

col_anno <- ct_rna[colnames(df),,drop=F]
row_anno <- ct_atac[rownames(df),,drop=F]

col_ha = HeatmapAnnotation(
    Celltype=col_anno[rna_od_time,], Pseudotime=(ptime %>% arrange(time) %>% pull(time)), 
    col=list(Celltype=c(`Cyc. Prog.`='#5470c6',`mGPC/OPC`='#3ba272',`RG/Astro`='#fac858',`nIPC/ExN`='#ee6666',`ExM`='#ea7ccc',`ExUp`='#91cc75',`ExDp`='#73c0de',`SP`='#9a60b4'), Pseudotime=colorRamp2(c(0, 1), c("#ffffff", "#6e7079"))), 
    show_legend=T, show_annotation_name=T, simple_anno_size = unit(0.3, "cm"),
    gap = unit(2, "mm"),
    annotation_legend_param = list(Celltype = list(ncol = 1, title = "Cell Type",at = c('Cyc. Prog.','RG/Astro','mGPC/OPC','nIPC/ExN','ExM','ExUp','ExDp','SP')))
)

row_ha = rowAnnotation(
    Celltype=row_anno[atac_od_time,], col=list(Celltype=c(`Cyc. Prog.`='#5470c6',`mGPC/OPC`='#3ba272',`RG/Astro`='#fac858',`nIPC/ExN`='#ee6666',`ExM`='#ea7ccc',`ExUp`='#91cc75',`ExDp`='#73c0de',`SP`='#9a60b4')), 
    show_annotation_name=F, show_legend=F, simple_anno_size = unit(0.3, "cm")
)

pdf('heatmap.pdf', width=5, height=4)
Heatmap(as.matrix(df[atac_od_time, rna_od_time]), 
    show_row_names = FALSE, show_column_names = FALSE, 
    col = colorRampPalette(c("white", "#FF4A4A"))(50), use_raster=F,
    border_gp = gpar(col="#aaaaaa"),  ## row_split = 7, 
    name = "Cosine Similarity", column_title='RNA', row_title='ATAC',
    top_annotation = col_ha, left_annotation = row_ha, cluster_columns=F, cluster_rows=F
)
dev.off()