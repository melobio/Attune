suppressPackageStartupMessages({
    library(dplyr)
    library(ggplot2)
    library(reshape2)
    library(tidyr)
})

res <- read.csv('residual_norm.txt', sep="\t", check.names=F)
rownames(res) <- res[,1]
res <- res[,-1]

ptime <- read.csv('data/pseudotime_palantir.csv', header=F)
colnames(ptime) <- c('cell', 'time')
rownames(ptime) <- ptime[,1]

common_cells <- intersect(rownames(res), rownames(ptime))

ptime <- ptime[common_cells,]
ptime <- ptime %>% arrange(time)
common_cells <- rownames(ptime)

res <- res[common_cells,]
res$time <- ptime$time

## order by standard deviation
res_plot <- melt(res, id.vars="time", variable.name='gene', value.name="residual")
p <- ggplot(res_plot, aes(x=time,y=residual)) + geom_point(alpha=0, size=0.5, color="#999999") + geom_smooth(se=F, span=0.3, color='#ee6666') + facet_wrap(~gene)
pdata <- ggplot_build(p)
df_sd <- pdata$data[[2]] %>% group_by(PANEL) %>% summarize(s=sd(y)) %>% as.data.frame
df_sd$gene <- colnames(res)[1:(ncol(res)-1)]
df_sd <- df_sd %>% arrange(desc(s))

## top 100 genes             
options(repr.plot.width=20, repr.plot.height=10)
png('top100_residual.png')
ggplot(melt(res[,c(df_sd$gene[1:100], 'time')], id.vars="time", variable.name='gene', value.name="residual"), aes(x=time,y=residual)) + geom_point(alpha=0, size=0.5, color="#999999") + geom_smooth(se=F, span=0.3, color='#ee6666') + facet_wrap(~gene, ncol=20) + 
    theme_bw() + theme(panel.background = element_blank(), panel.border = element_rect(color = "#6e7079"), strip.background=element_blank()) + ## , scales="free_y"
    scale_x_continuous(breaks=c(0,0.5,1), labels=c('0','0.5','1')) + xlab("Residual") + ylab("Pseudotime")
dev.off()

## bottom 100 genes                 
options(repr.plot.width=20, repr.plot.height=10)
png('bottom100_residual.png')
ggplot(melt(res[,c(df_sd$gene[1901:2000], 'time')], id.vars="time", variable.name='gene', value.name="residual"), aes(x=time,y=residual)) + geom_point(alpha=0, size=0.5, color="#999999") + geom_smooth(se=F, span=0.3, color='#ee6666') + facet_wrap(~gene, ncol=20) + 
    theme_bw() + theme(panel.background = element_blank(), panel.border = element_rect(color = "#6e7079"), strip.background=element_blank()) + ## , scales="free_y"
    scale_x_continuous(breaks=c(0,0.5,1), labels=c('0','0.5','1')) + xlab("Residual") + ylab("Pseudotime")
dev.off()