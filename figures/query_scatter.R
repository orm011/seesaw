library(tidyverse)
library(arrow)

df <- read_parquet('./query_scatter.parquet')
options(repr.plot.width=10, repr.plot.height=6)

jitter_sz <- .01
jitter <- position_jitter(width = jitter_sz, height = jitter_sz)
plot <- (ggplot(df, aes(x=ndcg_score_baseline, y=ndcg_score))
         + geom_point(aes(color=dataset), position = jitter, alpha=.7)
         + scale_x_continuous(breaks=seq(0,1,.2))
         + scale_y_continuous(breaks=seq(0,1,.2))
         + geom_abline(aes(slope=1, intercept=0))
         + labs(x='NDCG score - CLIP baseline', y='NDCG score - this work', 
                title='Scatter of query accuracy with Seesaw vs. CLIP baseline')
         + theme(axis.text=element_text(size=20),
                 legend.title=element_text(size=22),
                 legend.text=element_text(size=20),
                 axis.title=element_text(size=20),
                 plot.background=element_blank(),
         )
)

ggsave(plot = plot, filename='./query_scatter.pdf', width=10, height=6, units = 'in')
plot
