library(tidyverse)
library(arrow)

table <- read_parquet('./baseline_scores.parquet', as_data_frame=TRUE)

baseline_scores  <-  
  table %>% select(!c(method_config, other_params, hit_indices, index_spec) & !starts_with('__')) # %>% names

options(repr.plot.width=10, repr.plot.height=6)

plot <- (ggplot(data=baseline_scores)
         + geom_histogram(aes(x=ndcg_score, fill=dataset), binwidth=.1, color='black', size=.2)
         + scale_x_continuous(breaks=seq(0,1,.1))
         + labs(x='NDCG score', y='# of queries', title='Histogram of query result accuracy')
         + theme(axis.text=element_text(size=20),
                 legend.title=element_text(size=22),
                 legend.text=element_text(size=20),
                 axis.title=element_text(size=20),
                 plot.background=element_blank(),
         )
)

ggsave(plot = plot, filename='./baseline_histogram.pdf', width=10, height=6, units = 'in')
