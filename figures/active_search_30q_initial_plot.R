library(tidyverse)
library(arrow)

table <- read_parquet('./active_search_res_v0.parquet', as_data_frame=TRUE)
table0 <- tibble(table)

plot <- (ggplot(table0,  aes(x=variant, y=average_precision))
  + geom_boxplot())

plot1 <- (ggplot(table0,  aes(x=variant, y=rank_first))
          + geom_boxplot())

plot2 <- (ggplot(table0,  aes(x=variant, y=rank_second))
         + geom_boxplot())

plot_nfound <- (ggplot(table0,  aes(x=variant, y=nfound))
          + geom_boxplot())


## current takeaway: (30 randomly chosen classes)
## lknn is about as good as the baseline
## log reg2 beat both slightly
## active search l1 performs worse than all...
## part of it is the second result is much later.