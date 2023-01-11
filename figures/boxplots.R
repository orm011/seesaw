library(tidyverse)
library(arrow)

#library(ggplot2)
library(ggExtra)


df <- read_parquet('./query_scatter_multiscale.parquet', as_data_frame = TRUE)
options(repr.plot.width=6, repr.plot.height=6)

df <- tibble(df)

df <- (df %>% mutate(delta = variant - baseline))

sz <- 5
update_geom_defaults("text", list(size = sz))
update_geom_defaults("label", list(size = sz))

### same y scale. x-axis total, low. colors: dataset. total.
### show means, label quantiles.
## how about initial AP...

dftotal <- (df %>% mutate(dataset='all datasets'))

df = bind_rows(df, dftotal)
dfhard <- (df %>% filter(baseline < .5))

dfall <- (bind_rows(df %>% mutate(gp='all queries'), dfhard %>% mutate(gp='hard queries') ))

dfall <- (dfall %>%mutate(gp=factor(gp, c('hard queries', 'all queries')), 
                          dataset=factor(dataset, c('objectnet', 'lvis', 'bdd', 'coco', 'all datasets'))
          )
        )

dfagg <- (dfall %>%  group_by(gp, dataset) %>% summarise(delta_median=median(delta), delta_mean=mean(delta), n=n()))

plot <- (ggplot(dfall,  aes(x=gp, y=delta))
         + geom_boxplot(aes(color=dataset), position=position_dodge(.9),  coef=100)
         #+ geom_text(aes(x=gp, y=delta_median -.01, color=dataset, label=sprintf("%0.2f", round(delta_median, digits = 2))),
          #          position = position_dodge(.9),  data=dfagg, vjust='top')
         + geom_text(aes(x=gp, y=.9, color=dataset, label=paste("n =\n", n, '')), size=4,
                     position = position_dodge(.9),  data=dfagg, vjust='top', show.legend = FALSE)
        + geom_errorbar(aes(x=gp, ymin=delta_mean,  y = delta_mean, ymax=delta_mean, color=dataset),
                      position = position_dodge(.9),  data=dfagg, linetype='dashed', width=.75)
        + geom_text(aes(x=gp, y=delta_mean +.01, group=dataset,  label=sprintf("%0.2f", round(delta_mean, digits = 2))),
                     position = position_dodge(.9),  data=dfagg, vjust='bottom', show.legend = FALSE)
        + scale_y_continuous(breaks=seq(-.2, 1.,.2), limits = c(NA, 1.0), expand=c(.025, .01))
        + scale_linetype_discrete()
        + labs(y='change in AP (bigger is better)', 
                title='Change in AP by type of query and dataset',
                color='Dataset:',
         )
         + theme(
           axis.text=element_text(size=15),
           aspect.ratio=1.,
           axis.title.x=element_blank(),
           legend.title=element_text(size=15),
           legend.text=element_text(size=15),
           axis.title=element_text(size=15),
           plot.background=element_blank(),
           legend.position = 'bottom',
           legend.box.margin = unit(c(0,0,0,0), 'mm'), # = unit(,'native'),
           plot.margin=grid::unit(c(0,0,0,0), "mm"),
           #                axis.line=element_line(color="black", size = 2)
         )
)



ggsave(plot = plot, filename='./boxplot_results.pdf', bg = NULL, width=6, height=6, units = 'in')
plot