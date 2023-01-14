library(tidyverse)
library(arrow)

#library(ggplot2)
library(ggExtra)
library(grid)
library(gridExtra)


df <- read_parquet('./query_scatter_multiscale.parquet', as_data_frame = TRUE)

dev.new(width=3.8,height=3.8,  unit='in', noRStudioGD = TRUE)
df <- tibble(df)

df <- (df %>% mutate(delta = variant - baseline, dataset=recode(dataset, objectnet='ObjNet', coco='COCO', lvis='LVIS', bdd='BDD',  all='all')))

sz <- 3
update_geom_defaults("text", list(size = sz))
update_geom_defaults("label", list(size = sz))

### same y scale. x-axis total, low. colors: dataset. total.
### show means, label quantiles.
## how about initial AP...

dftotal <- (df %>% mutate(dataset='all'))

df = bind_rows(df, dftotal)
dfhard <- (df %>% filter(baseline < .5))

dfall <- (bind_rows(df %>% mutate(gp='all queries'), dfhard %>% mutate(gp='hard queries') ))

dfall <- (dfall %>%mutate(gp=factor(gp, c('hard queries', 'all queries')), 
                          dataset=factor(dataset, c('ObjNet', 'LVIS', 'BDD', 'COCO', 'all'))
          )
        )

dfagg <- (dfall %>%  group_by(gp, dataset) %>% summarise(delta_median=median(delta), delta_mean=mean(delta), n=n()))


#  str_remove(my_values, "^0+") 
plot <- (ggplot(dfall,  aes(x=dataset, y=delta))
         + facet_grid(cols = vars(gp))
         + geom_boxplot(position=position_dodge(.9),  coef=100, width=.9)
         #+ geom_text(aes(x=gp, y=delta_median -.01, color=dataset, label=sprintf("%0.2f", round(delta_median, digits = 2))),
          #          position = position_dodge(.9),  data=dfagg, vjust='top')
        + geom_text(aes(x=dataset, y=.85, label=paste('(', n, ')', sep='')), 
                       data=dfagg, size=2.5, vjust='center', hjust='center', show.legend = FALSE)
        # + annotate(geom='text', x=.6, y=.9, vjust='bottom', label='N=', )
        # + annotate(geom='text', x=.6, y=.8, vjust='bottom', label=expression(mu*'='), )
        + geom_errorbar(aes(x=dataset, ymin=delta_mean,  y = delta_mean, ymax=delta_mean),
                      position = position_dodge(.9),  data=dfagg, linetype='dashed', width=.9)
        #+ geom_text(aes(x=dataset, y=delta_mean +.01, group=dataset,  label=sprintf("%0.2f", round(delta_mean, digits = 2))),
        #             position = position_dodge(.9),  data=dfagg, vjust='bottom', show.legend = FALSE)
        + geom_text(aes(x=dataset, y=.9, label=str_remove(sprintf("%0.2f", round(delta_mean, digits = 2)), "^0+")),
                       data=dfagg, vjust='center', hjust='center', show.legend = FALSE)
        + scale_y_continuous(breaks=seq(-.2, .9,.2), limits = c(NA, .95), expand=c(.025, .01))
        + scale_linetype_discrete()
        + labs(y='change in AP  (bigger is better)', 
                title='Change in AP when using SeeSaw,\naggregated by type of query and dataset',
         )
         + theme(
           axis.title.y = element_text(size=9),
           axis.text=element_text(size=8),
           axis.text.x =element_text(size=8),
           #aspect.ratio=1.,
           plot.title = element_text(size=8),
           strip.text = element_text(size=8),
           panel.spacing.x = unit(1, units = 'mm'),
           # panel.border = element_rect(color='black', fill=NULL),
           axis.title.x=element_blank(),
           legend.title=element_text(size=8),
           legend.text=element_text(size=8),
           axis.title=element_text(size=8),
           plot.background=element_blank(),
           legend.position = 'bottom',
           legend.box.margin = unit(c(0,0,0,0), 'mm'), # = unit(,'native'),
           plot.margin=grid::unit(c(0,0,0,0), "mm"),
           #                axis.line=element_line(color="black", size = 2)
         )
)

## look into gridextra.

#grid.arrange(plot, plot2, nrow=2, 
#             heights=unit(c(10,2), c("cm", "cm")))

#grid.draw(rbind(ggplotGrob(plot), ggplotGrob(plot2), size = "last"))
## todo: instead of color, use position on x axis. do a faceted plot if needed.
#plot
ggsave(plot = plot, filename='./boxplot_results.pdf',  bg = NULL)
