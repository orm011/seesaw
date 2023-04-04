library(tidyverse)
library(arrow)


target = c(3.8,3.)
units = 'in'

if (all(near(dev.size(units=units), target))){
  print('no need to change device')
  
} else{
  graphics.off()
  print('changing device')
  dev.new(width=target[1], height=target[2],  unit=unit, noRStudioGD = TRUE)
}

sz <- 3
update_geom_defaults("text", list(size = sz))
update_geom_defaults("label", list(size = sz))

table <- read_parquet('./basic_scatter_data.parquet', as_data_frame=TRUE)
table <- tibble(table)

table <- (table %>% mutate(AP=average_precision))

data <- table %>% filter(`query group` == 'all queries')

hard_counts <- table %>% group_by(`query group`, dataset) %>% summarise(n=n())

#totals <- h
hard_counts <- hard_counts %>% pivot_wider(names_from=`query group`, values_from = n) %>% mutate(fraction=`hard subset`/`all queries`)


make_label <- function(delta_mean, n,t){
  strm <- str_remove(sprintf("%0.2f", round(delta_mean, digits = 2)), "^0+")
  paste(strm, ' (', n, '/', t, ')', sep='')
}

# label=make_label(delta_mean, n)),
# data=dfagg, vjust='center', hjust='left', 

theme_szs <- 9


plot <- (ggplot(data=data)
         + facet_grid(rows=vars(dataset) )
         + stat_ecdf(geom = "step", aes(x=AP, group=dataset))
         #+ geom_violin(aes(x=dataset, y=AP), scale='width', linewidth=.1)
         #+ geom_errorbar(aes(x=dataset, ymin=AP, ymax=AP), width=.05, linewidth=.4, alpha=.3)
         #+
         + annotate('rect', xmin=-Inf, ymin=0, xmax=.5, ymax=.0, fill='palegreen', alpha=.1)
         + geom_segment(aes(x=-Inf, xend=.5, y=fraction, yend=fraction), data=hard_counts,  linetype='21')
         + geom_text(aes(x=0, y=fraction + .05, label=make_label(fraction, `hard subset`, `all queries`)), vjust='bottom', hjust='left', data=hard_counts)
         #+ geom_text(aes(x=.2, y=fraction, label=`fraction`), data=hard_counts)
         
         #+ geom_text(aes(x=0, y=1, label=`all queries`), data=hard_counts, hjust='left', vjust='top')
         
         #+ coord_flip()
         + labs(x='AP', 
                y='fraction of queries (0 to 1)',
        #        title='CDF of zero-shot CLIP accuracy',
         )
       #  + scale_y_continuous(breaks=seq(1.))
         #, limits = c(NA,1.), expand=c(.025, .01))
         + theme(
           axis.title.x = element_text(size=theme_szs),
           axis.title.y = element_text(size=theme_szs),
           
           axis.text.y=element_blank(),
           axis.text.x =element_text(size=theme_szs),
           #aspect.ratio=1.,
           plot.title = element_text(size=theme_szs),
           strip.text = element_text(size=theme_szs),
           panel.spacing.x = unit(1, units = 'mm'),
           # panel.border = element_rect(color='black', fill=NULL),
           # axis.title.y=element_blank(),
           axis.ticks.y = element_blank(),   
           legend.title=element_text(size=theme_szs),
           legend.text=element_text(size=theme_szs),
           axis.title=element_text(size=theme_szs),
           plot.background=element_blank(),
           legend.position = 'bottom',
           legend.box.margin = unit(c(0,0,0,0), 'mm'), # = unit(,'native'),
           plot.margin=grid::unit(c(0,0,0,0), "mm"),
           #                axis.line=element_line(color="black", size = 2)
         )
)

ggsave(plot = plot, filename='./paper_figures/clip_zero_shot_variance.pdf',  bg = NULL)
plot

