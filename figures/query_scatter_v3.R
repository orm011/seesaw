library(tidyverse)
library(arrow)

#library(ggplot2)
library(ggExtra)

target = c(3.8,2.6)
units = 'in'
if (all(near(dev.size(units=units), target))){
  print('no need to change device')
  
} else{
  graphics.off()
  print('changing device')
  dev.new(width=target[1], height=target[2],  unit=unit, noRStudioGD = TRUE)
}


df <- read_parquet('../notebooks/main_results_scatter.parquet', as_data_frame = TRUE)
#options(repr.plot.width=6, repr.plot.height=6)


df <- tibble(df)
df <- (df %>% mutate(variant =  pseudo_lr) %>% mutate(delta = pseudo_lr - baseline))


# example <- (df %>% filter(dataset == 'lvis' & category %in% c('dishrag', 'pudding') ) 
#             %>% mutate(change= c('better', 'worse'))
# )

sz <- 3
update_geom_defaults("text", list(size = sz))
update_geom_defaults("label", list(size = sz))

theme_szs <- 9

plot <- (ggplot(df,  aes(x=baseline, y=delta))
         + geom_point(alpha=.7, size=.5)
         + scale_x_continuous(breaks=seq(0,1,.5), limits = c(0, 1), expand = c(.01, .01))
         + scale_y_continuous(breaks=seq(-.8, 1.,.2), limits = c(NA, 1.0), expand=c(.025, .01))
         + annotate('rect', xmin=-Inf, ymin=0, xmax=Inf, ymax=Inf, fill='palegreen', alpha=.2)
         + annotate('rect', xmin=-Inf, ymin=-Inf, xmax=Inf, ymax=0, fill='red', alpha=.2)
         + annotate('rect', xmin=-Inf, ymin=-Inf, xmax=.5, ymax=Inf, fill='gray', alpha=.4)
         
         
         #+ annotate('segment', x=.75, xend=.75, y=0, yend=.25,  arrow = arrow(ends = "last", length =unit(.02, 'native')))
         # + annotate('label', x=.75, y=.30, vjust='bottom', hjust='left', label='AP\nupper bound', labeler=label_wrap_gen(width=20))
         # + annotate('segment', x=., y=.30, vjust='bottom', hjust='left', label='AP\nupper bound', labeler=label_wrap_gen(width=20))
         
         + annotate('segment', x=0, xend=1, y=1, yend=0, linetype='dashed', color='white')
         + annotate('segment', x=0, xend=1, y=0, yend=0, linetype='dashed')
         + annotate('label', x=.25, y=-.45 - .05, label='Hard queries', hjust='center', 
                        vjust='top', )
         + annotate('segment', x=0, xend=.5, y=-.45, yend=-.45, 
                      arrow=arrow(ends='both', length =unit(.02, 'native')))
         
         
         #         + geom_abline(aes(slope=-1, intercept=1),linetype ='dashed')
         #         + geom_abline(aes(slope=0, intercept=0), linetype='dashed')
         # + geom_text(aes(label= paste('"',category,'"', sep=""), y=(delta + sign(delta)*.05)), hjust='center', vjust='center', data = example)
         # + geom_label(aes(label=change, y=delta/2, x=baseline+.01), hjust='left', vjust='center', data = example)
         #+ geom_annotate('segment', x=0, y=.5, )
         # + geom_segment(aes(y=0, yend=(delta - sign(delta)*.01), x=baseline, xend=baseline), arrow = arrow(ends = "last", length =unit(.02, 'native')), data=example)
         + labs(x='baseline AP', y='change in AP (more positive is better)', 
                title='Change in AP using SeeSaw vs. baseline AP',
                color='Dataset:'
         )
         + theme(
           axis.text=element_text(size=theme_szs),
           aspect.ratio=1.,
           plot.title = element_text(size=theme_szs),
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


plot <- ggMarginal(plot, type="boxplot", margin="both", size=20, )#, groupColour=TRUE)
# xparams = list(stat='identity')
plot



ggsave(plot = plot, filename='./main_results_scatter.pdf', bg = NULL, width=6, height=6, units = 'in')
#plot


### boxplots below

dftotal <- (df %>% mutate(dataset='all'))
df = bind_rows(df, dftotal)
dfhard <- (df %>% filter(baseline < .5))

dfall <- (bind_rows(df %>% mutate(gp='all queries'), 
                      dfhard %>% mutate(gp='hard queries') ))

dfall <- (dfall %>%mutate(gp=factor(gp, c('hard queries', 'all queries')), 
                          dataset=recode_factor(dataset, 'all'='ALL', objectnet='ObjNet', lvis='LVIS', 
                                                bdd='BDD', coco='COCO', ))
)

dfagg <- (dfall %>%  group_by(gp, dataset) %>% summarise(delta_median=median(delta), delta_mean=mean(delta), n=n()))


make_label <- function(delta_mean, n){
  strm <- str_remove(sprintf("%0.2f", round(delta_mean, digits = 2)), "^0+")
  paste(strm, ' (', n, ')', sep='')
}


theme_szs <- 10

#  str_remove(my_values, "^0+") 
boxplot <- (ggplot(dfall,  aes(x=dataset, y=delta))
         + facet_grid(rows = vars(gp), scales = 'free_y')
         + geom_boxplot(position=position_dodge(.9),  coef=100, width=.6)
         #+ geom_text(aes(x=gp, y=delta_median -.01, color=dataset, label=sprintf("%0.2f", round(delta_median, digits = 2))),
         #          position = position_dodge(.9),  data=dfagg, vjust='top')
         # + geom_text(aes(x=dataset, y=.85, label=paste('(', n, ')', sep='')),
         #             data=dfagg, size=2.5, vjust='center', hjust='center', show.legend = FALSE)
         # + annotate(geom='text', x=.6, y=.9, vjust='bottom', label='N=', )
         # + annotate(geom='text', x=.6, y=.8, vjust='bottom', label=expression(mu*'='), )
         + geom_errorbar(aes(x=dataset, ymin=delta_mean,  y = delta_mean, ymax=delta_mean),
                         position = position_dodge(.9),  data=dfagg, linetype='21', width=.6)
         #+ geom_text(aes(x=dataset, y=delta_mean +.01, group=dataset,  label=sprintf("%0.2f", round(delta_mean, digits = 2))),
         #             position = position_dodge(.9),  data=dfagg, vjust='bottom', show.legend = FALSE)
         + geom_text(aes(x=dataset, y=1., label=make_label(delta_mean, n)),
                     data=dfagg, vjust='center', hjust='left', show.legend = FALSE)
         + scale_y_continuous(breaks=seq(-.4, 1., .2), limits = c(NA, 1.35), expand=c(.025, .01))
         + annotate('rect', xmin=-Inf, ymin=0, xmax=Inf, ymax=Inf, fill='palegreen', alpha=.1)
#         + scale_linetype_discrete()
         + coord_flip()
        # + geom_text(aes(x=dataset, y=1.1, label=paste('(', n, ')', sep='')), 
        #     data=dfagg, size=2.5, vjust='center', hjust='center', show.legend = FALSE)

         + labs(y='change in AP  (bigger is better)', 
                title='Change in AP when using SeeSaw,\naggregated by type of query and dataset',
         )
         + theme(
           axis.title.x = element_text(size=theme_szs),
           axis.text.y=element_text(size=theme_szs),
           axis.text.x =element_text(size=theme_szs),
           #aspect.ratio=1.,
           plot.title = element_text(size=theme_szs),
           strip.text = element_text(size=theme_szs),
           panel.spacing.x = unit(1, units = 'mm'),
           # panel.border = element_rect(color='black', fill=NULL),
           axis.title.y=element_blank(),
           
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

## look into gridextra.

#grid.arrange(plot, plot2, nrow=2, 
#             heights=unit(c(10,2), c("cm", "cm")))

#grid.draw(rbind(ggplotGrob(plot), ggplotGrob(plot2), size = "last"))
## todo: instead of color, use position on x axis. do a faceted plot if needed.
#plot
ggsave(plot = boxplot, filename='./boxplot_results.pdf',  bg = NULL)
