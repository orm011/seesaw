library(tidyverse)
library(arrow)

df <- read_parquet('./seesaw_delta.parquet', as_data_frame = TRUE)
df <- tibble(df)

sz <- 3
update_geom_defaults("text", list(size = sz))
update_geom_defaults("label", list(size = sz))

theme_szs <- 9

quantile_df <- function(x, probs = c(0, 0.10, 0.5, .9, 1.)) {
  tibble(
    val = quantile(x, probs, na.rm = TRUE),
    quant = probs
  )
}

qdf <- df %>% group_by(`query group`, dataset) %>% reframe(quantile_df(delta))

pdf = pivot_wider(qdf, names_from='quant', values_from='val')

#  str_remove(my_values, "^0+") 
boxplot <- (ggplot(df)
            + facet_grid(rows=vars(dataset), cols=vars(`query group`) )
            + stat_ecdf(geom = "step", aes(x=delta))
            + geom_segment(aes(x=val, xend=val, y=0, yend=1.), data=qdf %>% filter(quant %in% c(0, .5, 1.)),  linewidth=.4)
            + geom_rect(aes(xmin=`0.1`, xmax=`0.9`, ymin=-Inf, ymax=Inf), data=pdf, alpha=.2, linewidth=.5 )
            
            + annotate('rect', xmin=-Inf, xmax=0, ymin=-Inf, ymax=Inf, fill='red', alpha=.2)
            #+ facet_grid(rows = vars(gp), scales = 'free_y', space = 'free_y')
            #+ geom_boxplot(position=position_dodge(.9),  coef=100, width=.6)
            #+ geom_text(aes(x=gp, y=delta_median -.01, color=dataset, label=sprintf("%0.2f", round(delta_median, digits = 2))),
            #          position = position_dodge(.9),  data=dfagg, vjust='top')
            # + geom_text(aes(x=dataset, y=.85, label=paste('(', n, ')', sep='')),
            #             data=dfagg, size=2.5, vjust='center', hjust='center', show.legend = FALSE)
            # + annotate(geom='text', x=.6, y=.9, vjust='bottom', label='N=', )
            # + annotate(geom='text', x=.6, y=.8, vjust='bottom', label=expression(mu*'='), )
            # + geom_errorbar(aes(x=dataset, ymin=delta_mean,  y = delta_mean, ymax=delta_mean),
            #                 position = position_dodge(.9),  data=dfagg, linetype='21', width=.6)
            # #+ geom_text(aes(x=dataset, y=delta_mean +.01, group=dataset,  label=sprintf("%0.2f", round(delta_mean, digits = 2))),
            # #             position = position_dodge(.9),  data=dfagg, vjust='bottom', show.legend = FALSE)
            # + geom_text(aes(x=dataset, y=1., label=make_label(delta_mean, n)),
            #             data=dfagg, vjust='center', hjust='left', show.legend = FALSE)
            # + scale_y_continuous(breaks=seq(-.4, 1., .2), limits = c(NA, 1.35), expand=c(.025, .01))
            # + annotate('rect', xmin=-Inf, ymin=0, xmax=Inf, ymax=Inf, fill='palegreen', alpha=.1)
            # #         + scale_linetype_discrete()
            # + coord_flip()
            # + geom_text(aes(x=dataset, y=1.1, label=paste('(', n, ')', sep='')), 
            #     data=dfagg, size=2.5, vjust='center', hjust='center', show.legend = FALSE)
            
            + labs(x='change in AP (bigger is better)', 
                   y='fraction of queries (0 to 1)'
                  # title='Change in AP when using SeeSaw,\naggregated by type of query and dataset',
            )
            + scale_y_continuous(breaks=seq())
            
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
              #axis.title.y=element_blank(),
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


target = c(3.8,2.6)
units = 'in'
if (all(near(dev.size(units=units), target))){
  print('no need to change device')
  
} else{
  graphics.off()
  print('changing device')
  dev.new(width=target[1], height=target[2],  unit=unit, noRStudioGD = TRUE)
}


## look into gridextra.

#grid.arrange(plot, plot2, nrow=2, 
#             heights=unit(c(10,2), c("cm", "cm")))

#grid.draw(rbind(ggplotGrob(plot), ggplotGrob(plot2), size = "last"))
## todo: instead of color, use position on x axis. do a faceted plot if needed.
#plot
ggsave(plot = boxplot, filename='./paper_figures/seesaw_delta_cdf.pdf',  bg = NULL)
