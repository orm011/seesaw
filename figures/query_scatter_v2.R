library(tidyverse)
library(arrow)

#library(ggplot2)
library(ggExtra)


df <- read_parquet('./query_scatter_multiscale.parquet', as_data_frame = TRUE)
options(repr.plot.width=6, repr.plot.height=6)


df <- tibble(df)

df <- (df %>% mutate(delta = variant - baseline))


example <- (df %>% filter(dataset == 'lvis' & category %in% c('dishrag', 'pudding') ) 
                           %>% mutate(change= c('better', 'worse'))
            )

sz <- 5
update_geom_defaults("text", list(size = sz))
update_geom_defaults("label", list(size = sz))


plot <- (ggplot(df,  aes(x=baseline, y=delta))
         + geom_point(aes(color=dataset),  alpha=.7)
         + scale_x_continuous(breaks=seq(0,1,.5), limits = c(0, 1), expand = c(.01, .01))
         + scale_y_continuous(breaks=seq(-.2, 1.,.2), limits = c(NA, 1.0), expand=c(.025, .01))
         + annotate('rect', xmin=-Inf, ymin=-Inf, xmax=.5, ymax=Inf, fill='palegreen', alpha=.2)
         + annotate('label', x=.25, y=.99, label='low initial AP', hjust='center', vjust='top')
         + annotate('segment', x=0, xend=.5, y=1., yend=1., arrow=arrow(ends='both', length =unit(.02, 'native')))
         + annotate('segment', x=.75, xend=.75, y=0, yend=.25,  arrow = arrow(ends = "last", length =unit(.02, 'native')))
         + annotate('label', x=.75, y=.25, vjust='bottom', hjust='left', label='upper bound\n(AP=1)', size=4.5)
         + annotate('segment', x=0, xend=1, y=1, yend=0, linetype='dashed')
         + annotate('segment', x=0, xend=1, y=0, yend=0, linetype='dashed')
#         + geom_abline(aes(slope=-1, intercept=1),linetype ='dashed')
#         + geom_abline(aes(slope=0, intercept=0), linetype='dashed')
          + geom_text(aes(label= paste('"',category,'"', sep=""), y=(delta + sign(delta)*.05)), hjust='center', vjust='center', data = example)
          + geom_label(aes(label=change, y=delta/2, x=baseline+.01), hjust='left', vjust='center', data = example)
         #+ geom_annotate('segment', x=0, y=.5, )
         + geom_segment(aes(y=0, yend=(delta - sign(delta)*.01), x=baseline, xend=baseline), arrow = arrow(ends = "last", length =unit(.02, 'native')), data=example)
         + labs(x='initial query AP', y='change in AP', 
                title='Change in AP using SeeSaw wrt. initial query AP'
                )
         + theme(
                 axis.text=element_text(size=15),
                 aspect.ratio=1.,
                 legend.title=element_text(size=12),
                 legend.text=element_text(size=10),
                 axis.title=element_text(size=15),
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



ggsave(plot = plot, filename='./query_scatter_multiscale.pdf', bg = NULL, width=6, height=6, units = 'in')
#plot