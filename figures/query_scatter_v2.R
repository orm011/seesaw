library(tidyverse)
library(arrow)

#library(ggplot2)
library(ggExtra)


df <- read_parquet('./query_scatter_multiscale.parquet', as_data_frame = TRUE)
options(repr.plot.width=10, repr.plot.height=6)


df <- tibble(df)

df <- (df %>% mutate(delta = variant - baseline))


example <- (df %>% filter(dataset == 'lvis' & category %in% c('dishrag', 'pudding') ) 
                           %>% mutate(change= c('better', 'worse'))
            )


plot <- (ggplot(df,  aes(x=baseline, y=delta))
         + geom_point(aes(color=dataset),  alpha=.7)
         + scale_x_continuous(breaks=seq(0,1,.5), limits = c(0, 1))
         + scale_y_continuous(breaks=seq(-.25, 1.,.25), limits = c(NA, 1.0))
         + annotate('rect', xmin=-Inf, ymin=-Inf, xmax=.5, ymax=Inf, fill='palegreen', alpha=.2)
         + annotate('label', x=.25, y=.99, label='low initial AP', hjust='center', vjust='top')
         + annotate('segment', x=0, xend=.5, y=1., yend=1., arrow=arrow(ends='both', length =unit(.02, 'native')))
         + annotate('segment', x=.75, xend=.75, y=0, yend=.25,  arrow = arrow(ends = "last", length =unit(.02, 'native')))
         + annotate('label', x=.75, y=.25, vjust='bottom', hjust='left', label='max possible\nincrease')
         + annotate('segment', x=0, xend=1, y=1, yend=0, linetype='dashed')
         + annotate('segment', x=0, xend=1, y=0, yend=0, linetype='dashed')
#         + geom_abline(aes(slope=-1, intercept=1),linetype ='dashed')
#         + geom_abline(aes(slope=0, intercept=0), linetype='dashed')
          + geom_text(aes(label= paste('category="',category,'"', sep=""), y=(delta + sign(delta)*.05)), hjust='center', vjust='center', data = example)
          + geom_label(aes(label=change, y=delta/2, x=baseline+.01), hjust='left', vjust='center', data = example)
         #+ geom_annotate('segment', x=0, y=.5, )
         + geom_segment(aes(y=0, yend=(delta - sign(delta)*.01), x=baseline, xend=baseline), arrow = arrow(ends = "last", length =unit(.02, 'native')), data=example)
         + labs(x='initial query AP', y='change in AP', 
                title='Changes in AP when using SeeSaw\nvs. initial query AP'
                )

         + theme(axis.text=element_text(size=20),
                 aspect.ratio=1.,
                 legend.title=element_blank(), #element_text(size=22),
                 legend.text=element_text(size=20),
                 axis.title=element_text(size=20),
                 plot.background=element_blank(),
                 legend.position = 'bottom'
         )
)

p2 <- ggMarginal(plot, type="boxplot", margin="both", groupColour=TRUE)
p2


ggsave(plot = p2, filename='./query_scatter_multiscale.pdf', width=10, height=6, units = 'in')
#plot