library(tidyverse)
library(arrow)


target = c(3.8,2.)
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

plot <- (ggplot(data=data)
         + geom_violin(aes(x=dataset, y=AP), scale='width', linewidth=.1)
         + geom_errorbar(aes(x=dataset, ymin=AP, ymax=AP), width=.05, linewidth=.4, alpha=.3)
         + annotate('rect', xmin=-Inf, ymin=0, xmax=Inf, ymax=.5, fill='palegreen', alpha=.1)      
         + coord_flip()
         
         + labs(y='AP (bigger is better)', 
                title='Density plot of zero-shot CLIP accuracy',
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

ggsave(plot = plot, filename='./paper_figures/clip_zero_shot_variance.pdf',  bg = NULL)
plot

