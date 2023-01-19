library(tidyverse)
library(arrow)


target = c(3.8, 2.4)
units = 'in'

if (all(near(dev.size(units=units), target))) {
  print('no need to change device')
  
} else{
  graphics.off()
  print('changing device')
  dev.new(width=target[1], height=target[2],  unit=unit, noRStudioGD = TRUE)
}

table <- read_parquet('../notebooks/find10_interval.parquet', as_data_frame=TRUE)
table0 <- tibble(table)

table <- replace_na(table0, replace=list(lower=360, high=360))
table <- (table %>% mutate(method=recode_factor(method, baseline='CLIP only'),
                           difficulty=recode_factor(qstr, wheelchair='hard',  dog='hard',
                                                    'spoon'='easy', 'melon'='hard',
                                                    'egg carton'='easy', 'dustpan'='easy',
                                                    'car with open door'='hard',
                                                  ),
                           qstr=recode_factor(qstr, 
                                              'egg carton'='egg carton',
                                              'dustpan'='dustpan',
                                              'spoon'='spoon',
                                              

                                              dog='dog',
                                              wheelchair='wheelchair', 
                                              
                                              melon='melon',
                                              'car with open door'='car with open door', 
                                              
                                              
                                              .ordered=TRUE))
          )


table <- table %>% filter(correction_n == 6)

pos <- position_dodge(width=1., )

plot <- (ggplot(data=table)
         #+ geom_vline(xintercept = 360, linetype='dashed', color='black', alpha=.5)
         
          + geom_errorbarh(aes(xmin=lower, xmax=high, y=qstr,  color=method, linetype=method),  position=pos, height=.6)
         + geom_point(aes(x=med, y=qstr, color=method,  fill=method), position=pos, shape=25)
         #+ geom_point(aes(x=mean, y=qstr, color=method, fill=method))
         #+ geom_errorbarh(aes(xmin=med, xmax=med, y=qstr, color=method), height=.6)
          + facet_grid(rows=vars(difficulty), scales='free_y', space='free_y')
         + xlab(label='time (seconds) - less is better')
         #+ annotate('vline', xintercept=c(360), linetype='dashed', color='black')
         + scale_x_continuous(breaks=seq(0, 360, 60), limits = c(0,364), expand=c(.0, .0))
       #  + scale_y_discrete(labels=label_wrap_gen(12))
         + geom_hline(yintercept =c(1.5, 2.5, 3.5), color='white', size=.5)
         
         + theme(legend.position = 'top', 
                 axis.title.y = element_blank(),
                 axis.ticks.y = element_blank(),
                 axis.text.y = element_text(size=9),
                 strip.text.y = element_text(size=9),
                 panel.grid.major.y =  element_blank(),
                 panel.spacing.y =  unit(1, unit='mm'),
                 panel.border = element_rect(colour="black", fill = NA),
                 #panel.border = element_line(color='white'),
                 legend.title = element_blank(),
                 legend.spacing = unit(.5, unit='mm'),
                 legend.margin = margin(c(0,0,0,0), unit='mm'),
                 #panel.background = element_blank(),
                 plot.margin=grid::unit(c(0,0,0,0), "mm"),
                strip.background = element_blank(),
                #legend.
            )
        )
  
ggsave(plot = plot, filename='./query_latency.pdf',  bg = NULL)
plot