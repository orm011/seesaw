library(tidyverse)
library(arrow)


target = c(3.8,1.5)
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

table <- read_parquet('./breakdown.parquet', as_data_frame=TRUE)
table <- tibble(table)

table <- (table %>% mutate(dataset=factor(dataset, c('objectnet', 'lvis', 'coco', 'bdd'))))

table <- (table %>% mutate(version=recode_factor(version, baseline='CLIP embedding', 
                                                 '+ multi-scale representation'='+ multi-scale',  
                                                 '+ CLIP-guided vector alignment'='+ CLIP alignment', 
                                                 '+ DB matched training'='+ DB matching')))


tableagg <- (table %>% group_by(gp, version) %>% summarise(AP=mean(AP)) %>% mutate(dataset='all'))

table <- (bind_rows(table, tableagg))
table <- (table %>% filter(gp=='all queries'))

table <- (table %>% mutate(dataset=factor(dataset, c('all', 'objectnet', 'lvis','bdd', 'coco'))))

bases <- (table %>% group_by(dataset) %>% summarise(AP=min(AP)))

## add baseline to table, column by group
table <- merge(table, bases, by.x = 'dataset', by.y ='dataset', suffixes = c('', '_base'))

table <- (table %>% mutate(dataset=recode_factor(dataset, all='all', objectnet='ObjNet', lvis='LVIS', coco='COCO', bdd='BDD')))

value_offset = .5

plot <- (ggplot(data=table)  + 
          geom_col(aes(x=version, y=(AP - AP_base*.99)*2), alpha=0.)
          + geom_col(aes(x=version, y=(AP - AP_base*.99)), width=1, color='black')
          + scale_x_discrete(limits=rev)
          # + ylim(.5,1.)
          + coord_flip()
          + facet_grid(cols=vars(dataset), scales = 'free_x')
          # + facet_grid(cols=vars(dataset), rows=vars(gp)) 
          + geom_text(aes(x=version, y=(AP - AP_base*.99)*1.05,  
                          label=str_remove(sprintf("%0.2f", round(AP, digits = 2)), "^0+")),
                      hjust='left', color='black')
          + theme(axis.title.x = element_blank(), 
                  axis.text.x = element_blank(),
                  axis.ticks.x = element_blank(),
                  axis.text.y = element_text(size=8),
                  axis.ticks.y = element_blank(),
                  axis.title.y = element_blank(),
                  panel.background = element_blank(),
                  panel.spacing.x = unit(0, units = 'mm'),
                  plot.margin=grid::unit(c(0,0,0,0), "mm"),
                  panel.spacing.y = unit(0, units='mm'),
                  strip.background = element_rect(color='white')
         ))




ggsave(plot = plot, filename='./breakdown.pdf',  bg = NULL)
plot