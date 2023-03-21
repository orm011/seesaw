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

table <- read_parquet('./breakdown_df.parquet', as_data_frame=TRUE)
table <- tibble(table)

#table <- (table %>% mutate(dataset=factor(dataset, c('objectnet', 'lvis', 'coco', 'bdd'))))

# table <- (table %>% mutate(version=recode_factor(version, baseline='CLIP embedding', 
#                                                  '+ multi-scale representation'='+ multi-scale',  
#                                                  '+ CLIP-guided vector alignment'='+ CLIP alignment', 
#                                                  '+ DB matched training'='+ DB matching')))


#tableagg <- (table %>% group_by(query_group, table_row, dataset) %>% summarise(average_precision=mean(average_precision))) 
#tableagg <- 
#globalagg <- tableagg %>% group_by(query_group, table_row) %>% summarise(average_precision=mean(average_precision))

#table <- bind_rows(tableagg, globalagg %>% mutate(dataset='avg'))
              
table <- (table %>% mutate(AP=average_precision))

bases <- (table %>% group_by(`query group`, dataset) %>% summarise(AP=min(AP)))

## add baseline to table, column by group
table <- merge(x = table, bases, by.x = c('query group', 'dataset'), by.y =c('query group', 'dataset'), suffixes = c('', '_base'))

#table <- (table %>% mutate(dataset=recode_factor(dataset, avg='avg', objectnet='ObjNet', lvis='LVIS', coco='COCO', bdd='BDD')))

value_offset = .5

plot <- (ggplot(data=table)  + 
          geom_col(aes(x=method, y=(AP - AP_base*.99)*2), alpha=0.)
          + geom_col(aes(x=method, y=(AP - AP_base*.99)), width=1, color='black')
          + scale_x_discrete(limits=rev)
          # + ylim(.5,1.)
          + coord_flip()
          + facet_grid(cols=vars(dataset), rows=vars(`query group`), scales = 'free')
          # + facet_grid(cols=vars(dataset), rows=vars(gp)) 
          + geom_text(aes(x=method, y=(AP - AP_base*.99)*1.05,  
                          label=str_remove(sprintf("%0.2f", round(AP, digits = 2)), "^0+")),
                      hjust='left', color='black')
          + theme(axis.title.x = element_blank(), 
                  axis.text.x = element_blank(),
                  axis.ticks.x = element_blank(),
                  axis.text.y = element_text(size=8),
                  axis.ticks.y = element_blank(),
                  axis.title.y = element_blank(),
                  panel.spacing.x = unit(0, units = 'mm'),
                  panel.background = element_blank(),
                  #panel.border = element_rect(color = "black", fill = NA, size = .1),
                  #plot.margin=grid::unit(c(0,0,0,0), "mm"),
                  panel.margin=unit(.05, 'lines'),
                  panel.spacing.y = unit(1, units='mm'),
                  strip.background = element_rect(color='gray', size = .1)
         ))

ggsave(plot = plot, filename='./paper_figures/breakdown.pdf',  bg = NULL)
plot