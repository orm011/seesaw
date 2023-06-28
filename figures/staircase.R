library(tidyverse)
library(latex2exp)

target = c(3.5,2.5)
units = 'in'
# 

if (TRUE){
if (all(near(dev.size(units=units), target))){
  print('no need to change device')

} else{
  graphics.off()
  print('changing device')
  dev.new(width=target[1], height=target[2],  unit=unit, noRStudioGD = TRUE)
}
}

deltas <- c(0,1,0,1,1,1,1,1,0) # frame

deltas <- c(0,1,0,1,1,0,0,0,0) # instance

deltas <- c(0,2,0,0,1,0,0,0,0) # ranomd
xs <- 1:length(deltas) - 1
ys<- cumsum(deltas)

tab1 <- data.frame(x=xs, y=ys )


base <- ggplot() +
  xlim(0, length(deltas)-1) + 
  ylim(0, max(ys)) +
  ylab('relevant frames found\n(reward)') + 
  ylab('unique instances found\n(reward)') +
  xlab('frames run through detector (cost)') + 
  scale_x_continuous(breaks=0:length(deltas)-1) +
  scale_y_continuous(breaks=0:max(ys)) +
  coord_fixed(ratio = 1) + 
  theme(  axis.title = element_text(size = 12), legend.position = 'right', legend.title = element_blank(),
        axis.text = element_text(size=12), panel.grid.minor=element_blank()) 


plot <- base +
  geom_step(data = tab1, mapping=aes(x,y, color='blue'),show.legend = FALSE)
  # geom_function(fun = random2, aes(color='chunk1'), linetype='21') +
  # stat_function(fun = random2, aes(color='chunk1'), geom = "point", n = 9) +
  # geom_function(fun = random1, aes(color='chunk2'), linetype='21') +
  # stat_function(fun = random1, aes(color='chunk2'), geom = "point", n = 9) 
ggsave(plot = plot, filename='~/Desktop/staircase1.pdf',  bg = NULL)
#plot