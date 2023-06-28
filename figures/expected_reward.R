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

#m <- 8
m <- 7
ps <- c(1,3,4)/m

ps2 <- c(1,2)/m
#ns <- 0:8
ns <- seq(0,m+1,by=.1)
ns2 <- seq(0,m+1,by=1)


bps <- matrix(ps, nrow=length(ps), ncol=length(ns))
bps2 <- matrix(ps2, nrow=length(ps2), ncol=length(ns))

bns <- t(matrix(ns, nrow = length(ns), ncol=length(ps)))
bns2 <- t(matrix(ns, nrow = length(ns), ncol=length(ps2)))
ys <- colSums(1 - (1 - bps) ^ bns)
ys2 <- colSums(1 - (1 - bps2) ^ bns2)

tab1 <- data.frame(x=ns, y=ys )


deltas <- c(0,1,0,1,1,1,1,1,0) # frame

deltas <- c(0,1,0,1,1,0,0,0,0) # instance

deltas <- c(0,2,0,0,1,0,0,0,0) # ranomd
#xs <- 1:length(deltas) - 1
stepys<- cumsum(deltas)

tab2 <- data.frame(x=ns2, y=stepys)

tab11 <- data.frame(x=ns, y=ys2)


tab11 <- data.frame(x=ns, y=ys2)

tab_kink1 <- data.frame(x=ns, y=ys)
tab_kink2 <- data.frame(x=ns+last(ns), y=last(ys)+ys2)

base <- ggplot() +
  #ylab('relevant frames found\n(reward)') + 
  ylab('unique instances found\n(reward)') +
  xlab('frames run through detector (cost)') + 
  xlim(0, max(ns)) + 
  ylim(0, 3) +
  scale_x_continuous(breaks=0:max(ns)) +
  scale_y_continuous(breaks=0:length(ps)) +
  coord_fixed(ratio = 1) + 
  theme(  axis.title = element_text(size = 12), legend.position = 'right', legend.title = element_blank(),
          axis.text = element_text(size=12), panel.grid.minor=element_blank(),)

plot <- base +
  geom_step(data = tab2, mapping=aes(x,y, color='blue'), show.legend = FALSE) +
  ylim(0, 3)


plot2 <- plot +
    geom_line(data = tab1, mapping=aes(x,y, color='blue'), linetype='21', show.legend = FALSE)+
    ylim(0, 3)



plot3 <- base + 
    geom_line(data = tab1, mapping=aes(x,y, color='blue'), linetype='21', show.legend = FALSE)+
    geom_line(data = tab11, mapping=aes(x,y, color='red'), linetype='21', show.legend = FALSE)+
    ylim(0, 3)

plot_kink <- base + 
    geom_line(data = tab_kink1, mapping=aes(x,y, color='blue'), linetype='21', show.legend = FALSE)+
    geom_line(data = tab_kink2, mapping=aes(x,y, color='red'), linetype='21', show.legend = FALSE)+
    xlim(0, 2*max(ns)) + 
    scale_x_continuous(breaks=0:2*max(ns)) +
    scale_y_continuous(breaks=0:length(ps) + length(ps2)) +
    ylim(0, 5)

  


# geom_function(fun = random2, aes(color='chunk1'), linetype='21') +
# stat_function(fun = random2, aes(color='chunk1'), geom = "point", n = 9) +
# geom_function(fun = random1, aes(color='chunk2'), linetype='21') +
# stat_function(fun = random1, aes(color='chunk2'), geom = "point", n = 9) 
ggsave(plot = plot, filename='~/Desktop/staircase_random.pdf',  bg = NULL)
ggsave(plot = plot2, filename='~/Desktop/plus_expected_random.pdf',  bg = NULL)
ggsave(plot= plot3, filename='~/Desktop/expected_chunks.pdf', bg=NULL)
ggsave(plot= plot_kink, filename='~/Desktop/plot_kink.pdf', bg=NULL)

#plot