library(tidyverse)
library(latex2exp)

target = c(5,3)
units = 'in'

if (all(near(dev.size(units=units), target))){
  print('no need to change device')
  
} else{
  graphics.off()
  print('changing device')
  dev.new(width=target[1], height=target[2],  unit=unit, noRStudioGD = TRUE)
}




base <- ggplot() +
  xlim(0, 8) + 
  ylim(0,4) +
  ylab(TeX(r'[ E[distinct results] ]')) + 
  xlab('frames inspected by detector\n(chosen at random)') + 
  coord_fixed(ratio = 1) + 
  theme(axis.title = element_text(size = 12), legend.position = 'right', legend.title = element_blank(),
        axis.text = element_text(size=12)
  )

p1 = .25
p2 = .5
N=4

random1 <- function(n) {N*(1 - exp( log(1-p1)*n ))}
random2 <- function(n) {N*(1 - exp( log(1-p2)*n ))}

linear1 <- function(n) {n/2}
linear2 <- function(n) {n}
linear3 <- function(n) {2*n}

plot <- base +
  geom_function(fun = random2, aes(color='chunk1'), linetype='21') +
  stat_function(fun = random2, aes(color='chunk1'), geom = "point", n = 9) +
  geom_function(fun = random1, aes(color='chunk2'), linetype='21') +
  stat_function(fun = random1, aes(color='chunk2'), geom = "point", n = 9) 

plot
ggsave(plot = plot, filename='~/Desktop/random_sample2.pdf',  bg = NULL)