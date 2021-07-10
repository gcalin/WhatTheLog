# Title     : TODO
# Objective : TODO
# Created by: Pandelis
# Created on: 27/06/2021

library(ggplot2)
library(tidyquant)
# install.packages('svglite')
metrics <- read.csv(file= 'resources/rewards_in_episode.csv')

plot <- ggplot(data=metrics, aes(x=step, y=reward)) +
            geom_line() +
            labs(x = "Step", y = "Reward", title= "Total reward over an episode") +
            theme_bw() +
            theme(text=element_text(size=13))

ggsave(filename= "out/plots/reward_in_episode.svg", width= 6, height = 4, device = 'svg', plot)