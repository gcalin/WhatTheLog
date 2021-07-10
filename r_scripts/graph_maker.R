# Title     : TODO
# Objective : TODO
# Created by: Pandelis
# Created on: 21/06/2021

library(ggplot2)
library(tidyquant)

metrics <- read.csv(file= 'out/metrics_0.csv')
print(names(metrics))
plot <- ggplot(data=metrics, aes(x=episode, y=f1_score)) +
            geom_line() +
            geom_ma(aema_fun = SMA, n = 30) +
            labs(x = "Episode", y = "F1 score") +
            theme_bw() +
            theme(text=element_text(size=15))

ggsave(filename= "out/plots/f1score.pdf", device = 'pdf', plot)

plot <- ggplot(data=metrics, aes(x=episode, y=size)) +
            geom_line() +
            geom_ma(aema_fun = SMA, n = 30) +
            labs(x = "Episode", y = "Size compression") +
            theme_bw() +
            theme(text=element_text(size=15))

ggsave(filename= "out/plots/compression.pdf", width= 6, height = 4, device = 'pdf', plot)

plot <- ggplot(data=metrics, aes(x=episode, y=reward)) +
            geom_line() +
            geom_ma(aema_fun = SMA, n = 30) +
            labs(x = "Episode", y = "Total reward") +
            theme_bw() +
            theme(text=element_text(size=15))

ggsave(filename= "out/plots/reward.pdf", width= 6, height = 4, device = 'pdf', plot)

plot <- ggplot(data=metrics, aes(x=episode, y=recall)) +
            geom_line() +
            geom_ma(aema_fun = SMA, n = 30) +
            labs(x = "Episode", y = "Recall") +
            theme_bw() +
            theme(text=element_text(size=15))

ggsave(filename= "out/plots/recall.pdf", width= 6, height = 4, device = 'pdf', plot)

plot <- ggplot(data=metrics, aes(x=episode, y=specificity)) +
            geom_line() +
            geom_ma(aema_fun = SMA, n = 30) +
            labs(x = "Episode", y = "Specificity") +
            theme_bw() +
            theme(text=element_text(size=15))

ggsave(filename= "out/plots/specificity.pdf", width= 6, height = 4, device = 'pdf', plot)

plot <- ggplot(data=metrics, aes(x=episode, y=precision)) +
            geom_line() +
            geom_ma(aema_fun = SMA, n = 30) +
            labs(x = "Episode", y = "Precision") +
            theme_bw() +
            theme(text=element_text(size=15))

ggsave(filename= "out/plots/precision.pdf", width= 6, height = 4, device = 'pdf', plot)

plot <- ggplot(data=metrics, aes(x=episode)) +
            geom_line(aes(y=nodes, color="nodes")) +
            geom_line(aes(y=transitions, color="transitions")) +
            labs(x = "Episode", y = "Size", color="variable") +
            theme_bw() +
            theme(text=element_text(size=15), legend.position = c(.95, .05))
                  # legend.box.just = "right",
                  # legend.margin = margin(4, 6, 6, 6))

ggsave(filename= "out/plots/size.pdf", width= 6, height = 4, device = 'pdf', plot)

plot <- ggplot(data=metrics, aes(x=episode, y=transitions)) +
            geom_line() +
            geom_ma(aema_fun = SMA, n = 30) +
            labs(x = "Episode", y = "Transitions") +
            theme_bw() +
            theme(text=element_text(size=15))

ggsave(filename= "out/plots/transitions.pdf", width= 6, height = 4, device = 'pdf', plot)

plot <- ggplot(data=metrics, aes(x=episode, y=steps)) +
            geom_line() +
            labs(x = "Episode", y = "Steps") +
            theme_bw() +
            theme(text=element_text(size=15))

ggsave(filename= "out/plots/steps.pdf", width= 6, height = 4, device = 'pdf', plot)

plot <- ggplot(data=metrics, aes(x=episode, y=duration)) +
            geom_line() +
            labs(x = "Episode", y = "Duration") +
            theme_bw() +
            theme(text=element_text(size=15))

ggsave(filename= "out/plots/durations.pdf", width= 6, height = 4, device = 'pdf', plot)


plot <- ggplot(data=metrics, aes(x=episode, y=transitions)) +
            geom_line() +
            geom_ma(aema_fun = SMA, n = 30) +
            labs(x = "Episode", y = "Transitions") +
            theme_bw() +
            theme(text=element_text(size=15))

ggsave(filename= "out/plots/transitions.pdf", width= 6, height = 4, device = 'pdf', plot)
