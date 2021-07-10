# Title     : TODO
# Objective : TODO
# Created by: Pandelis
# Created on: 08/07/2021


library(ggplot2)
library(tidyquant)

metrics <- read.csv(file= 'resources/k_fold_metrics/scalability.csv')
print(metrics)
plot <- ggplot(data=metrics, aes(x=data_set_size, y=duration)) +
            geom_line() +
            geom_point(size=3) +
            labs(x = "Data set size (amount of logs)", y = "Execution time (s)") +
            theme_bw() +
            theme(text=element_text(size=20))

ggsave(filename= "out/plots/scalability.pdf", device = 'pdf', plot)

