library(ggplot2)
library(readr)
library(readxl)
library(tidyquant)

test_data = read_csv("C:\\Users\\Tommaso\\Desktop\\Developer\\Uni\\WhatTheLog\\out\\size_reductions.csv")

plot <- ggplot(data=test_data, aes(x=n_traces, y=avg_compression, group=1)) + 
  geom_line() +
  scale_x_continuous(breaks = seq(50, 150, 10)) +
  scale_y_continuous(limits=c(1.0, 1.1)) +
  theme_bw()
print(plot)