library(ggplot2)
library(readr)
library(readxl)
library(tidyquant)
library(tidyverse)
library(DataCombine)
library(purrr)

# testdata <- read_csv("testdata2.csv")
setwd("C:\\Users\\Tommaso\\Desktop\\Developer\\Uni\\WhatTheLog\\out\\results")
file_list = list.files(path="C:\\Users\\Tommaso\\Desktop\\Developer\\Uni\\WhatTheLog\\out\\results")
print(file_list)
dataset <- list()
for (i in 1:length(file_list)) {
  data <- read_csv(file_list[i], col_names = FALSE)
  runtimes <- data %>% pull(1)
  dataset[[i]] <- data.frame(eval=1:length(runtimes), runtime=runtimes)
}

plot <- ggplot()
for (i in 1:length(dataset)) {
  plot <- plot + geom_line(data=dataset[[i]], aes(x=eval, y=runtime))
}
plot <- plot + xlab("Evaluation") + ylab("Runtime (s)") + theme_bw()
print(plot)