#!/usr/bin/env Rscript

library(tidyverse)

# Arguments
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]

# Read data
data <-
  read_csv(input_file) %>%
  mutate_all(cumsum)

p <-
  ggplot(data, aes(x = x, y = y)) +
  geom_path()

ggsave(output_file, p, width = 10, height = 10, dpi = 300)
