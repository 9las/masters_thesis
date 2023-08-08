#!/usr/bin/env Rscript
library(tidyverse)

# Read data ---------------------------------------------------------------
data <- read_tsv(file = "../results/s08_embedding_histograms.tsv")

data_sub <- data |>
  filter(sequence_type == "cdr",
         embedder_index == 3)

bin_edges <- data_sub |>
  pull(bin_edge)

bin_width = bin_edges[2]-bin_edges[1]

data_sub |>
  ggplot(mapping = aes(x = bin_edges + bin_width / 2,
                       y = fraction)) +
  geom_bar(stat = "identity")+
  scale_x_continuous(breaks = bin_edges |>
                       round(digits = 1))