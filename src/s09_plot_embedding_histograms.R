#!/usr/bin/env Rscript
library(tidyverse)
library(glue)

# Read data ---------------------------------------------------------------
data <- read_tsv(file = "../results/s08_embedding_histograms.tsv")

# Functions ---------------------------------------------------------------
plot_embedding_histogram <- function(data, sequence_type, embedder_name, x_break_increnment){
  bin_edges <- data |>
    pull(bin_edge)
  
  bin_width = bin_edges[2]-bin_edges[1]
  
  data |>
    ggplot(mapping = aes(x = bin_edges + bin_width / 2,
                         y = fraction)) +
    geom_col() +
    scale_x_continuous(breaks = seq(from = min(bin_edges),
                                    to = max(bin_edges) + bin_width,
                                    by = x_break_increnment) |>
                         round(1)) +
    labs(x = "Embedding value",
         y = "Count (fraction of total)",
         title = glue('{embedder_name} embedding - {sequence_type}'),
         fill = "Model")
}


# Make the plots ----------------------------------------------------------

data <- data |>
  group_by(sequence_type,
           embedder_index,
           embedder_source,
           embedder_name) |>
  nest() |>
  ungroup()

pwalk(.l = list(data = data |>
                  pull(data),
                sequence_type = data |>
                  pull(sequence_type),
                embedder_name = data |>
                  pull(embedder_name),
                x_break_increnment = c(1, 0.8, 0.1, 0.4, 0.5, 0.6, 1, 0.8)),
      .f = \(data,
             sequence_type,
             embedder_name,
             x_break_increnment){
        p <- data |>
          plot_embedding_histogram(sequence_type = sequence_type,
                                   embedder_name = embedder_name,
                                   x_break_increnment = x_break_increnment)
        
        embedder_name_filename <- embedder_name |>
          str_replace_all(pattern = "/",
                          replacement = "_")
        
        ggsave(filename = glue('s09_plot__embedding_histogram__{sequence_type}__{embedder_name_filename}.svg'),
               plot = p,
               path = "../results",
               width = 30,
               height = 20,
               units = "cm")
      })