#!/usr/bin/env Rscript
library(tidyverse)
library(svglite)
library(argparser)
library(config)

# Input arguments
parser <- arg_parser()
parser <- add_argument(parser = parser,
                  arg = "--config")
args <- parse_args(parser)
config_filename <- args$config

# Load config
config <- config::get(file = config_filename)

experiment_index = config$experiment_index

data <- read_tsv(file = str_glue("../data/s03_e{experiment_index}_performance.tsv"),
                 col_types = cols(peptide = col_factor(),
                                  count_positive = col_integer()))

plot_auc_bar <- function(data, partition) {
  p <- data |>
    ggplot(mapping = aes(x = auc,
                         y = peptide |>
                           fct_rev(),
                         fill = threshold))+
    geom_col(position = "dodge")+
    xlim(0, 1)+
    labs(title = str_c("Partition ",
                       partition),
         x = "AUC",
         y = "Peptide",
         fill = "Threshold")+
    theme(plot.title.position = "plot",
          legend.position = "top",
          legend.justification = "left")
  
  ggsave(filename = str_c("auc_",
                          partition,
                          ".svg"),
         plot = p,
         path = "../out")
}


plot_auc <- function(data, partition) {
  p <- data |>
    ggplot(mapping = aes(x = auc,
                         y = peptide |>
                           fct_rev(),
                         color = threshold,
                         group = threshold))+
    geom_point()+
    geom_path()+
    xlim(0, 1)+
    labs(title = str_c("Partition ",
                       partition),
         x = "AUC",
         y = "Peptide",
         color = "Threshold")+
    theme(plot.title.position = "plot",
          legend.position = "top",
          legend.justification = "left")
  
  ggsave(filename = str_c("auc_",
                          partition,
                          ".svg"),
         plot = p,
         path = "../out")
}

plot_ppv <- function(data, partition) {
  p <- data |>
    ggplot(mapping = aes(x = ppv,
                         y = peptide |>
                           fct_rev()))+
    geom_col()+
    xlim(0, 1)+
    labs(title = str_c("Partition ",
                       partition),
         x = "PPV",
         y = "Peptide")+
    theme(plot.title.position = "plot",
          legend.position = "top",
          legend.justification = "left")
  
  ggsave(filename = str_c("ppv_",
                          partition,
                          ".svg"),
         plot = p,
         path = "../out")
}

plot_performance <- function(data) {
  p <- data |>
    ggplot(mapping = aes(x = performance,
                         y = peptide_count_positive |>
                           as_factor() |> 
                           fct_rev(),
                         color = metric,
                         group = metric))+
    geom_point()+
    geom_path()+
    xlim(0, 1)+
    labs(x = "Performance",
         y = "Peptide",
         color = "Metric")+
    theme(legend.position = "top",
          legend.justification = "left")
  
  ggsave(filename = str_glue("s04_e{experiment_index}_performance_plot.svg"),
         plot = p,
         path = "../results")
}

# data |>
#   select(!ppv) |>
#   pivot_longer(cols = c(auc,
#                         auc01),
#                names_to = "threshold",
#                values_to = "auc") |>
#   mutate(threshold = if_else(condition = threshold == "auc",
#                              true = "1",
#                              false = "0.1")) |>
#   group_by(partition) |>
#   group_walk(.f = \(.x, .y) plot_auc(.x, .y))
# 
# data |>
#   group_by(partition) |>
#   group_walk(.f = \(.x, .y) plot_ppv(.x, .y))

data |>
  pivot_longer(cols = c(auc,
                        auc01,
                        ppv),
               names_to = "metric",
               values_to = "performance") |>
  mutate(metric = case_when(metric == "auc" ~ "AUC",
                            metric == "auc01" ~ "AUC 0.1",
                            metric == "ppv" ~ "PPV") |>
           as_factor(),
         peptide_count_positive = str_glue("{peptide} ({count_positive})")) |>
  plot_performance()