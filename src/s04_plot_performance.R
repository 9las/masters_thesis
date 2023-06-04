#!/usr/bin/env Rscript
library(tidyverse)
library(svglite)
library(argparser)
library(config)

# Input arguments
parser <- arg_parser(description = "Plot the model performance")
parser <- add_argument(parser = parser,
                       arg = "--config",
                       help = "Path to YAML configuration file")
args <- parse_args(parser)
config_filename <- args$config

# Load config
config <- config::get(file = config_filename)

experiment_index = config$experiment_index

data <- read_tsv(file = str_glue("../data/s03_e{experiment_index}_performance.tsv"),
                 col_types = cols(peptide = col_factor(),
                                  count_positive = col_integer()))

data_summary <- read_tsv(file = str_glue("../data/s03_e{experiment_index}_performance_summary.tsv"),
                         col_types = cols(statistic = col_factor()))

data <- data |>
  pivot_longer(cols = c(auc,
                        auc01,
                        ppv),
               names_to = "metric",
               values_to = "performance") |>
  mutate(metric = case_when(metric == "auc" ~ "AUC",
                            metric == "auc01" ~ "AUC 0.1",
                            metric == "ppv" ~ "PPV") |>
           as_factor(),
         peptide_count_positive = str_glue("{peptide} ({count_positive})"))

data_summary <- data_summary |>
  pivot_longer(cols = c(auc,
                        auc01,
                        ppv),
               names_to = "metric",
               values_to = "performance") |>
  mutate(metric = case_when(metric == "auc" ~ "AUC",
                            metric == "auc01" ~ "AUC 0.1",
                            metric == "ppv" ~ "PPV") |>
           as_factor(),
         statistic = if_else(condition = statistic == "mean",
                             true = "Mean",
                             false = "Weighted mean"))

p <- data |>
  ggplot(mapping = aes(x = performance,
                       y = peptide_count_positive |>
                         as_factor() |> 
                         fct_rev(),
                       color = metric,
                       group = metric))+
  geom_point()+
  geom_path()+
  geom_vline(mapping = aes(xintercept = performance,
                           color = metric,
                           linetype = statistic),
             data = data_summary)+
  xlim(0, 1)+
  labs(x = "Performance",
       y = "Peptide",
       color = "Metric",
       linetype = "Statistic")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = str_glue("s04_e{experiment_index}_performance_plot.svg"),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")