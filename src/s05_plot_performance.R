#!/usr/bin/env Rscript
library(tidyverse)
library(svglite)
library(fs)
library(patchwork)

# Read data

file_paths <- dir_ls(path = "../data/",
                     regexp = "s04_e\\d{2}_performance.tsv")

file_paths_summary <- dir_ls(path = "../data/",
                             regexp = "s04_e\\d{2}_performance_summary.tsv")

data <- read_tsv(file = file_paths,
                 col_types = cols(peptide = col_factor(),
                                  count_positive = col_integer()),
                 id = "path")

data_summary <- read_tsv(file = file_paths_summary,
                 col_types = cols(statistic = col_factor()),
                 id = "path")

data <- data |>
  mutate(experiment_index = str_sub(path,
                                    start = 13L,
                                    end = -17L) |>
           as_factor(),
         experiment_name = experiment_index |>
           fct_recode("Dropout and weighting" = "e01",
                      "Weighting only" = "e02",
                      "Dropout only" = "e03",
                      "No dropout and no weighting" = "e04",
                      "ESM1b" = "e05") |>
           fct_relevel("No dropout and no weighting",
                       "Weighting only",
                       "Dropout only",
                       "Dropout and weighting",
                       "ESM1b"),
         peptide_count_positive = str_glue("{peptide} ({count_positive})"))


data_summary <- data_summary |>
  mutate(experiment_index = str_sub(path,
                                    start = 13L,
                                    end = -25L) |>
           as_factor(),
         experiment_name = experiment_index |>
           fct_recode("Dropout and weighting" = "e01",
                      "Weighting only" = "e02",
                      "Dropout only" = "e03",
                      "No dropout and no weighting" = "e04",
                      "ESM1b" = "e05") |>
           fct_relevel("No dropout and no weighting",
                       "Weighting only",
                       "Dropout only",
                       "Dropout and weighting",
                       "ESM1b"),
         statistic = if_else(condition = statistic == "mean",
                             true = "Mean",
                             false = "Weighted mean"),
         statistic_count_positive = str_glue("{statistic} ({count_positive})"))

# AUC plot
p1 <- data |>
  ggplot(mapping = aes(x = peptide_count_positive |>
                         as_factor(),
                       y = auc,
                       fill = experiment_name))+
  geom_col(position = "dodge")+
  labs(x = "Peptide",
       fill = "Experiment")+
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
  

p2 <- data_summary |>
  ggplot(mapping = aes(x = statistic_count_positive,
                       y = auc,
                       fill = experiment_name))+
  geom_col(position = "dodge")+
  labs(x = "Statistic",
       y = "AUC",
       fill = "Experiment")

p <- p2+
  p1+
  plot_layout(guides = "collect",
              widths = c(0.075, 0.925))&
  theme(legend.position = "top",
        legend.justification = "left",
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))&
  ylim(0, 1)

ggsave(filename = "s05_auc_plot.svg",
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

# AUC 0.1 plot
p1 <- data |>
  ggplot(mapping = aes(x = peptide_count_positive |>
                         as_factor(),
                       y = auc01,
                       fill = experiment_name))+
  geom_col(position = "dodge")+
  labs(x = "Peptide",
       fill = "Experiment")+
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())


p2 <- data_summary |>
  ggplot(mapping = aes(x = statistic_count_positive,
                       y = auc01,
                       fill = experiment_name))+
  geom_col(position = "dodge")+
  labs(x = "Statistic",
       y = "AUC 0.1",
       fill = "Experiment")

p <- p2+
  p1+
  plot_layout(guides = "collect",
              widths = c(0.075, 0.925))&
  theme(legend.position = "top",
        legend.justification = "left",
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))&
  ylim(0, 1)

ggsave(filename = "s05_auc01_plot.svg",
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

# PPV plot
p1 <- data |>
  ggplot(mapping = aes(x = peptide_count_positive |>
                         as_factor(),
                       y = ppv,
                       fill = experiment_name))+
  geom_col(position = "dodge")+
  labs(x = "Peptide",
       fill = "Experiment")+
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())


p2 <- data_summary |>
  ggplot(mapping = aes(x = statistic_count_positive,
                       y = ppv,
                       fill = experiment_name))+
  geom_col(position = "dodge")+
  labs(x = "Statistic",
       y = "PPV",
       fill = "Experiment")

p <- p2+
  p1+
  plot_layout(guides = "collect",
              widths = c(0.075, 0.925))&
  theme(legend.position = "top",
        legend.justification = "left",
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))&
  ylim(0, 1)

ggsave(filename = "s05_ppv_plot.svg",
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")