#!/usr/bin/env Rscript
library(tidyverse)
library(svglite)
library(fs)
library(patchwork)

# Read data ---------------------------------------------------------------

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
                                    start = 14L,
                                    end = -17L) |>
           str_remove(pattern = "^0+") |>
           as.numeric(),
         peptide_count_positive = str_glue("{peptide} ({count_positive})"))

data_summary <- data_summary |>
  mutate(experiment_index = str_sub(path,
                                    start = 14L,
                                    end = -25L) |>
           str_remove(pattern = "^0+") |>
           as.numeric(),
         statistic = if_else(condition = statistic == "mean",
                             true = "Mean",
                             false = "Weighted mean"),
         statistic_count_positive = str_glue("{statistic} ({count_positive})"))


# Functions ---------------------------------------------------------------

plot_performance <- function(data, data_summary, metric) {
  metric_name <- metric |>
    substitute() |>
    deparse()
  
  if (metric_name == "auc") {
    y_label <- "AUC"
  } else if (metric_name == "auc01") {
    y_label <- "AUC 0.1"
  } else if (metric_name == "ppv") {
    y_label <- "PPV"
  }
  
  p1 <- data |>
    ggplot(mapping = aes(x = peptide_count_positive |>
                           as_factor(),
                         y = {{ metric }},
                         fill = experiment_name))+
    geom_col(position = "dodge")+
    labs(x = "Peptide",
         fill = "Experiment")+
    theme(axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
  
  
  p2 <- data_summary |>
    ggplot(mapping = aes(x = statistic_count_positive,
                         y = {{ metric }},
                         fill = experiment_name))+
    geom_col(position = "dodge")+
    labs(x = "Statistic",
         y = y_label,
         fill = "Experiment")
  
  p2+
    p1+
    plot_layout(guides = "collect",
                widths = c(0.075, 0.925))&
    theme(legend.position = "top",
          legend.justification = "left",
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))&
    ylim(0, 1)
}

get_delta_mean_performance_table <- function(data_summary,
                                             experiment_index_base){
  data_summary |>
    group_by(statistic) |>
    nest() |>
    mutate(data = data |>
             map(.f = \(x) x |>
                   mutate(across(.cols = c(auc,
                                           auc01,
                                           ppv),
                                 .fns = function(y){
                                   base_performance <- x |>
                                     filter(experiment_index == experiment_index_base) |>
                                     pull(y)
                                   
                                   y - base_performance
                                 },
                                 .names = "{.col}_delta")))) |>
    unnest(cols = data) |>
    ungroup() |>
    filter(experiment_index != experiment_index_base) |>
    select(c(experiment_index,
             experiment_name,
             statistic,
             ends_with("_delta"))) |>
    arrange(experiment_index)
}

get_delta_performance_count_sign_table <- function(data,
                                                   experiment_index_base){
  data |>
    mutate(across(.cols = c(auc,
                            auc01,
                            ppv),
                  .fns = \(x) x - (data_subset |>
                                     filter(experiment_index == experiment_index_base) |>
                                     pull(x)),
                  .names = "{.col}_delta")) |>
    filter(experiment_index != experiment_index_base) |>
    group_by(experiment_index,
             experiment_name) |>
    summarise(across(.cols = c(auc_delta,
                               auc01_delta,
                               ppv_delta),
                     .fns = list(count_greater = \(x) (x >= 0) |>
                                   sum(),
                                 count_lesser = \(x) (x < 0) |>
                                   sum()),
                     .names = "{.col}.{.fn}")) |>
    pivot_longer(cols = !c(experiment_index,
                           experiment_name),
                 names_to = c(".value",
                              "condition"),
                 names_sep = "\\.") |>
    mutate(condition = if_else(condition = condition == "count_greater",
                               true = "positive",
                               false = "negative")) |>
    rename_with(.fn = \(x) paste("count",
                                 x,
                                 sep = "_"),
                .cols = ends_with("_delta"))
}

# Dropout and weighting ---------------------------------------------------

## Subset data ------------------------------------------------------------
data_subset <- data |> 
  filter(experiment_index >= 1,
         experiment_index <= 4) |>
  mutate(experiment_name = experiment_index |>
           as_factor() |>
           fct_recode("Dropout and weighting" = "1",
                      "Weighting only" = "2",
                      "Dropout only" = "3",
                      "No dropout and no weighting" = "4") |>
           fct_relevel("No dropout and no weighting",
                       "Weighting only",
                       "Dropout only",
                       "Dropout and weighting"))

data_summary_subset <- data_summary |>
  filter(experiment_index >= 1,
         experiment_index <= 4) |>
  mutate(experiment_name = experiment_index |>
           as_factor() |>
           fct_recode("Dropout and weighting" = "1",
                      "Weighting only" = "2",
                      "Dropout only" = "3",
                      "No dropout and no weighting" = "4") |>
           fct_relevel("No dropout and no weighting",
                       "Weighting only",
                       "Dropout only",
                       "Dropout and weighting"))

## AUC plot ---------------------------------------------------------------
p <- plot_performance(data = data_subset,
                      data_summary = data_summary_subset,
                      metric = auc)

ggsave(filename = "s05_plot__dropout_and_weighting__auc.svg",
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")


## AUC 0.1 plot -----------------------------------------------------------
p <- plot_performance(data = data_subset,
                      data_summary = data_summary_subset,
                      metric = auc01)

ggsave(filename = "s05_plot__dropout_and_weighting__auc01.svg",
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")


## PPV plot ---------------------------------------------------------------
p <- plot_performance(data = data_subset,
                      data_summary = data_summary_subset,
                      metric = ppv)

ggsave(filename = "s05_plot__dropout_and_weighting__ppv.svg",
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")


## Table with delta mean performance metrics ------------------------------
delta_mean_performance_table <- get_delta_mean_performance_table(data_summary = data_summary_subset,
                                                                 experiment_index_base = 4)

delta_mean_performance_table |>
  write_tsv(file = "../results/s05_table__dropout_and_weighting__delta_mean_performance.tsv")



## Table where sign of deltas are counted ---------------------------------
delta_performance_count_sign_table <- get_delta_performance_count_sign_table(data = data_subset,
                                                                             experiment_index_base = 4)

delta_performance_count_sign_table |>
  write_tsv(file = "../results/s05_table__dropout_and_weighting__delta_performance_count_sign.tsv")


# ESM1b vs BLOSUM50 -------------------------------------------------------

## Subset data ------------------------------------------------------------
data_subset <- data |> 
  filter(experiment_index == 1
         | experiment_index == 5) |>
  mutate(experiment_name = experiment_index |>
           as_factor() |>
           fct_recode("BLOSUM50" = "1",
                      "ESM1b" = "5") |>
           fct_relevel("BLOSUM50",
                       "ESM1b"))

data_summary_subset <- data_summary |>
  filter(experiment_index == 1
         | experiment_index == 5) |>
  mutate(experiment_name = experiment_index |>
           as_factor() |>
           fct_recode("BLOSUM50" = "1",
                      "ESM1b" = "5") |>
           fct_relevel("BLOSUM50",
                       "ESM1b"))

## AUC plot ---------------------------------------------------------------
p <- plot_performance(data = data_subset,
                      data_summary = data_summary_subset,
                      metric = auc)

ggsave(filename = "s05_plot__esm1b_vs_blosum50__auc.svg",
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

## AUC 0.1 plot -----------------------------------------------------------
p <- plot_performance(data = data_subset,
                      data_summary = data_summary_subset,
                      metric = auc01)

ggsave(filename = "s05_plot__esm1b_vs_blosum50__auc01.svg",
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

## PPV plot ---------------------------------------------------------------
p <- plot_performance(data = data_subset,
                      data_summary = data_summary_subset,
                      metric = ppv)

ggsave(filename = "s05_plot__esm1b_vs_blosum50__ppv.svg",
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

## Table with delta mean performance metrics ------------------------------
delta_mean_performance_table <- get_delta_mean_performance_table(data_summary = data_summary_subset,
                                                                 experiment_index_base = 1)

delta_mean_performance_table |>
  write_tsv(file = "../results/s05_table__esm1b_vs_blosum50__delta_mean_performance.tsv")

## Table where sign of deltas are counted ---------------------------------
delta_performance_count_sign_table <- get_delta_performance_count_sign_table(data = data_subset,
                                                                             experiment_index_base = 1)

delta_performance_count_sign_table |>
  write_tsv(file = "../results/s05_table__esm1b_vs_blosum50__delta_performance_count_sign.tsv")


# ESM1b - compare different architecture sizes ----------------------------

## Subset data ------------------------------------------------------------
data_subset <- data |> 
  filter(experiment_index >= 5,
         experiment_index <= 7) |>
  mutate(experiment_name = experiment_index |>
           as_factor() |>
           fct_recode("ESM1b" = "5",
                      "ESM1b 2X" = "6",
                      "ESM1b 3X" = "7") |>
           fct_relevel("ESM1b",
                       "ESM1b 2X",
                       "ESM1b 3X"))

data_summary_subset <- data_summary |>
  filter(experiment_index >= 5,
         experiment_index <= 7) |>
  mutate(experiment_name = experiment_index |>
           as_factor() |>
           fct_recode("ESM1b" = "5",
                      "ESM1b 2X" = "6",
                      "ESM1b 3X" = "7") |>
           fct_relevel("ESM1b",
                       "ESM1b 2X",
                       "ESM1b 3X"))

## AUC plot ---------------------------------------------------------------
p <- plot_performance(data = data_subset,
                      data_summary = data_summary_subset,
                      metric = auc)

ggsave(filename = "s05_plot__esm1b_sizes__auc.svg",
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

## AUC 0.1 plot
p <- plot_performance(data = data_subset,
                      data_summary = data_summary_subset,
                      metric = auc01)

ggsave(filename = "s05_plot__esm1b_sizes__auc01.svg",
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

## PPV plot ---------------------------------------------------------------
p <- plot_performance(data = data_subset,
                      data_summary = data_summary_subset,
                      metric = ppv)

ggsave(filename = "s05_plot__esm1b_sizes__ppv.svg",
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

## Table with delta mean performance metrics ------------------------------
delta_mean_performance_table <- get_delta_mean_performance_table(data_summary = data_summary_subset,
                                                                 experiment_index_base = 5)

delta_mean_performance_table |>
  write_tsv(file = "../results/s05_table__esm1b_sizes__delta_mean_performance.tsv")

## Table where sign of deltas are counted ---------------------------------
delta_performance_count_sign_table <- get_delta_performance_count_sign_table(data = data_subset,
                                                                             experiment_index_base = 5)

delta_performance_count_sign_table |>
  write_tsv(file = "../results/s05_table__esm1b_sizes__delta_performance_count_sign.tsv")


# ESM1b 3X - compare different number of epochs ---------------------------
