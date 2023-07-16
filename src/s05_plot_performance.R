#!/usr/bin/env Rscript
library(tidyverse)
library(svglite)
library(patchwork)
library(glue)

# Read data ---------------------------------------------------------------

data <- read_tsv(file = "../data/s04_performance.tsv",
                 col_types = cols(peptide = col_factor(),
                                  count_positive = col_integer()))

data_summary <- read_tsv(file = "../data/s04_performance_summary.tsv",
                         col_types = cols(statistic = col_factor()))

data <- data |>
  mutate(peptide_count_positive = str_glue("{peptide} ({count_positive})"))

data_summary <- data_summary |>
  mutate(statistic = if_else(condition = statistic == "mean",
                             true = "Mean",
                             false = "Weighted mean"),
         statistic_count_positive = str_glue("{statistic} ({count_positive})"))


# Functions ---------------------------------------------------------------

plot_performance <- function(data, data_summary, metric) {

  if (metric == "auc") {
    y_label <- "AUC"
  } else if (metric == "auc01") {
    y_label <- "AUC 0.1"
  } else if (metric == "ppv") {
    y_label <- "PPV"
  }
  
  p1 <- data |>
    ggplot(mapping = aes(x = peptide_count_positive |>
                           as_factor(),
                         y = .data[[metric]],
                         fill = experiment_index_name))+
    geom_col(position = "dodge")+
    labs(x = "Peptide",
         fill = "Experiment")+
    theme(axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
  
  
  p2 <- data_summary |>
    ggplot(mapping = aes(x = statistic_count_positive,
                         y = .data[[metric]],
                         fill = experiment_index_name))+
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
                  .fns = \(x) x - (data |>
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

subset_data <- function(data, experiment_index_to_name_list){
  experiment_index_to_index_name_list <- experiment_index_to_name_list
  experiment_index_to_index_name_list |>
    names() <- str_c(experiment_index_to_name_list,
                     experiment_index_to_name_list |>
                       names(),
                     sep = " - ")
  
  data |>
    filter(experiment_index %in% experiment_index_to_name_list) |>
    mutate(experiment_name = experiment_index |>
             as_factor() |>
             fct_recode(!!!experiment_index_to_name_list) |>
             fct_relevel(experiment_index_to_name_list |>
                           names()),
           experiment_index_name = experiment_index |>
             as_factor() |>
             fct_recode(!!!experiment_index_to_index_name_list) |>
             fct_relevel(experiment_index_to_index_name_list |>
                           names()))
}

pwalk(.l = list(experiment_index_to_name_list = list(c("No dropout and no weighting" = "4",
                                                       "Weighting only" = "2",
                                                       "Dropout only" = "3",
                                                       "Dropout and weighting" = "1"),
                                                     c("BLOSUM50" = "1",
                                                       "ESM1b" = "5"),
                                                     c("ESM1b" = "5",
                                                       "ESM1b 2X" = "6",
                                                       "ESM1b 3X" = "7"),
                                                     c("200 epochs" = "7",
                                                       "400 epochs" = "8"),
                                                     c("CNN" = "5",
                                                       "FFNN" = "9"),
                                                     c("No dropout" = "10",
                                                       "With dropout" = "9"),
                                                     c("ESM1b" = "5",
                                                       "Protrans T5-XL-U50" = "11"),
                                                     c("Normalization divisor of 20" = "11",
                                                       "Normalization divisor of 1" = "12"),
                                                     c("ESM1b - TCR only" = "5",
                                                       "ESM1b - both" = "13"),
                                                     c("BLOSUM50" = "1",
                                                       "ESM1b" = "6",
                                                       "Protrans T5-XL-U50" = "11")),
                name = c("dropout_and_weighting",
                         "esm1b_vs_blosum50",
                         "esm1b_sizes",
                         "esm1b3x_epochs",
                         "esm1b_cnn_vs_ffnn",
                         "esm1b_ffnn_dropout",
                         "esm1b_vs_protrans_t5-xl-u50",
                         "protrans_t5-xl-u50_normalization_divisor",
                         "esm1b_peptide",
                         "best_embeddings"),
                experiment_index_base = c(4,
                                          1,
                                          5,
                                          7,
                                          5,
                                          10,
                                          5,
                                          11,
                                          5,
                                          1)),
      .f =  \(experiment_index_to_name_list,
              name,
              experiment_index_base){
        
        # Subset data
        data_subset <- data |>
          subset_data(experiment_index_to_name_list = experiment_index_to_name_list)
        
        data_summary_subset <- data_summary |>
          subset_data(experiment_index_to_name_list = experiment_index_to_name_list)
        
        # Plotting
        walk(.x = c("auc",
                    "auc01",
                    "ppv"),
             .f = \(metric){
               p <- plot_performance(data = data_subset,
                                     data_summary = data_summary_subset,
                                     metric = metric)
               
               ggsave(filename = glue('s05_plot__{name}__{metric}.svg'),
                      plot = p,
                      path = "../results",
                      width = 30,
                      height = 20,
                      units = "cm")
             })
        
        # Table with delta mean performance metrics
        delta_mean_performance_table <- get_delta_mean_performance_table(data_summary = data_summary_subset,
                                                                         experiment_index_base = experiment_index_base)
        
        delta_mean_performance_table |>
          write_tsv(file = glue('../results/s05_table__{name}__delta_mean_performance.tsv'))
        
        # Table where sign of deltas are counted
        delta_performance_count_sign_table <- get_delta_performance_count_sign_table(data = data_subset,
                                                                                     experiment_index_base = experiment_index_base)
        
        delta_performance_count_sign_table |>
          write_tsv(file = glue('../results/s05_table__{name}__delta_performance_count_sign.tsv'))
      })