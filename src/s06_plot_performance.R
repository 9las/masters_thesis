#!/usr/bin/env Rscript
library(tidyverse)
library(svglite)
library(patchwork)
library(glue)
library(broom)

# Settings ----------------------------------------------------------------
theme_set(theme_gray(base_size = 9))

# Read data ---------------------------------------------------------------

data <- read_tsv(file = "../data/s05_performance.tsv",
                 col_types = cols(peptide = col_factor(),
                                  count_positive = col_integer()))

data_summary <- read_tsv(file = "../data/s05_performance_summary.tsv",
                         col_types = cols(statistic = col_factor()))

data <- data |>
  mutate(peptide_count_positive = str_glue("{peptide} ({count_positive})"))

data_summary <- data_summary |>
  mutate(statistic = if_else(condition = statistic == "mean",
                             true = "Mean",
                             false = "Weighted\ Mean"),
         statistic_count_positive = str_glue("{statistic} ({count_positive})"))


# Functions ---------------------------------------------------------------
subset_data <- function(data, model_index_to_name_list){
  model_index_to_index_name_list <- model_index_to_name_list
  model_index_to_index_name_list |>
    names() <- str_c(model_index_to_name_list,
                     model_index_to_name_list |>
                       names(),
                     sep = " - ")
  
  data |>
    filter(model_index %in% model_index_to_name_list) |>
    mutate(model_name = model_index |>
             as_factor() |>
             fct_recode(!!!model_index_to_name_list) |>
             fct_relevel(model_index_to_name_list |>
                           names()),
           model_index_name = model_index |>
             as_factor() |>
             fct_recode(!!!model_index_to_index_name_list) |>
             fct_relevel(model_index_to_index_name_list |>
                           names()))
}

plot_performance <- function(data, data_summary, metric, index_model){
  
  if (metric == "auc") {
    y_label <- "AUC"
  } else if (metric == "auc01") {
    y_label <- "AUC 0.1"
  } else if (metric == "ppv") {
    y_label <- "PPV"
  }
  
  if (index_model) {
    model_name = "model_index_name"
  } else {
    model_name = "model_name"
  }
  
  p1 <- data |>
    ggplot(mapping = aes(x = peptide_count_positive |>
                           as_factor(),
                         y = .data[[metric]],
                         fill = .data[[model_name]]))+
    geom_col(position = "dodge")+
    labs(x = "Peptide",
         fill = "Model")+
    theme(axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
  
  
  p2 <- data_summary |>
    ggplot(mapping = aes(x = statistic_count_positive,
                         y = .data[[metric]],
                         fill = .data[[model_name]]))+
    geom_col(position = "dodge")+
    labs(x = "Statistic",
         y = y_label,
         fill = "Model")
  
  p2+
    p1+
    plot_layout(guides = "collect",
                widths = c(0.075, 0.925))&
    theme(legend.position = "top",
          legend.justification = "left",
          axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))&
    ylim(0, 1)
}

get_delta_performance_table <- function(data){
  data |>
    left_join(y = data |>
                select(!c(count_positive,
                          peptide_count_positive)),
              by = "peptide",
              relationship = "many-to-many") |>
    filter(model_index.x != model_index.y) |>
    mutate(auc_delta = auc.x - auc.y,
           auc01_delta = auc01.x - auc01.y,
           ppv_delta = ppv.x - ppv.y)
}

get_delta_mean_performance_table <- function(data_summary){
  data_summary |>
    left_join(y = data_summary |>
                select(!c(count_positive,
                          statistic_count_positive)),
              by = "statistic",
              relationship = "many-to-many") |>
    filter(model_index.x != model_index.y) |>
    mutate(auc_delta = auc.x - auc.y,
           auc01_delta = auc01.x - auc01.y,
           ppv_delta = ppv.x - ppv.y)
}

get_delta_performance_count_sign_table <- function(data_delta){
  data_delta |>
    group_by(model_index.x,
             model_name.x,
             model_index.y,
             model_name.y) |>
    summarise(across(.cols = c(auc_delta,
                               auc01_delta,
                               ppv_delta),
                     .fns = list(count.greater = \(x) (x > 0) |>
                                   sum(),
                                 count.lesser = \(x) (x <= 0) |>
                                   sum()),
                     .names = "{.col}.{.fn}")) |>
    pivot_longer(cols = !c(model_index.x,
                           model_name.x,
                           model_index.y,
                           model_name.y),
                 names_to = c("metric",
                              ".value"),
                 names_sep = "\\_") |>
    mutate(binom_test = map2(.x = delta.count.greater,
                             .y = delta.count.lesser,
                             .f = \(x, y){
                               c(x, y) |>
                                 binom.test(alternative = "greater",
                                            conf.level = 0.95) |>
                                 tidy()
                             }
    )) |>
    unnest(binom_test) |>
    select(!c(statistic,
              parameter,
              method,
              alternative)) |>
    arrange(model_index.x,
            model_index.y,
            metric)
}

plot_delta_performance <- function(data_delta, data_summary_delta, metric){
  
  metric <- str_c(metric,
                  "delta",
                  sep = "_")
  
  if (metric == "auc_delta") {
    y_label <- "\u0394AUC"
  } else if (metric == "auc01_delta") {
    y_label <- "\u0394AUC 0.1"
  } else if (metric == "ppv_delta") {
    y_label <- "\u0394PPV"
  }
  
  min_metric_delta <- c(data_delta |>
                       pull(.data[[metric]]),
                     data_summary_delta |>
                       pull(.data[[metric]])) |>
    min()
  
  max_metric_delta <- c(data_delta |>
                       pull(.data[[metric]]),
                     data_summary_delta |>
                       pull(.data[[metric]])) |>
    max()
  
  p1 <- data_delta |>
    ggplot(mapping = aes(x = peptide_count_positive |>
                           as_factor(),
                         y = .data[[metric]],
                         fill = model_index_name.y))+
    geom_col(position = "dodge")+
    facet_wrap(facets = vars(model_index_name.x))+
    labs(x = "Peptide",
         fill = "Model")+
    ylim(min_metric_delta,
         max_metric_delta)+
    theme(axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
  
  p2 <- data_summary_delta |>
    ggplot(mapping = aes(x = statistic_count_positive,
                         y = .data[[metric]],
                         fill = model_index_name.y))+
    geom_col(position = "dodge")+
    facet_wrap(facets = vars(model_index_name.x))+
    labs(x = "Statistic",
         y = y_label,
         fill = "Model")+
    ylim(min_metric_delta,
         max_metric_delta)
  
  p2+
    p1+
    plot_layout(guides = "collect",
                widths = c(0.35, 0.65))&
    theme(legend.position = "top",
          legend.justification = "left",
          axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
}


plot_delta_performance_2 <- function(data_delta, data_summary_delta, metric, base_model_index, index_model){
  
  metric <- str_c(metric,
                  "delta",
                  sep = "_")
  
  if (metric == "auc_delta") {
    y_label <- "\u0394AUC"
  } else if (metric == "auc01_delta") {
    y_label <- "\u0394AUC 0.1"
  } else if (metric == "ppv_delta") {
    y_label <- "\u0394PPV"
  }
  
  if (index_model) {
    model_x_name = "model_index_name.x"
  } else {
    model_x_name = "model_name.x"
  }
  
  model_count <- data_delta |>
    pull(model_index.x) |>
    n_distinct()
  
  data_delta <- data_delta |>
    filter(model_index.y == base_model_index)
  
  data_summary_delta <- data_summary_delta |>
    filter(model_index.y == base_model_index)
  
  min_metric_delta <- c(data_delta |>
                          pull(.data[[metric]]),
                        data_summary_delta |>
                          pull(.data[[metric]])) |>
    min()
  
  max_metric_delta <- c(data_delta |>
                          pull(.data[[metric]]),
                        data_summary_delta |>
                          pull(.data[[metric]])) |>
    max()
  
  data_delta |> pull(model_index.x) |> n_distinct()
  
  if(model_count > 2){
    p1 <- data_delta |>
      ggplot(mapping = aes(x = peptide_count_positive |>
                             as_factor(),
                           y = .data[[metric]],
                           fill = .data[[model_x_name]]))
    p2 <- data_summary_delta |>
      ggplot(mapping = aes(x = statistic_count_positive,
                           y = .data[[metric]],
                           fill = .data[[model_x_name]]))
  } else {
    p1 <- data_delta |>
      ggplot(mapping = aes(x = peptide_count_positive |>
                             as_factor(),
                           y = .data[[metric]]))
    
    p2 <- data_summary_delta |>
      ggplot(mapping = aes(x = statistic_count_positive,
                           y = .data[[metric]]))
  }
  
  p1 <- p1 +
    geom_col(position = "dodge")+
    labs(x = "Peptide",
         fill = "Model")+
    ylim(min_metric_delta,
         max_metric_delta)+
    theme(axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
  
  p2 <- p2 +
    geom_col(position = "dodge")+
    labs(x = "Statistic",
         y = y_label,
         fill = "Model")+
    ylim(min_metric_delta,
         max_metric_delta)
  
  p2+
    p1+
    plot_layout(guides = "collect",
                widths = c(0.075, 0.925))&
    theme(legend.position = "top",
          legend.justification = "left",
          axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
}

# axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1

pwalk(.l = list(model_index_to_name_list = list(c("Weighting: -\nDropout: -" = "4",
                                                  "Weighting: +\nDropout: -" = "2",
                                                  "Weighting: -\nDropout: +" = "3",
                                                  "Weighting: +\nDropout: +" = "1"),
                                                c("BLOSUM50" = "1",
                                                  "ESM1-b" = "36",
                                                  "ProtT5-XL-U50" = "55",
                                                  "ESM-2" = "92"),
                                                c("BLOSUM50" = "1",
                                                  "ESM1-b" = "18",
                                                  "ProtT5-XL-U50" = "21",
                                                  "ESM-2" = "102"),
                                                c("CNN" = "18",
                                                  "FNN" = "8"),
                                                c("BLOSUM50" = "1",
                                                  "ESM1-b" = "122",
                                                  "ProtT5-XL-U50" = "139",
                                                  "ESM-2" = "130")),
                name = c("dropout_and_weighting",
                         "initial_embedding_comparison",
                         "comparsion_after_tuning_scaling_and_activation_function",
                         "esm1b_cnn_vs_ffnn",
                         "best_embeddings"),
                base_model_index = c(4,
                                     1,
                                     1,
                                     18,
                                     1)),
      .f =  \(model_index_to_name_list,
              name,
              base_model_index){
        
        # Subset data
        data_subset <- data |>
          subset_data(model_index_to_name_list = model_index_to_name_list)
        
        data_summary_subset <- data_summary |>
          subset_data(model_index_to_name_list = model_index_to_name_list)
        
        # Table with delta performance metrics
        data_subset_delta <- get_delta_performance_table(data = data_subset)
        
        # Table with delta mean performance metrics
        data_summary_subset_delta <- get_delta_mean_performance_table(data_summary = data_summary_subset)
        
        # Table where sign of deltas are counted
        data_count_sign <- get_delta_performance_count_sign_table(data_delta = data_subset_delta)
        
        # Write tables to files
        data_summary_subset_delta |>
          select(model_index.x,
                 model_name.x,
                 model_index.y,
                 model_name.y,
                 statistic,
                 auc_delta,
                 auc01_delta,
                 ppv_delta) |>
          arrange(model_index.x,
                  model_index.y,
                  statistic) |>
          write_tsv(file = glue('../results/s06_table__{name}__delta_mean_performance.tsv'))
        
        data_count_sign |>
          write_tsv(file = glue('../results/s06_table__{name}__delta_performance_count_sign.tsv'))
        
        # Plotting
        walk(.x = c("auc",
                    "auc01",
                    "ppv"),
             .f = \(metric){
               p <- plot_performance(data = data_subset,
                                     data_summary = data_summary_subset,
                                     metric = metric,
                                     index_model = TRUE)
               
               ggsave(filename = glue('s06_plot__{name}__{metric}.png'),
                      plot = p,
                      path = "../results",
                      width = 16,
                      height = 10.5,
                      units = "cm",
                      dpi = 600)
               
               p <- plot_performance(data = data_subset,
                                     data_summary = data_summary_subset,
                                     metric = metric,
                                     index_model = FALSE)
               
               ggsave(filename = glue('s06_plot__{name}__no_index__{metric}.png'),
                      plot = p,
                      path = "../results",
                      width = 16,
                      height = 10.5,
                      units = "cm",
                      dpi = 600)
               
               p <- plot_delta_performance_2(data_delta = data_subset_delta,
                                           data_summary_delta = data_summary_subset_delta,
                                           metric = metric,
                                           base_model_index = base_model_index,
                                           index_model = TRUE)
               
               ggsave(filename = glue('s06_plot__{name}__{metric}__delta.png'),
                      plot = p,
                      path = "../results",
                      width = 16,
                      height = 10.5,
                      units = "cm",
                      dpi = 600)
               
               p <- plot_delta_performance_2(data_delta = data_subset_delta,
                                             data_summary_delta = data_summary_subset_delta,
                                             metric = metric,
                                             base_model_index = base_model_index,
                                             index_model = FALSE)
               
               ggsave(filename = glue('s06_plot__{name}__no_index__{metric}__delta.png'),
                      plot = p,
                      path = "../results",
                      width = 16,
                      height = 10.5,
                      units = "cm",
                      dpi = 600)
             })
      })

# Handle Cases where there are only two models ----------------------------

# model_index_to_name_list <- c("CNN" = "18",
#                               "FNN" = "8")
# 
# name <- "esm1b_cnn_vs_ffnn"
# 
# base_model_index <- 18
# 
# # Subset data
# data_subset <- data |>
#   subset_data(model_index_to_name_list = model_index_to_name_list)
# 
# data_summary_subset <- data_summary |>
#   subset_data(model_index_to_name_list = model_index_to_name_list)
# 
# # Table with delta performance metrics
# data_subset_delta <- get_delta_performance_table(data = data_subset)
# 
# # Table with delta mean performance metrics
# data_summary_subset_delta <- get_delta_mean_performance_table(data_summary = data_summary_subset)
# 
# # Table where sign of deltas are counted
# data_count_sign <- get_delta_performance_count_sign_table(data_delta = data_subset_delta)
# 
# # Write tables to files
# data_summary_subset_delta |>
#   select(model_index.x,
#          model_name.x,
#          model_index.y,
#          model_name.y,
#          statistic,
#          auc_delta,
#          auc01_delta,
#          ppv_delta) |>
#   arrange(model_index.x,
#           model_index.y,
#           statistic) |>
#   write_tsv(file = glue('../results/s06_table__{name}__delta_mean_performance.tsv'))
# 
# data_count_sign |>
#   write_tsv(file = glue('../results/s06_table__{name}__delta_performance_count_sign.tsv'))
# 
# walk(.x = c("auc",
#             "auc01",
#             "ppv"),
#      .f = \(metric){
#        p <- plot_performance(data = data_subset,
#                              data_summary = data_summary_subset,
#                              metric = metric)
#        
#        ggsave(filename = glue('s06_plot__{name}__{metric}.svg'),
#               plot = p,
#               path = "../results",
#               width = 30,
#               height = 20,
#               units = "cm")
#        
#        p <- plot_delta_performance_2(data_delta = data_subset_delta,
#                                      data_summary_delta = data_summary_subset_delta,
#                                      metric = metric,
#                                      base_model_index = base_model_index)
#        
#        ggsave(filename = glue('s06_plot__{name}__{metric}__delta.svg'),
#               plot = p,
#               path = "../results",
#               width = 30,
#               height = 20,
#               units = "cm")
#      })