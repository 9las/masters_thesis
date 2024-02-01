#!/usr/bin/env Rscript
library(tidyverse)
library(fs)
library(glue)
library(argparser)

parser <- arg_parser(description = "Plot training history")
parser <- parser |>
  add_argument(arg = "model_index",
               help = "Index for model to plot training history for")

args <- parse_args(parser)
model_index <- args$model_index

theme_set(theme_gray(base_size = 9))

file_paths <- dir_ls(path = "../results",
                     glob = glue('../results/s02_m{model_index}_training_history_t?v?.tsv'))

data <- read_tsv(file = file_paths,
                 id = "path")

plot_training_history <- function(data,
                                  data_max_auc01,
                                  model_index,
                                  metric){
  if (metric == "auc") {
    y_label <- "AUC"
  } else if (metric == "auc01") {
    y_label <- "AUC 0.1"
  } else if (metric == "loss") {
    y_label <- "Loss"
  }
  
  data |>
    select(epoch,
           ends_with(metric),
           test_set_index,
           validation_set_index,
           test_validation_set_index) |>
    pivot_longer(cols = ends_with(metric),
                 names_to = "data_set",
                 values_to = metric) |>
    mutate(data_set = if_else(condition = data_set |>
                                str_starts(pattern = "val_"),
                              true = "Validation",
                              false = "Training")) |>
    ggplot(mapping = aes(x = epoch,
                         y = .data[[metric]],
                         color = data_set))+
    geom_vline(mapping = aes(xintercept = epoch),
               data = data_max_auc01)+
    geom_line()+
    facet_wrap(facets = vars(test_validation_set_index),
               ncol = 4)+
    labs(x = "Epoch",
         y = y_label,
         color = "Set")+
    theme(legend.position = "top",
          legend.justification = "left",
          strip.text.x = element_blank(),
          plot.margin = margin(t = 0.3, r = 0.3, b = 0.3, l = 0.3, "cm"))
}

data <- data |>
  mutate(model_index = path |>
           str_extract(pattern = "(?<=_m)\\d{3}(?=_)") |>
           str_remove(pattern = "^0+") |>
           as.numeric(),
         test_set_index = path |>
           str_extract(pattern = "(?<=_t)\\d{1}"),
         validation_set_index = path |>
           str_extract(pattern = "(?<=v)\\d{1}"),
         test_validation_set_index = glue('t{test_set_index}v{validation_set_index}'),
         .before = path) |>
  select(!path) |>
  group_by(model_index) |>
  nest() |>
  mutate(data_max_auc01 = data |>
           map(.f = \(x) x |>
                 group_by(test_set_index,
                          validation_set_index) |>
                 slice_max(val_auc01) |>
                 select(test_set_index,
                        validation_set_index,
                        test_validation_set_index,
                        epoch)))

pwalk(.l = list(model_index = data |>
                  pull(model_index),
                data = data |>
                  pull(data),
                data_max_auc01 = data |>
                  pull(data_max_auc01)),
      .f = \(model_index,
             data,
             data_max_auc01) {
        model_index_padded <- model_index |>
          str_pad(width = 3,
                  pad = 0)
        walk(.x = c("auc",
                    "auc01",
                    "loss"),
             .f = \(metric){
               p <- plot_training_history(data = data,
                                          data_max_auc01 = data_max_auc01,
                                          model_index = model_index,
                                          metric = metric)
               
               ggsave(filename = glue('s03_m{model_index_padded}_plot__training_history__{metric}.png'),
                      plot = p,
                      path = "../results",
                      width = 16,
                      height = 10,
                      units = "cm",
                      dpi = 600)
               
               p <- p+
                 theme_gray(base_size = 11)
               
               ggsave(filename = glue('s03_m{model_index_padded}_plot__training_history__{metric}.svg'),
                      plot = p,
                      path = "../results",
                      width = 30,
                      height = 20,
                      units = "cm")
             })
      })