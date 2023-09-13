#!/usr/bin/env Rscript
library(tidyverse)
library(glue)

# Read data ---------------------------------------------------------------

data <- read_tsv(file = "../data/s12_validation_performance_summary.tsv")

data <- data |>
  mutate(test_validation_partition = str_glue("t{test_partition}v{validation_partition}"))

data |>
  filter(model_index %in% c(19, 25, 34)) |>
  ggplot(mapping = aes(x = tcr_normalization_divisor,
                       y = auc,
                       color = test_validation_partition,
                       group = test_validation_partition)) +
  geom_line()+
  geom_point()

data |>
  filter(model_index %in% c(5, 16, 36, 37 ,38)) |>
  ggplot(mapping = aes(x = tcr_normalization_divisor,
                       y = auc,
                       color = test_validation_partition,
                       group = test_validation_partition)) +
  geom_line()+
  geom_point()

data |>
  filter(model_index %in% c(17, 18, 39, 40 ,41)) |>
  ggplot(mapping = aes(x = tcr_normalization_divisor,
                       y = auc,
                       color = test_validation_partition,
                       group = test_validation_partition)) +
  geom_line()+
  geom_point()