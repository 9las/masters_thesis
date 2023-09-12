#!/usr/bin/env Rscript
library(tidyverse)
library(glue)

# Read data ---------------------------------------------------------------

data <- read_tsv(file = "../data/s12_validation_performance_summary.tsv",
                 col_types = cols(tcr_normalization_divisor = col_factor()))

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