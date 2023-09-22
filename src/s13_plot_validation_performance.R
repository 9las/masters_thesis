#!/usr/bin/env Rscript
library(tidyverse)
library(glue)

# Read data ---------------------------------------------------------------

data <- read_tsv(file = "../data/s12_validation_performance_summary.tsv")

data <- data |>
  mutate(test_validation_partition = str_glue("t{test_partition}v{validation_partition}"))

# ESM1b 2X (sigmoid)

data_filtered <- data |>
  filter(model_index %in% c(19, 25, 34, 35))

p <- data_filtered |>
  ggplot(mapping = aes(x = tcr_normalization_divisor,
                       y = auc,
                       color = test_validation_partition,
                       group = test_validation_partition)) +
  geom_line()+
  geom_point()

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x_sigmoid__auc.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

p <- data_filtered |>
  ggplot(mapping = aes(x = tcr_normalization_divisor,
                       y = auc01,
                       color = test_validation_partition,
                       group = test_validation_partition)) +
  geom_line()+
  geom_point()

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x_sigmoid__auc01.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

p <- data_filtered |>
  ggplot(mapping = aes(x = tcr_normalization_divisor,
                       y = ppv,
                       color = test_validation_partition,
                       group = test_validation_partition)) +
  geom_line()+
  geom_point()

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x_sigmoid__ppv.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

# ESM1b
data_filtered <- data |>
  filter(model_index %in% c(5, 36, 37 ,38, 42, 43, 44, 45, 17, 18, 39, 40, 46, 47, 48, 49, 50, 51, 52, 53 ,54))

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__esm1b__auc.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc01,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__esm1b__auc01.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = ppv,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__esm1b__ppv.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")



# Prottrans
data_filtered <- data |>
  filter(model_index %in% c(10, 21, 26, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71))

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__prottrans_t5_xl_u50__auc.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc01,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__prottrans_t5_xl_u50__auc01.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = ppv,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__prottrans_t5_xl_u50__ppv.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")