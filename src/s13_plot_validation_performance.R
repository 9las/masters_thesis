#!/usr/bin/env Rscript
library(tidyverse)
library(glue)

# Read data ---------------------------------------------------------------

data <- read_tsv(file = "../data/s12_validation_performance_summary.tsv")

data <- data |>
  mutate(test_validation_partition = str_glue("t{test_partition}v{validation_partition}"))

# ESM1b 2X
data_filtered <- data |>
  filter(model_index %in% c(25, 35, 112:120))

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x__auc.svg'),
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

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x__auc01.svg'),
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

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x__ppv.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

# ESM1b 2X - half learning rate
data_filtered <- data |>
  filter(model_index %in% c(121:128))

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x_half_learning_rate__auc.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc01)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x_half_learning_rate__auc01.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = ppv)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x_half_learning_rate__ppv.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

# ESM2 t33 2X
data_filtered <- data |>
  filter(model_index %in% c(129:136))

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__esm2_t33_650M_UR50D_2x__auc.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc01)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__esm2_t33_650M_UR50D_2x__auc01.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = ppv)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__esm2_t33_650M_UR50D_2x__ppv.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")

# BLOSUM50_20aa
data_filtered <- data |>
  filter(model_index %in% c(1, 27, 75:89))

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__blosum50_20aa__auc.svg'),
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

ggsave(filename = glue('s13_plot__normalisation__blosum50_20aa__auc01.svg'),
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

ggsave(filename = glue('s13_plot__normalisation__blosum50_20aa__ppv.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")


# ESM1b
data_filtered <- data |>
  filter(model_index %in% c(5, 17:18, 36:40, 42:54))

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
  filter(model_index %in% c(10, 21, 26, 55:71))

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

# ESM2 t33
data_filtered <- data |>
  filter(model_index %in% c(13, 92:108))

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross")

ggsave(filename = glue('s13_plot__normalisation__esm2_t33_650M_UR50D__auc.svg'),
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

ggsave(filename = glue('s13_plot__normalisation__esm2_t33_650M_UR50D__auc01.svg'),
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

ggsave(filename = glue('s13_plot__normalisation__esm2_t33_650M_UR50D__ppv.svg'),
       plot = p,
       path = "../results",
       width = 30,
       height = 20,
       units = "cm")