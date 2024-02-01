#!/usr/bin/env Rscript
library(tidyverse)
library(glue)
# library(MASS)

# Settings ----------------------------------------------------------------
theme_set(theme_gray(base_size = 9))

# Read data ---------------------------------------------------------------

data <- read_tsv(file = "../data/s12_validation_performance_summary.tsv")

data <- data |>
  mutate(test_validation_partition = str_glue("t{test_partition}v{validation_partition}"),
         cdr_conv_activation = if_else(condition = cdr_conv_activation == "relu",
                                       true = "ReLU",
                                       false = "Sigmoid"),
         architecture_size = if_else(condition = hidden_units_count == 64,
                                     true = "1X",
                                     false = "2X"))


# Functions ---------------------------------------------------------------
# divisor_to_factor <- function(divisor){
#   1 / as.numeric(divisor) |>
#     fractions()
# }

divisor_to_factor <- function(divisor){
  divisor |>
    map(.f = \(x){
      if (x == 1){
        factor_ <- x
      } else {
        factor_ <- glue('frac(1,{x})')
      }
      factor_
    }) |>
    parse(text = _)
}

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
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC",
       color = "Activation Function")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__blosum50_20aa__auc.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc01,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC 0.1",
       color = "Activation Function")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__blosum50_20aa__auc01.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = ppv,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "PPV",
       color = "Activation Function")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__blosum50_20aa__ppv.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)


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
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC",
       color = "Activation Function")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm1b__auc.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc01,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC 0.1",
       color = "Activation Function")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm1b__auc01.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = ppv,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "PPV",
       color = "Activation Function")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm1b__ppv.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

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
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC",
       color = "Activation Function")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__prottrans_t5_xl_u50__auc.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc01,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC 0.1",
       color = "Activation Function")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__prottrans_t5_xl_u50__auc01.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = ppv,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "PPV",
       color = "Activation Function")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__prottrans_t5_xl_u50__ppv.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

# ESM2 t33
data_filtered <- data |>
  filter(model_index %in% c(13, 92:104, 106:108))

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC",
       color = "Activation Function")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm2_t33_650M_UR50D__auc.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc01,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC 0.1",
       color = "Activation Function")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm2_t33_650M_UR50D__auc01.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = ppv,
                       color = cdr_conv_activation)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "PPV",
       color = "Activation Function")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm2_t33_650M_UR50D__ppv.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

# ESM1b FFNN - dropout
data_filtered <- data |>
  filter(model_index %in% c(8:9))

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(dropout_rate),
                       y = auc)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  labs(x = "Dropout Rate",
       y = "AUC")

ggsave(filename = glue('s13_plot__ffnn_dropout__esm1b__auc.png'),
       plot = p,
       path = "../results",
       width = 9,
       height = 6.75,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(dropout_rate),
                       y = auc01)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  labs(x = "Dropout Rate",
       y = "AUC 0.1")

ggsave(filename = glue('s13_plot__ffnn_dropout__esm1b__auc01.png'),
       plot = p,
       path = "../results",
       width = 9,
       height = 6.75,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(dropout_rate),
                       y = ppv)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  labs(x = "Dropout Rate",
       y = "PPV")

ggsave(filename = glue('s13_plot__ffnn_dropout__esm1b__ppv.png'),
       plot = p,
       path = "../results",
       width = 9,
       height = 6.75,
       units = "cm",
       dpi = 600)

# ESM1b 2X
data_filtered <- data |>
  filter(model_index %in% c(5, 6, 17:18, 36:40, 42:54, 25, 35, 112:120))

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc,
                       color = architecture_size)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  facet_grid(cols = vars(cdr_conv_activation),
             scales = "free_x",
             space = "free_x")+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC",
       color = "Architecture Size")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x__auc.png'),
       plot = p,
       path = "../results",
       width = 16,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc01,
                       color = architecture_size)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  facet_grid(cols = vars(cdr_conv_activation),
             scales = "free_x",
             space = "free_x")+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC 0.1",
       color = "Architecture Size")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x__auc01.png'),
       plot = p,
       path = "../results",
       width = 16,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = ppv,
                       color = architecture_size)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  facet_grid(cols = vars(cdr_conv_activation),
             scales = "free_x",
             space = "free_x")+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "PPV",
       color = "Architecture Size")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x__ppv.png'),
       plot = p,
       path = "../results",
       width = 16,
       height = 10.5,
       units = "cm",
       dpi = 600)

# ESM1b 2X - half learning rate
data_filtered <- data |>
  filter(model_index %in% c(25, 35, 17:18, 39:40, 46:49, 121:128))

data_filtered <- data_filtered |>
  mutate(model = case_when(hidden_units_count == 64 ~ "1X",
                           hidden_units_count == 128 & learning_rate == 0.001 ~ "2X",
                           hidden_units_count == 128 & learning_rate == 0.0005 ~ "2X - Half Learning Rate"))

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc,
                       color = model)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC",
       color = "Model")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x_half_learning_rate__auc.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc01,
                       color = model)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC 0.1",
       color = "Model")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x_half_learning_rate__auc01.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = ppv,
                       color = model)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "PPV",
       color = "Model")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm1b_2x_half_learning_rate__ppv.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

# ESM2 t33 2X
data_filtered <- data |>
  filter(model_index %in% c(98:104, 129:136))

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc,
                       color = architecture_size)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC",
       color = "Architecture Size")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm2_t33_650M_UR50D_2x__auc.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc01,
                       color = architecture_size)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC 0.1",
       color = "Architecture Size")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm2_t33_650M_UR50D_2x__auc01.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = ppv,
                       color = architecture_size)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "PPV",
       color = "Architecture Size")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__esm2_t33_650M_UR50D_2x__ppv.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

# Prottrans 2x
data_filtered <- data |>
  filter(model_index %in% c(21, 26, 67:71, 22, 139:145))

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc,
                       color = architecture_size)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC",
       color = "Architecture Size")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__prottrans_t5_xl_u50_2x__auc.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = auc01,
                       color = architecture_size)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "AUC 0.1",
       color = "Architecture Size")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__prottrans_t5_xl_u50_2x__auc01.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)

p <- data_filtered |>
  ggplot(mapping = aes(x = factor(tcr_normalization_divisor),
                       y = ppv,
                       color = architecture_size)) +
  geom_boxplot()+
  stat_summary(fun=mean,
               geom="point",
               shape = "cross",
               position = position_dodge(width = 0.75))+
  scale_x_discrete(labels = divisor_to_factor)+
  labs(x = "Scaling Factor",
       y = "PPV",
       color = "Architecture Size")+
  theme(legend.position = "top",
        legend.justification = "left")

ggsave(filename = glue('s13_plot__normalisation__prottrans_t5_xl_u50_2x__ppv.png'),
       plot = p,
       path = "../results",
       width = 14,
       height = 10.5,
       units = "cm",
       dpi = 600)
