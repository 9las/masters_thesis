#!/usr/bin/env Rscript
library(tidyverse)
library(patchwork)
library(scales)
library(yardstick)

# Settings ----------------------------------------------------------------
theme_set(theme_gray(base_size = 9))

# Activation functions ----------------------------------------------------

logistic <- function(x){
  plogis(q = x)
}

logistic_derivative <- function(x){
  logistic(x)*(1 - logistic(x))
}

relu <- function(x){
  ((x > 0) * x) |>
    as.numeric()
}

relu_derivative <- function(x){
  (x > 0) |>
    as.numeric()
}

logisitc_functions <- tibble(z = seq(from = -8,
                                       to = 8,
                                       by = 0.1),
                               logistic = logistic(z),
                               logistic_derivative = logistic_derivative(z)) |>
  pivot_longer(cols = !z)

relu_functions <- tibble(z = seq(from = -8,
                                     to = 8,
                                     by = 0.1),
                             relu = relu(z),
                         relu_derivative = relu_derivative(z)) |>
  mutate(piece = if_else(condition = relu_derivative == 0,
                         true = 1,
                         false = 2)) |>
  pivot_longer(cols = c(relu,
                        relu_derivative))

p1 <- logisitc_functions |>
  ggplot(mapping = aes(x = z,
                       y = value,
                       color = name,
                       linetype = name))+
  geom_line()+
  scale_linetype_manual(values = c("solid",
                                 "dashed"),
                        labels = expression(sigma(z),sigma*"′"*(z)))+
  scale_color_discrete(labels = expression(sigma(z),sigma*"′"*(z)))+
  theme(legend.position = "top",
        legend.justification = "left",
        axis.title.y = element_blank(),
        legend.title = element_blank())

color_palette <- hue_pal()(2)
p2 <- tibble(z = c(-8, 8, -8, 8),
       y = c(0, 8, 0, 8),
       name = c("relu", "relu", "relu_derivative", "relu_derivative")) |> 
  ggplot(aes(x = z,
             y = y,
             color = name,
             linetype = name))+
  geom_line(alpha = 0)+
  annotate(geom = "segment",
           x = -8,
           y = 0,
           xend = 0,
           yend = 0,
           color = color_palette |>
             pluck(1))+
  annotate(geom = "segment",
           x = 0,
           y = 0,
           xend = 8,
           yend = 8,
           color = color_palette |>
             pluck(1))+
  annotate(geom = "segment",
           x = -8,
           y = 0,
           xend = 0,
           yend = 0,
           color = color_palette |>
             pluck(2),
           linetype = "dashed")+
  annotate(geom = "segment",
           x = 0,
           y = 1,
           xend = 8,
           yend = 1,
           color = color_palette |>
             pluck(2),
           linetype = "dashed")+
  annotate(geom = "point",
           x = 0,
           y = 0,
           color = color_palette |>
             pluck(2))+
  annotate(geom = "point",
           x = 0,
           y = 1,
           color = color_palette |>
             pluck(2),
           shape = 1)+
  guides(color = guide_legend(override.aes = list(alpha=1)))+
  scale_linetype_manual(values=c("solid",
                                 "dashed"),
                        labels = expression(ReLU(z),ReLU*"′"*(z)))+
  scale_color_discrete(labels = expression(ReLU(z),ReLU*"′"*(z)))+
  theme(legend.position = "top",
        legend.justification = "left",
        axis.title.y = element_blank(),
        legend.title = element_blank())

p <- p1 + p2 +
  plot_annotation(tag_levels = 'A')

ggsave(filename = "s15_plot__activation_functions.png",
       plot = p,
       path = "../results",
       width = 14,
       height = 6,
       units = "cm",
       dpi = 600,
       device=png)


# ROC curve ---------------------------------------------------------------

get_random_score <- function(positve_class){
  positve_class_length <- length(positve_class)
  positve_class * rbeta(n = positve_class_length,
                        shape1 = 6,
                        shape2 = 3) +
    (1 - positve_class) * rbeta(n = positve_class_length,
                                shape1 = 3,
                                shape2 = 6)
}

data <- tibble(actual_class = rep(x = c(FALSE,
                                        TRUE),
                                  times = c(1000,
                                            1000)),
               score = actual_class |>
                 get_random_score())

roc_curve_data <- data |>
  roc_curve(truth = actual_class |>
              factor(levels = c(TRUE,
                                FALSE)),
            score) |>
  mutate(fpr = 1 - specificity)

auc <- data |>
  roc_auc(truth = actual_class |>
              factor(levels = c(TRUE,
                                FALSE)),
            score)


p <- roc_curve_data |>
  ggplot(mapping = aes(x = fpr,
                       y = sensitivity))+
  geom_line()+
  annotate(geom = "segment",
           x = 0,
           y = 0,
           xend = 1,
           yend = 1,
           linetype = "dashed")+
  labs(x = "False Positive Rate",
       y = "True Positive Rate")

ggsave(filename = "s15_plot__roc_curve.png",
       plot = p,
       path = "../results",
       width = 6,
       height = 6,
       units = "cm",
       dpi = 600,
       device=png)

p1 <- data |>
  ggplot(mapping = aes(x = score,
                       color = actual_class))+
  geom_freqpoly(bins = 22)+
  scale_x_continuous(limits = c(0, 1))+
  theme(axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x=element_blank(),
        legend.position = "top",
        legend.justification = "left")

p2 <- data |>
  ggplot(mapping = aes(x = score,
                       y = 0,
                       color = actual_class))+
  geom_jitter()+
  scale_y_continuous(breaks = NULL)+
  scale_x_continuous(limits = c(0, 1))+
  theme(axis.title.y = element_blank(),
        legend.position = "none")

p <- p1 / p2 +
  plot_layout(heights = c(3, 1))


precision_top <- function(data, truth, score){
  positive_count <- data |>
    pull({{truth}}) |>
    sum()
  
  data <- data |>
    arrange(desc({{score}})) |>
    head(n = positive_count)
  
  postive_top_count <- data |>
    pull({{truth}}) |>
    sum()
  
  precision_top <- postive_top_count / positive_count
  
  threshold <- data |>
    pull({{score}}) |>
    min()
  
  tibble(precision = precision_top,
         threshold = threshold)
}

ppv_top <- data |>
  precision_top(truth = actual_class,
                score = score)

ppv <- data |>
  precision(truth = actual_class |>
              factor(levels = c(TRUE,
                                FALSE)),
            estimate = (score >= 0.446) |>
              factor(levels = c(TRUE,
                                FALSE)))

tpr <- data |>
  sens(truth = actual_class |>
         factor(levels = c(TRUE,
                           FALSE)),
       estimate = (score >= 0.446) |>
         factor(levels = c(TRUE,
                           FALSE)))