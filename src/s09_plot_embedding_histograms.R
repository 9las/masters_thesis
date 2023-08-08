#!/usr/bin/env Rscript
library(tidyverse)

# Read data ---------------------------------------------------------------
data <- read_tsv(file = "../results/s08_embedding_histograms.tsv")