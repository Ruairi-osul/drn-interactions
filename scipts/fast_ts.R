library(tidyverse)
library(patchwork)
library(bbplot)
library(latex2exp)
library(lme4)
library(lmerTest)
library(patchwork)
library(emmeans)
library(arrow)
library(pander)
library(clipr)


####### Load data

data_dir <- "/Users/ruairiosullivan/repos/DRN Interactions/data/derived"


read_csv("data/derived/neuron_types.csv") %>% 
  mutate(neuron_type = factor(neuron_type, levels=c("SR", "SIR", "FF"), ordered = T)) -> neuron_types

read_parquet(
  file.path(data_dir, "base_shock_counts.parquet")
)  %>%
  mutate(Window = factor(if_else(window == 'pre', 'Pre', "Post"), ordered=T))  %>%
  mutate(counts = counts * 100) %>%
  left_join(select(neuron_types, neuron_type, neuron_id)) %>%
  drop_na() -> fast_counts


### SR
mod_lmer <- lmer(
  counts ~ Window  + (Window | neuron_id),
  data=filter(fast_counts, neuron_type == "SR")
)
anova(mod_lmer)
ranova(mod_lmer, reduce.terms = T)


mod_lmer <- lmer(
  counts ~ Window  + (Window | neuron_id),
  data=filter(fast_counts, neuron_type == "SIR")
)
anova(mod_lmer)
ranova(mod_lmer, reduce.terms = T)

mod_lmer <- lmer(
  counts ~ Window  + (Window | neuron_id),
  data=filter(fast_counts, neuron_type == "FF")
)
anova(mod_lmer)
ranova(mod_lmer, reduce.terms = T)
