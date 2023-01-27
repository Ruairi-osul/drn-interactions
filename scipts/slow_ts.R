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
  file.path(data_dir, "slow_ts_fs_counts.parquet")
)  %>%
  mutate(Block = factor(if_else(bin <= 0, "Pre", if_else(bin <=600, "Shock", "Post")), levels=c("Pre", "Shock", "Post"), ordered=T))  %>%
  mutate(counts = counts / 30) %>%
  left_join(select(neuron_types, neuron_type, neuron_id)) %>%
  drop_na() -> slow_counts

slow_counts 

## MODS
mod_lmer <- lmer(
  counts ~ Block  + (Block | neuron_id),
  data=filter(slow_counts, neuron_type == "SR")
)
anova(mod_lmer)
lmerTest::ranova(mod_lmer, reduce.terms =F)


mod_lmer <- lmer(
  counts ~ Block  + (Block | neuron_id),
  data=filter(slow_counts, neuron_type == "FF")
)
anova(mod_lmer)
lmerTest::ranova(mod_lmer, reduce.terms =F)


mod_lm <- lm(
  counts ~ Block,
  data=filter(slow_counts, neuron_type == "SR")
)
anova(mod_lm)
ems <- emmeans(mod_lm, ~Block, pbkrtest.limit = 6549)
pairs(ems)



mod_lmer <- lmer(
  counts ~ Block  + (Block | neuron_id),
  data=filter(slow_counts, neuron_type == "SIR")
)
anova(mod_lmer)
lmerTest::ranova(mod_lmer, reduce.terms =F)
mod_lm <- lm(
  counts ~ Block,
  data=filter(slow_counts, neuron_type == "SIR")
)
anova(mod_lm)
ems <- emmeans(mod_lm, ~Block, pbkrtest.limit = 6549)
pairs(ems)
ems


mod_lmer <- lmer(
  counts ~ Block  + (Block | neuron_id),
  data=filter(slow_counts, neuron_type == "FF")
)
anova(mod_lmer)
lmerTest::ranova(mod_lmer, reduce.terms =F)
mod_lm <- lm(
  counts ~ Block,
  data=filter(slow_counts, neuron_type == "FF")
)
anova(mod_lm)
ems <- emmeans(mod_lm, ~Block, pbkrtest.limit = 6549)
pairs(ems)
ems
