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
  filter(group_name %in% c('acute_cit', 'acute_citalopram', 'acute_sal', 'acute_citalopram')) %>%
  mutate(group = if_else(group_name %in% c('acute_cit', 'acute_citalopram'), 'CIT', 'SAL')) %>%
  mutate(neuron_type = factor(neuron_type, levels=c("SR", "SIR", "FF"), ordered = T)) -> neuron_types




read_parquet(
  file.path(data_dir, "chal_binned.parquet")
) %>%
  filter(bin < 1200) %>%
  mutate(Block = factor(if_else(bin <= 0, "Pre", "Drug"), levels=c("Pre", "Drug"), ordered=T)) %>% 
  mutate(counts = counts / 60) %>%
  left_join(select(neuron_types, group, neuron_type, neuron_id)) %>%
  drop_na() -> cit_counts


######

mod_lmer <- lmer(
  counts ~ block * neuron_type * group + (1 | neuron_type),
  data=cit_counts
)
anova(mod_lmer)

anova(mod_lmer) %>% write_clip()



mod <- lm(
  zcounts ~ block * neuron_type * group,
  data=cit_counts
)
as.data.frame(anova(mod)) %>%
  

emms <- emmeans(
  mod, ~ block | group | neuron_type,
)

emms %>% as_tibble()
prepost_by_type_by_group <- pairs(emms) %>% as_tibble()
prepost_by_type_by_group



cont <- pairs(pairs(emms), by='neuron_type')

cont
pairs(cont, by=NULL)


clipr::write_clip(cont)




read_parquet(
  file.path(data_dir, "way_binned.parquet")
) %>%
  filter(bin < 1200) %>%
  mutate(Block = factor(if_else(bin <= 0, "Pre", "Drug"), levels=c("Pre", "Drug"), ordered=T)) %>% 
  mutate(counts = counts / 60) %>%
  left_join(select(neuron_types, group, neuron_type, neuron_id)) %>%
  drop_na() -> way_counts

mod_lmer <- lmer(
  zcounts ~ block * neuron_type * group + (1 | neuron_type),
  data=way_counts
)
anova(mod_lmer)

anova(mod_lmer) %>% write_clip()

mod <- lm(
  zcounts ~ block * neuron_type * group,
  data=way_counts
)
anova(mod)
emms 


emms <- emmeans(
  mod, ~ block | group | neuron_type,
)

emms %>% as_tibble()
prepost_by_type_by_group <- pairs(emms) %>% as_tibble()
prepost_by_type_by_group %>% mutate(p = round(p.value, 3))


pairs(emms) 
pairs(pairs(emms) , by="neuron_type")
