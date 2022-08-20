library(tidyverse)
library(emmeans)
library(lme4)
library(lmerTest)
library(bbplot)
library(latex2exp)
library(patchwork)
emm_options(lmerTest.limit = 6281)

theme_set(
  bbc_style() +
    theme(
      text=element_text(family="Arial"),
      strip.text.x = element_text(size = 10, angle=0, hjust=0.5, margin = margin(t = 0, r = 0, b = 10, l = 0)),
      strip.text.y = element_text(size = 10, angle=0, hjust=0.5, margin = margin(t = 0, r = 0, b = 10, l = 0)),
      panel.spacing = unit(1, "lines"),
      axis.text.x = element_text(size=10, angle=0),
      axis.text.y = element_text(size=9),
      legend.text = element_text(size=10),
      legend.key.size = unit(0.5, 'cm'),
      legend.position = "right",
      axis.line=element_line(),
      axis.title.y = element_text(size=10, angle=90, margin = margin(t = 0, r = 5, b = 0, l = 0)),
      panel.border = element_blank(),
      strip.background = element_blank(),
      plot.title = element_text(size=9, face="plain", margin = margin(t = 0, r = 0, b = 5, l = 0))
    )
)

preprocess_fs_block <- function(df){
  mutate(
    df, 
    block = factor(
      block, 
      levels=c("pre", "base_shock"), 
      labels=c("Pre", "Shock")
      )
    )
}
preprocess_response_fs_slow <- function(df){
  mutate(
    df,
    response_fs_slow = factor(
      response_fs_slow, 
      levels=c("inhibited", "activated"),
      labels=c("Shock-\nInhibited", "Shock-\nActivated")
    )
  )
}
preprocess_nt_combs <- function(df){
  df %>% mutate(
    nt_comb = factor(
      nt_comb, 
      levels=c("SR-SR", "SR-SIR", "SR-FF", "SIR-SIR", "SIR-FF", "FF-FF")
      )
    )
}
preprocess_neuron_types <- function(df){
  df %>% mutate(
    neuron_type = factor(
      neuron_type, 
      levels=c("SR", "SIR", "FF")
    )
  )
}
bin_width_distinct <- function(df){
  df %>% filter(.[["bin_width"]] == 1) %>% distinct()
}


df_responders <- read_csv("data/derived/graph/fs - responders.csv") %>%
  filter(response_fs_slow != "no_response")


df_node <- read_csv("data/derived/graph/fs - node.csv") %>% 
  bin_width_distinct() %>%
  left_join(df_responders) %>% 
  preprocess_fs_block() %>%
  preprocess_response_fs_slow() %>%
  preprocess_neuron_types()


df_edge <- read_csv("data/derived/graph/fs - edge.csv")  %>%
  bin_width_distinct() %>%
  preprocess_fs_block() %>%
  preprocess_nt_combs()


######  NODE DEGREE

# ylab_node <- TeX(r"(\overset{\normalsize{Normalized Degree}}{\overset{\normalsize{($\sum{R_{SC}} \cdot N^{-1}$)}}})")
ylab_node <- "Neuron\nNormalized\nDegree"

degree_mod <- lmer(
  degree ~ block + neuron_type + response_fs_slow + 
    block:response_fs_slow + (1 | neuron_id),
  data=df_node
)
anova(degree_mod)

degree_emms_nt <- emmeans(
  degree_mod, 
  specs= ~ block * neuron_type
)
res <- pairs(degree_emms_nt, by="block")
res
pairs(res, by=NULL)

p_node_nt <- degree_emms_nt %>%
  as_tibble() %>%
  ggplot(aes(x=block, y=emmean, fill=block, ymin=emmean - SE, ymax=emmean + SE)) + 
  geom_bar(
    stat="identity",  
    color="black", 
    width=0.7, 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  geom_errorbar(
    width=0.28, 
    color='#5c5c5c', 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  facet_grid(cols=vars(neuron_type)) +
  scale_fill_manual(values=c(Shock="black", Pre="grey")) +
  guides(fill="none") +
  labs(y=ylab_node) +
  theme(
    axis.text.x = element_text(size=10, angle=45)
  )
p_node_nt

degree_emms_response <- emmeans(
  degree_mod, 
  specs= ~ block | response_fs_slow
)
res <- pairs(degree_emms_response)
res
pairs(res, by=NULL)

p_node_responsivity <- degree_emms_response %>%
  as_tibble() %>%
  ggplot(aes(x=response_fs_slow, y=emmean, fill=block, ymin=emmean - SE, ymax=emmean + SE)) + 
  geom_bar(
    stat="identity",  
    color="black", 
    width=0.7, 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  geom_errorbar(
    width=0.28, 
    color='#5c5c5c', 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  scale_fill_manual(values=c(Shock="black", Pre="grey"), labels=c("Pre-Shock", "Shock")) +
  facet_grid(cols=vars(block)) +
  labs(y=ylab_node) +
  guides(fill = guide_legend(byrow = TRUE)) +
  theme(
    axis.text.x = element_text(size=9, angle=45, hjust=0.5, vjust=0.5, margin=margin(t=0, b=0)),
  )
p_node_responsivity

p_node_nt + p_node_responsivity
###### Edge Model

ylab = TeX(r"(\overset{\normalsize{Pair Interaction}}{\overset{\normalsize{Strength ($R_{SC}$)}}} )")

df_edge_mod <- df_edge %>% 
  select(comb_id, a, b, same_ensemble, nt_comb, distance, weight, block) %>% 
  distinct() %>%
  left_join(
    df_responders %>% select(neuron_id, response_fs_slow) %>% rename(a=neuron_id, response_a=response_fs_slow)
  ) %>%
  left_join(
    df_responders %>% select(neuron_id, response_fs_slow) %>% rename(b=neuron_id, response_b=response_fs_slow)
  ) %>%
  mutate(
    same_response = factor(response_a == response_b, levels=c(FALSE, TRUE), labels=c("Different\nResponsivity", "Same\nResponsivity")),
    both_activated = (response_a == "activated") & (response_a == response_b),
    both_inhibited = (response_a == "inhibited")& (response_a == response_b)
  ) %>%
  drop_na()

df_edge_mod %>%
  select(response_a, response_b, same_response) %>%
  sample_n(10)

mod_edge <- lmer(
  weight ~ nt_comb + block + same_response + 
    distance + 
    nt_comb:block + same_response:block + 
    (1 | comb_id), 
  data=df_edge_mod
)
summary(mod_edge)
anova(mod_edge)

ems_edge_nt <- emmeans(
  mod_edge, 
  specs = ~ block | nt_comb,
  pbkrtest.limit = 6281
)

pairs(ems_edge_nt, by="block")
pairs(pairs(ems_edge_nt), by=NULL)

p_edge_nt <-ems_edge_nt %>%
  as_tibble() %>%
  ggplot(aes(x=block, y=emmean, fill=block, ymin=emmean - SE, ymax=emmean + SE)) +
  geom_bar(
    stat="identity",  
    color="black", 
    width=0.5, 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  geom_errorbar(
    width=0.28, 
    color='#5c5c5c', 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  scale_fill_manual(values=c(Shock="black", Pre="grey")) +
  facet_grid(cols=vars(nt_comb)) +
  guides(fill="none") +
  labs(y=ylab) +
  theme(
    axis.text.x = element_text(angle=45),
    panel.border = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(margin = margin(t = 0, r = 0, b = 10, l = 0))
  )

p_edge_nt

ems_sameresponse <- emmeans(
  mod_edge, 
  specs = ~ block | same_response, pbkrtest.limit = 6281)

p_edge_response <- ems_sameresponse %>%
  as_tibble() %>%
  ggplot(aes(x=same_response, y=emmean, fill=block, ymin=emmean - SE, ymax=emmean + SE)) +
  geom_bar(
    stat="identity",  
    color="black", 
    width=0.7, 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  geom_errorbar(
    width=0.28, 
    color='#5c5c5c', 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  scale_fill_manual(values=c(Shock="black", Pre="grey")) +
  labs(y=ylab) +
  guides(fill="none") +
  theme(
    axis.text.x = element_text(size=10, angle=0),
  )

res <- pairs(ems_sameresponse, by="block")
res
pairs(res, by=NULL)

layout <- "
CCCCCC#DDDDD
EEEEEEEEEEE#
"
out <- p_node_nt + p_node_responsivity + p_edge_nt  + plot_layout(design = layout)
out



########
df_ensemble_stats <- read_csv("data/derived/ensembles/fs - descripted.csv") %>%
select(
  ensemble_id, 
  block,
  size,
  internal_density,
  normalized_cut_size,
  cut_size,
  conductance,
  neuron_type_entropy, 
  slow_response_entropy,) %>%
  preprocess_fs_block() %>%
  pivot_longer(-c(block, ensemble_id), names_to="metric") %>%
  mutate(
    metric=factor(
      metric, 
      levels=c(
        "size",
        "conductance", 
        "neuron_type_entropy",
        "slow_response_entropy"
      ),
      labels=c(
        "Ensemble\nSize",
        "Ensemble\nConductance",
        "Ensemble\nNeuron Type\nEntropy",
        "Ensemble\nFoot Shock\nResponse Entropy"
      )
    )
  ) %>%
  drop_na() %>%
  group_by(block, metric) %>%
  summarise(mean=mean(value, na.rm = TRUE), se=sd(value)/sqrt(n()))

p_ensembles_size <- df_ensemble_stats %>%
  filter(metric== "Ensemble\nSize") %>%
  ggplot(aes(x=block, fill=block, y=mean, ymin=mean-se, ymax=mean+se)) +
  geom_bar(
    stat="identity",  
    color="black", 
    width=0.52, 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  geom_errorbar(
    width=0.28,
    color='#5c5c5c',
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  scale_fill_manual(values=c(Shock="black", Pre="grey")) +
  guides(fill="none") + labs(y="") +
  ggtitle("Ensemble\nSize")

p_conductance <- df_ensemble_stats %>%
  filter(metric== "Ensemble\nConductance") %>%
  ggplot(aes(x=block, fill=block, y=mean, ymin=mean-se, ymax=mean+se)) +
  geom_bar(
    stat="identity",  
    color="black", 
    width=0.52, 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  geom_errorbar(
    width=0.28,
    color='#5c5c5c',
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  ggtitle("Ensemble\nConductance") +
  scale_fill_manual(values=c(Shock="black", Pre="grey")) +
  guides(fill="none") + labs(y="")

p_nt_entropy <- df_ensemble_stats %>%
  filter(metric== "Ensemble\nNeuron Type\nEntropy") %>%
  ggplot(aes(x=block, fill=block, y=mean, ymin=mean-se, ymax=mean+se)) +
  geom_bar(
    stat="identity",  
    color="black", 
    width=0.52, 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  geom_errorbar(
    width=0.28,
    color='#5c5c5c',
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  ggtitle("Ensemble\nNeuron Type\nEntropy") +
  scale_fill_manual(values=c(Shock="black", Pre="grey")) +
  guides(fill="none") + labs(y="")


p_response_entropy <- df_ensemble_stats %>%
  filter(metric== "Ensemble\nFoot Shock\nResponse Entropy") %>%
  ggplot(aes(x=block, fill=block, y=mean, ymin=mean-se, ymax=mean+se)) +
  geom_bar(
    stat="identity",  
    color="black", 
    width=0.52, 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  geom_errorbar(
    width=0.28,
    color='#5c5c5c',
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  ggtitle("Ensemble\nResponse Entropy") +
  scale_fill_manual(values=c(Shock="black", Pre="grey")) +
  guides(fill="none") + labs(y="") 

out <- (p_node_nt | p_node_responsivity ) / (p_edge_nt) / ( p_ensembles_size | p_conductance | p_nt_entropy | p_response_entropy  )
out


ggsave(
  "FS Interactions.png",
  width = 8,
  height = 5,
  unit="in",
  dpi=300
)

