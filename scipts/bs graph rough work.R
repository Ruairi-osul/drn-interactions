library(tidyverse)
library(emmeans)
library(lme4)
library(lmerTest)
library(bbplot)
library(extrafont)
library(latex2exp)
library(patchwork)

## Relabel states and BS responsivity to be less confusing 

df_responders <- read_csv("data/derived/graph/bs - responders.csv") %>%
  mutate(response_bs = factor(response_bs, levels=c("inhibited", "activated"), labels=c("Inactivated\nPreferring", "Activated\nPreferring")))


theme_set(
  bbc_style() +
    theme(
      strip.text.x = element_text(size = 9, angle=0, hjust=0.5),
      axis.text.x = element_text(size=9, angle=0),
      axis.text.y = element_text(size=9),
      legend.text = element_text(size=9),
      legend.key.size = unit(0.5, 'cm'),
      legend.position = "right",
      axis.line=element_line(),
      axis.title.y = element_text(size=10, angle=90, margin = margin(t = 0, r = 5, b = 0, l = 0))
      )
)

df_node <- read_csv("data/derived/graph/bs - node.csv") %>% 
  filter(bin_width == 1) %>%
  left_join(df_responders) %>% 
  distinct()  %>%
  filter(response_bs != "no_response") %>%
  mutate(response_bs = factor(response_bs, levels=c("inhibited", "activated"), labels=c("Inactivated\nPreferring", "Activated\nPreferring"))) %>%
  mutate(neuron_type = factor(neuron_type, levels=c("SR", "SIR", "FF"))) %>%
  mutate(state = factor(state, levels=c("sw", "act"), labels=c("Inact", "Act"))) 


df_graph <- read_csv("data/derived/graph/bs - graph.csv") %>%
  filter(bin_width == 1)

df_edge <- read_csv("data/derived/graph/bs - edge.csv")  %>%
  filter(bin_width == 1) %>%
  mutate(nt_comb = factor(nt_comb, levels=c("SR-SR", "SR-SIR", "SR-FF", "SIR-SIR", "SIR-FF", "FF-FF"))) %>%
  mutate(state = factor(state, levels=c("sw", "act"), labels=c("Inact", "Act")))

df_ensembles <-  read_csv("data/derived/ensembles/bs - stats - true.csv")  %>%
  filter(bin_width == 1) 


######  NODE DEGREE

ylab_node <- "Neuron\nNormalized\nDegree"
ylab_edge <- "Neuron Pair\nInteraction\nWeight"

mod_node <- lmer(
  degree ~ state + neuron_type + response_bs +
           state:neuron_type + state:response_bs + 
           (1 | neuron_id),
  data=df_node
)
anova_node <- anova(mod_node)

emms_node_nt <- emmeans(
  mod_node, 
  specs= ~ state + neuron_type
)
contrasts_node_nt <- pairs(emms_node_nt)

emms_node_response <- emmeans(
  mod_node, 
  specs= ~response_bs * state
)
contrasts_node_response <- pairs(emms_node_response, by="response_bs")

# stats 
anova_node
emms_node_nt
contrasts_node_nt
emms_node_response
contrasts_node_response
pairs(contrasts_node_response, by=NULL)



##### EDGE WEIGHT

df_edge_mod <- df_edge %>% 
  select(comb_id, a, b, same_ensemble, nt_comb, distance, weight, state) %>% 
  distinct() %>%
  left_join(
    df_responders %>% select(neuron_id, response_bs) %>% rename(a=neuron_id, response_a=response_bs)
  ) %>%
  left_join(
    df_responders %>% select(neuron_id, response_bs) %>% rename(b=neuron_id, response_b=response_bs)
  ) %>%
  mutate(
    same_response = factor(response_a == response_b, labels=c("Same\nPreferred\nState", "Different\nPreferred\nState")),
    both_activated = (response_a == "activated") & (response_a == response_b),
    both_inhibited = (response_a == "inhibited")& (response_a == response_b)
  )


mod_edge <- lmer(
  weight ~ nt_comb + state +
    distance + 
    state:nt_comb +
    (1 | comb_id), 
  data=df_edge_mod
)
summary_edge_mode <- summary(mod_edge)
anova_edge_mod <- anova(mod_edge)


emms_edge_nt <- emmeans(
  mod_edge, 
  specs = ~ state*nt_comb, 
  pbkrtest.limit = 4818
  )

contrasts_emms_edge_nt <- contrast(emms_edge_nt, by="nt_comb")  

# stats 
summary_edge_mode
anova_edge_mod
ems_edge_nt
contrasts_emms_edge_nt


##### PLOTS

p_node_nt <- emms_node_nt %>%
  as_tibble() %>%
  ggplot(aes(x=neuron_type, y=emmean, fill=state, ymin=emmean - SE, ymax=emmean + SE)) + 
  geom_bar(stat="identity",  position=position_dodge(preserve = "single", width=0.8), color="black", width=0.7) +
  geom_errorbar(width=0.28, color='#5c5c5c', position=position_dodge(preserve = "single", width=0.8)) +
  scale_fill_manual(values=c(Inact="grey", Act="black"), labels=c("Inactivated\nBrain State", "Activated\nBrain State")) +
  guides(fill="none") +
  labs(y=ylab_edge) +
  theme(
    axis.text.x = element_text(size=10, angle=0),
  )
p_node_nt

p_node_response <- emms_node_response %>%
  as_tibble() %>%
  ggplot(aes(x=response_bs, y=emmean, fill=state,  ymin=emmean - SE, ymax=emmean + SE)) + 
  geom_bar(stat="identity",  position=position_dodge(preserve = "single", width=0.8), color="black", width=0.7) +
  geom_errorbar(width=0.28, color='#5c5c5c', position=position_dodge(preserve = "single", width=0.8)) +
  scale_fill_manual(values=c(Inact="grey", Act="black"), labels=c("Inactivated\nBrain State", "Activated\nBrain State")) +
  labs(y=ylab_node) +
  guides(fill = guide_legend(byrow = TRUE)) +
  theme(
    axis.text.x = element_text(size=9, angle=45, hjust=0.5, vjust=0.5, margin=margin(t=0, b=0)),
    panel.border = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_blank()
  )
p_node_response

p_edge <-emms_edge_nt %>%
  as_tibble() %>%
  ggplot(aes(x=state, y=emmean, fill=state, ymin=emmean - SE, ymax=emmean + SE)) +
  geom_bar(stat="identity",  position=position_dodge(preserve = "single", width=0.9), color="black", width=0.7) +
  geom_errorbar(width=0.28, color='#5c5c5c', position=position_dodge(preserve = "single", width=0.9)) +
  scale_fill_manual(values=c(Inact="grey", Act="black")) +
  facet_grid(cols=vars(nt_comb)) +
  guides(fill="none") +
  labs(y=ylab) +
  theme(
    axis.text.x = element_text(angle=45),
    panel.border = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(margin = margin(t = 0, r = 0, b = 10, l = 0))
  )

p_edge 

layout <- "
CCCCCCCCDDDDD
EEEEEEEEEEEEE
"
out <- p_node_nt + p_node_response + p_edge +
  plot_layout(design = layout)
out


ggsave(
  "BS Ensembles.png",
  width = 8,
  height = 4,
  unit="in",
  dpi=300
)


##### Ensembles


df_ensembles %>%
  select(session_name, size, average_weight, average_weight_out, response_bs_entropy, neuron_type_entropy)

