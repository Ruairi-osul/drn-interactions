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
      axis.text.x = element_text(size=10, angle=0),
      axis.text.y = element_text(size=9),
      legend.text = element_text(size=10),
      legend.key.size = unit(0.5, 'cm'),
      legend.position = "right",
      axis.line=element_line(),
      axis.title.y = element_text(size=10, angle=90, margin = margin(t = 0, r = 0, b = 0, l = 0))
      )
)

df_node <- read_csv("data/derived/graph/bs - node.csv") %>% 
  left_join(df_responders) %>% 
  distinct()  %>%
  filter(response_bs != "no_response") %>%
  mutate(response_bs = factor(response_bs, levels=c("inhibited", "activated"), labels=c("Inactivated\nPreferring", "Activated\nPreferring"))) %>%
  mutate(neuron_type = factor(neuron_type, levels=c("SR", "SIR", "FF"))) %>%
  mutate(state = factor(state, levels=c("sw", "act"), labels=c("Inact", "Act"))) 


df_graph <- read_csv("data/derived/graph/bs - graph.csv")

df_edge <- read_csv("data/derived/graph/bs - edge.csv")  %>%
  mutate(nt_comb = factor(nt_comb, levels=c("SR-SR", "SR-SIR", "SR-FF", "SIR-SIR", "SIR-FF", "FF-FF"))) %>%
  mutate(state = factor(state, levels=c("sw", "act"), labels=c("Inact", "Act")))


######  NODE DEGREE

#  The Quantity of interactions is lower during activated brain states. 
#  But only for neurons who increase their spike rate during this period

ylab <- TeX(r"(\overset{\normalsize{Normalized Degree}}{\overset{\normalsize{($\sum{R_{SC}} \cdot N^{-1}$)}}})")

mod_state <- lmer(
  degree ~ state + neuron_type + state + response_bs + state:neuron_type + state:response_bs + (1 | neuron_id),
  data=df_node
)

anova(mod_state)

ems_nt <- emmeans(
  mod_state, 
  specs= ~state + neuron_type
)



p3 <- ems_nt %>%
  as_tibble() %>%
  ggplot(aes(x=neuron_type, y=emmean, fill=state, ymin=emmean - SE, ymax=emmean + SE)) + 
  geom_bar(stat="identity",  position=position_dodge(preserve = "single", width=0.8), color="black", width=0.7) +
  geom_errorbar(width=0.28, color='#5c5c5c', position=position_dodge(preserve = "single", width=0.8)) +
  scale_fill_manual(values=c(Inact="grey", Act="black"), labels=c("Inactivated\nBrain State", "Activated\nBrain State")) +
  guides(fill="none") +
  labs(y=ylab) +
  theme(
    axis.text.x = element_text(size=10, angle=0),
  )
p3
res <- pairs(ems_nt, by="neuron_type")
res
pairs(res, by=NULL)


ems_response <- emmeans(
  mod_state, 
  specs= ~state * response_bs
)

p4 <- ems_response %>%
  as_tibble() %>%
  ggplot(aes(x=response_bs, y=emmean, fill=state, ymin=emmean - SE, ymax=emmean + SE)) + 
  geom_bar(stat="identity",  position=position_dodge(preserve = "single", width=0.7), color="black", width=0.7) +
  geom_errorbar(width=0.28, color='#5c5c5c', position=position_dodge(preserve = "single", width=0.7)) +
  scale_fill_manual(values=c(Inact="grey", Act="black"), labels=c("Inactivated\nBrain State", "Activated\nBrain State")) +
  scale_y_continuous(
    breaks=c(0, 0.07, 0.15), limits=c(0, 0.18)
  ) +
  facet_grid(cols=vars(state)) +
  labs(y=ylab) +
  guides(fill = guide_legend(byrow = TRUE)) +
  theme(
    axis.text.x = element_text(size=9, angle=45, hjust=0.5, vjust=0.5, margin=margin(t=0, b=0)),
    panel.border = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_blank()
  )

p4
ems_response
res <- pairs(ems_response, by="response_bs")
res
pairs(res, by=NULL)


res2 <- pairs(ems_response, by="state")
res2


p3 + p4


##### EDGE WEIGHT
# The strength of interactions differed between states. Among all of neuron types,
# interactions were stronger in SW than act. The opposite effect was observed in 
# Interactions were stronger among neuron pairs which had the same brain state responsivity
# There was no interaction between brain state and the effect of distance between nerons on
# Interaction strength.
ylab = TeX(r"(\overset{\normalsize{Pair Interaction}}{\overset{\normalsize{Strength ($R_{SC}$)}}} )")


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
  weight ~ nt_comb + state + same_response + 
    distance + 
    state:nt_comb + same_response:state +
    (1 | comb_id), 
  data=df_edge_mod
)
summary(mod_edge)
anova(mod_edge)


ems_edge_nt <- emmeans(
  mod_edge, 
  specs = ~ state*nt_comb, 
  pbkrtest.limit = 4818
  )

p5 <-ems_edge_nt %>%
  as_tibble() %>%
  ggplot(aes(x=state, y=emmean, fill=state, ymin=emmean - SE, ymax=emmean + SE)) +
  geom_bar(stat="identity",  position=position_dodge(preserve = "single", width=0.8), color="black", width=0.7) +
  geom_errorbar(width=0.28, color='#5c5c5c', position=position_dodge(preserve = "single", width=0.8)) +
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
  
p5

res <- contrast(ems_edge_nt, by="nt_comb")  
pairs(res)

pairs(pairs(res), by=NULL)



ems_sameresponse <- emmeans(
  mod_edge, 
  specs = ~ same_response * state, pbkrtest.limit = 4818)

p6 <- ems_sameresponse %>%
  as_tibble() %>%
  ggplot(aes(x=same_response, y=emmean, fill=state, ymin=emmean - SE, ymax=emmean + SE)) +
  geom_bar(stat="identity",  position=position_dodge(preserve = "single", width=0.8), color="black", width=0.7) +
  geom_errorbar(width=0.28, color='#5c5c5c', position=position_dodge(preserve = "single", width=0.8)) +
  scale_fill_manual(values=c(Inact="grey", Act="black")) +
  labs(y=ylab) +
  guides(fill="none") +
  theme(
    axis.text.x = element_text(size=10, angle=0),
  ) 

p5 + p6


layout <- "
CCCCCCCCDDDD#
EEEEEEEEEEEFF
"
out <- p3 + p4 + p5 + p6 +
  plot_layout(design = layout)
out


ggsave(
  "tes.png",
  width = 8,
  height = 4,
  unit="in",
  dpi=300
)

