library(tidyverse)
library(emmeans)
library(extrafont)
loadfonts(device = "win")
library(bbplot)
library(latex2exp)
library(patchwork)
library(mgcv)
library(arrow)




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


df_responders <- read_csv("data/derived/graph/spont - responders.csv") %>%
  mutate(
    response_bs = 
      factor(
        response_bs, 
        levels=c("inhibited", "activated"), 
        labels=c("Inactivated\nPreferring", "Activated\nPreferring")
        ),
    response_fs_slow = 
      factor(
        response_fs_slow, 
        levels=c("inhibited", "activated"), 
        labels=c("Sghock-\nInhibited", "Shock-\nActivated")
      ),
    )

df_node <- read_csv("data/derived/graph/spont - node.csv") %>% 
  filter(shuffle == FALSE) %>%
  left_join(df_responders) %>% 
  preprocess_neuron_types()

df_edge <- read_csv("data/derived/graph/spont - edge.csv")  %>%
  filter(shuffle == FALSE) %>%
  preprocess_nt_combs() %>%
  mutate(distance = distance / 1000)  # convert um to mm


#######

df_pcup <- read_parquet("data/derived/corrs/spont - pcup.parquet")

df_pcup %>%
  mutate(
    neuron_type = factor(
      neuron_type, 
      levels=c("SR", "SIR", "FF")
      ),
    sig=factor(
      sig, 
      levels=c(FALSE, TRUE),
      ordered=T
    )
    ) %>%
  filter(bin_width == 1, shuffle == F) %>%
  ggplot(aes(x=cc, fill=sig)) +
  geom_histogram(breaks=seq(-0.5, 0.5, length.out=30)) +
  scale_fill_manual(values=c("FALSE"="grey", "TRUE"="black")) +
  facet_grid(cols=vars(neuron_type))


df_pcup %>%
  mutate(
    neuron_type = factor(
      neuron_type, 
      levels=c("SR", "SIR", "FF")
      ),
    neg = (sig == TRUE) & (cc < 0)
  ) %>%
  filter(bin_width == 1, shuffle == F) %>%
  group_by(neuron_type) %>%
  summarise(neg=mean(neg == T)) %>%
  ggplot(aes(y=neg, x=neuron_type)) +
  geom_bar(stat="identity")
  # scale_fill_manual(values=c("FALSE"="grey", "TRUE"="black")) +


df_pcup %>%
  mutate(
    neuron_type = factor(
      neuron_type, 
      levels=c("SR", "SIR", "FF")
    )
  ) %>%
  filter(bin_width == 1, shuffle == F) %>%
  ggplot(aes(y=cc, x=neuron_type)) +
  geom_boxplot()



########## Neuron Degree Mod

degree_mod <- lm(
  degree ~ neuron_type,
  data=df_node
)
anova(degree_mod)
degree_emms_nt <- emmeans(
  degree_mod, 
  specs= ~ neuron_type
)
res <- pairs(degree_emms_nt)


########## Edge Mod

edge_mod_base <- gam(
  weight ~ nt_comb * distance,
  data=df_edge
)
summary(edge_mod_base)
anova.gam(edge_mod_base)

edge_mod <- gam(
  weight ~ nt_comb + s(distance,  bs = "cr", by=nt_comb),
  data=df_edge
)
summary(edge_mod)
anova.gam(edge_mod)

emms_emms_nt <- emmeans(
  edge_mod, 
  specs= ~ nt_comb
)
emms_emms_nt
res <- pairs(emms_emms_nt)
res

###### Plots

ylab_node <- "Neuron\nNormalized\nDegree"
ylab_edge <- "Neuron Pair\nInteraction\nWeight"

p_nt <- degree_emms_nt %>%
  as_tibble() %>%
  ggplot(aes(x=neuron_type, y=emmean, ymin=emmean - SE, ymax=emmean + SE)) + 
  geom_bar(
    stat="identity",  
    color="black", 
    fill="black",
    width=0.7, 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  geom_errorbar(
    width=0.2,
    color='#5c5c5c', 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  guides(fill="none") +
  labs(y=ylab_node)

p_edge_comb <- emms_emms_nt %>%
  as_tibble() %>%
  ggplot(aes(x=nt_comb, y=emmean, ymin=emmean - SE, ymax=emmean + SE)) + 
  geom_bar(
    stat="identity",  
    color="black", 
    fill="black",
    width=0.5, 
    alpha=0.9,
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  geom_errorbar(
    width=0.18, 
    color='#5c5c5c', 
    position=position_dodge(preserve = "single", width=0.8)
  ) +
  guides(fill="none") +
  labs(y=ylab_edge)



p_distance <- df_edge %>%
  ggplot(aes(x=distance, y=weight)) + 
  geom_smooth(color="black", fill="black", alpha=0.3, formula= y ~ s(x, bs=("cr"))) +
  facet_grid(cols=vars(nt_comb)) +
  scale_x_continuous(breaks=c(0, 0.1, 0.2, 0.3, 0.4)) +
  labs(y=ylab_edge) +
  theme(axis.text.x=element_text(angle=45))


layout <- "
ABBB
CCCC
"
out <- p_nt + p_edge_comb + p_distance + patchwork::plot_layout(design=layout)

out

ggsave(
  "spont.png",
  width = 8,
  height = 4,
  unit="in",
  dpi=300
)
