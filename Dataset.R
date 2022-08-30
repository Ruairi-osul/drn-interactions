library(tidyverse)
library(bbplot)
library(extrafont)
library(patchwork)

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
      axis.title.y = element_text(size=10, angle=90, margin = margin(t = 0, r = 5, b = 0, l = 0)),
      axis.title.x = element_text(size=10, angle=0, margin = margin(t = 0, r = 5, b = 0, l = 0)),
      
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_line(size=0.3, color="grey")
    )
)

df_neurons <- read_csv("data/derived/neuron_types.csv")

df_neurons %>%
  count(session_name) %>%
  drop_na() %>%
  arrange(n) %>% 
  mutate(session_name=factor(session_name, levels=session_name)) %>%
  ggplot(aes(y=session_name, x=n)) +
  geom_bar(stat="identity", color="white", fill="black") +
  theme(axis.text.y = element_blank()) +
  labs(y="Individual\nRecording Sessions", x = "Number of Isolated\n Neurons")

ggsave(
  "yeild.png",
  width = 3,
  height = 2,
  unit="in",
  dpi=300
)



  