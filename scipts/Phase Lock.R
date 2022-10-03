library(tidyverse)
library(patchwork)
library(circular)


df <- read_csv("data/derived/brain_states_phase_responders.csv") %>%
  mutate(
    ang = circular(mean_angle, units="radians"),
    oscillation=factor(oscillation, levels=c("delta", "theta")),
    neuron_type=factor(neuron_type, levels=c("SR", "SIR", "FF"))
  )


## Summary Stats

df %>%
  filter(oscillation=="delta", p < 0.05) %>%
  group_by(neuron_type) %>%
  summarise(
    theta=circular::mean.circular(ang),
    var=circular::var.circular(ang),
    rho=circular::rho.circular(ang)
  )

df %>%
  filter(oscillation=="theta", p < 0.05) %>%
  group_by(neuron_type) %>%
  summarise(
    theta=circular::mean.circular(ang),
    var=circular::var.circular(ang),
    rho=circular::rho.circular(ang)
  )

## delta

invisible(capture.output(
  mod_delta <- bpnreg::bpnr(
    ang ~ neuron_type,
    data=filter(df, oscillation=="delta", p < 0.05)
  )
))

coef_delta <- bpnreg::coef_circ(
  mod_delta, type="categorical", units="radians"
)
print(round(coef_delta$Means, 2))
print(round(coef_delta$Differences, 2))

## theta

invisible(capture.output(
  mod_delta <- bpnreg::bpnr(
    ang ~ neuron_type,
    data=filter(df, oscillation=="theta", p < 0.05)
  )
))

coef_delta <- bpnreg::coef_circ(
  mod_delta, type="categorical", units="radians"
)
print(round(coef_delta$Means, 2))
print(round(coef_delta$Differences, 2))


circular::plot.circular(c( -2.74, 0.01, 2.11), ticks=T, axes=F)


circular::plot.circular(c(-2.09, 2.68, 6.07), ticks=T, axes=F)


circular::plot.circular(c(1.73, 1.07, 2.34), ticks=T, axes=F)


