library(ggplot2)
library(reshape)
library(dplyr)
require(scales)

t <- read.csv("/home/giacomo/PycharmProjects/trace_learning/hist_cyber.csv",header = F)
names(t) <- c("Sample","x")
t$Sample <- as.character(t$Sample)

ggplot(t,aes(x,group=Sample)) + geom_histogram(aes(fill=Sample), alpha=.2)+
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  facet_wrap(~ Sample)