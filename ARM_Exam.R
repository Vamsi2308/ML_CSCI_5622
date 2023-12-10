# Load the arules package
# install.packages("arules")
# install.packages("arulesViz", dependencies = TRUE)
# Load the arules package
library(viridis)
library(arules)
library(TSP)
library(data.table)
library(dplyr)
library(devtools)
library(purrr)
library(tidyr)
library(arulesViz)

#Setting working directory
setwd("/Users/vamsigontu/Documents/ML/Exam_ML/Exam_2")

# Load your data
data <- read.transactions("transactional_dataset_new.csv",
                                       rm.duplicates = FALSE,
                                       format = "basket",  ##if you use "single" also use cols=c(1,2)
                                       sep=",",  ## csv file
                                       cols=NULL)
inspect(data)
rules <- arules::apriori(data, parameter = list(support=0.01, confidence=0.01, minlen=2))
summary(rules)
# Display the rules
inspect(rules)

itemFrequencyPlot(data, topN=5, type="absolute", cex.names = 0.6)
# ## Plot of which items are most frequent
# itemFrequencyPlot(items(rules), topN=30, type="absolute", main = "Item Frequency Plot")

# Sort rules by confidence in descending order
sorted_rules_conf <- sort(rules, by = "confidence", decreasing = TRUE)
# Print the sorted rules
inspect(sorted_rules_conf)

# Sort rules by support in descending order
sorted_rules_sup <- sort(rules, by = "support", decreasing = TRUE)
# Print the sorted rules
inspect(sorted_rules_sup)

# Sort rules by lift in descending order
sorted_rules_lift <- sort(rules, by = "lift", decreasing = TRUE)
# Print the sorted rules
inspect(sorted_rules_lift)

# Visualize top rules sorted on the basis of lift.
subrules_lift <- head(sort(sorted_rules_lift, by="lift"), 15)
# plot(subrules_lift)
plot(subrules_lift, method="graph", engine="htmlwidget")
