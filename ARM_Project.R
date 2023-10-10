# Load the arules package
# install.packages("arules")
# install.packages("arulesViz", dependencies = TRUE)
# Load the arules package
library(arules)
library(data.table)
library(arulesViz)

#Setting working directory
setwd("/Users/vamsigontu/Documents/ML")

# Load your data
data <- read.csv("cleaned_data.csv", header = TRUE)
head(data)

# Preprocess categorical variables (convert them to factors)
data$STATUS <- as.factor(data$STATUS)
data$STREET_NAME <- as.factor(data$STREET_NAME)
data$PAVETYPE <- as.factor(data$PAVETYPE)
data$FUNCTIONAL_CLASS <- as.factor(data$FUNCTIONAL_CLASS)
head(data)

# Create a transaction dataset
transactions <- as(data[, c("STATUS", "STREET_NAME", "PAVETYPE", "FUNCTIONAL_CLASS")], "transactions")
# View the first few transactions
head(transactions)

# Generate association rules
rules <- apriori(transactions, parameter = list(support = 0.1, confidence = 0.5))
# Print the generated rules
inspect(rules)

## Plot of which items are most frequent
itemFrequencyPlot(items(rules), topN=5, type="absolute", main = "Item Frequency Plot")

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

subrules <- head(sort(sorted_rules_conf, by="lift"),15)
plot(subrules)

# Create a scatterplot of association rules
plot(subrules, method = "scatter", measure = "lift", shading = "confidence", jitter = 0, main = 'Scatter plot for 15 rules(jitter)')

# Create a scatter plot and vary pointers based on the "solution" attribute
plot(subrules, method = "scatter", measure = "lift", shading = "confidence",jitter = 0,
  control = list(
   col = c("red", "blue"),  # Specify different colors for different solutions
   pch = c(19, 3),          # Specify different point types (shapes) for different solutions
   cex = c(1, 1)            # Specify different point sizes for different solutions
 )
 )
#################### This Part didnt worked for me
plot(subrules, method="graph", engine="interactive")
