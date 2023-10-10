# Install and load the proxy package 
# install.packages("proxy")
library(proxy)

# setting up the working directory
setwd("/Users/vamsigontu/Documents/ML")

# Load your data
data <- read.csv("cleaned_data.csv", header = TRUE)
head(data)

# Preprocess categorical variables (convert them to factors)
data$STATUS <- as.factor(data$STATUS)
data$STREET_NAME <- as.factor(data$STREET_NAME)
data$PAVETYPE <- as.factor(data$PAVETYPE)
data$FUNCTIONAL_CLASS <- as.factor(data$FUNCTIONAL_CLASS)
######################

# Performing Hierarchical Clustering

#################

# using cosine similarity matrix

################
# Select the numeric columns for similarity calculation
numeric_data <- data[, c("TRAFFIC_COUNT","STATION_NUMBER" )]

# Calculate cosine similarity matrix
cosine_similarity_matrix <- proxy::simil(numeric_data, method = "cosine")

# Perform hierarchical clustering using the cosine similarity matrix with hclust
hclust_result_cosine <- hclust(proxy::as.dist(1 - cosine_similarity_matrix), method = "complete")
plot(hclust_result_cosine, main = "Hierarchical Clustering Dendrogram (Cosine Similarity)")
clusters_cosine <- cutree(hclust_result_cosine, k = 3)
clustered_data_cosine <- cbind(data, Cluster = clusters_cosine)

################

# using Distnace Matrix

################

# Assuming only numeric variables are used for clustering
distance_matrix <- dist(data[, c("TRAFFIC_COUNT", "STATION_NUMBER")])
#(e.g., "complete," "single," "average")
hclust_result <- hclust(distance_matrix, method = "complete")
# plotting Dendogram
plot(hclust_result, main = "Hierarchical Clustering Dendrogram(Distnace Matrix)")
# assigning Clusters
clusters <- cutree(hclust_result, k = 3)
# analysing clusters
clustered_data <- cbind(data, Cluster = clusters)

#####################

# Performing Elbow and Silhouette Method

#####################

# Select the numeric columns for clustering
numeric_data <- data[, c("TRAFFIC_COUNT", "TRAFFIC_YEAR_COUNTED", "CHRIS_NUMB", "STATION_NUMBER")]

# Perform elbow method and silhouette analysis to determine k
wss <- numeric(10)
silhouette_scores <- numeric(10)

for (i in 1:10) {
  kmeans_temp <- kmeans(numeric_data, centers = i)
  wss[i] <- sum(kmeans_temp$withinss)
  if (i > 1) {
    silhouette_scores[i] <- silhouette(kmeans_temp$cluster, dist(numeric_data))
  }
}
#####################

# Elbow Method Plot

#####################
plot(1:10, wss, type = "b", xlab = "Number of Clusters (k)", ylab = "Within-cluster Sum of Squares (WSS)", main = "Elbow Method")

#####################

# Silhouette Method Plot

#####################
if (any(!is.na(silhouette_scores))) {
  plot(2:10, silhouette_scores[2:10], type = "b", xlab = "Number of Clusters (k)", ylab = "Average Silhouette Width", main = "Silhouette Method")
}

# Based on the visual inspection of the plots, choose an appropriate value of k
# For example, you can choose the k value where the elbow occurs in the WSS plot or where silhouette score is highest.
chosen_k <- 3  # Replace with your chosen value of k

# Perform k-means clustering with the chosen k
kmeans_result <- kmeans(numeric_data, centers = chosen_k, nstart = 25)

# Add cluster assignments to the original dataset
data$Cluster <- kmeans_result$cluster

# View cluster centers
cluster_centers <- kmeans_result$centers
print(cluster_centers)

# View cluster assignments
print(table(data$Cluster))

