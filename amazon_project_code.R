
# Load required libraries
library(dplyr)
library(randomForest)
library(caret)
library(ggplot2)
library(tidyr)

# Load the dataset
data <- read.csv("amazon_reviews.csv")

# Handle missing values
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
data$reviewText[is.na(data$reviewText)] <- "No Review"
data$reviewerName[is.na(data$reviewerName)] <- Mode(data$reviewerName)

# Process helpfulness column into numeric values
data <- data %>%
  separate(helpful, into = c("helpful_yes", "helpful_total"), sep = ", ", remove = FALSE) %>%
  mutate(across(c(helpful_yes, helpful_total), ~gsub("\\[|\\]", "", .))) %>%
  mutate(across(c(helpful_yes, helpful_total), as.numeric)) %>%
  mutate(helpfulness_ratio = ifelse(helpful_total == 0, 0, helpful_yes / helpful_total))

# Create review length feature
data$review_length <- nchar(as.character(data$reviewText))

# Final dataset for modeling
model_data <- data %>%
  select(overall, helpfulness_ratio, review_length, unixReviewTime) %>%
  na.omit()

# Train-test split (80/20)
set.seed(123)
train_index <- createDataPartition(model_data$overall, p = 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# Train Random Forest regression model
rf_model <- randomForest(overall ~ ., data = train_data, ntree = 100)

# Make predictions
predictions <- predict(rf_model, test_data)

# Model evaluation
mae <- mean(abs(predictions - test_data$overall))
rmse <- sqrt(mean((predictions - test_data$overall)^2))
r2 <- 1 - sum((test_data$overall - predictions)^2) / sum((test_data$overall - mean(test_data$overall))^2)

print(paste("MAE:", round(mae, 2)))
print(paste("RMSE:", round(rmse, 2)))
print(paste("R-squared:", round(r2, 2)))

# Residual analysis
residuals <- predictions - test_data$overall
plot(residuals, main = "Residual Plot", ylab = "Residuals", xlab = "Index", col = "blue")

# Actual vs Predicted
plot(test_data$overall, predictions,
     xlab = "Actual Ratings", ylab = "Predicted Ratings",
     main = "Actual vs Predicted Ratings", col = "darkgreen", pch = 19)
abline(0, 1, col = "red")












