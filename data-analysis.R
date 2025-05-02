# Load necessary libraries
library(dplyr)
library(readr)

# Load the CSV file
data <- read_csv("/mnt/data/balanced_reviews_labeled-abrv1.csv")

# Normalize labels to lowercase
data <- data %>%
  mutate(actual_label = tolower(actual_label))

# Define model columns
model_columns <- c(
  "openai_o3_neutral",
  "openai_o3_veracity",
  "openai_o3_base_rate",
  "openai_o4-mini_neutral",
  "openai_o4-mini_veracity",
  "openai_o4-mini_base_rate",
  "claude_neutral",
  "claude_veracity",
  "claude_base_rate"
)

# Function to calculate metrics
calculate_metrics <- function(predictions, actuals) {
  predictions <- tolower(predictions)
  actuals <- tolower(actuals)
  
  total <- length(actuals)
  correct_truths <- sum(predictions == "truthful" & actuals == "truthful")
  correct_lies <- sum(predictions == "deceptive" & actuals == "deceptive")
  total_truths <- sum(actuals == "truthful")
  total_lies <- sum(actuals == "deceptive")
  judged_truths <- sum(predictions == "truthful")
  
  list(
    overall_accuracy = (correct_truths + correct_lies) / total,
    truth_bias = judged_truths / total,
    truth_accuracy = ifelse(total_truths == 0, NA, correct_truths / total_truths),
    deception_accuracy = ifelse(total_lies == 0, NA, correct_lies / total_lies)
  )
}

# Run the metrics for all models
results <- lapply(model_columns, function(col) {
  metrics <- calculate_metrics(data[[col]], data$actual_label)
  c(model = col, metrics)
})

# Convert results into a data frame
results_df <- do.call(rbind, lapply(results, as.data.frame))
rownames(results_df) <- NULL

# Print the results
print(results_df)
