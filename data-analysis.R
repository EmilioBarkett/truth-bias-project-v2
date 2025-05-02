# Load necessary libraries
library(dplyr)
library(readr)

# Load the CSV file
data <- read_csv("balanced_reviews_labeled-abrv1.csv")

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

# Calculate the mean for each type of prompt (neutral, veracity, and base-rate)
mean_results <- data.frame(
  prompt_type = c("Neutral", "Veracity", "Base-rate"),
  mean_neutral = c(
    mean(data$openai_o3_neutral, na.rm = TRUE),
    mean(data$openai_o4-mini_neutral, na.rm = TRUE),
    mean(data$claude_neutral, na.rm = TRUE)
  ),
  mean_veracity = c(
    mean(data$openai_o3_veracity, na.rm = TRUE),
    mean(data$openai_o4-mini_veracity, na.rm = TRUE),
    mean(data$claude_veracity, na.rm = TRUE)
  ),
  mean_base_rate = c(
    mean(data$openai_o3_base_rate, na.rm = TRUE),
    mean(data$openai_o4-mini_base_rate, na.rm = TRUE),
    mean(data$claude_base_rate, na.rm = TRUE)
  )
)

# Print mean results explicitly
print(mean_results)

# Load required library
library(effsize)

# Function to compute Welch's t-test and Cohen's d
compute_stats <- function(predictions, actuals) {
  # Convert predictions to binary: 1 = "truthful", 0 = "deceptive"
  binary_preds <- ifelse(tolower(predictions) == "truthful", 1, 0)
  
  # Split predictions by actual label
  group_truthful <- binary_preds[tolower(actuals) == "truthful"]
  group_deceptive <- binary_preds[tolower(actuals) == "deceptive"]
  
  # Welch's t-test
  t_result <- t.test(group_truthful, group_deceptive)
  
  # Cohen's d
  d_result <- cohen.d(group_truthful, group_deceptive)
  
  list(
    t_stat = t_result$statistic,
    p_value = t_result$p.value,
    cohens_d = d_result$estimate
  )
}

# Run stats for all models
stat_results <- lapply(model_columns, function(col) {
  stats <- compute_stats(data[[col]], data$actual_label)
  c(model = col, stats)
})

# Convert stats to data frame
stat_results_df <- do.call(rbind, lapply(stat_results, as.data.frame))
rownames(stat_results_df) <- NULL

# Print the stats
print(stat_results_df)
