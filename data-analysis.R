# Load necessary libraries
library(dplyr)
library(readr)
library(effsize)

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

# Group the columns by prompt type
neutral_columns <- c("openai_o3_neutral", "openai_o4-mini_neutral", "claude_neutral")
veracity_columns <- c("openai_o3_veracity", "openai_o4-mini_veracity", "claude_veracity")
base_rate_columns <- c("openai_o3_base_rate", "openai_o4-mini_base_rate", "claude_base_rate")

# Function to calculate means for each metric by prompt type
calculate_prompt_type_means <- function(columns) {
  prompt_results <- results_df[results_df$model %in% columns, ]
  
  data.frame(
    overall_accuracy = mean(prompt_results$overall_accuracy, na.rm = TRUE),
    truth_bias = mean(prompt_results$truth_bias, na.rm = TRUE),
    truth_accuracy = mean(prompt_results$truth_accuracy, na.rm = TRUE),
    deception_accuracy = mean(prompt_results$deception_accuracy, na.rm = TRUE)
  )
}

# Calculate means by prompt type
neutral_means <- calculate_prompt_type_means(neutral_columns)
veracity_means <- calculate_prompt_type_means(veracity_columns)
base_rate_means <- calculate_prompt_type_means(base_rate_columns)

# Combine into a single data frame
prompt_type_means <- rbind(
  data.frame(prompt_type = "Neutral", neutral_means),
  data.frame(prompt_type = "Veracity", veracity_means),
  data.frame(prompt_type = "Base-rate", base_rate_means)
)

# Print the prompt type means
print("Mean metrics by prompt type:")
print(prompt_type_means)

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
print("Statistical tests by model:")
print(stat_results_df)

# Calculate means of statistical measures by prompt type
calculate_stats_means <- function(columns) {
  stats_results <- stat_results_df[stat_results_df$model %in% columns, ]
  
  data.frame(
    t_stat = mean(stats_results$t_stat, na.rm = TRUE),
    p_value = mean(stats_results$p_value, na.rm = TRUE),
    cohens_d = mean(stats_results$cohens_d, na.rm = TRUE)
  )
}

# Calculate statistical means by prompt type
neutral_stats_means <- calculate_stats_means(neutral_columns)
veracity_stats_means <- calculate_stats_means(veracity_columns)
base_rate_stats_means <- calculate_stats_means(base_rate_columns)

# Combine into a single data frame
prompt_type_stats_means <- rbind(
  data.frame(prompt_type = "Neutral", neutral_stats_means),
  data.frame(prompt_type = "Veracity", veracity_stats_means),
  data.frame(prompt_type = "Base-rate", base_rate_stats_means)
)

# Print the prompt type statistical means
print("Mean statistical measures by prompt type:")
print(prompt_type_stats_means)