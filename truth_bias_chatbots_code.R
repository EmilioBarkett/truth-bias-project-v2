library(tidyverse)
library(psych)
library(crosstable)
library(interactions)
library(gridExtra)
library(lme4)
library(lmerTest)
library(emmeans)
library(rstatix)

# study 1 ai data
study1_ai <- read.csv("study1_neutral_prompt_data_OSF.csv")
nrow(study1_ai)

# accuracy
round(sum(study1_ai$accuracy_2)/nrow(study1_ai)*100, digits = 2)
# overall truth-bias
round(sum(study1_ai$model_result == "Truthful")/nrow(study1_ai)*100, digits = 2)

truth_accuracy_study1_ai <- study1_ai %>% filter(study1_ai$ground_truth == "Truthful")
round(sum(truth_accuracy_study1_ai$model_result == "Truthful")/nrow(truth_accuracy_study1_ai)*100, digits = 2)

deception_accuracy_study1_ai <- study1_ai %>% filter(study1_ai$ground_truth == "Deceptive")
round(sum(deception_accuracy_study1_ai$model_result == "Deceptive")/nrow(deception_accuracy_study1_ai)*100, digits = 2)

# study 1 human data

study1_humans <- read.csv("study1_human_detection_OSF.csv")
nrow(study1_humans)

# accuracy
round(sum(study1_humans$accuracy_2)/nrow(study1_humans)*100, digits = 2)
# overall truth-bias
round(sum(study1_humans$participant_response == "Truthful")/nrow(study1_humans)*100, digits = 2)

truth_accuracy_study1_humans <- study1_humans %>% filter(study1_humans$ground_truth == "Truthful")
round(sum(truth_accuracy_study1_humans$participant_response == "Truthful")/nrow(truth_accuracy_study1_humans)*100, digits = 2)

deception_accuracy_study1_humans <- study1_humans %>% filter(study1_humans$ground_truth == "Deceptive")
round(sum(deception_accuracy_study1_humans$participant_response == "Deceptive")/nrow(deception_accuracy_study1_humans)*100, digits = 2)

humans_vs_ai_study1 <- read.csv('study1_human_vs_ai_OSF.csv')
t.test(humans_vs_ai_study1$truth_bias ~ humans_vs_ai_study1$type)
cohens_d(
  humans_vs_ai_study1,
  truth_bias ~ type,
  ci = TRUE,
  conf.level = 0.95,
  ci.type = "perc",
  nboot = 5000
)

# study 2

study2_ai <- read.csv("study2_veracity_prompt_data_OSF.csv")
nrow(study2_ai)

# accuracy
round(sum(study2_ai$accuracy_2)/nrow(study2_ai)*100, digits = 2)
# overall truth-bias
round(sum(study2_ai$model_result == "Truthful")/nrow(study2_ai)*100, digits = 2)

truth_accuracy_study2_ai <- study2_ai %>% filter(study2_ai$ground_truth == "Truthful")
round(sum(truth_accuracy_study2_ai$model_result == "Truthful")/nrow(truth_accuracy_study2_ai)*100, digits = 2)

deception_accuracy_study2_ai <- study2_ai %>% filter(study2_ai$ground_truth == "Deceptive")
round(sum(deception_accuracy_study2_ai$model_result == "Deceptive")/nrow(deception_accuracy_study2_ai)*100, digits = 2)

# study 3 ai
study3_ai <- read.csv("study3_baserate_prompt_data_OSF.csv")
nrow(study3_ai)

# accuracy
round(sum(study3_ai$accuracy_2)/nrow(study3_ai)*100, digits = 2)
# overall truth-bias
round(sum(study3_ai$model_result == "Truthful")/nrow(study3_ai)*100, digits = 2)

truth_accuracy_study3_ai <- study3_ai %>% filter(study3_ai$ground_truth == "Truthful")
round(sum(truth_accuracy_study3_ai$model_result == "Truthful")/nrow(truth_accuracy_study3_ai)*100, digits = 2)

deception_accuracy_study3_ai <- study3_ai %>% filter(study3_ai$ground_truth == "Deceptive")
round(sum(deception_accuracy_study3_ai$model_result == "Deceptive")/nrow(deception_accuracy_study3_ai)*100, digits = 2)

# study 3 human data

study3_humans <- read.csv("study3_human_detection_OSF.csv")
nrow(study3_humans)

# accuracy
round(sum(study3_humans$accuracy_2)/nrow(study3_humans)*100, digits = 2)
# overall truth-bias
round(sum(study3_humans$participant_response == "Truthful")/nrow(study3_humans)*100, digits = 2)

truth_accuracy_study3_humans <- study3_humans %>% filter(study3_humans$ground_truth == "Truthful")
round(sum(truth_accuracy_study3_humans$participant_response == "Truthful")/nrow(truth_accuracy_study3_humans)*100, digits = 2)

deception_accuracy_study3_humans <- study3_humans %>% filter(study3_humans$ground_truth == "Deceptive")
round(sum(deception_accuracy_study3_humans$participant_response == "Deceptive")/nrow(deception_accuracy_study3_humans)*100, digits = 2)

# model specific findings
# change X in model == "X" to GPT3.5, Bard, or GPT4
# change X in sample == to hotel miami, or friends

# study 1 model specific findings
study1_neutral_model <- study1_ai %>% filter(sample == "hotel") %>% filter(model == "GPT4")
nrow(study1_neutral_model)
# accuracy
round(sum(study1_neutral_model$accuracy_2)/nrow(study1_neutral_model)*100, digits = 2)
# truth-bias
round(sum(study1_neutral_model$model_result == "Truthful")/nrow(study1_neutral_model)*100, digits = 2)

truth_accuracy_model_specific <- study1_neutral_model %>% filter(study1_neutral_model$ground_truth == "Truthful")
round(sum(truth_accuracy_model_specific$model_result == "Truthful")/nrow(truth_accuracy_model_specific)*100, digits = 2)

deception_accuracy_model_specific <- study1_neutral_model %>% filter(study1_neutral_model$ground_truth == "Deceptive")
round(sum(deception_accuracy_model_specific$model_result == "Deceptive")/nrow(deception_accuracy_model_specific)*100, digits = 2)

# study 2 model specific findings
study2_veracity_model <- study2_ai %>% filter(sample == "hotel") %>% filter(model == "GPT4")
nrow(study2_veracity_model)
# accuracy
round(sum(study2_veracity_model$accuracy_2)/nrow(study2_veracity_model)*100, digits = 2)
# truth-bias
round(sum(study2_veracity_model$model_result == "Truthful")/nrow(study2_veracity_model)*100, digits = 2)

truth_accuracy_model_specific <- study2_veracity_model %>% filter(study2_veracity_model$ground_truth == "Truthful")
round(sum(truth_accuracy_model_specific$model_result == "Truthful")/nrow(truth_accuracy_model_specific)*100, digits = 2)

deception_accuracy_model_specific <- study2_veracity_model %>% filter(study2_veracity_model$ground_truth == "Deceptive")
round(sum(deception_accuracy_model_specific$model_result == "Deceptive")/nrow(deception_accuracy_model_specific)*100, digits = 2)

# study 3 model specific findings
study3_baserate_model <- study3_ai %>% filter(sample == "hotel") %>% filter(model == "GPT4")
nrow(study3_baserate_model)
# accuracy
round(sum(study3_baserate_model$accuracy_2)/nrow(study3_baserate_model)*100, digits = 2)
# truth-bias
round(sum(study3_baserate_model$model_result == "Truthful")/nrow(study3_baserate_model)*100, digits = 2)

truth_accuracy_model_specific <- study3_baserate_model %>% filter(study3_baserate_model$ground_truth == "Truthful")
round(sum(truth_accuracy_model_specific$model_result == "Truthful")/nrow(truth_accuracy_model_specific)*100, digits = 2)

deception_accuracy_model_specific <- study3_baserate_model %>% filter(study3_baserate_model$ground_truth == "Deceptive")
round(sum(deception_accuracy_model_specific$model_result == "Deceptive")/nrow(deception_accuracy_model_specific)*100, digits = 2)

