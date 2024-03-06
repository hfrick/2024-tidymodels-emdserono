library(tidymodels)
library(censored)
library(survminer)

theme_set(theme_minimal())

data("wa_churn", package = "modeldata")

wa_churn <- wa_churn %>%
  drop_na() %>%
  mutate(
    multiple_lines = if_else(multiple_lines == "No phone service", "No", multiple_lines),
    across(all_of(c("online_security", "online_backup",
                    "device_protection", "tech_support", "streaming_tv", 
                    "streaming_movies")), 
           ~ if_else(.x == "No internet service", "No", .x))
  ) %>% 
  select(-contract, -total_charges)

# large amount of censoring
ggplot(wa_churn) + geom_bar(aes(churn))

# make Surv object at the start
telco_churn <- wa_churn %>% 
  mutate(
    churn_surv = Surv(tenure, if_else(churn == "Yes", 1, 0)),
    .keep = "unused"
  )


# Split the data ----------------------------------------------------------

set.seed(403)
telco_split <- initial_split(telco_churn)
telco_train <- training(telco_split)
telco_test <- testing(telco_split)

telco_rs <- vfold_cv(telco_train)


# EDA ---------------------------------------------------------------------


# response distribution
survfit(churn_surv ~ 1, data = telco_train) %>% ggsurvplot(legend = "none")

# see some pretty clear effects
survfit(churn_surv ~ partner, data = telco_train) %>% ggsurvplot() 

# possibly combine these?
survfit(churn_surv ~ streaming_tv, data = telco_train) %>% ggsurvplot()
survfit(churn_surv ~ streaming_movies, data = telco_train) %>% ggsurvplot() 


# preprocessing -----------------------------------------------------------

telco_rec <- recipe(churn_surv ~ ., data = telco_train) %>% 
  step_zv(all_predictors()) 

telco_rec_streaming <- telco_rec %>%
  step_mutate(streaming = factor(if_else(streaming_tv == "Yes" | 
                                           streaming_movies == "Yes", "Yes", "No"))) %>% 
  step_rm(streaming_tv, streaming_movies)


# baseline model ----------------------------------------------------------

spec_surv_reg <- survival_reg()

wflow_surv_reg <- workflow() %>%
  add_recipe(telco_rec) %>%
  add_model(spec_surv_reg)

set.seed(12)
sr_rs_fit <- fit_resamples(
  wflow_surv_reg, 
  telco_rs, 
  metrics = metric_set(brier_survival_integrated, brier_survival, 
                       roc_auc_survival, concordance_survival), 
  eval_time = c(1, 6, 12, 18, 24, 36, 48, 60)
)

collect_metrics(sr_rs_fit)

show_best(sr_rs_fit, metric = "brier_survival_integrated")


# tune a model ------------------------------------------------------------

spec_tree <- decision_tree(tree_depth = tune(), min_n = tune(), cost_complexity = tune()) %>% 
  set_engine("rpart") %>%
  set_mode("censored regression")

wflow_tree <- workflow() %>%
  add_recipe(telco_rec) %>%
  add_model(spec_tree)

set.seed(12)
tree_res <- tune_grid(
  wflow_tree, 
  telco_rs, 
  grid = 10,
  metrics = metric_set(brier_survival_integrated, brier_survival, roc_auc_survival, concordance_survival), 
  eval_time = c(1, 6, 12, 18, 24, 36, 48, 60)
)

collect_metrics(tree_res)
show_best(tree_res, metric = "brier_survival_integrated")


# workflowsets ------------------------------------------------------------

# reduced version of the kitchen sink 

spec_cox <- proportional_hazards() %>% 
  set_engine("survival")

spec_oblique_forest <- rand_forest(min_n = tune(), mtry = tune()) %>% 
  set_engine("aorsf") %>% 
  set_mode("censored regression")

wflow_set <- workflow_set(
  list(recipe_base = telco_rec, recipe_streaming = telco_rec_streaming), 
  list(spec_surv_reg, spec_cox, spec_tree, spec_oblique_forest)
)

set.seed(403)
wflow_set_res <- workflow_map(
  wflow_set,
  "tune_grid", 
  resamples = telco_rs,
  grid = 10,
  metrics = metric_set(brier_survival_integrated, brier_survival, roc_auc_survival, concordance_survival), 
  eval_time = c(1, 6, 12, 18, 24, 36, 48, 60),
  control = control_grid(save_workflow = TRUE)
)

rank_results(wflow_set_res, rank_metric = "brier_survival_integrated", select_best = TRUE) %>% 
  filter(.metric == "brier_survival_integrated") 


# final model ------------------------------------------------------------

churn_mod <- fit_best(wflow_set_res, metric = "brier_survival_integrated")

test_predictions <- augment(churn_mod, telco_test, 
                            eval_time = c(1, 6, 12, 18, 24, 36, 48, 60)) 
brier_survival_integrated(test_predictions, truth = churn_surv, .pred)
