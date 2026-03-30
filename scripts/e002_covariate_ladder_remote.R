args <- commandArgs(trailingOnly = TRUE)

get_arg_value <- function(flag, default = NULL) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (length(hit) == 0) {
    return(default)
  }
  sub(paste0("^", flag, "="), "", hit[[1]])
}

.libPaths(c(path.expand(get_arg_value("--lib-dir", "~/R/x86_64-redhat-linux-gnu-library/4.3")), .libPaths()))
suppressPackageStartupMessages(library(lme4))

input_path <- path.expand(get_arg_value("--input"))
slice_name <- get_arg_value("--slice-name", "year2020_states_12_14_16_remote")
output_dir <- path.expand(get_arg_value("--output-dir", "~/heat_mnch_remote/artifacts/remote_benchmarks/e002/output"))

if (is.null(input_path) || !file.exists(input_path)) {
  stop("Valid --input path is required")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(input_path, stringsAsFactors = FALSE)
df$maternal_schooling_code <- factor(df$maternal_schooling_code)
df$parity_total_prior <- factor(df$parity_total_prior)
df$plurality_code <- factor(df$plurality_code)
df$newborn_sex_code <- factor(df$newborn_sex_code)
df$maternal_race_color_code <- factor(df$maternal_race_color_code)
df$municipality_code <- factor(df$municipality_code)
df$state_code <- factor(df$state_code)

fit_specs <- list(
  list(
    model_label = "base_min",
    fixed_terms = c("tmax_weekly_mean", "maternal_age_years"),
    random_terms = c("(1 | state_code)", "(1 | municipality_code)")
  ),
  list(
    model_label = "base_intermediate",
    fixed_terms = c(
      "tmax_weekly_mean",
      "maternal_age_years",
      "maternal_schooling_code",
      "parity_total_prior"
    ),
    random_terms = c("(1 | state_code)", "(1 | municipality_code)")
  ),
  list(
    model_label = "base_intermediate_plus_sex",
    fixed_terms = c(
      "tmax_weekly_mean",
      "maternal_age_years",
      "maternal_schooling_code",
      "parity_total_prior",
      "newborn_sex_code"
    ),
    random_terms = c("(1 | state_code)", "(1 | municipality_code)")
  ),
  list(
    model_label = "no_race_random_state",
    fixed_terms = c(
      "tmax_weekly_mean",
      "maternal_age_years",
      "maternal_schooling_code",
      "parity_total_prior",
      "newborn_sex_code"
    ),
    random_terms = c("(1 | state_code)", "(1 | municipality_code)")
  ),
  list(
    model_label = "no_race_fixed_state",
    fixed_terms = c(
      "tmax_weekly_mean",
      "maternal_age_years",
      "maternal_schooling_code",
      "parity_total_prior",
      "newborn_sex_code",
      "state_code"
    ),
    random_terms = c("(1 | municipality_code)")
  ),
  list(
    model_label = "full_current",
    fixed_terms = c(
      "tmax_weekly_mean",
      "maternal_age_years",
      "maternal_schooling_code",
      "parity_total_prior",
      "newborn_sex_code",
      "maternal_race_color_code"
    ),
    random_terms = c("(1 | state_code)", "(1 | municipality_code)")
  )
)

droppable_terms <- c(
  "maternal_schooling_code",
  "parity_total_prior",
  "plurality_code",
  "newborn_sex_code",
  "maternal_race_color_code",
  "state_code"
)

results <- list()

for (spec in fit_specs) {
  warns <- character()
  invariant_terms <- character()
  for (term in intersect(spec$fixed_terms, droppable_terms)) {
    if (length(unique(df[[term]])) < 2) {
      invariant_terms <- c(invariant_terms, term)
    }
  }
  active_fixed_terms <- setdiff(spec$fixed_terms, invariant_terms)
  formula_parts <- c(active_fixed_terms, spec$random_terms)
  formula_text <- paste("preterm_event_flag ~", paste(formula_parts, collapse = " + "))
  start <- proc.time()[["elapsed"]]
  fit <- withCallingHandlers(
    glmer(
      as.formula(formula_text),
      data = df,
      family = binomial(link = "cloglog"),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    ),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  elapsed <- proc.time()[["elapsed"]] - start
  coef_row <- summary(fit)$coefficients["tmax_weekly_mean", , drop = FALSE]
  results[[length(results) + 1]] <- data.frame(
    slice_name = slice_name,
    model_label = spec$model_label,
    formula_text = formula_text,
    row_count = nrow(df),
    event_count = sum(df$preterm_event_flag),
    municipality_count = length(unique(df$municipality_code)),
    state_count = length(unique(df$state_code)),
    estimate = unname(coef_row[1, "Estimate"]),
    std_error = unname(coef_row[1, "Std. Error"]),
    rr = exp(unname(coef_row[1, "Estimate"])),
    ci_low = exp(unname(coef_row[1, "Estimate"]) - 1.96 * unname(coef_row[1, "Std. Error"])),
    ci_high = exp(unname(coef_row[1, "Estimate"]) + 1.96 * unname(coef_row[1, "Std. Error"])),
    singular_flag = isSingular(fit, tol = 1e-5),
    elapsed_seconds = elapsed,
    dropped_invariant_terms = paste(invariant_terms, collapse = " | "),
    warning_text = paste(unique(warns), collapse = " | "),
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, results)
output_path <- file.path(output_dir, sprintf("%s_covariate_ladder.csv", slice_name))
write.csv(out, output_path, row.names = FALSE)
print(out)
