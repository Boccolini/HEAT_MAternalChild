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
slice_name <- get_arg_value("--slice-name", "year2024_all_ufs_base")
output_dir <- path.expand(get_arg_value("--output-dir", "~/heat_mnch_remote/artifacts/remote_benchmarks/e002/output"))

if (is.null(input_path) || !file.exists(input_path)) {
  stop("Valid --input path is required")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(input_path, stringsAsFactors = FALSE)
df$municipality_code <- factor(df$municipality_code)
df$state_code <- factor(df$state_code)

tmax_mean <- mean(df$tmax_weekly_mean, na.rm = TRUE)
tmax_sd <- sd(df$tmax_weekly_mean, na.rm = TRUE)
age_mean <- mean(df$maternal_age_years, na.rm = TRUE)
age_sd <- sd(df$maternal_age_years, na.rm = TRUE)

df$tmax_weekly_mean_z <- (df$tmax_weekly_mean - tmax_mean) / tmax_sd
df$maternal_age_years_z <- (df$maternal_age_years - age_mean) / age_sd

df$maternal_age_group <- cut(
  df$maternal_age_years,
  breaks = c(-Inf, 19, 34, Inf),
  labels = c("<20", "20_34", "35plus"),
  right = TRUE,
  ordered_result = FALSE
)
df$maternal_age_group <- factor(df$maternal_age_group, levels = c("20_34", "<20", "35plus"))

fit_specs <- list(
  list(
    model_label = "model2_zscore",
    formula_text = "preterm_event_flag ~ tmax_weekly_mean_z + maternal_age_years_z + (1 | state_code) + (1 | municipality_code)"
  ),
  list(
    model_label = "model3_zscore_tmax_age_group",
    formula_text = "preterm_event_flag ~ tmax_weekly_mean_z + maternal_age_group + (1 | state_code) + (1 | municipality_code)"
  )
)

results <- list()

for (spec in fit_specs) {
  warns <- character()
  start <- proc.time()[["elapsed"]]
  fit <- withCallingHandlers(
    glmer(
      as.formula(spec$formula_text),
      data = df,
      family = binomial(link = "cloglog"),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 5e5))
    ),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  elapsed <- proc.time()[["elapsed"]] - start
  coef_row <- summary(fit)$coefficients["tmax_weekly_mean_z", , drop = FALSE]
  conv_messages <- fit@optinfo$conv$lme4$messages
  if (is.null(conv_messages)) {
    conv_messages <- character()
  }
  all_warnings <- unique(c(warns, as.character(conv_messages)))
  results[[length(results) + 1]] <- data.frame(
    slice_name = slice_name,
    model_label = spec$model_label,
    formula_text = spec$formula_text,
    row_count = nrow(df),
    event_count = sum(df$preterm_event_flag),
    municipality_count = length(unique(df$municipality_code)),
    state_count = length(unique(df$state_code)),
    estimate = unname(coef_row[1, "Estimate"]),
    std_error = unname(coef_row[1, "Std. Error"]),
    rr_per_sd = exp(unname(coef_row[1, "Estimate"])),
    ci_low_per_sd = exp(unname(coef_row[1, "Estimate"]) - 1.96 * unname(coef_row[1, "Std. Error"])),
    ci_high_per_sd = exp(unname(coef_row[1, "Estimate"]) + 1.96 * unname(coef_row[1, "Std. Error"])),
    beta_per_1c = unname(coef_row[1, "Estimate"]) / tmax_sd,
    rr_per_1c = exp(unname(coef_row[1, "Estimate"]) / tmax_sd),
    singular_flag = isSingular(fit, tol = 1e-5),
    elapsed_seconds = elapsed,
    tmax_mean = tmax_mean,
    tmax_sd = tmax_sd,
    age_mean = age_mean,
    age_sd = age_sd,
    convergence_code = fit@optinfo$conv$opt,
    warning_text = paste(all_warnings, collapse = " | "),
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, results)
output_path <- file.path(output_dir, sprintf("%s_diagnostics_result.csv", slice_name))
write.csv(out, output_path, row.names = FALSE)
print(out)
