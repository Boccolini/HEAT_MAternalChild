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
slice_name <- get_arg_value("--slice-name", "stepup_slice")
output_dir <- path.expand(get_arg_value("--output-dir", "~/heat_mnch_remote/artifacts/remote_benchmarks/e002"))

df <- read.csv(input_path, stringsAsFactors = FALSE)
df$maternal_schooling_code <- factor(df$maternal_schooling_code)
df$parity_total_prior <- factor(df$parity_total_prior)
df$newborn_sex_code <- factor(df$newborn_sex_code)
df$maternal_race_color_code <- factor(df$maternal_race_color_code)
df$municipality_code <- factor(df$municipality_code)
df$state_code <- factor(df$state_code)

warns <- character()
start <- proc.time()[["elapsed"]]
fit <- withCallingHandlers(
  glmer(
    preterm_event_flag ~ tmax_weekly_mean + maternal_age_years +
      maternal_schooling_code + parity_total_prior +
      newborn_sex_code + maternal_race_color_code +
      (1 | state_code) + (1 | municipality_code),
    data = df,
    family = binomial(link = "cloglog"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  ),
  warning = function(w) {
    warns <<- c(warns, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
)
elapsed <- proc.time()[["elapsed"]] - start

coef_row <- summary(fit)$coefficients["tmax_weekly_mean", , drop = FALSE]
out <- data.frame(
  slice_name = slice_name,
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
  warning_text = paste(unique(warns), collapse = " | "),
  stringsAsFactors = FALSE
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(out, file.path(output_dir, sprintf("%s_stepup_result.csv", slice_name)), row.names = FALSE)
print(out)
