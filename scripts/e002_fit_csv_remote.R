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
slice_name <- get_arg_value("--slice-name", "unnamed_slice")
output_dir <- path.expand(get_arg_value("--output-dir", "~/heat_mnch_remote/artifacts/remote_benchmarks/e002"))
input_profile <- get_arg_value("--input-profile", "full_stage")

if (is.null(input_path) || !file.exists(input_path)) {
  stop("Valid --input path is required")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

read_e002_input <- function(path, profile) {
  if (identical(profile, "minfit")) {
    col_classes <- c(
      "integer",
      "numeric",
      "numeric",
      "character",
      "character",
      "character",
      "character",
      "character",
      "character",
      "character"
    )
  } else {
    col_classes <- c(
      "NULL",
      "character",
      "character",
      "NULL",
      "integer",
      "numeric",
      "numeric",
      "character",
      "character",
      "character",
      "character",
      "character",
      "NULL",
      "NULL",
      "NULL"
    )
  }
  read.csv(path, stringsAsFactors = FALSE, colClasses = col_classes)
}

df <- read_e002_input(input_path, input_profile)
df$maternal_schooling_code <- factor(df$maternal_schooling_code)
df$parity_total_prior <- factor(df$parity_total_prior)
df$plurality_code <- factor(df$plurality_code)
df$newborn_sex_code <- factor(df$newborn_sex_code)
df$maternal_race_color_code <- factor(df$maternal_race_color_code)
df$municipality_code <- factor(df$municipality_code)
df$state_code <- factor(df$state_code)

warns <- character()
fixed_terms <- c(
  "tmax_weekly_mean",
  "maternal_age_years",
  "maternal_schooling_code",
  "parity_total_prior",
  "plurality_code",
  "newborn_sex_code",
  "maternal_race_color_code"
)
invariant_terms <- character()
for (term in c("maternal_schooling_code", "parity_total_prior", "plurality_code", "newborn_sex_code", "maternal_race_color_code")) {
  if (length(unique(df[[term]])) < 2) {
    invariant_terms <- c(invariant_terms, term)
  }
}
if ("plurality_code" %in% invariant_terms) {
  warns <- c(
    warns,
    "plurality_code was invariant in the singleton-restricted slice and was dropped from the fixed-effects benchmark formula."
  )
}
active_terms <- setdiff(fixed_terms, invariant_terms)
formula_text <- paste(
  "preterm_event_flag ~",
  paste(active_terms, collapse = " + "),
  "+ (1 | state_code) + (1 | municipality_code)"
)
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
estimate <- unname(coef_row[1, "Estimate"])
std_error <- unname(coef_row[1, "Std. Error"])

out <- data.frame(
  slice_name = slice_name,
  input_profile = input_profile,
  row_count = nrow(df),
  event_count = sum(df$preterm_event_flag),
  municipality_count = length(unique(df$municipality_code)),
  state_count = length(unique(df$state_code)),
  estimate = estimate,
  std_error = std_error,
  rr = exp(estimate),
  ci_low = exp(estimate - 1.96 * std_error),
  ci_high = exp(estimate + 1.96 * std_error),
  singular_flag = isSingular(fit, tol = 1e-5),
  elapsed_seconds = elapsed,
  dropped_invariant_terms = paste(invariant_terms, collapse = " | "),
  warning_text = paste(unique(warns), collapse = " | "),
  stringsAsFactors = FALSE
)

output_path <- file.path(output_dir, sprintf("%s_result.csv", slice_name))
write.csv(out, output_path, row.names = FALSE)
print(out)
