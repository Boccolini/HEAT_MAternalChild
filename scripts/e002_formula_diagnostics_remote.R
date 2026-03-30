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
if (is.null(input_path) || !file.exists(input_path)) {
  stop("Valid --input path is required")
}

df <- read.csv(input_path, stringsAsFactors = FALSE)
df$maternal_schooling_code <- factor(df$maternal_schooling_code)
df$parity_total_prior <- factor(df$parity_total_prior)
df$plurality_code <- factor(df$plurality_code)
df$newborn_sex_code <- factor(df$newborn_sex_code)
df$maternal_race_color_code <- factor(df$maternal_race_color_code)
df$municipality_code <- factor(df$municipality_code)
df$state_code <- factor(df$state_code)

cat("Factor levels\n")
for (term in c(
  "maternal_schooling_code",
  "parity_total_prior",
  "plurality_code",
  "newborn_sex_code",
  "maternal_race_color_code",
  "municipality_code",
  "state_code"
)) {
  cat(term, "->", nlevels(df[[term]]), "\n")
}

formulas <- list(
  simple = preterm_event_flag ~ tmax_weekly_mean + maternal_age_years + (1 | state_code) + (1 | municipality_code),
  schooling = preterm_event_flag ~ tmax_weekly_mean + maternal_age_years + maternal_schooling_code + (1 | state_code) + (1 | municipality_code),
  parity = preterm_event_flag ~ tmax_weekly_mean + maternal_age_years + maternal_schooling_code + parity_total_prior + (1 | state_code) + (1 | municipality_code),
  sex = preterm_event_flag ~ tmax_weekly_mean + maternal_age_years + maternal_schooling_code + parity_total_prior + newborn_sex_code + (1 | state_code) + (1 | municipality_code),
  race = preterm_event_flag ~ tmax_weekly_mean + maternal_age_years + maternal_schooling_code + parity_total_prior + newborn_sex_code + maternal_race_color_code + (1 | state_code) + (1 | municipality_code)
)

for (name in names(formulas)) {
  cat("\nFormula:", name, "\n")
  formula_obj <- formulas[[name]]
  mm <- try(model.matrix(lme4::nobars(formula_obj), data = df), silent = TRUE)
  if (inherits(mm, "try-error")) {
    cat("model.matrix failed\n")
    print(mm)
    next
  }
  cat("model.matrix ok:", nrow(mm), "rows,", ncol(mm), "cols\n")
  fit <- try(
    glmer(
      formula_obj,
      data = df,
      family = binomial(link = "cloglog"),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 5e4))
    ),
    silent = TRUE
  )
  if (inherits(fit, "try-error")) {
    cat("glmer failed\n")
    print(fit)
    next
  }
  coef_row <- summary(fit)$coefficients["tmax_weekly_mean", , drop = FALSE]
  cat("glmer ok, estimate =", coef_row[1, "Estimate"], "stderr =", coef_row[1, "Std. Error"], "\n")
}
