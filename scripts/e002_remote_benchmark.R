args <- commandArgs(trailingOnly = TRUE)

get_arg_value <- function(flag, default = NULL) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (length(hit) == 0) {
    return(default)
  }
  sub(paste0("^", flag, "="), "", hit[[1]])
}

root_dir <- normalizePath(get_arg_value("--root", "."), mustWork = TRUE)
output_dir <- get_arg_value(
  "--output-dir",
  file.path(root_dir, "artifacts/remote_benchmarks/e002")
)
slice_name <- get_arg_value("--slice", "micro_validation")
lib_dir <- path.expand(get_arg_value(
  "--lib-dir",
  "~/R/x86_64-redhat-linux-gnu-library/4.3"
))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib_dir, .libPaths()))

suppressPackageStartupMessages({
  library(duckdb)
  library(lme4)
})

mart_glob <- file.path(
  root_dir,
  "artifacts/analysis/shared_marts/flagship_gestational_weekly_person_period_mart/YEAR=*/*.parquet"
)

slice_registry <- list(
  micro_validation = list(
    years = c(2020L),
    states = c("12", "14", "16"),
    municipality_limit = 2L
  ),
  year2020_three_states_full = list(
    years = c(2020L),
    states = c("12", "14", "16"),
    municipality_limit = NA_integer_
  ),
  year2020_state12_full = list(
    years = c(2020L),
    states = c("12"),
    municipality_limit = NA_integer_
  )
)

if (!slice_name %in% names(slice_registry)) {
  stop(
    sprintf(
      "Unknown --slice value '%s'. Valid values: %s",
      slice_name,
      paste(names(slice_registry), collapse = ", ")
    )
  )
}

slice_spec <- slice_registry[[slice_name]]

state_in_sql <- paste(sprintf("'%s'", slice_spec$states), collapse = ", ")
year_in_sql <- paste(slice_spec$years, collapse = ", ")

con <- dbConnect(duckdb::duckdb(), dbdir = ":memory:")
on.exit(dbDisconnect(con, shutdown = TRUE), add = TRUE)

scratch_dir <- file.path(output_dir, "duckdb_tmp")
dir.create(scratch_dir, recursive = TRUE, showWarnings = FALSE)
dbExecute(con, "PRAGMA threads=4")
dbExecute(con, "SET preserve_insertion_order = false")
dbExecute(con, "SET memory_limit = '20GB'")
dbExecute(con, "PRAGMA max_temp_directory_size='120GiB'")
dbExecute(con, sprintf("SET temp_directory = '%s'", normalizePath(scratch_dir)))

base_filtered_sql <- sprintf(
  "
  SELECT
    pregnancy_id,
    municipality_code,
    state_code,
    gestational_week_completed,
    preterm_event_flag,
    tmax_weekly_mean,
    maternal_age_years,
    maternal_schooling_code,
    parity_total_prior,
    plurality_code,
    newborn_sex_code,
    maternal_race_color_code
  FROM read_parquet('%s', hive_partitioning = true)
  WHERE birth_record_year IN (%s)
    AND state_code IN (%s)
    AND tmax_weekly_mean IS NOT NULL
    AND maternal_age_years IS NOT NULL
    AND maternal_schooling_code IS NOT NULL
    AND parity_total_prior IS NOT NULL
    AND plurality_code IS NOT NULL
    AND newborn_sex_code IS NOT NULL
  ",
  mart_glob,
  year_in_sql,
  state_in_sql
)

if (!is.na(slice_spec$municipality_limit)) {
  municipality_sql <- sprintf(
    "
    WITH base AS (
      %s
    ),
    municipalities AS (
      SELECT municipality_code
      FROM base
      GROUP BY municipality_code
      ORDER BY municipality_code
      LIMIT %d
    )
    SELECT b.*
    FROM base AS b
    INNER JOIN municipalities AS m
      ON b.municipality_code = m.municipality_code
    ",
    base_filtered_sql,
    slice_spec$municipality_limit
  )
} else {
  municipality_sql <- base_filtered_sql
}

summary_sql <- sprintf(
  "
  WITH filtered AS (
    %s
  )
  SELECT
    COUNT(*) AS row_count,
    SUM(CAST(preterm_event_flag AS BIGINT)) AS event_count,
    COUNT(DISTINCT municipality_code) AS municipality_count,
    COUNT(DISTINCT state_code) AS state_count,
    MIN(gestational_week_completed) AS min_week,
    MAX(gestational_week_completed) AS max_week
  FROM filtered
  ",
  municipality_sql
)

summary_df <- dbGetQuery(con, summary_sql)

message(
  sprintf(
    "Slice %s -> rows=%s events=%s municipalities=%s states=%s",
    slice_name,
    format(summary_df$row_count[[1]], big.mark = ","),
    format(summary_df$event_count[[1]], big.mark = ","),
    summary_df$municipality_count[[1]],
    summary_df$state_count[[1]]
  )
)

dataset_query <- sprintf(
  "
  WITH filtered AS (
    %s
  )
  SELECT
    CAST(preterm_event_flag AS INTEGER) AS preterm_event_flag,
    CAST(tmax_weekly_mean AS DOUBLE) AS tmax_weekly_mean,
    CAST(maternal_age_years AS DOUBLE) AS maternal_age_years,
    CAST(maternal_schooling_code AS VARCHAR) AS maternal_schooling_code,
    CAST(parity_total_prior AS VARCHAR) AS parity_total_prior,
    CAST(plurality_code AS VARCHAR) AS plurality_code,
    CAST(newborn_sex_code AS VARCHAR) AS newborn_sex_code,
    CAST(COALESCE(maternal_race_color_code, 'unknown') AS VARCHAR) AS maternal_race_color_code,
    CAST(municipality_code AS VARCHAR) AS municipality_code,
    CAST(state_code AS VARCHAR) AS state_code
  FROM filtered
  ORDER BY state_code, municipality_code, gestational_week_completed
  ",
  municipality_sql
)

dataset_df <- dbGetQuery(con, dataset_query)

dataset_df$maternal_schooling_code <- factor(dataset_df$maternal_schooling_code)
dataset_df$parity_total_prior <- factor(dataset_df$parity_total_prior)
dataset_df$plurality_code <- factor(dataset_df$plurality_code)
dataset_df$newborn_sex_code <- factor(dataset_df$newborn_sex_code)
dataset_df$maternal_race_color_code <- factor(dataset_df$maternal_race_color_code)
dataset_df$municipality_code <- factor(dataset_df$municipality_code)
dataset_df$state_code <- factor(dataset_df$state_code)

formula <- stats::as.formula(
  "preterm_event_flag ~ tmax_weekly_mean + maternal_age_years + maternal_schooling_code + parity_total_prior + plurality_code + newborn_sex_code + maternal_race_color_code + (1 | state_code) + (1 | municipality_code)"
)

fit_warnings <- character()
warning_handler <- function(w) {
  fit_warnings <<- c(fit_warnings, conditionMessage(w))
  invokeRestart("muffleWarning")
}

start_time <- proc.time()[["elapsed"]]
fit <- withCallingHandlers(
  glmer(
    formula = formula,
    data = dataset_df,
    family = binomial(link = "cloglog"),
    control = glmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = 2e5)
    )
  ),
  warning = warning_handler
)
elapsed_seconds <- proc.time()[["elapsed"]] - start_time

fit_summary <- summary(fit)
coef_row <- fit_summary$coefficients["tmax_weekly_mean", , drop = FALSE]
estimate <- unname(coef_row[1, "Estimate"])
std_error <- unname(coef_row[1, "Std. Error"])
rr <- exp(estimate)
ci_low <- exp(estimate - 1.96 * std_error)
ci_high <- exp(estimate + 1.96 * std_error)
singular_flag <- isSingular(fit, tol = 1e-5)
log_lik <- as.numeric(logLik(fit))

result_df <- data.frame(
  slice_name = slice_name,
  row_count = summary_df$row_count[[1]],
  event_count = summary_df$event_count[[1]],
  municipality_count = summary_df$municipality_count[[1]],
  state_count = summary_df$state_count[[1]],
  min_week = summary_df$min_week[[1]],
  max_week = summary_df$max_week[[1]],
  estimate = estimate,
  std_error = std_error,
  rr = rr,
  ci_low = ci_low,
  ci_high = ci_high,
  log_likelihood = log_lik,
  singular_flag = singular_flag,
  elapsed_seconds = elapsed_seconds,
  warning_count = length(fit_warnings),
  warning_text = paste(unique(fit_warnings), collapse = " | "),
  stringsAsFactors = FALSE
)

csv_path <- file.path(output_dir, sprintf("e002_%s_result.csv", slice_name))
txt_path <- file.path(output_dir, sprintf("e002_%s_result.txt", slice_name))
stats::write.csv(result_df, csv_path, row.names = FALSE)

lines_out <- c(
  sprintf("Slice: %s", slice_name),
  sprintf("Rows: %s", format(summary_df$row_count[[1]], big.mark = ",")),
  sprintf("Events: %s", format(summary_df$event_count[[1]], big.mark = ",")),
  sprintf("Municipalities: %s", summary_df$municipality_count[[1]]),
  sprintf("States: %s", summary_df$state_count[[1]]),
  sprintf("Weeks covered: %s-%s", summary_df$min_week[[1]], summary_df$max_week[[1]]),
  sprintf("Estimate (tmax_weekly_mean): %.8f", estimate),
  sprintf("RR: %.8f", rr),
  sprintf("95%% CI: %.8f to %.8f", ci_low, ci_high),
  sprintf("Singular flag: %s", singular_flag),
  sprintf("Elapsed seconds: %.2f", elapsed_seconds),
  sprintf("Warnings: %s", ifelse(length(fit_warnings) == 0, "none", paste(unique(fit_warnings), collapse = " | ")))
)
writeLines(lines_out, txt_path)

print(result_df)
