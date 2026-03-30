args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(name, default = "") {
  index <- match(name, args)
  if (is.na(index) || index == length(args)) {
    return(default)
  }
  args[[index + 1]]
}

if (!requireNamespace("fixest", quietly = TRUE)) {
  stop("Package 'fixest' is required for phase3")
}

registry_path <- parse_arg("--registry")
input_dir <- parse_arg("--input-dir")
output_dir <- parse_arg("--output-dir")

if (!nzchar(registry_path) || !nzchar(input_dir) || !nzchar(output_dir)) {
  stop("Missing required arguments: --registry, --input-dir, --output-dir")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
registry <- read.csv(registry_path, stringsAsFactors = FALSE)
registry <- subset(registry, case_crossover_required_flag %in% c(TRUE, "True", "TRUE", 1, "1"))

results <- list()
comparisons <- list()
diagnostic_rows <- list()

record_diagnostic <- function(
  analysis_unit_id,
  status,
  failure_stage,
  row_count = NA_integer_,
  warning_messages = character(),
  error_message = "",
  note = ""
) {
  diagnostic_rows[[length(diagnostic_rows) + 1]] <<- data.frame(
    analysis_unit_id = analysis_unit_id,
    status = status,
    failure_stage = failure_stage,
    row_count = row_count,
    warning_messages = paste(unique(Filter(nzchar, warning_messages)), collapse = " | "),
    error_message = error_message,
    note = note,
    stringsAsFactors = FALSE
  )
}

safe_log_effect <- function(value) {
  numeric_value <- suppressWarnings(as.numeric(value))
  if (!is.finite(numeric_value) || numeric_value <= 0) {
    return(NA_real_)
  }
  log(numeric_value)
}

for (i in seq_len(nrow(registry))) {
  row <- registry[i, ]
  analysis_unit_id <- row$analysis_unit_id
  input_path <- file.path(input_dir, paste0(analysis_unit_id, ".csv"))
  warning_messages <- character()
  message(sprintf("[phase3] outcome=%s stage=start", analysis_unit_id))

  if (!file.exists(input_path)) {
    record_diagnostic(
      analysis_unit_id = analysis_unit_id,
      status = "skipped",
      failure_stage = "input_missing",
      note = "Input CSV not found"
    )
    message(sprintf("[phase3] outcome=%s stage=input_missing", analysis_unit_id))
    next
  }

  frame <- tryCatch(
    read.csv(input_path, stringsAsFactors = FALSE),
    error = function(e) {
      record_diagnostic(
        analysis_unit_id = analysis_unit_id,
        status = "error",
        failure_stage = "read_csv",
        warning_messages = warning_messages,
        error_message = conditionMessage(e)
      )
      message(sprintf("[phase3] outcome=%s stage=read_csv status=error error=%s", analysis_unit_id, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(frame)) {
    next
  }
  if (nrow(frame) == 0) {
    record_diagnostic(
      analysis_unit_id = analysis_unit_id,
      status = "skipped",
      failure_stage = "empty_input",
      row_count = 0,
      note = "No rows available after upstream filters"
    )
    message(sprintf("[phase3] outcome=%s stage=empty_input", analysis_unit_id))
    next
  }

  frame$denominator_count <- pmax(as.numeric(frame$denominator_count), 1)
  frame$outcome_count <- as.numeric(frame$outcome_count)
  frame$exposure_value <- as.numeric(frame$exposure_value)
  frame$log_denominator <- log(frame$denominator_count)
  row_count <- nrow(frame)

  strata_result <- tryCatch(
    {
      frame$strata_id <- interaction(
        frame$municipality_code,
        frame$calendar_year,
        frame$calendar_month,
        frame$day_of_week,
        drop = TRUE,
        lex.order = TRUE
      )
      frame
    },
    error = function(e) {
      record_diagnostic(
        analysis_unit_id = analysis_unit_id,
        status = "error",
        failure_stage = "strata_build",
        row_count = row_count,
        warning_messages = warning_messages,
        error_message = conditionMessage(e)
      )
      message(sprintf("[phase3] outcome=%s stage=strata_build status=error error=%s", analysis_unit_id, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(strata_result)) {
    rm(frame)
    invisible(gc(verbose = FALSE))
    next
  }
  frame <- strata_result

  fit <- tryCatch(
    withCallingHandlers(
      fixest::feglm(
        fml = outcome_count ~ exposure_value | strata_id,
        family = "poisson",
        offset = ~ log_denominator,
        data = frame,
        lean = TRUE,
        mem.clean = TRUE,
        warn = TRUE
      ),
      warning = function(w) {
        warning_messages <<- c(warning_messages, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      record_diagnostic(
        analysis_unit_id = analysis_unit_id,
        status = "error",
        failure_stage = "feglm",
        row_count = row_count,
        warning_messages = warning_messages,
        error_message = conditionMessage(e)
      )
      message(sprintf("[phase3] outcome=%s stage=feglm status=error error=%s", analysis_unit_id, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(fit)) {
    rm(frame)
    invisible(gc(verbose = FALSE))
    next
  }

  estimate_result <- tryCatch(
    {
      coefficient <- stats::coef(fit)[["exposure_value"]]
      vcov_matrix <- stats::vcov(fit)
      if (!("exposure_value" %in% rownames(vcov_matrix))) {
        stop("Exposure coefficient missing from covariance matrix")
      }
      se <- sqrt(vcov_matrix["exposure_value", "exposure_value"])
      if (!is.finite(coefficient) || !is.finite(se)) {
        stop("Non-finite coefficient or standard error")
      }
      p_value <- 2 * stats::pnorm(abs(coefficient / se), lower.tail = FALSE)
      list(coefficient = coefficient, se = se, p_value = p_value)
    },
    error = function(e) {
      record_diagnostic(
        analysis_unit_id = analysis_unit_id,
        status = "error",
        failure_stage = "coef_summary",
        row_count = row_count,
        warning_messages = warning_messages,
        error_message = conditionMessage(e)
      )
      message(sprintf("[phase3] outcome=%s stage=coef_summary status=error error=%s", analysis_unit_id, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(estimate_result)) {
    rm(frame, fit)
    invisible(gc(verbose = FALSE))
    next
  }

  estimate <- estimate_result$coefficient
  se <- estimate_result$se
  rr <- exp(estimate)
  lower_ci <- exp(estimate - 1.96 * se)
  upper_ci <- exp(estimate + 1.96 * se)
  p_value <- estimate_result$p_value
  current_effect <- safe_log_effect(row$current_estimate)
  case_effect <- estimate
  magnitude_ratio <- if (is.finite(current_effect) && abs(current_effect) > 0) abs(case_effect) / abs(current_effect) else NA_real_
  current_sign <- sign(current_effect)
  case_sign <- sign(case_effect)
  concordance_classification <- if (!is.finite(current_effect)) {
    "partial_concordance"
  } else if (!identical(current_sign, case_sign)) {
    "discordant"
  } else if (is.finite(magnitude_ratio) && magnitude_ratio >= 0.5 && magnitude_ratio <= 1.5) {
    "concordant"
  } else {
    "partial_concordance"
  }

  results[[length(results) + 1]] <- data.frame(
    analysis_unit_id = analysis_unit_id,
    case_crossover_estimate = rr,
    case_crossover_lower_ci = lower_ci,
    case_crossover_upper_ci = upper_ci,
    case_crossover_p_value = p_value,
    observed_rows = row_count,
    stringsAsFactors = FALSE
  )
  comparisons[[length(comparisons) + 1]] <- data.frame(
    analysis_unit_id = analysis_unit_id,
    current_estimate = as.numeric(row$current_estimate),
    current_lower_ci = as.numeric(row$current_lower_ci),
    current_upper_ci = as.numeric(row$current_upper_ci),
    case_crossover_estimate = rr,
    case_crossover_lower_ci = lower_ci,
    case_crossover_upper_ci = upper_ci,
    log_effect_ratio = magnitude_ratio,
    concordance_classification = concordance_classification,
    stringsAsFactors = FALSE
  )
  record_diagnostic(
    analysis_unit_id = analysis_unit_id,
    status = "success",
    failure_stage = "completed",
    row_count = row_count,
    warning_messages = warning_messages,
    note = concordance_classification
  )
  message(sprintf("[phase3] outcome=%s stage=completed classification=%s", analysis_unit_id, concordance_classification))
  rm(frame, fit)
  invisible(gc(verbose = FALSE))
}

results_frame <- if (length(results) > 0) do.call(rbind, results) else data.frame()
comparison_frame <- if (length(comparisons) > 0) do.call(rbind, comparisons) else data.frame()
diagnostic_frame <- if (length(diagnostic_rows) > 0) {
  do.call(rbind, diagnostic_rows)
} else {
  data.frame(
    analysis_unit_id = character(),
    status = character(),
    failure_stage = character(),
    row_count = integer(),
    warning_messages = character(),
    error_message = character(),
    note = character(),
    stringsAsFactors = FALSE
  )
}

write.csv(results_frame, file.path(output_dir, "case_crossover_results_table.csv"), row.names = FALSE)
write.csv(comparison_frame, file.path(output_dir, "fe_proxy_vs_case_crossover_comparison_table.csv"), row.names = FALSE)
write.csv(diagnostic_frame, file.path(output_dir, "case_crossover_outcome_diagnostics.csv"), row.names = FALSE)

classification_count <- function(classification_label) {
  if (!("concordance_classification" %in% names(comparison_frame))) {
    return(0)
  }
  sum(comparison_frame$concordance_classification == classification_label, na.rm = TRUE)
}

failed_rows <- diagnostic_frame[diagnostic_frame$status != "success", , drop = FALSE]
failure_stage_lines <- character()
if (nrow(failed_rows) > 0) {
  failure_stage_counts <- sort(table(failed_rows$failure_stage), decreasing = TRUE)
  failure_stage_lines <- c(
    "",
    "- Failure stages:",
    sprintf("  - `%s`: `%s`", names(failure_stage_counts), as.integer(failure_stage_counts))
  )
}

sample_error_lines <- character()
if (nrow(failed_rows) > 0 && any(nzchar(failed_rows$error_message))) {
  error_messages <- unique(failed_rows$error_message[nzchar(failed_rows$error_message)])
  sample_error_lines <- c(
    "",
    "- Sample errors:",
    sprintf("  - `%s`", utils::head(error_messages, 5))
  )
}

note_lines <- c(
  "# Acute Design Concordance Note",
  "",
  sprintf("- Outcomes attempted: `%s`", nrow(diagnostic_frame)),
  sprintf("- Outcomes analysed: `%s`", nrow(comparison_frame)),
  sprintf("- Successful outcomes: `%s`", sum(diagnostic_frame$status == "success", na.rm = TRUE)),
  sprintf("- Failed outcomes: `%s`", sum(diagnostic_frame$status == "error", na.rm = TRUE)),
  sprintf("- Skipped outcomes: `%s`", sum(diagnostic_frame$status == "skipped", na.rm = TRUE)),
  sprintf("- Concordant: `%s`", classification_count("concordant")),
  sprintf("- Partial concordance: `%s`", classification_count("partial_concordance")),
  sprintf("- Discordant: `%s`", classification_count("discordant")),
  failure_stage_lines,
  sample_error_lines
)
writeLines(note_lines, con = file.path(output_dir, "acute_design_concordance_note.md"))
