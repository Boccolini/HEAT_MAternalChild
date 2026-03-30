args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(name, default = "") {
  index <- match(name, args)
  if (is.na(index) || index == length(args)) {
    return(default)
  }
  args[[index + 1]]
}

if (!requireNamespace("dlnm", quietly = TRUE)) {
  stop("Package 'dlnm' is required for phase2")
}
if (!requireNamespace("fixest", quietly = TRUE)) {
  stop("Package 'fixest' is required for phase2")
}

registry_path <- parse_arg("--registry")
input_dir <- parse_arg("--input-dir")
output_dir <- parse_arg("--output-dir")
lag_max_default <- as.integer(parse_arg("--lag-max-default", "7"))
lag_max_extended <- as.integer(parse_arg("--lag-max-extended", "21"))
extended_lag_outcomes <- Filter(nzchar, trimws(strsplit(parse_arg("--extended-lag-outcomes", ""), ",", fixed = TRUE)[[1]]))
reference_quantile <- as.numeric(parse_arg("--reference-quantile", "0.5"))
hot_quantile <- as.numeric(parse_arg("--hot-quantile", "0.95"))

if (!nzchar(registry_path) || !nzchar(input_dir) || !nzchar(output_dir)) {
  stop("Missing required arguments: --registry, --input-dir, --output-dir")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
registry <- read.csv(registry_path, stringsAsFactors = FALSE)
registry <- subset(registry, dlnm_required_flag %in% c(TRUE, "True", "TRUE", 1, "1"))

summary_rows <- list()
compare_rows <- list()
exposure_grids <- list()
lag_grids <- list()
diagnostic_rows <- list()

record_diagnostic <- function(
  analysis_unit_id,
  status,
  failure_stage,
  row_count = NA_integer_,
  lag_max = NA_integer_,
  warning_messages = character(),
  error_message = "",
  note = ""
) {
  diagnostic_rows[[length(diagnostic_rows) + 1]] <<- data.frame(
    analysis_unit_id = analysis_unit_id,
    status = status,
    failure_stage = failure_stage,
    row_count = row_count,
    lag_max = lag_max,
    warning_messages = paste(unique(Filter(nzchar, warning_messages)), collapse = " | "),
    error_message = error_message,
    note = note,
    stringsAsFactors = FALSE
  )
}

safe_sign <- function(value) {
  numeric_value <- suppressWarnings(as.numeric(value))
  if (!is.finite(numeric_value) || numeric_value <= 0) {
    return(NA_real_)
  }
  sign(log(numeric_value))
}

for (i in seq_len(nrow(registry))) {
  row <- registry[i, ]
  analysis_unit_id <- row$analysis_unit_id
  input_path <- file.path(input_dir, paste0(analysis_unit_id, ".csv"))
  warning_messages <- character()
  message(sprintf("[phase2] outcome=%s stage=start", analysis_unit_id))
  if (!file.exists(input_path)) {
    record_diagnostic(
      analysis_unit_id = analysis_unit_id,
      status = "skipped",
      failure_stage = "input_missing",
      note = "Input CSV not found"
    )
    message(sprintf("[phase2] outcome=%s stage=input_missing", analysis_unit_id))
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
      message(sprintf("[phase2] outcome=%s stage=read_csv status=error error=%s", analysis_unit_id, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(frame)) {
    next
  }
  if (nrow(frame) < 30) {
    record_diagnostic(
      analysis_unit_id = analysis_unit_id,
      status = "skipped",
      failure_stage = "insufficient_rows",
      row_count = nrow(frame),
      note = "Fewer than 30 rows available after filters"
    )
    message(sprintf("[phase2] outcome=%s stage=insufficient_rows row_count=%s", analysis_unit_id, nrow(frame)))
    next
  }
  frame$observation_date <- as.Date(frame$observation_date)
  frame$municipality_code <- factor(frame$municipality_code)
  frame$calendar_year <- factor(frame$calendar_year)
  frame$calendar_month <- factor(frame$calendar_month)
  frame$day_of_week <- factor(frame$day_of_week)
  frame$outcome_count <- as.numeric(frame$outcome_count)
  frame$denominator_count <- pmax(as.numeric(frame$denominator_count), 1)
  frame$exposure_value <- as.numeric(frame$exposure_value)
  frame$log_denominator <- log(frame$denominator_count)
  lag_max <- if (analysis_unit_id %in% extended_lag_outcomes) lag_max_extended else lag_max_default
  row_count <- nrow(frame)

  cb <- tryCatch(
    withCallingHandlers(
      dlnm::crossbasis(
        frame$exposure_value,
        lag = lag_max,
        argvar = list(fun = "lin"),
        arglag = list(fun = "ns", df = min(3, lag_max))
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
        failure_stage = "crossbasis",
        row_count = row_count,
        lag_max = lag_max,
        warning_messages = warning_messages,
        error_message = conditionMessage(e)
      )
      message(sprintf("[phase2] outcome=%s stage=crossbasis status=error error=%s", analysis_unit_id, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(cb)) {
    rm(frame)
    invisible(gc(verbose = FALSE))
    next
  }
  cb_matrix <- as.matrix(cb)
  cb_column_names <- paste0("cb_", seq_len(ncol(cb_matrix)))
  colnames(cb_matrix) <- cb_column_names
  cb_frame <- as.data.frame(cb_matrix, stringsAsFactors = FALSE)
  frame <- cbind(frame, cb_frame)

  fit_formula <- stats::as.formula(
    paste(
      "outcome_count ~",
      paste(cb_column_names, collapse = " + "),
      "| municipality_code + calendar_year + calendar_month + day_of_week"
    )
  )
  fit <- tryCatch(
    withCallingHandlers(
      fixest::feglm(
        fml = fit_formula,
        family = "poisson",
        offset = ~ log_denominator,
        data = frame
        ,
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
        lag_max = lag_max,
        warning_messages = warning_messages,
        error_message = conditionMessage(e)
      )
      message(sprintf("[phase2] outcome=%s stage=feglm status=error error=%s", analysis_unit_id, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(fit)) {
    rm(frame, cb, cb_matrix, cb_frame, fit_formula)
    invisible(gc(verbose = FALSE))
    next
  }

  coef_values <- tryCatch(
    {
      values <- stats::coef(fit)[cb_column_names]
      if (length(values) != length(cb_column_names) || any(!is.finite(values))) {
        stop("Failed to recover a complete coefficient vector for the crossbasis terms")
      }
      values
    },
    error = function(e) {
      record_diagnostic(
        analysis_unit_id = analysis_unit_id,
        status = "error",
        failure_stage = "coef_vcov",
        row_count = row_count,
        lag_max = lag_max,
        warning_messages = warning_messages,
        error_message = conditionMessage(e)
      )
      message(sprintf("[phase2] outcome=%s stage=coef_vcov status=error error=%s", analysis_unit_id, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(coef_values)) {
    rm(frame, cb, cb_matrix, cb_frame, fit_formula, fit)
    invisible(gc(verbose = FALSE))
    next
  }

  vcov_matrix <- tryCatch(
    {
      matrix_value <- as.matrix(stats::vcov(fit))[cb_column_names, cb_column_names, drop = FALSE]
      if (any(!is.finite(matrix_value))) {
        stop("Failed to recover a finite covariance matrix for the crossbasis terms")
      }
      matrix_value
    },
    error = function(e) {
      record_diagnostic(
        analysis_unit_id = analysis_unit_id,
        status = "error",
        failure_stage = "coef_vcov",
        row_count = row_count,
        lag_max = lag_max,
        warning_messages = warning_messages,
        error_message = conditionMessage(e)
      )
      message(sprintf("[phase2] outcome=%s stage=coef_vcov status=error error=%s", analysis_unit_id, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(vcov_matrix)) {
    rm(frame, cb, cb_matrix, cb_frame, fit_formula, fit, coef_values)
    invisible(gc(verbose = FALSE))
    next
  }

  ref_value <- as.numeric(stats::quantile(frame$exposure_value, probs = reference_quantile, na.rm = TRUE))
  hot_value <- as.numeric(stats::quantile(frame$exposure_value, probs = hot_quantile, na.rm = TRUE))
  single_pred <- tryCatch(
    withCallingHandlers(
      dlnm::crosspred(
        cb,
        coef = coef_values,
        vcov = vcov_matrix,
        model.link = "log",
        cen = ref_value,
        at = c(hot_value),
        cumul = TRUE
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
        failure_stage = "crosspred_single",
        row_count = row_count,
        lag_max = lag_max,
        warning_messages = warning_messages,
        error_message = conditionMessage(e)
      )
      message(sprintf("[phase2] outcome=%s stage=crosspred_single status=error error=%s", analysis_unit_id, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(single_pred)) {
    rm(frame, cb, cb_matrix, cb_frame, fit_formula, fit, coef_values, vcov_matrix)
    invisible(gc(verbose = FALSE))
    next
  }
  cumfit <- single_pred$cumfit
  cumrrfit <- single_pred$cumRRfit
  cumrrlow <- single_pred$cumRRlow
  cumrrhigh <- single_pred$cumRRhigh
  if (!is.matrix(cumfit)) {
    cumfit <- matrix(cumfit, nrow = 1)
  }
  if (is.null(cumrrfit)) {
    cumrrfit <- exp(cumfit)
  }
  if (!is.matrix(cumrrfit)) {
    cumrrfit <- matrix(cumrrfit, nrow = nrow(cumfit))
  }
  if (is.null(cumrrlow)) {
    cumrrlow <- matrix(NA_real_, nrow = nrow(cumrrfit), ncol = ncol(cumrrfit))
  }
  if (is.null(cumrrhigh)) {
    cumrrhigh <- matrix(NA_real_, nrow = nrow(cumrrfit), ncol = ncol(cumrrfit))
  }
  if (!is.matrix(cumrrlow)) {
    cumrrlow <- matrix(cumrrlow, nrow = nrow(cumrrfit))
  }
  if (!is.matrix(cumrrhigh)) {
    cumrrhigh <- matrix(cumrrhigh, nrow = nrow(cumrrfit))
  }
  lag_index_short <- min(2, ncol(cumfit))
  lag_index_full <- ncol(cumfit)
  cum01_rr <- cumrrfit[1, lag_index_short]
  cum01_low <- cumrrlow[1, lag_index_short]
  cum01_high <- cumrrhigh[1, lag_index_short]
  cumfull_rr <- cumrrfit[1, lag_index_full]
  cumfull_low <- cumrrlow[1, lag_index_full]
  cumfull_high <- cumrrhigh[1, lag_index_full]
  cumulative_ratio <- if (abs(log(cumfull_rr)) > 0) abs(log(cum01_rr)) / abs(log(cumfull_rr)) else NA_real_
  current_sign <- safe_sign(row$current_estimate)
  dlnm_sign <- safe_sign(cumfull_rr)
  dlnm_classification <- if (!is.finite(current_sign) || !is.finite(dlnm_sign)) {
    "refines_retained_signal"
  } else if (!identical(current_sign, dlnm_sign)) {
    "contradicts_retained_signal"
  } else if (is.finite(cumulative_ratio) && cumulative_ratio >= 0.5) {
    "supports_retained_simple_signal"
  } else {
    "refines_retained_signal"
  }

  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    analysis_unit_id = analysis_unit_id,
    lag_max = lag_max,
    cumulative_rr_lag0_1 = cum01_rr,
    cumulative_rr_lag0_1_lower_ci = cum01_low,
    cumulative_rr_lag0_1_upper_ci = cum01_high,
    cumulative_rr_full = cumfull_rr,
    cumulative_rr_full_lower_ci = cumfull_low,
    cumulative_rr_full_upper_ci = cumfull_high,
    cumulative_ratio_lag0_1_vs_full = cumulative_ratio,
    dlnm_classification = dlnm_classification,
    stringsAsFactors = FALSE
  )
  compare_rows[[length(compare_rows) + 1]] <- data.frame(
    analysis_unit_id = analysis_unit_id,
    current_estimate = as.numeric(row$current_estimate),
    current_lower_ci = as.numeric(row$current_lower_ci),
    current_upper_ci = as.numeric(row$current_upper_ci),
    dlnm_cumulative_rr_full = cumfull_rr,
    dlnm_cumulative_rr_full_lower_ci = cumfull_low,
    dlnm_cumulative_rr_full_upper_ci = cumfull_high,
    dlnm_classification = dlnm_classification,
    stringsAsFactors = FALSE
  )

  exposure_grid_values <- seq(
    stats::quantile(frame$exposure_value, probs = 0.05, na.rm = TRUE),
    stats::quantile(frame$exposure_value, probs = 0.95, na.rm = TRUE),
    length.out = 25
  )
  exposure_pred <- tryCatch(
    withCallingHandlers(
      dlnm::crosspred(
        cb,
        coef = coef_values,
        vcov = vcov_matrix,
        model.link = "log",
        cen = ref_value,
        at = exposure_grid_values,
        cumul = TRUE
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
        failure_stage = "crosspred_grid",
        row_count = row_count,
        lag_max = lag_max,
        warning_messages = warning_messages,
        error_message = conditionMessage(e)
      )
      message(sprintf("[phase2] outcome=%s stage=crosspred_grid status=error error=%s", analysis_unit_id, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(exposure_pred)) {
    rm(frame, cb, cb_matrix, cb_frame, fit_formula, fit, coef_values, vcov_matrix, single_pred)
    invisible(gc(verbose = FALSE))
    next
  }
  overall_rr <- if (is.matrix(exposure_pred$allRRfit)) exposure_pred$allRRfit[, 1] else exposure_pred$allRRfit
  exposure_grids[[length(exposure_grids) + 1]] <- data.frame(
    analysis_unit_id = analysis_unit_id,
    exposure_value = exposure_grid_values,
    overall_cumulative_rr = as.numeric(overall_rr),
    stringsAsFactors = FALSE
  )
  lag_grids[[length(lag_grids) + 1]] <- data.frame(
    analysis_unit_id = analysis_unit_id,
    lag = seq(0, lag_max),
    cumulative_rr = as.numeric(cumrrfit[1, ]),
    stringsAsFactors = FALSE
  )
  record_diagnostic(
    analysis_unit_id = analysis_unit_id,
    status = "success",
    failure_stage = "completed",
    row_count = row_count,
    lag_max = lag_max,
    warning_messages = warning_messages,
    note = dlnm_classification
  )
  message(sprintf("[phase2] outcome=%s stage=completed classification=%s", analysis_unit_id, dlnm_classification))
  rm(frame, cb, cb_matrix, cb_frame, fit_formula, fit, coef_values, vcov_matrix, single_pred, exposure_pred)
  invisible(gc(verbose = FALSE))
}

summary_frame <- if (length(summary_rows) > 0) do.call(rbind, summary_rows) else data.frame()
compare_frame <- if (length(compare_rows) > 0) do.call(rbind, compare_rows) else data.frame()
exposure_grid_frame <- if (length(exposure_grids) > 0) do.call(rbind, exposure_grids) else data.frame()
lag_grid_frame <- if (length(lag_grids) > 0) do.call(rbind, lag_grids) else data.frame()
diagnostic_frame <- if (length(diagnostic_rows) > 0) {
  do.call(rbind, diagnostic_rows)
} else {
  data.frame(
    analysis_unit_id = character(),
    status = character(),
    failure_stage = character(),
    row_count = integer(),
    lag_max = integer(),
    warning_messages = character(),
    error_message = character(),
    note = character(),
    stringsAsFactors = FALSE
  )
}

write.csv(summary_frame, file.path(output_dir, "dlnm_lag_response_summary_table.csv"), row.names = FALSE)
write.csv(compare_frame, file.path(output_dir, "dlnm_vs_current_simplified_estimand_table.csv"), row.names = FALSE)
write.csv(exposure_grid_frame, file.path(output_dir, "dlnm_exposure_response_grid.csv"), row.names = FALSE)
write.csv(lag_grid_frame, file.path(output_dir, "dlnm_lag_response_grid.csv"), row.names = FALSE)
write.csv(diagnostic_frame, file.path(output_dir, "dlnm_outcome_diagnostics.csv"), row.names = FALSE)

classification_count <- function(classification_label) {
  if (!("dlnm_classification" %in% names(summary_frame))) {
    return(0)
  }
  sum(summary_frame$dlnm_classification == classification_label, na.rm = TRUE)
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
  "# DLNM Editorial Note",
  "",
  sprintf("- Outcomes attempted: `%s`", nrow(diagnostic_frame)),
  sprintf("- Outcomes analysed: `%s`", nrow(summary_frame)),
  sprintf("- Successful outcomes: `%s`", sum(diagnostic_frame$status == "success", na.rm = TRUE)),
  sprintf("- Failed outcomes: `%s`", sum(diagnostic_frame$status == "error", na.rm = TRUE)),
  sprintf("- Skipped outcomes: `%s`", sum(diagnostic_frame$status == "skipped", na.rm = TRUE)),
  sprintf("- Supports retained simple signal: `%s`", classification_count("supports_retained_simple_signal")),
  sprintf("- Refines retained signal: `%s`", classification_count("refines_retained_signal")),
  sprintf("- Contradicts retained signal: `%s`", classification_count("contradicts_retained_signal")),
  failure_stage_lines,
  sample_error_lines
)
writeLines(note_lines, con = file.path(output_dir, "dlnm_editorial_note.md"))
