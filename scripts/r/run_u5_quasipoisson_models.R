args <- commandArgs(trailingOnly = TRUE)

root <- if (length(args) >= 1) normalizePath(args[[1]], mustWork = TRUE) else getwd()
output_dir <- if (length(args) >= 2) normalizePath(args[[2]], mustWork = FALSE) else file.path(root, "artifacts", "analysis", "ad_hoc_quasipoisson_u5_20260326")
input_path <- if (length(args) >= 3) normalizePath(args[[3]], mustWork = TRUE) else file.path(output_dir, "u5_municipality_day_model_input.csv.gz")

suppressPackageStartupMessages(library(mgcv))

if (!file.exists(input_path)) {
  stop("Analytical input not found: ", input_path)
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

message("Reading analytical dataset from ", input_path)
dat <- read.csv(gzfile(input_path), stringsAsFactors = FALSE)
message("Rows read: ", format(nrow(dat), big.mark = ","))

dat$DATE <- as.Date(dat$DATE)
dat$DATE_NUM <- as.numeric(dat$DATE - min(dat$DATE, na.rm = TRUE))
dat$DOW <- factor(as.POSIXlt(dat$DATE)$wday)
dat$MUNICIPALITY_CODE <- factor(dat$MUNICIPALITY_CODE)
dat$UF <- factor(dat$UF)
dat$DEATHS_U5 <- as.numeric(dat$DEATHS_U5)
dat$POP_U5 <- as.numeric(dat$POP_U5)
dat$DAYS_IN_YEAR <- as.numeric(dat$DAYS_IN_YEAR)
dat$OFFSET_LOG <- log(dat$POP_U5 / dat$DAYS_IN_YEAR)
dat$TMAX_C <- as.numeric(dat$TMAX_C)
dat$RH_MEAN_PCT <- as.numeric(dat$RH_MEAN_PCT)
dat$HEAT_INDEX_APPROX_C <- as.numeric(dat$HEAT_INDEX_APPROX_C)
dat$TMAX_C_C <- dat$TMAX_C - mean(dat$TMAX_C, na.rm = TRUE)
dat$RH_MEAN_PCT_C <- dat$RH_MEAN_PCT - mean(dat$RH_MEAN_PCT, na.rm = TRUE)
dat$HEAT_INDEX_APPROX_C_C <- dat$HEAT_INDEX_APPROX_C - mean(dat$HEAT_INDEX_APPROX_C, na.rm = TRUE)

fit_stats <- list()
coef_tables <- list()
smooth_tables <- list()
residual_tables <- list()

make_ci_table <- function(summary_obj, model_id) {
  ptab <- as.data.frame(summary_obj$p.table)
  ptab$term <- rownames(ptab)
  names(ptab)[1:4] <- c("estimate", "std_error", "statistic", "p_value")
  ptab$conf_low <- ptab$estimate - 1.96 * ptab$std_error
  ptab$conf_high <- ptab$estimate + 1.96 * ptab$std_error
  ptab$rr <- exp(ptab$estimate)
  ptab$rr_conf_low <- exp(ptab$conf_low)
  ptab$rr_conf_high <- exp(ptab$conf_high)
  ptab$model_id <- model_id
  ptab <- ptab[, c("model_id", "term", "estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high", "rr", "rr_conf_low", "rr_conf_high")]
  rownames(ptab) <- NULL
  ptab
}

make_smooth_table <- function(summary_obj, model_id) {
  if (is.null(summary_obj$s.table)) {
    return(data.frame())
  }
  stab <- as.data.frame(summary_obj$s.table)
  stab$term <- rownames(stab)
  names(stab) <- c("edf", "ref_df", "statistic", "p_value", "term")
  stab$model_id <- model_id
  stab <- stab[, c("model_id", "term", "edf", "ref_df", "statistic", "p_value")]
  rownames(stab) <- NULL
  stab
}

sample_residual_data <- function(fit, model_id) {
  set.seed(20260326)
  n <- length(fit$fitted.values)
  sample_n <- min(200000L, n)
  idx <- sort(sample.int(n, sample_n))
  data.frame(
    model_id = model_id,
    fitted = fit$fitted.values[idx],
    response = fit$y[idx],
    pearson_residual = residuals(fit, type = "pearson")[idx],
    deviance_residual = residuals(fit, type = "deviance")[idx]
  )
}

residual_summary <- function(fit, model_id, model_dat) {
  pearson_resid <- residuals(fit, type = "pearson")
  deviance_resid <- residuals(fit, type = "deviance")
  mu <- fit$fitted.values
  disp <- sum(pearson_resid^2, na.rm = TRUE) / fit$df.residual
  qs_pearson <- as.numeric(quantile(pearson_resid, probs = c(0.01, 0.05, 0.5, 0.95, 0.99), na.rm = TRUE))
  qs_deviance <- as.numeric(quantile(deviance_resid, probs = c(0.01, 0.05, 0.5, 0.95, 0.99), na.rm = TRUE))
  date_means <- aggregate(pearson_resid, by = list(DATE = model_dat$DATE), FUN = mean, na.rm = TRUE)
  acf_vals <- acf(date_means$x, lag.max = 7, plot = FALSE, na.action = na.pass)$acf
  data.frame(
    model_id = model_id,
    n_obs = length(mu),
    mean_outcome = mean(fit$y, na.rm = TRUE),
    mean_fitted = mean(mu, na.rm = TRUE),
    pearson_dispersion = disp,
    pearson_mean = mean(pearson_resid, na.rm = TRUE),
    pearson_sd = sd(pearson_resid, na.rm = TRUE),
    pearson_abs_gt2 = mean(abs(pearson_resid) > 2, na.rm = TRUE),
    pearson_abs_gt3 = mean(abs(pearson_resid) > 3, na.rm = TRUE),
    deviance_mean = mean(deviance_resid, na.rm = TRUE),
    deviance_sd = sd(deviance_resid, na.rm = TRUE),
    corr_fitted_pearson = suppressWarnings(cor(mu, pearson_resid, use = "complete.obs")),
    pearson_q01 = qs_pearson[[1]],
    pearson_q05 = qs_pearson[[2]],
    pearson_q50 = qs_pearson[[3]],
    pearson_q95 = qs_pearson[[4]],
    pearson_q99 = qs_pearson[[5]],
    deviance_q01 = qs_deviance[[1]],
    deviance_q05 = qs_deviance[[2]],
    deviance_q50 = qs_deviance[[3]],
    deviance_q95 = qs_deviance[[4]],
    deviance_q99 = qs_deviance[[5]],
    acf_lag1 = acf_vals[[2]],
    acf_lag7 = acf_vals[[8]]
  )
}

save_residual_plots <- function(resid_sample, model_id) {
  png(file.path(output_dir, paste0(model_id, "_residual_plots.png")), width = 1800, height = 1400, res = 180)
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  plot(
    resid_sample$fitted,
    resid_sample$pearson_residual,
    pch = 16,
    cex = 0.25,
    col = rgb(0, 0, 0, 0.08),
    xlab = "Fitted values",
    ylab = "Pearson residuals",
    main = paste(model_id, "Residuals vs Fitted")
  )
  abline(h = 0, col = "red", lwd = 2)
  hist(
    resid_sample$pearson_residual,
    breaks = 80,
    main = paste(model_id, "Pearson residuals"),
    xlab = "Pearson residual"
  )
  qqnorm(resid_sample$deviance_residual, pch = 16, cex = 0.3, main = paste(model_id, "QQ deviance residuals"))
  qqline(resid_sample$deviance_residual, col = "red", lwd = 2)
  plot(
    resid_sample$response,
    resid_sample$fitted,
    pch = 16,
    cex = 0.25,
    col = rgb(0, 0, 0, 0.08),
    xlab = "Observed count",
    ylab = "Fitted count",
    main = paste(model_id, "Observed vs fitted")
  )
  dev.off()
}

fit_one_model <- function(model_id, exposure_terms, required_columns) {
  model_dat <- dat[stats::complete.cases(dat[, required_columns]), ]
  message(model_id, ": rows used = ", format(nrow(model_dat), big.mark = ","))

  formula_text <- paste(
    "DEATHS_U5 ~",
    exposure_terms,
    "+ s(DATE_NUM, k = 70, bs = 'cr')",
    "+ DOW",
    "+ s(UF, bs = 're')",
    "+ s(MUNICIPALITY_CODE, bs = 're')"
  )

  fit <- bam(
    as.formula(formula_text),
    family = quasipoisson(link = "log"),
    offset = model_dat$OFFSET_LOG,
    data = model_dat,
    method = "fREML",
    discrete = TRUE,
    nthreads = max(1L, parallel::detectCores(logical = FALSE) - 1L),
    gc.level = 1
  )

  sfit <- summary(fit)
  coef_tables[[model_id]] <<- make_ci_table(sfit, model_id)
  smooth_tables[[model_id]] <<- make_smooth_table(sfit, model_id)
  residual_tables[[model_id]] <<- residual_summary(fit, model_id, model_dat)

  sampled <- sample_residual_data(fit, model_id)
  write.csv(sampled, file.path(output_dir, paste0(model_id, "_residual_sample.csv")), row.names = FALSE)
  save_residual_plots(sampled, model_id)

  fit_stats[[model_id]] <<- data.frame(
    model_id = model_id,
    n_obs = nrow(model_dat),
    deviance_explained = unname(sfit$dev.expl),
    adj_r_sq = unname(sfit$r.sq),
    scale_estimate = unname(sfit$scale),
    aic = suppressWarnings(tryCatch(AIC(fit), error = function(e) NA_real_))
  )

  saveRDS(fit, file.path(output_dir, paste0(model_id, "_fit.rds")))
}

fit_one_model(
  model_id = "qp_tmax_only",
  exposure_terms = "TMAX_C_C",
  required_columns = c("DEATHS_U5", "OFFSET_LOG", "TMAX_C_C")
)

fit_one_model(
  model_id = "qp_tmax_rh_interaction",
  exposure_terms = "TMAX_C_C + RH_MEAN_PCT_C + TMAX_C_C:RH_MEAN_PCT_C",
  required_columns = c("DEATHS_U5", "OFFSET_LOG", "TMAX_C_C", "RH_MEAN_PCT_C")
)

fit_one_model(
  model_id = "qp_heat_index",
  exposure_terms = "HEAT_INDEX_APPROX_C_C",
  required_columns = c("DEATHS_U5", "OFFSET_LOG", "HEAT_INDEX_APPROX_C_C")
)

write.csv(do.call(rbind, coef_tables), file.path(output_dir, "quasipoisson_parametric_estimates.csv"), row.names = FALSE)
write.csv(do.call(rbind, smooth_tables), file.path(output_dir, "quasipoisson_smooth_terms.csv"), row.names = FALSE)
write.csv(do.call(rbind, residual_tables), file.path(output_dir, "quasipoisson_residual_summary.csv"), row.names = FALSE)
write.csv(do.call(rbind, fit_stats), file.path(output_dir, "quasipoisson_fit_stats.csv"), row.names = FALSE)

message("Saved outputs under ", output_dir)
