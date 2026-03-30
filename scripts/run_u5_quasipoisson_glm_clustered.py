from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Compressed CSV produced by build_u5_quasipoisson_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store estimates and residual diagnostics.",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip")
    frame["DATE"] = pd.to_datetime(frame["DATE"])
    frame["DATE_NUM"] = (frame["DATE"] - frame["DATE"].min()).dt.days.astype(int)
    frame["DOW"] = frame["DATE"].dt.dayofweek.astype(str)
    frame["OFFSET_LOG"] = np.log(frame["POP_U5"] / frame["DAYS_IN_YEAR"])
    frame["TMAX_C_C"] = frame["TMAX_C"] - frame["TMAX_C"].mean()
    frame["RH_MEAN_PCT_C"] = frame["RH_MEAN_PCT"] - frame["RH_MEAN_PCT"].mean()
    frame["HEAT_INDEX_APPROX_C_C"] = frame["HEAT_INDEX_APPROX_C"] - frame["HEAT_INDEX_APPROX_C"].mean()
    return frame


def fit_quasipoisson(frame: pd.DataFrame, model_id: str, formula_rhs: str, needed: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_df = frame.dropna(subset=needed).copy()
    formula = f"DEATHS_U5 ~ {formula_rhs} + C(UF) + C(DOW) + bs(DATE_NUM, df=24, degree=3)"
    y, X = patsy.dmatrices(formula, model_df, return_type="dataframe")

    glm = sm.GLM(
        y,
        X,
        family=sm.families.Poisson(),
        offset=model_df["OFFSET_LOG"].to_numpy(),
    )
    result = glm.fit(
        scale="X2",
        cov_type="cluster",
        cov_kwds={"groups": model_df["MUNICIPALITY_CODE"].to_numpy()},
    )

    coef = pd.DataFrame(
        {
            "model_id": model_id,
            "term": result.params.index,
            "estimate": result.params.values,
            "std_error": result.bse.values,
            "z_value": result.tvalues.values,
            "p_value": result.pvalues.values,
        }
    )
    coef["conf_low"] = coef["estimate"] - 1.96 * coef["std_error"]
    coef["conf_high"] = coef["estimate"] + 1.96 * coef["std_error"]
    coef["rr"] = np.exp(coef["estimate"])
    coef["rr_conf_low"] = np.exp(coef["conf_low"])
    coef["rr_conf_high"] = np.exp(coef["conf_high"])

    pearson_resid = result.resid_pearson
    deviance_resid = result.resid_deviance
    fitted = result.fittedvalues
    dispersion = float(np.sum(np.square(pearson_resid)) / result.df_resid)
    date_mean_resid = pd.DataFrame({"DATE": model_df["DATE"], "pearson_resid": pearson_resid}).groupby("DATE", as_index=False)["pearson_resid"].mean()
    acf_lag1 = float(date_mean_resid["pearson_resid"].autocorr(lag=1))
    acf_lag7 = float(date_mean_resid["pearson_resid"].autocorr(lag=7))

    residual_summary = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "n_obs": int(model_df.shape[0]),
                "mean_outcome": float(model_df["DEATHS_U5"].mean()),
                "mean_fitted": float(np.mean(fitted)),
                "pearson_dispersion": dispersion,
                "pearson_mean": float(np.mean(pearson_resid)),
                "pearson_sd": float(np.std(pearson_resid, ddof=1)),
                "pearson_abs_gt2": float(np.mean(np.abs(pearson_resid) > 2)),
                "pearson_abs_gt3": float(np.mean(np.abs(pearson_resid) > 3)),
                "deviance_mean": float(np.mean(deviance_resid)),
                "deviance_sd": float(np.std(deviance_resid, ddof=1)),
                "corr_fitted_pearson": float(np.corrcoef(fitted, pearson_resid)[0, 1]),
                "pearson_q01": float(np.quantile(pearson_resid, 0.01)),
                "pearson_q05": float(np.quantile(pearson_resid, 0.05)),
                "pearson_q50": float(np.quantile(pearson_resid, 0.50)),
                "pearson_q95": float(np.quantile(pearson_resid, 0.95)),
                "pearson_q99": float(np.quantile(pearson_resid, 0.99)),
                "deviance_q01": float(np.quantile(deviance_resid, 0.01)),
                "deviance_q05": float(np.quantile(deviance_resid, 0.05)),
                "deviance_q50": float(np.quantile(deviance_resid, 0.50)),
                "deviance_q95": float(np.quantile(deviance_resid, 0.95)),
                "deviance_q99": float(np.quantile(deviance_resid, 0.99)),
                "acf_lag1": acf_lag1,
                "acf_lag7": acf_lag7,
                "pearson_chi2": float(result.pearson_chi2),
                "df_resid": float(result.df_resid),
                "pseudo_r2_cs": float(result.pseudo_rsquared(kind="cs")),
            }
        ]
    )

    sample_n = min(200000, model_df.shape[0])
    rng = np.random.default_rng(20260326)
    sample_pos = np.sort(rng.choice(model_df.shape[0], size=sample_n, replace=False))
    fitted_array = np.asarray(fitted)
    pearson_array = np.asarray(pearson_resid)
    deviance_array = np.asarray(deviance_resid)
    residual_sample = pd.DataFrame(
        {
            "model_id": model_id,
            "DATE": model_df.iloc[sample_pos]["DATE"].dt.strftime("%Y-%m-%d").to_numpy(),
            "MUNICIPALITY_CODE": model_df.iloc[sample_pos]["MUNICIPALITY_CODE"].to_numpy(),
            "fitted": fitted_array[sample_pos],
            "response": model_df.iloc[sample_pos]["DEATHS_U5"].to_numpy(),
            "pearson_residual": pearson_array[sample_pos],
            "deviance_residual": deviance_array[sample_pos],
        }
    )

    return coef, residual_summary, residual_sample


def save_plots(sample: pd.DataFrame, output_dir: Path, model_id: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].scatter(sample["fitted"], sample["pearson_residual"], s=2, alpha=0.08, color="black")
    axes[0, 0].axhline(0, color="red", linewidth=1.5)
    axes[0, 0].set_title(f"{model_id}: Pearson residuals vs fitted")
    axes[0, 0].set_xlabel("Fitted")
    axes[0, 0].set_ylabel("Pearson residual")

    axes[0, 1].hist(sample["pearson_residual"], bins=80, color="#4c78a8")
    axes[0, 1].set_title(f"{model_id}: Pearson residual distribution")
    axes[0, 1].set_xlabel("Pearson residual")

    sm.qqplot(sample["deviance_residual"], line="45", ax=axes[1, 0], alpha=0.08, marker=".")
    axes[1, 0].set_title(f"{model_id}: QQ plot deviance residuals")

    axes[1, 1].scatter(sample["response"], sample["fitted"], s=2, alpha=0.08, color="black")
    axes[1, 1].set_title(f"{model_id}: observed vs fitted")
    axes[1, 1].set_xlabel("Observed")
    axes[1, 1].set_ylabel("Fitted")

    fig.tight_layout()
    fig.savefig(output_dir / f"{model_id}_residual_plots.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_data(input_path)

    model_specs = [
        ("qp_glm_tmax_only", "TMAX_C_C", ["TMAX_C_C"]),
        ("qp_glm_tmax_rh_interaction", "TMAX_C_C + RH_MEAN_PCT_C + TMAX_C_C:RH_MEAN_PCT_C", ["TMAX_C_C", "RH_MEAN_PCT_C"]),
        ("qp_glm_heat_index", "HEAT_INDEX_APPROX_C_C", ["HEAT_INDEX_APPROX_C_C"]),
    ]

    coef_tables: list[pd.DataFrame] = []
    resid_summaries: list[pd.DataFrame] = []

    for model_id, rhs, needed in model_specs:
        print(f"Fitting {model_id} with {len(frame.dropna(subset=needed)):,} rows")
        coef, resid_summary, resid_sample = fit_quasipoisson(frame, model_id, rhs, needed)
        coef_tables.append(coef)
        resid_summaries.append(resid_summary)
        resid_sample.to_csv(output_dir / f"{model_id}_residual_sample.csv", index=False)
        save_plots(resid_sample, output_dir, model_id)

    pd.concat(coef_tables, ignore_index=True).to_csv(output_dir / "quasipoisson_glm_clustered_estimates.csv", index=False)
    pd.concat(resid_summaries, ignore_index=True).to_csv(output_dir / "quasipoisson_glm_clustered_residual_summary.csv", index=False)
    print(f"Saved outputs under {output_dir}")


if __name__ == "__main__":
    main()
