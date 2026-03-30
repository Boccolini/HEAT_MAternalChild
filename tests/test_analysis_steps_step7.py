from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from heat_mnch.analysis_steps.config import load_step7_route_catalog
from heat_mnch.analysis_steps.step7 import build_step7_escalation_round
from heat_mnch.cli import main


def _real_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_step7_configs(root: Path) -> None:
    repo_root = _real_repo_root()
    for relative_path in (
        Path("config/analysis/step7_escalation.yaml"),
        Path("config/analysis/step7_route_catalog.yaml"),
    ):
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((repo_root / relative_path).read_text(encoding="utf-8"), encoding="utf-8")


def _write_round_config(root: Path) -> None:
    _write_json_yaml(
        root / "config" / "analysis" / "round_config.yaml",
        {
            "analysis_id": "heat_mnch_incremental_round",
            "analysis_title": "HEAT-MNCH Incremental Analysis Round",
            "years": [2020, 2021, 2022, 2023, 2024],
        },
    )


def _group_for_analysis_unit(analysis_unit_id: str) -> tuple[str, str]:
    maternal = {
        "antepartum_haemorrhage_admission_count",
        "gestational_diabetes_admission_count",
        "hypertensive_disorder_of_pregnancy_count",
        "obstetric_admission_count",
        "preterm_labor_admission_count",
        "severe_hypertensive_disorder_of_pregnancy_count",
    }
    perinatal = {
        "preterm_birth_count",
        "very_preterm_birth_count",
        "low_birth_weight_lt2500_count",
        "stillbirth_primary_gte28_count",
        "stillbirth_sensitivity_gte22_count",
        "gestational_days_lost_below_37_weeks",
        "prenatal_eight_plus_visit_count",
        "preterm_low_birth_weight_count",
    }
    hospitalization = {
        "child_dehydration_admission_count",
        "child_diarrhea_gi_admission_count",
        "child_acute_respiratory_admission_count",
        "child_asthma_wheeze_admission_count",
        "child_direct_heat_hospitalization_count",
        "child_indirect_heat_hospitalization_count",
    }
    if analysis_unit_id in maternal:
        return "maternal", "Maternal"
    if analysis_unit_id in perinatal:
        return "perinatal_nascimento", "Perinatal and Birth"
    if analysis_unit_id in hospitalization:
        return "hospitalizacao_infantil", "Child Hospitalization"
    return "mortalidade_infantil", "Child Mortality"


def _selected_spec(analysis_unit_id: str, group_id: str) -> tuple[str, str, str]:
    if group_id == "maternal":
        return "family_1_tmax_primary", "Maximum daily temperature", "0_6"
    if analysis_unit_id in {
        "preterm_birth_count",
        "very_preterm_birth_count",
        "low_birth_weight_lt2500_count",
        "stillbirth_primary_gte28_count",
        "stillbirth_sensitivity_gte22_count",
        "gestational_days_lost_below_37_weeks",
        "prenatal_eight_plus_visit_count",
        "preterm_low_birth_weight_count",
    }:
        return "family_4_heat_index_parallel", "Approximate heat index", "0_6"
    return "family_4_heat_index_parallel", "Approximate heat index", "lag0"


def _final_translation_stage(analysis_unit_id: str) -> str:
    if analysis_unit_id in {
        "preterm_birth_count",
        "under_five_death_count",
        "child_dehydration_admission_count",
    }:
        return "headline_absolute"
    if analysis_unit_id in {
        "very_preterm_birth_count",
        "low_birth_weight_lt2500_count",
        "stillbirth_primary_gte28_count",
        "stillbirth_sensitivity_gte22_count",
    }:
        return "supplementary_absolute"
    return "secondary_parallel"


def _recommended_public_use(analysis_unit_id: str) -> str:
    if analysis_unit_id == "gestational_days_lost_below_37_weeks":
        return "continuous_internal_translation_only"
    if analysis_unit_id in {
        "prenatal_eight_plus_visit_count",
        "antepartum_haemorrhage_admission_count",
        "gestational_diabetes_admission_count",
        "hypertensive_disorder_of_pregnancy_count",
        "obstetric_admission_count",
        "preterm_labor_admission_count",
        "severe_hypertensive_disorder_of_pregnancy_count",
    }:
        return "relative_only"
    return "relative_plus_observed_hot_volume"


def _write_step5_step6_artifacts(root: Path) -> None:
    route_payload = load_step7_route_catalog(_real_repo_root())
    route_rows = route_payload["analysis_units"]
    step5_root = root / "artifacts" / "analysis" / "step5"
    step6_root = root / "artifacts" / "analysis" / "step6"
    step4_root = root / "artifacts" / "analysis" / "step4"
    step3_root = root / "artifacts" / "analysis" / "step3"
    step5_root.mkdir(parents=True, exist_ok=True)
    step6_root.mkdir(parents=True, exist_ok=True)
    step4_root.mkdir(parents=True, exist_ok=True)
    step3_root.mkdir(parents=True, exist_ok=True)

    step5_rows: list[dict[str, object]] = []
    step6_rows: list[dict[str, object]] = []
    secondary_rows: list[dict[str, object]] = []
    step4_rows: list[dict[str, object]] = []
    step3_rows: list[dict[str, object]] = []

    for item in route_rows:
        analysis_unit_id = str(item["analysis_unit_id"])
        group_id, group_label = _group_for_analysis_unit(analysis_unit_id)
        exposure_family_id, exposure_family_name, preferred_window = _selected_spec(analysis_unit_id, group_id)
        final_translation_stage = _final_translation_stage(analysis_unit_id)
        public_burden_gate_pass = final_translation_stage in {"headline_absolute", "supplementary_absolute"}
        estimate = 0.4 if analysis_unit_id == "gestational_days_lost_below_37_weeks" else (0.95 if group_id == "maternal" else 1.25)
        lower_ci = 0.2 if analysis_unit_id == "gestational_days_lost_below_37_weeks" else (0.90 if group_id == "maternal" else 1.10)
        upper_ci = 0.6 if analysis_unit_id == "gestational_days_lost_below_37_weeks" else (1.00 if group_id == "maternal" else 1.40)
        p_value = 0.20 if group_id == "maternal" else 0.01

        step5_rows.append(
            {
                "analysis_unit_id": analysis_unit_id,
                "has_any_significant_reversal_flag": analysis_unit_id in {
                    "antepartum_haemorrhage_admission_count",
                    "child_acute_respiratory_admission_count",
                },
                "significant_reversal_modifier_count": 1 if analysis_unit_id in {
                    "antepartum_haemorrhage_admission_count",
                    "child_acute_respiratory_admission_count",
                } else 0,
                "decision_reason": "promoted_to_step6_consistent" if group_id != "maternal" else "promoted_to_step6_with_significant_reversal",
            }
        )
        step6_rows.append(
            {
                "analysis_unit_id": analysis_unit_id,
                "group_id": group_id,
                "group_label": group_label,
                "public_label": analysis_unit_id.replace("_", " ").capitalize(),
                "final_translation_stage": final_translation_stage,
                "selected_exposure_family_id": exposure_family_id,
                "selected_exposure_family_name": exposure_family_name,
                "preferred_lag_or_window": preferred_window,
                "estimate": estimate,
                "lower_ci": lower_ci,
                "upper_ci": upper_ci,
                "p_value": p_value,
                "public_burden_gate_pass": public_burden_gate_pass,
                "has_any_significant_reversal_flag": analysis_unit_id in {
                    "antepartum_haemorrhage_admission_count",
                    "child_acute_respiratory_admission_count",
                },
                "significant_reversal_modifier_count": 1 if analysis_unit_id in {
                    "antepartum_haemorrhage_admission_count",
                    "child_acute_respiratory_admission_count",
                } else 0,
            }
        )
        step4_rows.append(
            {
                "analysis_unit_id": analysis_unit_id,
                "selected_exposure_family_id": exposure_family_id,
                "preferred_lag_or_window": preferred_window,
            }
        )
        step3_rows.append(
            {
                "analysis_unit_id": analysis_unit_id,
                "headline_family_recommendation": "tmax" if group_id == "maternal" else "heat_index",
                "family_comparison_reason": "fixed for Step 7 testing",
            }
        )
        if final_translation_stage == "secondary_parallel":
            secondary_rows.append(
                {
                    "analysis_unit_id": analysis_unit_id,
                    "recommended_public_use": _recommended_public_use(analysis_unit_id),
                }
            )

    pd.DataFrame(step5_rows).to_csv(step5_root / "step5_scorecard.csv", index=False)
    pd.DataFrame(step6_rows).to_csv(step6_root / "step6_candidate_catalog.csv", index=False)
    pd.DataFrame(
        [
            {
                "group_id": "maternal",
                "group_label": "Maternal",
                "headline_analysis_unit_id": "",
                "headline_public_label": "",
                "selected_exposure_family_id": "",
                "selected_exposure_family_name": "",
                "preferred_lag_or_window": "",
                "estimate": "",
                "p_value": "",
                "signed_excess": "",
                "headline_selected_flag": False,
                "headline_missing_flag": True,
            },
            {
                "group_id": "perinatal_nascimento",
                "group_label": "Perinatal and Birth",
                "headline_analysis_unit_id": "preterm_birth_count",
                "headline_public_label": "Preterm birth count",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "selected_exposure_family_name": "Approximate heat index",
                "preferred_lag_or_window": "0_6",
                "estimate": 1.25,
                "p_value": 0.01,
                "signed_excess": 25.0,
                "headline_selected_flag": True,
                "headline_missing_flag": False,
            },
            {
                "group_id": "mortalidade_infantil",
                "group_label": "Child Mortality",
                "headline_analysis_unit_id": "under_five_death_count",
                "headline_public_label": "Under-five death count",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "selected_exposure_family_name": "Approximate heat index",
                "preferred_lag_or_window": "lag0",
                "estimate": 1.25,
                "p_value": 0.01,
                "signed_excess": 15.0,
                "headline_selected_flag": True,
                "headline_missing_flag": False,
            },
            {
                "group_id": "hospitalizacao_infantil",
                "group_label": "Child Hospitalization",
                "headline_analysis_unit_id": "child_dehydration_admission_count",
                "headline_public_label": "Child dehydration admission count",
                "selected_exposure_family_id": "family_4_heat_index_parallel",
                "selected_exposure_family_name": "Approximate heat index",
                "preferred_lag_or_window": "lag0",
                "estimate": 1.25,
                "p_value": 0.01,
                "signed_excess": 20.0,
                "headline_selected_flag": True,
                "headline_missing_flag": False,
            },
        ]
    ).to_csv(step6_root / "step6_group_headlines.csv", index=False)
    pd.DataFrame(
        [
            {"analysis_unit_id": "preterm_birth_count"},
            {"analysis_unit_id": "under_five_death_count"},
            {"analysis_unit_id": "child_dehydration_admission_count"},
        ]
    ).to_csv(step6_root / "headline_burden_estimates.csv", index=False)
    pd.DataFrame(
        [
            {"analysis_unit_id": "very_preterm_birth_count"},
            {"analysis_unit_id": "stillbirth_primary_gte28_count"},
        ]
    ).to_csv(step6_root / "supplementary_burden_estimates.csv", index=False)
    pd.DataFrame(secondary_rows).to_csv(step6_root / "secondary_parallel_translation.csv", index=False)
    pd.DataFrame(step4_rows).to_csv(step4_root / "step4_promotion_decisions.csv", index=False)
    pd.DataFrame(step3_rows).to_csv(step3_root / "step3_family_threshold_comparison.csv", index=False)


def _write_lookup(root: Path) -> None:
    lookup = pd.DataFrame(
        [
            {
                "atlas_id": 1,
                "MUNICIPALITY_CODE_IBGE7": 1100015,
                "UF": "RO",
                "NM_MUNICIPIO": "Alta Floresta D'Oeste",
            },
            {
                "atlas_id": 2,
                "MUNICIPALITY_CODE_IBGE7": 1100023,
                "UF": "RO",
                "NM_MUNICIPIO": "Ariquemes",
            },
        ]
    )
    path = root / "artifacts" / "reference" / "municipality_context" / "atlas_brasil_municipality_lookup.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(path, index=False)


def _write_climate(root: Path) -> None:
    for year in range(2020, 2025):
        rows: list[dict[str, object]] = []
        for municipality_code, offset in [("110001", 0.0), ("110002", 2.0)]:
            for day_index, date in enumerate(pd.date_range(f"{year}-01-20", f"{year}-03-10", freq="D")):
                rows.append(
                    {
                        "COD_MUNICIPIO": municipality_code,
                        "DATE": date.strftime("%Y-%m-%d"),
                        "TMAX_C": 30.0 + offset + (day_index * 0.1),
                        "HEAT_INDEX_C": 36.0 + (offset * 1.5) + (day_index * 0.15),
                    }
                )
        climate = pd.DataFrame(rows)
        path = root / "data" / "silver" / "climate" / f"YEAR={year}" / f"climate_canonical_daily_{year}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        climate.to_parquet(path, index=False)


def _write_live_births(root: Path) -> None:
    for year in range(2020, 2025):
        rows: list[dict[str, object]] = []
        for municipality_code, offset in [("110001", 0), ("110002", 5)]:
            for index, date in enumerate(pd.date_range(f"{year}-02-01", periods=24, freq="D")):
                rows.append(
                    {
                        "MUNICIPALITY_CODE": municipality_code,
                        "DATE": date.strftime("%Y-%m-%d"),
                        "YEAR": year,
                        "MONTH": date.month,
                        "GA_DAYS_FINAL": 252 + (index % 12),
                        "GA_WEEKS_COMPLETED_FINAL": 36 + (index % 4),
                        "GA_SOURCE": "LMP_EXACT",
                        "PRETERM_LT37_FLAG": 1 if index % 4 == 0 else 0,
                        "PRETERM_LT32_FLAG": 1 if index % 10 == 0 else 0,
                        "PRETERM_DAYS_SHORT_OF_37W_FINAL": float((index % 4) * 2),
                        "LOW_BIRTH_WEIGHT_LT2500_FLAG": 1 if (index + offset) % 5 == 0 else 0,
                        "GA_VALID_FOR_ANALYSIS_FLAG": 1,
                    }
                )
        live_births = pd.DataFrame(rows)
        path = root / "data" / "silver" / "sinasc" / "silver_sinasc_event" / f"YEAR={year}" / f"silver_sinasc_event_{year}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        live_births.to_parquet(path, index=False)


def _write_stillbirths(root: Path) -> None:
    for year in range(2020, 2025):
        rows: list[dict[str, object]] = []
        for municipality_code, offset in [("110001", 0), ("110002", 1)]:
            for index, date in enumerate(pd.date_range(f"{year}-02-05", periods=8, freq="4D")):
                rows.append(
                    {
                        "MUNICIPALITY_CODE": municipality_code,
                        "EVENT_DATE": date.strftime("%Y-%m-%d"),
                        "YEAR": year,
                        "MONTH": date.month,
                        "GA_WEEKS_COMPLETED_FINAL": 30 + offset + (index % 3),
                        "GA_SOURCE": "SEMAGESTAC_EXACT_WEEKS",
                        "GA_VALID_FOR_ANALYSIS_FLAG": 1,
                        "STILLBIRTH_GTE28_PRIMARY_FLAG": 1,
                        "STILLBIRTH_GTE22_SENSITIVITY_FLAG": 1,
                    }
                )
        stillbirths = pd.DataFrame(rows)
        path = root / "data" / "silver" / "stillbirth" / "silver_stillbirth_individual" / f"YEAR={year}" / f"silver_stillbirth_individual_{year}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        stillbirths.to_parquet(path, index=False)


def _write_gold_mortality(root: Path) -> None:
    for year in range(2020, 2025):
        rows: list[dict[str, object]] = []
        for municipality_code, offset in [("1100015", 0.0), ("1100023", 1.5)]:
            for index, date in enumerate(pd.date_range(f"{year}-02-01", periods=35, freq="D")):
                rows.append(
                    {
                        "MUNICIPALITY_CODE": municipality_code,
                        "DATE": date.strftime("%Y-%m-%d"),
                        "YEAR": year,
                        "MONTH": date.month,
                        "DEATHS_NEONATAL": 1 if (index + int(offset)) % 11 == 0 else 0,
                        "LIVE_BIRTHS_TOTAL_ANNUAL": 220.0,
                        "TMAX_C": 31.0 + offset + (index * 0.1),
                        "HEAT_INDEX_APPROX_C": 38.0 + offset + (index * 0.12),
                        "QC_STATUS": "PASS",
                        "OFFSET_VALID_FLAG": 1,
                        "DENOMINATOR_ZERO_FLAG": 0,
                        "DENOMINATOR_MISSING_FLAG": 0,
                        "RATE_NOT_COMPUTABLE_FLAG": 0,
                    }
                )
        mortality = pd.DataFrame(rows)
        path = root / "data" / "gold" / "gold_mnch_mortality" / f"YEAR={year}" / f"gold_mnch_mortality_{year}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        mortality.to_parquet(path, index=False)


def _prepare_step7_fixture(root: Path) -> Path:
    _copy_step7_configs(root)
    _write_round_config(root)
    _write_step5_step6_artifacts(root)
    _write_lookup(root)
    _write_climate(root)
    _write_live_births(root)
    _write_stillbirths(root)
    _write_gold_mortality(root)
    return root / "config" / "analysis" / "round_config.yaml"


def test_step7_route_catalog_matches_locked_counts() -> None:
    repo_root = _real_repo_root()
    payload = load_step7_route_catalog(repo_root)
    frame = pd.DataFrame(payload["analysis_units"])
    counts = frame["step7_route_class"].value_counts().to_dict()
    assert counts == {
        "aggregated_inherited": 17,
        "event_level_escalated": 7,
        "relative_only_inherited": 6,
    }


def test_build_step7_escalation_round_writes_dual_surfaces() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config_path = _prepare_step7_fixture(root)
        with patch.dict("os.environ", {"HEAT_MNCH_STEP7_COOLDOWN_SECONDS": "0"}):
            report = build_step7_escalation_round(
                root=root,
                config_path=config_path,
                step6_report={"status": "FAIL"},
            )

        assert report["status"] == "PASS"
        assert report["step"] == "step7_escalation_round"
        assert report["summary"]["candidate_count"] == 30
        assert report["summary"]["headline_group_surface_count"] == 24
        assert report["summary"]["all_candidate_surface_count"] == 30
        assert report["summary"]["event_level_escalation_count"] == 7
        assert report["summary"]["aggregated_inherited_count"] == 17
        assert report["summary"]["relative_only_inherited_count"] == 6
        assert all(report["gates"].values())

        candidate_catalog = pd.read_csv(root / report["outputs"]["step7_candidate_catalog"])
        event_results = pd.read_csv(root / report["outputs"]["step7_event_level_results"])
        headline_surface = pd.read_csv(root / report["outputs"]["step7_headline_group_surface"])
        all_surface = pd.read_csv(root / report["outputs"]["step7_all_candidate_surface"])
        manifest = json.loads((root / "artifacts" / "analysis" / "step7" / "_runtime" / "step7_round_robin_manifest.json").read_text(encoding="utf-8"))
        job_status = pd.read_csv(root / "artifacts" / "analysis" / "step7" / "_runtime" / "step7_job_status.csv")

        assert candidate_catalog.shape[0] == 30
        assert headline_surface.shape[0] == 24
        assert all_surface.shape[0] == 30
        assert manifest["job_count"] == 62
        assert manifest["completed_count"] == 62
        assert manifest["failed_count"] == 0
        assert manifest["skipped_count"] == 0
        assert set(job_status["status"].astype(str)) == {"completed"}
        assert "maternal" not in set(headline_surface["group_id"].astype(str))
        maternal = all_surface.loc[all_surface["group_id"].astype(str).eq("maternal")].copy()
        assert maternal.shape[0] == 6
        assert maternal["public_use"].astype(str).eq("relative_only").all()
        assert maternal["headline_group_allowed"].fillna(True).astype(bool).eq(False).all()
        assert maternal["absolute_burden_allowed"].fillna(True).astype(bool).eq(False).all()
        primary_rows = event_results.loc[event_results["result_origin"].astype(str).eq("event_level_primary")]
        assert {"preterm_birth_count", "stillbirth_primary_gte28_count", "neonatal_death_count"}.issubset(
            set(primary_rows["analysis_unit_id"].astype(str))
        )
        assert (root / report["outputs"]["step7_escalation_bundle_xlsx"]).exists()
        assert (root / "reports" / "publications" / "incremental_analysis_round" / "14_step7_escalation_scope.md").exists()
        assert (root / "reports" / "publications" / "incremental_analysis_round" / "15_step7_headline_group_surface.md").exists()
        assert (root / "reports" / "publications" / "incremental_analysis_round" / "16_step7_all_candidate_surface.md").exists()


def test_step7_resume_skips_completed_jobs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config_path = _prepare_step7_fixture(root)
        with patch.dict("os.environ", {"HEAT_MNCH_STEP7_COOLDOWN_SECONDS": "0"}):
            first = build_step7_escalation_round(
                root=root,
                config_path=config_path,
                step6_report={"status": "FAIL"},
            )
            second = build_step7_escalation_round(
                root=root,
                config_path=config_path,
                step6_report={"status": "FAIL"},
                resume=True,
            )

        manifest = json.loads((root / "artifacts" / "analysis" / "step7" / "_runtime" / "step7_round_robin_manifest.json").read_text(encoding="utf-8"))
        assert first["status"] == "PASS"
        assert second["status"] == "PASS"
        assert manifest["completed_count"] == 62
        assert all(int(job["attempt_count"]) == 1 for job in manifest["jobs"])


def test_build_step7_escalation_round_fails_when_non_maternal_headline_is_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config_path = _prepare_step7_fixture(root)
        headlines_path = root / "artifacts" / "analysis" / "step6" / "step6_group_headlines.csv"
        headlines = pd.read_csv(headlines_path)
        mask = headlines["group_id"].astype(str).eq("perinatal_nascimento")
        headlines.loc[mask, "headline_selected_flag"] = False
        headlines.loc[mask, "headline_missing_flag"] = True
        headlines.to_csv(headlines_path, index=False)
        with patch.dict("os.environ", {"HEAT_MNCH_STEP7_COOLDOWN_SECONDS": "0"}):
            report = build_step7_escalation_round(
                root=root,
                config_path=config_path,
                step6_report={"status": "FAIL"},
            )

        assert report["status"] == "FAIL"
        assert report["gates"]["step6_start_condition_acceptable"] is False


def test_step7_cli_command_runs_on_temp_fixture() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config_path = _prepare_step7_fixture(root)
        stdout = io.StringIO()
        with (
            patch.dict("os.environ", {"HEAT_MNCH_STEP7_COOLDOWN_SECONDS": "0"}),
            patch("heat_mnch.cli.root_path", return_value=root),
            redirect_stdout(stdout),
        ):
            exit_code = main(["step-7-escalation-round", "--config-path", str(config_path)])
        payload = json.loads(stdout.getvalue())
        assert exit_code == 0
        assert payload["status"] == "PASS"
        assert payload["summary"]["headline_group_surface_count"] == 24
