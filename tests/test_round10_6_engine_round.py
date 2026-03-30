from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/Users/digicode.agent/Documents/HEAT_MAternalChild")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_round10_6_registered_and_round11_depends_on_it() -> None:
    round_registry = _read_csv(
        ROOT / "artifacts/analysis/step9_rebuild/registries/round_registry.csv"
    )
    row_10_6 = next(row for row in round_registry if row["round_id"] == "10.6")
    assert row_10_6["round_label"] == "E002 Exact-Engine Unblock"
    assert "mixed-effects cloglog engine" in row_10_6["gate_rule"]

    row_11 = next(row for row in round_registry if row["round_id"] == "11")
    assert "Round 10.6 exact-engine unblock decision" in row_11["gate_rule"]
    assert "full primary replay" in row_11["gate_rule"]


def test_gate_log_and_preterm_block_now_point_to_round10_6() -> None:
    gate_log = _read_csv(
        ROOT / "artifacts/analysis/step9_rebuild/gates/gate_verdict_log.csv"
    )
    row_10_6 = next(row for row in gate_log if row["round_id"] == "10.6")
    assert row_10_6["current_status"] == "advance_with_restrictions"
    assert "validated an exact local mixed-effects cloglog route" in row_10_6["blocking_issue"]

    blocked = _read_csv(
        ROOT
        / "artifacts/analysis/step9_rebuild/results/round10/round10_blocked_domain_register.csv"
    )
    preterm = next(row for row in blocked if row["domain_id"] == "flagship_preterm_primary")
    assert preterm["blocker_class"] == "local_compute_block_after_engine_validation"
    assert preterm["next_unblock_round"] == "infrastructure_provision_then_round10_refresh"


def test_round10_6_engine_option_registry_has_expected_candidates() -> None:
    options = _read_csv(
        ROOT
        / "artifacts/analysis/step9_rebuild/registries/round10_6_engine_option_registry.csv"
    )
    by_id = {row["option_id"]: row for row in options}
    assert by_id["e002_r_lme4_glmer_cloglog"]["current_local_status"] == (
        "runtime_present_package_installed_validated"
    )
    assert by_id["e002_r_lme4_glmer_cloglog"]["decision_state"] == (
        "validated_exact_primary_engine"
    )
    assert by_id["e002_r_glmmtmb_cloglog"]["current_local_status"] == (
        "runtime_present_package_installed_tmb_version_warning"
    )
    assert by_id["e002_python_statsmodels_bayes"]["round5_compliance_status"] == (
        "non_compliant_round5"
    )
    assert by_id["e002_convenience_fallbacks"]["decision_state"] == "forbidden"


def test_round10_6_engine_decision_memo_records_local_scan() -> None:
    memo = (
        ROOT
        / "artifacts/analysis/step9_rebuild/reviews/round10_6_engine_decision_memo.md"
    ).read_text(encoding="utf-8")
    assert "`Rscript` is available locally." in memo
    assert "`lme4 2.0.1` is now installed locally." in memo
    assert "`glmmTMB 1.1.14` is now installed locally" in memo
    assert "Rows after frozen Round 5 missingness policy: `8,560`" in memo
    assert "165,745,287" in memo
    assert "0.986762" in memo
