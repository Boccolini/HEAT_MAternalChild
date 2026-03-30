from heat_mnch.round10_execution import OUTCOME_SPECS, _exposure_column, _should_suppress


def _spec(outcome_id: str):
    for spec in OUTCOME_SPECS:
        if spec.outcome_id == outcome_id:
            return spec
    raise AssertionError(f"Outcome spec not found: {outcome_id}")


def test_exposure_column_uses_round6_frozen_windows() -> None:
    u5 = _spec("child_under5_deaths")
    dehydration = _spec("child_dehydration_admissions")

    assert _exposure_column(u5, "primary") == "TMAX_LAG0_3"
    assert _exposure_column(u5, "heat_index_repeat") == "HEAT_INDEX_LAG0_3"
    assert _exposure_column(u5, "lead_day_placebo_repeat") == "TMAX_LEAD1_4"

    assert _exposure_column(dehydration, "primary") == "TMAX_LAG0_1"
    assert _exposure_column(dehydration, "heat_index_repeat") == "HEAT_INDEX_LAG0_1"
    assert _exposure_column(dehydration, "lead_day_placebo_repeat") == "TMAX_LEAD1_2"


def test_round8_split_thresholds_apply_to_acute_domains() -> None:
    suppress, reason = _should_suppress(
        "acute_companion_primary",
        "macroregion_repeat",
        {"total_events": 99, "event_days": 100, "municipalities_with_event_day": 100},
    )
    assert suppress is True
    assert reason == "suppressed_by_round8_split_threshold"

    suppress, reason = _should_suppress(
        "acute_companion_primary",
        "subperiod_repeat_2022_2024",
        {"total_events": 200, "event_days": 29, "municipalities_with_event_day": 100},
    )
    assert suppress is True
    assert reason == "suppressed_by_round8_split_threshold"


def test_round8_interaction_thresholds_apply_to_acute_domains() -> None:
    suppress, reason = _should_suppress(
        "acute_companion_core_morbidity",
        "idh_interaction_repeat",
        {"total_events": 499, "event_days": 300, "municipalities_with_event_day": 200},
    )
    assert suppress is True
    assert reason == "suppressed_by_round8_interaction_threshold"

    suppress, reason = _should_suppress(
        "acute_companion_core_morbidity",
        "idh_interaction_repeat",
        {"total_events": 600, "event_days": 300, "municipalities_with_event_day": 99},
    )
    assert suppress is True
    assert reason == "suppressed_by_round8_interaction_threshold"

    suppress, reason = _should_suppress(
        "acute_companion_core_morbidity",
        "idh_interaction_repeat",
        {"total_events": 600, "event_days": 300, "municipalities_with_event_day": 100},
    )
    assert suppress is False
    assert reason == ""
