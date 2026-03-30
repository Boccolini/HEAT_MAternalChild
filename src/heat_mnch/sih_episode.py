from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SILVER_SIH_AIH_COLLAPSED = "silver_sih_aih_collapsed"
SILVER_SIH_EPISODE_LABELED = "silver_sih_episode_labeled"
SILVER_SIH_EPISODE_CANONICAL = "silver_sih_episode_canonical"
SILVER_SIH_EPISODE_EXCLUDED = "silver_sih_episode_excluded"

COLLAPSE_SELECTION_METHOD = "min_dt_int_then_min_dt_saida_then_source_row_order"
EPISODE_SELECTION_BASIS = "min_dt_int_then_min_dt_saida_max_then_representative_row_order"


@dataclass(frozen=True)
class SihEpisodeConfig:
    continuation_ident_values: frozenset[str] = frozenset({"3", "5"})
    same_episode_gap_days: int = 1
    patient_key_candidates: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("cns", ("CNS",)),
        ("cpf_paciente", ("CPF_PACIENTE", "CPF_PAC")),
        ("prontuario", ("PRONTUARIO", "NUM_PRONTUARIO")),
    )
    patient_fallback_columns: tuple[str, ...] = ("NASC", "SEXO", "MUNIC_RES", "CEP")
    facility_key_candidates: tuple[str, ...] = ("CNES", "CGC_HOSP")
    followon_reason_columns: tuple[str, ...] = ("MOTIVO_APRESENTACAO", "MOT_APRES", "TIPO_SAIDA", "MOT_SAIDA")
    intercorrencia_reason_values: tuple[str, ...] = ("2.2",)
    procedure_change_reason_values: tuple[str, ...] = ("2.6",)
    reoperation_reason_values: tuple[str, ...] = ("2.7",)
    transfer_reason_prefixes: tuple[str, ...] = ("3.",)
    uti_signal_numeric_columns: tuple[str, ...] = ("UTI_MES_TO", "UTI_INT_TO", "VAL_UTI", "VAL_UCI")
    uti_signal_flag_columns: tuple[str, ...] = ("MARCA_UTI", "MARCA_UCI")
    procedure_change_pairs: tuple[tuple[str, str], ...] = (("PROC_SOLIC", "PROC_REA"),)


def _coalesce_config(config: SihEpisodeConfig | dict[str, object] | None) -> SihEpisodeConfig:
    if config is None:
        return SihEpisodeConfig()
    if isinstance(config, SihEpisodeConfig):
        return config
    return SihEpisodeConfig(**config)


def _clean_text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.strip()


def _parse_date_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(_clean_text_series(frame, column).replace("", pd.NA), format="%Y-%m-%d", errors="coerce")


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    return pd.to_numeric(_clean_text_series(frame, column).replace("", pd.NA), errors="coerce")


def _series_unique_values(values: pd.Series) -> list[str]:
    unique_values = {str(value).strip() for value in values.tolist() if str(value).strip()}

    def sort_key(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value)

    return sorted(unique_values, key=sort_key)


def _join_sorted_unique(values: pd.Series) -> str:
    return "|".join(_series_unique_values(values))


def _count_unique_nonblank(values: pd.Series) -> int:
    return len(_series_unique_values(values))


def _format_date_series(values: pd.Series) -> pd.Series:
    formatted = values.dt.strftime("%Y-%m-%d")
    return formatted.fillna("")


def _first_nonblank_from_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    result = pd.Series("", index=frame.index, dtype="object")
    for column in available:
        values = _clean_text_series(frame, column)
        result = result.where(result.ne(""), values)
    return result


def _build_patient_key(frame: pd.DataFrame, config: SihEpisodeConfig) -> tuple[pd.Series, pd.Series]:
    patient_key = pd.Series("", index=frame.index, dtype="object")
    patient_key_basis = pd.Series("", index=frame.index, dtype="object")
    for basis_name, columns in config.patient_key_candidates:
        values = _first_nonblank_from_columns(frame, columns)
        use_mask = patient_key.eq("") & values.ne("")
        patient_key.loc[use_mask] = basis_name + "::" + values.loc[use_mask]
        patient_key_basis.loc[use_mask] = basis_name

    fallback_columns = [column for column in config.patient_fallback_columns if column in frame.columns]
    if fallback_columns:
        fallback_frame = frame[fallback_columns].fillna("").astype(str)
        for column in fallback_columns:
            fallback_frame[column] = fallback_frame[column].str.strip()
        fallback_hash = pd.util.hash_pandas_object(fallback_frame, index=False).astype(str)
        fallback_blank = fallback_frame.eq("").all(axis=1)
        use_mask = patient_key.eq("") & ~fallback_blank
        patient_key.loc[use_mask] = "fallback_hash::" + fallback_hash.loc[use_mask]
        patient_key_basis.loc[use_mask] = "fallback_hash:" + ",".join(fallback_columns)

    missing_mask = patient_key.eq("")
    patient_key.loc[missing_mask] = "missing_patient::" + _clean_text_series(frame, "N_AIH").loc[missing_mask]
    patient_key_basis.loc[missing_mask] = "missing_patient_fields"
    return patient_key, patient_key_basis


def _build_facility_key(frame: pd.DataFrame, config: SihEpisodeConfig) -> pd.Series:
    facility_key = _first_nonblank_from_columns(frame, config.facility_key_candidates)
    missing_mask = facility_key.eq("")
    facility_key.loc[missing_mask] = "missing_facility::" + _clean_text_series(frame, "N_AIH").loc[missing_mask]
    return facility_key


def _has_positive_numeric_signal(row: pd.Series, columns: tuple[str, ...]) -> bool:
    for column in columns:
        value = str(row.get(column, "")).strip()
        if not value:
            continue
        try:
            if float(value) > 0:
                return True
        except ValueError:
            continue
    return False


def _has_flag_signal(row: pd.Series, columns: tuple[str, ...]) -> bool:
    for column in columns:
        value = str(row.get(column, "")).strip()
        if value and value not in {"0", "00"}:
            return True
    return False


def _reason_code_from_row(row: pd.Series, columns: tuple[str, ...]) -> str:
    for column in columns:
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return ""


def _is_procedure_change(row: pd.Series, config: SihEpisodeConfig) -> bool:
    for left_column, right_column in config.procedure_change_pairs:
        left_value = str(row.get(left_column, "")).strip()
        right_value = str(row.get(right_column, "")).strip()
        if left_value and right_value and left_value != right_value:
            return True
    return False


def collapse_same_n_aih(df: pd.DataFrame, config: SihEpisodeConfig | dict[str, object] | None = None) -> pd.DataFrame:
    config = _coalesce_config(config)
    if "N_AIH" not in df.columns:
        raise KeyError("collapse_same_n_aih requires N_AIH")

    working = df.copy().reset_index(drop=True)
    working["REPRESENTATIVE_SOURCE_ROW_ORDER"] = working.index
    working["__DT_INTER_PARSED"] = _parse_date_series(working, "DT_INTER")
    working["__DT_SAIDA_PARSED"] = _parse_date_series(working, "DT_SAIDA")
    working["__VAL_TOT_NUM"] = _numeric_series(working, "VAL_TOT")

    representative = (
        working.sort_values(
            ["N_AIH", "__DT_INTER_PARSED", "__DT_SAIDA_PARSED", "REPRESENTATIVE_SOURCE_ROW_ORDER"],
            kind="mergesort",
            na_position="last",
        )
        .groupby("N_AIH", sort=False, group_keys=False)
        .head(1)
        .set_index("N_AIH")
    )
    grouped = working.groupby("N_AIH", sort=False, dropna=False)

    collapsed = representative.copy()
    collapsed["N_SOURCE_ROWS"] = grouped.size()
    collapsed["IDENT_VALUES"] = grouped["IDENT"].agg(_join_sorted_unique) if "IDENT" in working.columns else ""
    collapsed["HAS_CONTINUATION_IDENT"] = (
        grouped["IDENT"].agg(lambda values: int(any(value in config.continuation_ident_values for value in _series_unique_values(values))))
        if "IDENT" in working.columns
        else 0
    )
    collapsed["DT_INT_MIN"] = _format_date_series(grouped["__DT_INTER_PARSED"].min())
    collapsed["DT_SAIDA_MAX"] = _format_date_series(grouped["__DT_SAIDA_PARSED"].max())
    collapsed["SEQUENCIA_VALUES"] = grouped["SEQUENCIA"].agg(_join_sorted_unique) if "SEQUENCIA" in working.columns else ""
    collapsed["VAL_TOT_MIN"] = grouped["__VAL_TOT_NUM"].min()
    collapsed["VAL_TOT_MAX"] = grouped["__VAL_TOT_NUM"].max()
    collapsed["N_UNIQUE_DT_INT"] = grouped["DT_INTER"].agg(_count_unique_nonblank) if "DT_INTER" in working.columns else 0
    collapsed["N_UNIQUE_DT_SAIDA"] = grouped["DT_SAIDA"].agg(_count_unique_nonblank) if "DT_SAIDA" in working.columns else 0
    collapsed["CANONICAL_ROW_SELECTION_METHOD"] = COLLAPSE_SELECTION_METHOD
    collapsed["CANONICAL_ROW_SELECTION_USES_SEQUENCIA_ONLY"] = 0

    drop_columns = [column for column in ("__DT_INTER_PARSED", "__DT_SAIDA_PARSED", "__VAL_TOT_NUM") if column in collapsed.columns]
    collapsed = collapsed.drop(columns=drop_columns).reset_index()
    return collapsed.sort_values(["REPRESENTATIVE_SOURCE_ROW_ORDER", "N_AIH"], kind="mergesort").reset_index(drop=True)


def assign_episode_ids(collapsed_df: pd.DataFrame, config: SihEpisodeConfig | dict[str, object] | None = None) -> pd.DataFrame:
    config = _coalesce_config(config)
    working = collapsed_df.copy().reset_index(drop=True)
    if "REPRESENTATIVE_SOURCE_ROW_ORDER" not in working.columns:
        working["REPRESENTATIVE_SOURCE_ROW_ORDER"] = working.index

    working["__DT_INT_MIN_PARSED"] = _parse_date_series(working, "DT_INT_MIN")
    working["__DT_SAIDA_MAX_PARSED"] = _parse_date_series(working, "DT_SAIDA_MAX")
    working["PATIENT_KEY"], working["PATIENT_KEY_BASIS"] = _build_patient_key(working, config)
    working["FACILITY_KEY"] = _build_facility_key(working, config)

    ordered = working.sort_values(
        ["PATIENT_KEY", "FACILITY_KEY", "__DT_INT_MIN_PARSED", "__DT_SAIDA_MAX_PARSED", "REPRESENTATIVE_SOURCE_ROW_ORDER", "N_AIH"],
        kind="mergesort",
        na_position="last",
    ).copy()

    episode_ids = pd.Series("", index=ordered.index, dtype="object")
    episode_sequence = 0
    max_gap = pd.Timedelta(days=config.same_episode_gap_days)
    for (_, _), group in ordered.groupby(["PATIENT_KEY", "FACILITY_KEY"], sort=False):
        current_episode_id = ""
        current_episode_end = pd.NaT
        for index, row in group.iterrows():
            start = row["__DT_INT_MIN_PARSED"]
            end = row["__DT_SAIDA_MAX_PARSED"]
            if pd.isna(end):
                end = start
            same_episode = (
                bool(current_episode_id)
                and pd.notna(start)
                and pd.notna(current_episode_end)
                and start <= current_episode_end + max_gap
            )
            if not same_episode:
                episode_sequence += 1
                current_episode_id = f"SIH_EPISODE_{episode_sequence:012d}"
                current_episode_end = end
            elif pd.notna(end) and (pd.isna(current_episode_end) or end > current_episode_end):
                current_episode_end = end
            episode_ids.loc[index] = current_episode_id

    ordered["EPISODE_ID"] = episode_ids
    return (
        ordered.sort_values(["REPRESENTATIVE_SOURCE_ROW_ORDER", "N_AIH"], kind="mergesort")
        .drop(columns=["__DT_INT_MIN_PARSED", "__DT_SAIDA_MAX_PARSED"])
        .reset_index(drop=True)
    )


def _infer_no_go_reason(row: pd.Series, config: SihEpisodeConfig) -> str:
    if int(row["PROMOTE_TO_CANONICAL"]) == 1 and int(row["EPISODE_SINGLE_AIH"]) == 1:
        return "single_aih_episode_promoted"
    if int(row["PROMOTE_TO_CANONICAL"]) == 1:
        return "canonical_original_admission_promoted"

    reason_code = _reason_code_from_row(row, config.followon_reason_columns)
    if reason_code in config.intercorrencia_reason_values:
        return "intercorrencia_same_episode"
    if reason_code in config.procedure_change_reason_values:
        return "procedure_change_same_episode"
    if reason_code in config.reoperation_reason_values:
        return "reoperacao_same_episode"
    if any(reason_code.startswith(prefix) for prefix in config.transfer_reason_prefixes):
        return "transfer_same_episode"
    if int(row.get("HAS_CONTINUATION_IDENT", 0)) == 1:
        return "continuation_same_n_aih"
    if _has_positive_numeric_signal(row, ("VAL_TRANSP",)):
        return "transfer_same_episode"
    if _is_procedure_change(row, config):
        return "procedure_change_same_episode"
    if _has_positive_numeric_signal(row, config.uti_signal_numeric_columns) or _has_flag_signal(row, config.uti_signal_flag_columns):
        return "uti_or_bed_movement_followon"
    return "followon_aih_same_episode"


def label_episode_roles(episode_df: pd.DataFrame, config: SihEpisodeConfig | dict[str, object] | None = None) -> pd.DataFrame:
    config = _coalesce_config(config)
    working = episode_df.copy().reset_index(drop=True)
    if "REPRESENTATIVE_SOURCE_ROW_ORDER" not in working.columns:
        working["REPRESENTATIVE_SOURCE_ROW_ORDER"] = working.index

    working["__DT_INT_MIN_PARSED"] = _parse_date_series(working, "DT_INT_MIN")
    working["__DT_SAIDA_MAX_PARSED"] = _parse_date_series(working, "DT_SAIDA_MAX")

    canonical_rows = (
        working.sort_values(
            ["EPISODE_ID", "__DT_INT_MIN_PARSED", "__DT_SAIDA_MAX_PARSED", "REPRESENTATIVE_SOURCE_ROW_ORDER", "N_AIH"],
            kind="mergesort",
            na_position="last",
        )
        .groupby("EPISODE_ID", sort=False, group_keys=False)
        .head(1)[["EPISODE_ID", "N_AIH"]]
        .rename(columns={"N_AIH": "CANONICAL_N_AIH"})
    )
    episode_counts = working.groupby("EPISODE_ID", dropna=False)["N_AIH"].nunique()
    canonical_map = canonical_rows.set_index("EPISODE_ID")["CANONICAL_N_AIH"]

    working["EPISODE_N_UNIQUE_AIH"] = working["EPISODE_ID"].map(episode_counts).fillna(0).astype(int)
    working["EPISODE_SINGLE_AIH"] = (working["EPISODE_N_UNIQUE_AIH"] == 1).astype(int)
    working["CANONICAL_N_AIH"] = working["EPISODE_ID"].map(canonical_map)
    working["PROMOTE_TO_CANONICAL"] = working["N_AIH"].eq(working["CANONICAL_N_AIH"]).astype(int)
    working["PROMOTE_SINGLE_AIH_TO_CANONICAL"] = (
        (working["PROMOTE_TO_CANONICAL"] == 1) & (working["EPISODE_SINGLE_AIH"] == 1)
    ).astype(int)
    working["NO_GO_TO_SILVER"] = (
        (working["EPISODE_N_UNIQUE_AIH"] > 1) & (working["PROMOTE_TO_CANONICAL"] == 0)
    ).astype(int)
    working["LIKELY_COMPLICATION_OR_TRANSFER"] = (
        (working["EPISODE_N_UNIQUE_AIH"] > 1) & (working["PROMOTE_TO_CANONICAL"] == 0)
    ).astype(int)
    working["AIH_EPISODE_ROLE"] = "canonical_original_admission"
    working.loc[working["PROMOTE_TO_CANONICAL"] == 0, "AIH_EPISODE_ROLE"] = "likely_complication_transfer_or_followon"
    working["CANONICAL_SELECTION_BASIS"] = EPISODE_SELECTION_BASIS
    working["CANONICAL_SELECTION_USES_SEQUENCIA_ONLY"] = 0
    working["NO_GO_REASON"] = working.apply(lambda row: _infer_no_go_reason(row, config), axis=1)

    return (
        working.sort_values(["REPRESENTATIVE_SOURCE_ROW_ORDER", "N_AIH"], kind="mergesort")
        .drop(columns=["__DT_INT_MIN_PARSED", "__DT_SAIDA_MAX_PARSED"])
        .reset_index(drop=True)
    )


def build_canonical_silver(df: pd.DataFrame, config: SihEpisodeConfig | dict[str, object] | None = None) -> pd.DataFrame:
    config = _coalesce_config(config)
    collapsed = collapse_same_n_aih(df, config=config)
    episode_labeled = assign_episode_ids(collapsed, config=config)
    return label_episode_roles(episode_labeled, config=config)
