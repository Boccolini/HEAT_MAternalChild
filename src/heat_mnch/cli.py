from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LEGACY_DISABLED_COMMANDS = {
    "track-a-freeze",
    "track-a-build-semantic-layer",
    "track-a-build-marts",
    "track-a-descriptive",
    "track-a-screening",
    "track-a-dags",
    "track-a-benchmark",
    "track-a-sensitivity",
    "track-a-synthesis",
    "track-a-paper-package",
    "track-a-validity-report-v2",
    "track-a-run",
    "track-b-editorial-freeze",
    "track-b-build-window-matrix",
    "track-b-preterm-models",
    "track-b-stillbirth-models",
    "track-b-neonatal-models",
    "track-b-burden",
    "track-b-inequity",
    "track-b-robustness",
    "track-b-paper-package",
    "track-b-run",
}


def root_path() -> Path:
    return Path(__file__).resolve().parents[2]


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _add_steps_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heat-mnch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preflight")
    subparsers.add_parser("inventory")

    manifest_parser = subparsers.add_parser("build-manifest")
    manifest_parser.add_argument("--layer", required=True, choices=["raw", "bronze", "silver", "gold"])
    manifest_parser.add_argument("--dataset", required=True)

    contract_parser = subparsers.add_parser("validate-contract")
    contract_parser.add_argument("--data-path", required=True)
    contract_parser.add_argument("--schema-path", required=True)

    run_log_parser = subparsers.add_parser("init-run-log")
    run_log_parser.add_argument("--label", required=True)

    event_parser = subparsers.add_parser("log-event")
    event_parser.add_argument("--log-path", required=True)
    event_parser.add_argument("--phase", required=True)
    event_parser.add_argument("--dataset", required=True)
    event_parser.add_argument("--year", type=int)
    event_parser.add_argument("--executed-command", dest="executed_command", required=True)
    event_parser.add_argument("--decision", required=True)
    event_parser.add_argument("--status", required=True)

    extract_parser = subparsers.add_parser("extract-raw")
    extract_parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=["sim", "sinasc", "sih", "inmet", "era5_land", "ibge"],
    )
    extract_parser.add_argument("--start-year", type=int, required=True)
    extract_parser.add_argument("--end-year", type=int, required=True)
    extract_parser.add_argument("--runtime-root", default="/tmp/heat_maternal_child_runtime")
    extract_parser.add_argument("--max-workers", type=int, default=1)
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.add_argument("--log-path")

    bronze_parser = subparsers.add_parser("promote-bronze")
    bronze_parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "sim",
            "sinasc",
            "sih",
            "stillbirth",
            "inmet",
            "era5_land",
            "ibge",
            "ibge_total",
            "ibge_sidra",
            "datasus_popsvs",
            "pop_under5",
            "pop_wra",
        ],
    )
    bronze_parser.add_argument("--start-year", type=int, required=True)
    bronze_parser.add_argument("--end-year", type=int, required=True)
    bronze_parser.add_argument("--runtime-root", default="/tmp/heat_maternal_child_runtime")
    bronze_parser.add_argument("--log-path", required=True)
    bronze_parser.add_argument("--force", action="store_true")

    silver_parser = subparsers.add_parser("promote-silver")
    silver_parser.add_argument("--dataset", required=True, choices=["climate", "sinasc", "sim", "sih", "stillbirth"])
    silver_parser.add_argument("--start-year", type=int, required=True)
    silver_parser.add_argument("--end-year", type=int, required=True)
    silver_parser.add_argument("--runtime-root", default="/tmp/heat_maternal_child_runtime")
    silver_parser.add_argument("--log-path", required=True)
    silver_parser.add_argument("--force", action="store_true")

    release_parser = subparsers.add_parser("build-silver-release-candidate")
    release_parser.add_argument("--start-year", type=int, required=True)
    release_parser.add_argument("--end-year", type=int, required=True)
    release_parser.add_argument("--runtime-root", default="/tmp/heat_maternal_child_runtime")
    release_parser.add_argument("--log-path", required=True)
    release_parser.add_argument("--force", action="store_true")

    gold_parser = subparsers.add_parser("promote-gold")
    gold_parser.add_argument("--start-year", type=int, required=True)
    gold_parser.add_argument("--end-year", type=int, required=True)
    gold_parser.add_argument("--runtime-root", default="/tmp/heat_maternal_child_runtime")
    gold_parser.add_argument("--log-path", required=True)
    gold_parser.add_argument("--force", action="store_true")

    gold_perinatal_parser = subparsers.add_parser("promote-gold-perinatal")
    gold_perinatal_parser.add_argument("--start-year", type=int, required=True)
    gold_perinatal_parser.add_argument("--end-year", type=int, required=True)
    gold_perinatal_parser.add_argument("--runtime-root", default="/tmp/heat_maternal_child_runtime")
    gold_perinatal_parser.add_argument("--log-path", required=True)
    gold_perinatal_parser.add_argument("--force", action="store_true")

    gold_mortality_parser = subparsers.add_parser("promote-gold-mortality")
    gold_mortality_parser.add_argument("--start-year", type=int, required=True)
    gold_mortality_parser.add_argument("--end-year", type=int, required=True)
    gold_mortality_parser.add_argument("--runtime-root", default="/tmp/heat_maternal_child_runtime")
    gold_mortality_parser.add_argument("--log-path", required=True)
    gold_mortality_parser.add_argument("--force", action="store_true")

    gold_panel_parser = subparsers.add_parser("promote-gold-panel")
    gold_panel_parser.add_argument("--start-year", type=int, required=True)
    gold_panel_parser.add_argument("--end-year", type=int, required=True)
    gold_panel_parser.add_argument("--runtime-root", default="/tmp/heat_maternal_child_runtime")
    gold_panel_parser.add_argument("--log-path", required=True)
    gold_panel_parser.add_argument("--force", action="store_true")

    gold_derived_parser = subparsers.add_parser("promote-gold-derived")
    gold_derived_parser.add_argument("--start-year", type=int, required=True)
    gold_derived_parser.add_argument("--end-year", type=int, required=True)
    gold_derived_parser.add_argument("--runtime-root", default="/tmp/heat_maternal_child_runtime")
    gold_derived_parser.add_argument("--log-path", required=True)
    gold_derived_parser.add_argument("--force", action="store_true")

    gold_readiness_parser = subparsers.add_parser("promote-gold-readiness")
    gold_readiness_parser.add_argument("--start-year", type=int, required=True)
    gold_readiness_parser.add_argument("--end-year", type=int, required=True)
    gold_readiness_parser.add_argument("--runtime-root", default="/tmp/heat_maternal_child_runtime")
    gold_readiness_parser.add_argument("--log-path", required=True)
    gold_readiness_parser.add_argument("--force", action="store_true")

    gold_release_parser = subparsers.add_parser("promote-gold-release")
    gold_release_parser.add_argument("--start-year", type=int, required=True)
    gold_release_parser.add_argument("--end-year", type=int, required=True)
    gold_release_parser.add_argument("--runtime-root", default="/tmp/heat_maternal_child_runtime")
    gold_release_parser.add_argument("--log-path", required=True)
    gold_release_parser.add_argument("--force", action="store_true")

    municipality_context_parser = subparsers.add_parser("build-gold-municipality-context")
    municipality_context_parser.add_argument("--start-year", type=int, default=2020)
    municipality_context_parser.add_argument("--end-year", type=int, default=2024)
    municipality_context_parser.add_argument("--runtime-root", default="/tmp/heat_maternal_child_runtime")
    municipality_context_parser.add_argument("--log-path")
    municipality_context_parser.add_argument("--force", action="store_true")

    audit_parser = subparsers.add_parser("audit-gold-municipality-context")
    audit_parser.add_argument("--output-dir")

    dashboard_marts_parser = subparsers.add_parser("build-dashboard-marts")
    dashboard_marts_parser.add_argument("--start-year", type=int, required=True)
    dashboard_marts_parser.add_argument("--end-year", type=int, required=True)
    dashboard_marts_parser.add_argument("--output-dir", required=True)

    dashboard_geometry_parser = subparsers.add_parser("build-dashboard-geometry")
    dashboard_geometry_parser.add_argument("--source", required=True)
    dashboard_geometry_parser.add_argument("--output-dir", required=True)
    dashboard_geometry_parser.add_argument("--tolerance", type=float, default=0.01)

    serve_dashboard_api_parser = subparsers.add_parser("serve-dashboard-api")
    serve_dashboard_api_parser.add_argument("--dashboard-dir", required=True)
    serve_dashboard_api_parser.add_argument("--host", default="0.0.0.0")
    serve_dashboard_api_parser.add_argument("--port", type=int, default=8000)

    serve_statcalc_api_parser = subparsers.add_parser("serve-statcalc-api")
    serve_statcalc_api_parser.add_argument("--host", default="127.0.0.1")
    serve_statcalc_api_parser.add_argument("--port", type=int, default=8010)
    serve_statcalc_api_parser.add_argument("--storage-dir")
    serve_statcalc_api_parser.add_argument("--frontend-dist")

    statcalc_run_job_parser = subparsers.add_parser("statcalc-run-job")
    statcalc_run_job_parser.add_argument("--run-id", required=True)
    statcalc_run_job_parser.add_argument("--storage-dir", required=True)
    statcalc_run_job_parser.add_argument("--root", required=True)

    step0_parser = subparsers.add_parser("step-0-freeze")
    _add_steps_config_argument(step0_parser)

    step1_parser = subparsers.add_parser("step-1-descriptive-atlas")
    _add_steps_config_argument(step1_parser)

    step2_parser = subparsers.add_parser("step-2-baseline-models")
    _add_steps_config_argument(step2_parser)

    step3_parser = subparsers.add_parser("step-3-threshold-discovery")
    _add_steps_config_argument(step3_parser)

    step4_parser = subparsers.add_parser("step-4-lag-refinement")
    _add_steps_config_argument(step4_parser)

    step5_parser = subparsers.add_parser("step-5-heterogeneity-synthesis")
    _add_steps_config_argument(step5_parser)

    step6_parser = subparsers.add_parser("step-6-absolute-burden")
    _add_steps_config_argument(step6_parser)

    step7_parser = subparsers.add_parser("step-7-escalation-round")
    _add_steps_config_argument(step7_parser)
    step7_parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    step7_parser.add_argument("--max-active-jobs", type=int, default=1)
    step7_parser.add_argument("--execution-mode", default="round_robin", choices=["round_robin"])
    step7_parser.add_argument("--job-id")
    step7_parser.add_argument("--root-override", help=argparse.SUPPRESS)

    steps_run_parser = subparsers.add_parser("steps-run")
    _add_steps_config_argument(steps_run_parser)
    steps_run_parser.add_argument(
        "--through",
        default="step0",
        choices=["step0", "step1", "step2", "step3", "step4", "step5", "step6", "step7"],
    )

    subparsers.add_parser("init-status-note")
    return parser


def main(argv: list[str] | None = None, invocation: str | None = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if args_list:
        legacy_command = args_list[0]
        if legacy_command in LEGACY_DISABLED_COMMANDS:
            _print_json(
                {
                    "status": "BLOCKED",
                    "reason": "legacy_pipeline_disabled",
                    "command": legacy_command,
                    "replacement": "Use step-* or steps-run only.",
                }
            )
            return 2

    parser = build_parser()
    args = parser.parse_args(args_list)
    root_override = getattr(args, "root_override", None)
    root = Path(root_override) if root_override else root_path()

    if args.command == "preflight":
        from .preflight import run_preflight

        _print_json(run_preflight(root))
        return 0

    if args.command == "inventory":
        from .preflight import build_inventory

        _print_json(build_inventory(root))
        return 0

    if args.command == "build-manifest":
        from .manifest import build_manifest

        _print_json(build_manifest(root, args.layer, args.dataset))
        return 0

    if args.command == "validate-contract":
        from .contracts import validate_csv_contract

        result = validate_csv_contract(Path(args.data_path), root / args.schema_path if not Path(args.schema_path).is_absolute() else Path(args.schema_path))
        _print_json(result)
        return 0 if result["valid"] else 1

    if args.command == "init-run-log":
        from .run_log import init_run_log

        path = init_run_log(root, args.label)
        print(path)
        return 0

    if args.command == "extract-raw":
        from .raw_extract import extract_dataset
        from .run_log import init_run_log

        years = list(range(args.start_year, args.end_year + 1))
        log_path = Path(args.log_path) if args.log_path else init_run_log(root, f"raw_extraction_{args.start_year}_{args.end_year}")
        command = invocation if invocation else " ".join(argv if argv is not None else sys.argv[1:])
        payload = {
            "log_path": str(log_path),
            "datasets": [],
        }
        for dataset in args.datasets:
            payload["datasets"].append(
                extract_dataset(
                    root=root,
                    runtime_root=Path(args.runtime_root),
                    dataset=dataset,
                    years=years,
                    log_path=log_path,
                    command=command,
                    force=args.force,
                    max_workers=args.max_workers,
                )
            )
        _print_json(payload)
        return 0

    if args.command == "promote-bronze":
        from .bronze import promote_bronze_range

        command = invocation if invocation else " ".join(argv if argv is not None else sys.argv[1:])
        reports = promote_bronze_range(
            root=root,
            runtime_root=Path(args.runtime_root),
            dataset=args.dataset,
            start_year=args.start_year,
            end_year=args.end_year,
            log_path=Path(args.log_path),
            command=command,
            force=args.force,
        )
        _print_json({"dataset": args.dataset, "reports": reports})
        return 0

    if args.command == "promote-silver":
        from .silver_climate import promote_climate_silver_range
        from .silver_sih import promote_sih_silver_range
        from .silver_sim import promote_sim_silver_range
        from .silver_stillbirth import promote_stillbirth_silver_range
        from .silver_sinasc import promote_sinasc_silver_range

        command = invocation if invocation else " ".join(argv if argv is not None else sys.argv[1:])
        if args.dataset == "climate":
            report = promote_climate_silver_range(
                root=root,
                runtime_root=Path(args.runtime_root),
                start_year=args.start_year,
                end_year=args.end_year,
                log_path=Path(args.log_path),
                command=command,
                force=args.force,
            )
        elif args.dataset == "sinasc":
            report = promote_sinasc_silver_range(
                root=root,
                runtime_root=Path(args.runtime_root),
                start_year=args.start_year,
                end_year=args.end_year,
                log_path=Path(args.log_path),
                command=command,
                force=args.force,
            )
        elif args.dataset == "sim":
            report = promote_sim_silver_range(
                root=root,
                runtime_root=Path(args.runtime_root),
                start_year=args.start_year,
                end_year=args.end_year,
                log_path=Path(args.log_path),
                command=command,
                force=args.force,
            )
        elif args.dataset == "sih":
            report = promote_sih_silver_range(
                root=root,
                runtime_root=Path(args.runtime_root),
                start_year=args.start_year,
                end_year=args.end_year,
                log_path=Path(args.log_path),
                command=command,
                force=args.force,
            )
        elif args.dataset == "stillbirth":
            report = promote_stillbirth_silver_range(
                root=root,
                runtime_root=Path(args.runtime_root),
                start_year=args.start_year,
                end_year=args.end_year,
                log_path=Path(args.log_path),
                command=command,
                force=args.force,
            )
        else:
            parser.error(f"Unsupported silver dataset: {args.dataset}")
        _print_json(report)
        return 0

    if args.command == "build-silver-release-candidate":
        from .silver_release_candidate import build_silver_release_candidate

        command = invocation if invocation else " ".join(argv if argv is not None else sys.argv[1:])
        report = build_silver_release_candidate(
            root=root,
            runtime_root=Path(args.runtime_root),
            start_year=args.start_year,
            end_year=args.end_year,
            log_path=Path(args.log_path),
            command=command,
            force=args.force,
        )
        _print_json(report)
        return 0

    if args.command == "promote-gold":
        from .gold_promotion import promote_gold_range

        command = invocation if invocation else " ".join(argv if argv is not None else sys.argv[1:])
        report = promote_gold_range(
            root=root,
            runtime_root=Path(args.runtime_root),
            start_year=args.start_year,
            end_year=args.end_year,
            log_path=Path(args.log_path),
            command=command,
            force=args.force,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "promote-gold-perinatal":
        from .gold_promotion import promote_gold_perinatal_range

        command = invocation if invocation else " ".join(argv if argv is not None else sys.argv[1:])
        report = promote_gold_perinatal_range(
            root=root,
            runtime_root=Path(args.runtime_root),
            start_year=args.start_year,
            end_year=args.end_year,
            log_path=Path(args.log_path),
            command=command,
            force=args.force,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "promote-gold-mortality":
        from .gold_promotion import promote_gold_mortality_range

        command = invocation if invocation else " ".join(argv if argv is not None else sys.argv[1:])
        report = promote_gold_mortality_range(
            root=root,
            runtime_root=Path(args.runtime_root),
            start_year=args.start_year,
            end_year=args.end_year,
            log_path=Path(args.log_path),
            command=command,
            force=args.force,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "promote-gold-panel":
        from .gold_promotion import promote_gold_panel_range

        command = invocation if invocation else " ".join(argv if argv is not None else sys.argv[1:])
        report = promote_gold_panel_range(
            root=root,
            runtime_root=Path(args.runtime_root),
            start_year=args.start_year,
            end_year=args.end_year,
            log_path=Path(args.log_path),
            command=command,
            force=args.force,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "promote-gold-derived":
        from .gold_promotion import promote_gold_derived_range

        command = invocation if invocation else " ".join(argv if argv is not None else sys.argv[1:])
        report = promote_gold_derived_range(
            root=root,
            runtime_root=Path(args.runtime_root),
            start_year=args.start_year,
            end_year=args.end_year,
            log_path=Path(args.log_path),
            command=command,
            force=args.force,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "promote-gold-readiness":
        from .gold_promotion import promote_gold_dictionary_range

        command = invocation if invocation else " ".join(argv if argv is not None else sys.argv[1:])
        report = promote_gold_dictionary_range(
            root=root,
            runtime_root=Path(args.runtime_root),
            start_year=args.start_year,
            end_year=args.end_year,
            log_path=Path(args.log_path),
            command=command,
            force=args.force,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "promote-gold-release":
        from .gold_promotion import promote_gold_release_range

        command = invocation if invocation else " ".join(argv if argv is not None else sys.argv[1:])
        report = promote_gold_release_range(
            root=root,
            runtime_root=Path(args.runtime_root),
            start_year=args.start_year,
            end_year=args.end_year,
            log_path=Path(args.log_path),
            command=command,
            force=args.force,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "build-gold-municipality-context":
        from .gold_municipality_context import build_gold_municipality_context

        command = invocation if invocation else " ".join(argv if argv is not None else sys.argv[1:])
        report = build_gold_municipality_context(
            root=root,
            runtime_root=Path(args.runtime_root),
            start_year=args.start_year,
            end_year=args.end_year,
            log_path=Path(args.log_path) if args.log_path else None,
            command=command,
            force=args.force,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "audit-gold-municipality-context":
        from .municipality_context_audit import audit_gold_municipality_context

        report = audit_gold_municipality_context(
            root=root,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "build-dashboard-marts":
        from .dashboard_build import build_dashboard_marts

        report = build_dashboard_marts(
            root=root,
            start_year=args.start_year,
            end_year=args.end_year,
            output_dir=Path(args.output_dir),
        )
        _print_json(report)
        return 0

    if args.command == "build-dashboard-geometry":
        from .dashboard_geometry import build_dashboard_geometry

        report = build_dashboard_geometry(
            root=root,
            source_zip=Path(args.source),
            output_dir=Path(args.output_dir),
            tolerance=args.tolerance,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "serve-dashboard-api":
        import uvicorn

        from .dashboard_api.app import create_app

        app = create_app(Path(args.dashboard_dir))
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if args.command == "serve-statcalc-api":
        import uvicorn

        from .statcalc_api.app import create_app

        app = create_app(
            root=root,
            storage_dir=Path(args.storage_dir) if args.storage_dir else None,
            frontend_dist=Path(args.frontend_dist) if args.frontend_dist else None,
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if args.command == "statcalc-run-job":
        from .statcalc_api.runner import run_queued_job

        return run_queued_job(
            root=Path(args.root),
            storage_dir=Path(args.storage_dir),
            run_id=args.run_id,
        )

    if args.command == "step-0-freeze":
        from .analysis_steps.step0 import build_step0_freeze

        report = build_step0_freeze(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "step-1-descriptive-atlas":
        from .analysis_steps.step1 import build_step1_descriptive_atlas

        report = build_step1_descriptive_atlas(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "step-2-baseline-models":
        from .analysis_steps.step2 import build_step2_baseline_models

        report = build_step2_baseline_models(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "step-3-threshold-discovery":
        from .analysis_steps.step3 import build_step3_threshold_discovery

        report = build_step3_threshold_discovery(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "step-4-lag-refinement":
        from .analysis_steps.step4 import build_step4_lag_refinement

        report = build_step4_lag_refinement(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "step-5-heterogeneity-synthesis":
        from .analysis_steps.step5 import build_step5_heterogeneity_synthesis

        report = build_step5_heterogeneity_synthesis(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "step-6-absolute-burden":
        from .analysis_steps.step6 import build_step6_absolute_burden

        report = build_step6_absolute_burden(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "step-7-escalation-round":
        from .analysis_steps.step7 import build_step7_escalation_round

        report = build_step7_escalation_round(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
            resume=bool(args.resume),
            max_active_jobs=int(args.max_active_jobs),
            execution_mode=str(args.execution_mode),
            job_id=str(args.job_id) if args.job_id else None,
        )
        _print_json(report)
        return 0 if report["status"] == "PASS" else 1

    if args.command == "steps-run":
        from .analysis_steps.step0 import build_step0_freeze
        from .analysis_steps.step1 import build_step1_descriptive_atlas
        from .analysis_steps.step2 import build_step2_baseline_models
        from .analysis_steps.step3 import build_step3_threshold_discovery
        from .analysis_steps.step4 import build_step4_lag_refinement
        from .analysis_steps.step5 import build_step5_heterogeneity_synthesis
        from .analysis_steps.step6 import build_step6_absolute_burden
        from .analysis_steps.step7 import build_step7_escalation_round

        step0_report = build_step0_freeze(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
        )
        if args.through == "step0":
            payload = {
                "run_mode": "through_step0",
                "step0": step0_report,
            }
            _print_json(payload)
            return 0 if step0_report["status"] == "PASS" else 1

        step1_report = build_step1_descriptive_atlas(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
            step0_report=step0_report,
        )
        if args.through == "step1":
            payload = {
                "run_mode": "through_step1",
                "step0": step0_report,
                "step1": step1_report,
            }
            _print_json(payload)
            return 0 if step0_report["status"] == "PASS" and step1_report["status"] == "PASS" else 1

        step2_report = build_step2_baseline_models(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
            step1_report=step1_report,
        )
        if args.through == "step2":
            payload = {
                "run_mode": "through_step2",
                "step0": step0_report,
                "step1": step1_report,
                "step2": step2_report,
            }
            _print_json(payload)
            return 0 if step0_report["status"] == "PASS" and step1_report["status"] == "PASS" and step2_report["status"] == "PASS" else 1

        step3_report = build_step3_threshold_discovery(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
            step2_report=step2_report,
        )
        if args.through == "step3":
            payload = {
                "run_mode": "through_step3",
                "step0": step0_report,
                "step1": step1_report,
                "step2": step2_report,
                "step3": step3_report,
            }
            _print_json(payload)
            return 0 if (
                step0_report["status"] == "PASS"
                and step1_report["status"] == "PASS"
                and step2_report["status"] == "PASS"
                and step3_report["status"] == "PASS"
            ) else 1

        step4_report = build_step4_lag_refinement(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
            step3_report=step3_report,
        )
        if args.through == "step4":
            payload = {
                "run_mode": "through_step4",
                "step0": step0_report,
                "step1": step1_report,
                "step2": step2_report,
                "step3": step3_report,
                "step4": step4_report,
            }
            _print_json(payload)
            return 0 if (
                step0_report["status"] == "PASS"
                and step1_report["status"] == "PASS"
                and step2_report["status"] == "PASS"
                and step3_report["status"] == "PASS"
                and step4_report["status"] == "PASS"
            ) else 1

        step5_report = build_step5_heterogeneity_synthesis(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
            step4_report=step4_report,
        )
        if args.through == "step5":
            payload = {
                "run_mode": "through_step5",
                "step0": step0_report,
                "step1": step1_report,
                "step2": step2_report,
                "step3": step3_report,
                "step4": step4_report,
                "step5": step5_report,
            }
            _print_json(payload)
            return 0 if (
                step0_report["status"] == "PASS"
                and step1_report["status"] == "PASS"
                and step2_report["status"] == "PASS"
                and step3_report["status"] == "PASS"
                and step4_report["status"] == "PASS"
                and step5_report["status"] == "PASS"
            ) else 1

        step6_report = build_step6_absolute_burden(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
            step5_report=step5_report,
        )
        if args.through == "step6":
            payload = {
                "run_mode": "through_step6",
                "step0": step0_report,
                "step1": step1_report,
                "step2": step2_report,
                "step3": step3_report,
                "step4": step4_report,
                "step5": step5_report,
                "step6": step6_report,
            }
            _print_json(payload)
            return 0 if (
                step0_report["status"] == "PASS"
                and step1_report["status"] == "PASS"
                and step2_report["status"] == "PASS"
                and step3_report["status"] == "PASS"
                and step4_report["status"] == "PASS"
                and step5_report["status"] == "PASS"
                and step6_report["status"] == "PASS"
            ) else 1

        step7_report = build_step7_escalation_round(
            root=root,
            config_path=Path(args.config_path) if args.config_path else None,
            step6_report=step6_report,
        )
        payload = {
            "run_mode": "through_step7",
            "step0": step0_report,
            "step1": step1_report,
            "step2": step2_report,
            "step3": step3_report,
            "step4": step4_report,
            "step5": step5_report,
            "step6": step6_report,
            "step7": step7_report,
        }
        _print_json(payload)
        return 0 if (
            step0_report["status"] == "PASS"
            and step1_report["status"] == "PASS"
            and step2_report["status"] == "PASS"
            and step3_report["status"] == "PASS"
            and step4_report["status"] == "PASS"
            and step5_report["status"] == "PASS"
            and step7_report["status"] == "PASS"
        ) else 1

    if args.command == "log-event":
        from .run_log import append_run_event

        append_run_event(
            Path(args.log_path),
            phase=args.phase,
            dataset=args.dataset,
            year=args.year,
            command=args.executed_command,
            decision=args.decision,
            status=args.status,
        )
        print(args.log_path)
        return 0

    if args.command == "init-status-note":
        from .status import init_status_note

        print(init_status_note(root))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
