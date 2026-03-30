# HEAT Maternal Child

This repository contains the reproducibility-facing codebase for the Brazil 2020-2024 heat and maternal, newborn, and child health study. The active scope is the executable analytical pipeline from source inventory and promotion contracts through the official `step0-step7` flow, plus the dashboard application that exposes the frozen study outputs.

## Repository scope

This repository is intentionally code-first.

- It includes executable code, configs, schemas, tests, and the dashboard application.
- It excludes editorial working files, report drafts, large artifacts, and non-essential local outputs.
- Its purpose is reproducibility, operational traceability, and frozen-release governance.

## What is implemented

- Strict `raw -> bronze -> silver -> gold` pipeline boundaries in code and config.
- A Python package and CLI for preflight checks, inventory, manifest generation, schema validation, run logging, and official analytical steps.
- The official analytical flow from `step-0-freeze` through `step-7-escalation-round`.
- A dashboard application with a frozen baseline, release metadata, citation-aware downloads, and an internal 32-outcome progression matrix.
- Automated tests for the active operational tooling and analytical orchestration.

## Official analytical boundary

The executable analysis pipeline is `steps-only`.

- Supported analytical commands are `step-0-freeze` through `step-7-escalation-round` and `steps-run --through step7`.
- `step8` and `step9` are not executable analytical commands in this repository.
- Legacy `track_a` and `track_b` are not part of the active workflow, public interface, or expected execution path.

## Current limitations

- No raw data are distributed in this repository.
- No promoted bronze, silver, or gold datasets are distributed in this repository.
- Reproducing final results still depends on the governed source data environment and the frozen analytical inputs created outside the public Git payload.
- Dashboard DOI archival is still pending; current downloads carry citation metadata and a placeholder DOI status.

## Quick start

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .[dev]
make preflight
make inventory
make init-status
make test
```

## Official analysis commands

```bash
PYTHONPATH=src .venv/bin/python -m heat_mnch.cli step-0-freeze
PYTHONPATH=src .venv/bin/python -m heat_mnch.cli step-1-descriptive-atlas
PYTHONPATH=src .venv/bin/python -m heat_mnch.cli step-2-baseline-models
PYTHONPATH=src .venv/bin/python -m heat_mnch.cli step-3-threshold-discovery
PYTHONPATH=src .venv/bin/python -m heat_mnch.cli step-4-lag-refinement
PYTHONPATH=src .venv/bin/python -m heat_mnch.cli step-5-heterogeneity-synthesis
PYTHONPATH=src .venv/bin/python -m heat_mnch.cli step-6-absolute-burden
PYTHONPATH=src .venv/bin/python -m heat_mnch.cli step-7-escalation-round
PYTHONPATH=src .venv/bin/python -m heat_mnch.cli steps-run --through step7
```

## Core operational commands

```bash
PYTHONPATH=src python3 scripts/heat_mnch.py preflight
PYTHONPATH=src python3 scripts/heat_mnch.py inventory
PYTHONPATH=src python3 scripts/heat_mnch.py build-manifest --layer raw --dataset sinasc
PYTHONPATH=src python3 scripts/heat_mnch.py validate-contract --data-path /abs/path/file.csv --schema-path schemas/gold_mnch_perinatal_diario_municipio.schema.csv
PYTHONPATH=src python3 scripts/heat_mnch.py init-run-log --label bootstrap
PYTHONPATH=src python3 scripts/heat_mnch.py init-status-note
```

## Repository layout

```text
apps/
config/
schemas/
scripts/
src/
tests/
```

## Dashboard release baseline

The dashboard is currently frozen at `freeze-2026-03-29`.

- A frozen baseline banner is shown in the application shell.
- Generated dashboard downloads append the suggested citation:
  `Boccolini, CS. Brazil's heat exposure on maternal, newborn, and child health - Dashboard.`
- The exact DOI will be attached after archival of the frozen dashboard release.
