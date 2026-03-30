# HEAT Maternal Child

This repository was restarted from zero and rebuilt as a reproducible scaffold for a Brazilian heat and maternal, neonatal, and child health pipeline. The current baseline follows the operating prompt provided in `codex_gold_ready_pipeline_prompt.docx`, but adapts it to the reality of an empty repository: the project now has the execution framework, contracts, manifests, run logging, and gold-readiness bookkeeping needed to rebuild the pipeline safely.

## What is implemented now

- Local-first repository structure with strict `raw -> bronze -> silver -> gold` boundaries.
- Python package and CLI for preflight checks, inventory, manifest generation, contract validation, run logging, and outcome status notes.
- Externalized configuration and rules under `config/`.
- Initial gold schema contracts under `schemas/`.
- Configuration files stored as JSON-compatible YAML, so the scaffold runs with only the Python standard library.
- Documentation for operating phases, rollback rules, and current blockers.
- Automated unit tests for the scaffolded operational tooling.

## Current limitation

The dataset-specific ingestion and promotion scripts referenced by the DOCX prompt do not exist in this fresh restart. No raw data snapshots are present either. Because of that, endpoint production is currently blocked upstream, and the repository records that state explicitly instead of fabricating outputs.

## Official analytical flow

The official analytical flow is now `steps-only`.

- Supported analytical commands are `step-0-freeze` through `step-7-escalation-round` and `steps-run --through step7`.
- `track_a` and `track_b` are retained on disk only as archived historical code and are not part of the official CLI, test suite, or execution path.
- Shared internal marts for the official flow are materialized under `artifacts/analysis/shared_marts`.
- Official analytical tests live under `tests/`; archived legacy tests live under `archive/legacy_tests/` and are intentionally excluded from the active suite.
- `step8` and `step9` are reserved for post-analytic editorial closure and next-cycle planning under `reports/publications/incremental_analysis_round/`; they are not CLI commands and do not extend the executable analysis pipeline.

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

## Core commands

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
artifacts/
config/
data/
docs/
reports/
references/
schemas/
scripts/
src/
tests/
```

## Operating references

- [docs/OPERATING_PLAYBOOK.md](docs/OPERATING_PLAYBOOK.md)
- [docs/GEOGRAPHIC_CONTRACT.md](docs/GEOGRAPHIC_CONTRACT.md)
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md)
- [docs/DEPLOY_MAC_MINI_CLOUDFLARE.md](docs/DEPLOY_MAC_MINI_CLOUDFLARE.md)
- [reports/README.md](reports/README.md)
- [references/README.md](references/README.md)
