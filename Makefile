PYTHONPATH := src
PYTHON := .venv/bin/python

.PHONY: test preflight inventory init-status dashboard-frontend-build dashboard-api-prod dashboard-caddy dashboard-stack-check

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

preflight:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/heat_mnch.py preflight

inventory:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/heat_mnch.py inventory

init-status:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/heat_mnch.py init-status-note

dashboard-frontend-build:
	./scripts/build_dashboard_frontend.sh

dashboard-api-prod:
	./scripts/run_dashboard_api_prod.sh

dashboard-caddy:
	./scripts/run_dashboard_caddy.sh

dashboard-stack-check:
	./scripts/check_dashboard_stack.sh
