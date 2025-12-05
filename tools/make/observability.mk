# ====================== observability.mk ======================
# = Observability targets for semantic-router monitoring       =
# ====================== observability.mk ======================

##@ Observability

# Observability directories
OBS_CONFIG_DIR = tools/observability
OBS_SCRIPTS_DIR = tools/observability/scripts

.PHONY: run-observability stop-observability \
	o11y-local o11y-compose o11y-logs o11y-status o11y-clean open-observability

## run-observability: Start observability stack (alias for o11y-local)
run-observability: o11y-local

## o11y-local: Start observability in LOCAL mode (router on host, o11y in Docker)
o11y-local:
o11y-local: ## Start observability in LOCAL mode (router on host, o11y in Docker)
	@$(call log, Starting observability in LOCAL mode...)
	@$(OBS_SCRIPTS_DIR)/start-observability.sh local

## o11y-compose: Start observability in COMPOSE mode (all services in Docker)
o11y-compose:
o11y-compose: ## Start observability in COMPOSE mode (all services in Docker)
	@$(call log, Starting observability in COMPOSE mode...)
	@$(OBS_SCRIPTS_DIR)/start-observability.sh compose

## stop-observability: Stop and remove observability containers
stop-observability:
stop-observability: ## Stop and remove observability containers
	@$(call log, Stopping observability stack...)
	@$(OBS_SCRIPTS_DIR)/stop-observability.sh

## open-observability: Open Prometheus and Grafana in browser
open-observability:
open-observability: Open Prometheus and Grafana in browser
	@echo "Opening Prometheus and Grafana..."
	@open http://localhost:9090 2>/dev/null || xdg-open http://localhost:9090 2>/dev/null || echo "Please open http://localhost:9090"
	@open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null || echo "Please open http://localhost:3000"

## o11y-logs: Show logs from observability containers
o11y-logs:
o11y-logs: ## Show logs from observability containers
	@docker compose -f tools/observability/docker-compose.obs.yml logs -f 2>/dev/null || docker compose -f deploy/docker-compose/docker-compose.yml logs prometheus grafana -f

## o11y-status: Check status of observability containers
o11y-status:
o11y-status: ## Check status of observability containers
	@echo "==> Local mode:"
	@docker compose -f tools/observability/docker-compose.obs.yml ps 2>/dev/null || echo "  Not running"
	@echo ""
	@echo "==> Compose mode:"
	@docker compose -f deploy/docker-compose/docker-compose.yml ps prometheus grafana 2>/dev/null || echo "  Not running"

## o11y-clean: Remove observability data volumes
o11y-clean:
o11y-clean: ## Remove observability data volumes
	@echo "⚠️  Removing all observability data volumes..."
	@docker volume rm prometheus-local-data grafana-local-data prometheus-data grafana-data 2>/dev/null || true
	@echo "✓ Done"
