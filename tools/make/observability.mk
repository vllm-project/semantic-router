# ====================== observability.mk ======================
# = Observability targets for semantic-router monitoring       =
# ====================== observability.mk ======================

# Observability directories
OBS_CONFIG_DIR = tools/observability
OBS_SCRIPTS_DIR = tools/observability/scripts

.PHONY: run-observability stop-observability obs-local obs-compose open-observability obs-logs obs-status obs-clean

## run-observability: Start observability stack (alias for obs-local)
run-observability: obs-local

## obs-local: Start observability in LOCAL mode (router on host, obs in Docker)
obs-local:
	@$(call log, Starting observability in LOCAL mode...)
	@$(OBS_SCRIPTS_DIR)/start-observability.sh local

## obs-compose: Start observability in COMPOSE mode (all services in Docker)
obs-compose:
	@$(call log, Starting observability in COMPOSE mode...)
	@$(OBS_SCRIPTS_DIR)/start-observability.sh compose

## stop-observability: Stop and remove observability containers
stop-observability:
	@$(call log, Stopping observability stack...)
	@$(OBS_SCRIPTS_DIR)/stop-observability.sh

## open-observability: Open Prometheus and Grafana in browser
open-observability:
	@echo "Opening Prometheus and Grafana..."
	@open http://localhost:9090 2>/dev/null || xdg-open http://localhost:9090 2>/dev/null || echo "Please open http://localhost:9090"
	@open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null || echo "Please open http://localhost:3000"

## obs-logs: Show logs from observability containers
obs-logs:
	@docker compose -f $(PWD)/docker-compose.obs.yml logs -f 2>/dev/null || docker compose logs prometheus grafana -f

## obs-status: Check status of observability containers
obs-status:
	@echo "==> Local mode:"
	@docker compose -f $(PWD)/docker-compose.obs.yml ps 2>/dev/null || echo "  Not running"
	@echo ""
	@echo "==> Compose mode:"
	@docker compose ps prometheus grafana 2>/dev/null || echo "  Not running"

## obs-clean: Remove observability data volumes
obs-clean:
	@echo "⚠️  Removing all observability data volumes..."
	@docker volume rm prometheus-local-data grafana-local-data prometheus-data grafana-data 2>/dev/null || true
	@echo "✓ Done"
