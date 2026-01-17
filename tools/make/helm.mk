# ======== helm.mk ============
# = Helm deployment targets  =
# ======== helm.mk ============

##@ Helm

# Configuration
HELM_RELEASE_NAME ?= semantic-router
HELM_NAMESPACE ?= vllm-semantic-router-system
HELM_CHART_PATH ?= deploy/helm/semantic-router
HELM_VALUES_FILE ?=
HELM_SET_VALUES ?=
HELM_TIMEOUT ?= 10m

# Colors for output (reuse from common.mk if available, otherwise define)
BLUE ?= \033[0;34m
GREEN ?= \033[0;32m
YELLOW ?= \033[1;33m
RED ?= \033[0;31m
NC ?= \033[0m

.PHONY: helm-lint helm-template helm-install helm-upgrade helm-install-or-upgrade \
	helm-uninstall helm-status helm-list helm-history helm-rollback helm-test \
	helm-package helm-dev helm-prod helm-values helm-manifest \
	helm-port-forward-api helm-port-forward-grpc helm-port-forward-metrics \
	helm-logs helm-clean helm-setup helm-cleanup helm-reinstall help-helm _check-k8s

helm-lint: ## Lint the Helm chart
helm-lint:
	@$(LOG_TARGET)
	@helm lint $(HELM_CHART_PATH)
	@echo "$(GREEN)[SUCCESS]$(NC) Helm chart linted successfully"

helm-template: ## Template the Helm chart (dry-run)
helm-template:
	@$(LOG_TARGET)
	@helm template $(HELM_RELEASE_NAME) $(HELM_CHART_PATH) \
		$(if $(HELM_VALUES_FILE),-f $(HELM_VALUES_FILE)) \
		$(if $(HELM_SET_VALUES),--set $(HELM_SET_VALUES)) \
		--namespace $(HELM_NAMESPACE)

helm-install: ## Install the Helm chart
helm-install: _check-k8s
	@$(LOG_TARGET)
	@echo "Installing Helm release: $(HELM_RELEASE_NAME)"
	@if helm list -n $(HELM_NAMESPACE) 2>/dev/null | grep -q "^$(HELM_RELEASE_NAME)"; then \
		echo "$(YELLOW)[WARNING]$(NC) Release $(HELM_RELEASE_NAME) already exists in namespace $(HELM_NAMESPACE)"; \
		echo "$(BLUE)[INFO]$(NC) Use 'make helm-upgrade' to upgrade or 'make helm-uninstall' to remove it first"; \
		exit 1; \
	fi
	@echo "$(BLUE)[INFO]$(NC) Ensuring namespace $(HELM_NAMESPACE) exists..."
	@kubectl get namespace $(HELM_NAMESPACE) &>/dev/null || kubectl create namespace $(HELM_NAMESPACE)
	@helm install $(HELM_RELEASE_NAME) $(HELM_CHART_PATH) \
		$(if $(HELM_VALUES_FILE),-f $(HELM_VALUES_FILE)) \
		$(if $(HELM_SET_VALUES),--set $(HELM_SET_VALUES)) \
		--namespace $(HELM_NAMESPACE) \
		--wait \
		--timeout $(HELM_TIMEOUT)
	@echo "$(GREEN)[SUCCESS]$(NC) Helm chart installed successfully"
	@$(MAKE) helm-status

helm-upgrade: ## Upgrade the Helm release
helm-upgrade: _check-k8s
	@$(LOG_TARGET)
	@echo "Upgrading Helm release: $(HELM_RELEASE_NAME)"
	@helm upgrade $(HELM_RELEASE_NAME) $(HELM_CHART_PATH) \
		$(if $(HELM_VALUES_FILE),-f $(HELM_VALUES_FILE)) \
		$(if $(HELM_SET_VALUES),--set $(HELM_SET_VALUES)) \
		--namespace $(HELM_NAMESPACE) \
		--wait \
		--timeout $(HELM_TIMEOUT)
	@echo "$(GREEN)[SUCCESS]$(NC) Helm release upgraded successfully"
	@$(MAKE) helm-status

helm-install-or-upgrade: ## Install or upgrade the Helm release (idempotent)
helm-install-or-upgrade: _check-k8s
	@if helm list -n $(HELM_NAMESPACE) 2>/dev/null | grep -q "^$(HELM_RELEASE_NAME)"; then \
		echo "$(BLUE)[INFO]$(NC) Release exists, upgrading..."; \
		$(MAKE) helm-upgrade; \
	else \
		echo "$(BLUE)[INFO]$(NC) Release does not exist, installing..."; \
		$(MAKE) helm-install; \
	fi

helm-uninstall: ## Uninstall the Helm release
helm-uninstall:
	@$(LOG_TARGET)
	@helm uninstall $(HELM_RELEASE_NAME) --namespace $(HELM_NAMESPACE)
	@echo "$(GREEN)[SUCCESS]$(NC) Helm release uninstalled successfully"

helm-status: ## Show Helm release status
helm-status:
	@$(LOG_TARGET)
	@helm status $(HELM_RELEASE_NAME) --namespace $(HELM_NAMESPACE) || echo "$(RED)[ERROR]$(NC) Release not found"
	@echo ""
	@echo "$(BLUE)[INFO]$(NC) Deployed resources:"
	@kubectl get all -n $(HELM_NAMESPACE) -l app.kubernetes.io/instance=$(HELM_RELEASE_NAME) || echo "$(YELLOW)[WARNING]$(NC) No resources found"

helm-list: ## List all Helm releases
helm-list:
	@$(LOG_TARGET)
	@helm list --all-namespaces

helm-history: ## Show Helm release history
helm-history:
	@$(LOG_TARGET)
	@helm history $(HELM_RELEASE_NAME) --namespace $(HELM_NAMESPACE)

helm-rollback: ## Rollback to previous Helm release
helm-rollback:
	@$(LOG_TARGET)
	@helm rollback $(HELM_RELEASE_NAME) --namespace $(HELM_NAMESPACE) --wait
	@echo "$(GREEN)[SUCCESS]$(NC) Helm release rolled back successfully"
	@$(MAKE) helm-status

helm-test: ## Test the Helm release
helm-test: _check-k8s
	@$(LOG_TARGET)
	@if ! helm list -n $(HELM_NAMESPACE) 2>/dev/null | grep -q "^$(HELM_RELEASE_NAME)"; then \
		echo "$(RED)[ERROR]$(NC) Release $(HELM_RELEASE_NAME) not found in namespace $(HELM_NAMESPACE)"; \
		echo "$(BLUE)[INFO]$(NC) Please run 'make helm-install' or 'make helm-setup' first"; \
		exit 1; \
	fi
	@echo "$(BLUE)[INFO]$(NC) Checking deployment status..."
	@kubectl wait --for=condition=Available deployment/$(HELM_RELEASE_NAME) \
		-n $(HELM_NAMESPACE) --timeout=300s || echo "$(RED)[ERROR]$(NC) Deployment not ready"
	@echo "$(BLUE)[INFO]$(NC) Checking pod status..."
	@kubectl get pods -n $(HELM_NAMESPACE) -l app.kubernetes.io/instance=$(HELM_RELEASE_NAME) || echo "$(RED)[ERROR]$(NC) Cannot get pods"
	@echo "$(BLUE)[INFO]$(NC) Checking services..."
	@kubectl get svc -n $(HELM_NAMESPACE) -l app.kubernetes.io/instance=$(HELM_RELEASE_NAME) || echo "$(RED)[ERROR]$(NC) Cannot get services"
	@echo "$(BLUE)[INFO]$(NC) Checking PVC..."
	@kubectl get pvc -n $(HELM_NAMESPACE) || echo "$(YELLOW)[WARNING]$(NC) Cannot get PVC"
	@echo "$(GREEN)[SUCCESS]$(NC) Helm release test completed"

helm-package: ## Package the Helm chart
helm-package:
	@$(LOG_TARGET)
	@mkdir -p ./dist
	@tmp_dir=$$(mktemp -d); \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	cp -a $(HELM_CHART_PATH) "$$tmp_dir/chart"; \
	helm package -u "$$tmp_dir/chart" --destination ./dist
	@echo "$(GREEN)[SUCCESS]$(NC) Helm chart packaged successfully"
	@ls -lh ./dist/semantic-router-*.tgz

helm-dev: ## Deploy with development configuration
helm-dev:
	@$(LOG_TARGET)
	@$(MAKE) helm-install HELM_VALUES_FILE=$(HELM_CHART_PATH)/values-dev.yaml
	@echo ""
	@echo "$(GREEN)[SUCCESS]$(NC) Development deployment completed!"
	@echo "$(BLUE)[INFO]$(NC) Next steps:"
	@echo "  - Test deployment: make helm-test"
	@echo "  - Port forward API: make helm-port-forward-api"
	@echo "  - View logs: make helm-logs"

helm-prod: ## Deploy with production configuration
helm-prod:
	@$(LOG_TARGET)
	@$(MAKE) helm-install HELM_VALUES_FILE=$(HELM_CHART_PATH)/values-prod.yaml

helm-values: ## Show computed Helm values
helm-values:
	@$(LOG_TARGET)
	@helm get values $(HELM_RELEASE_NAME) --namespace $(HELM_NAMESPACE) --all

helm-manifest: ## Show deployed Helm manifest
helm-manifest:
	@$(LOG_TARGET)
	@helm get manifest $(HELM_RELEASE_NAME) --namespace $(HELM_NAMESPACE)

helm-port-forward-api: ## Port forward Classification API (8080)
helm-port-forward-api:
	@$(LOG_TARGET)
	@echo "$(YELLOW)[INFO]$(NC) Access API at: http://localhost:8080"
	@echo "$(YELLOW)[INFO]$(NC) Health check: curl http://localhost:8080/health"
	@echo "$(YELLOW)[INFO]$(NC) Press Ctrl+C to stop port forwarding"
	@kubectl port-forward -n $(HELM_NAMESPACE) svc/$(HELM_RELEASE_NAME) 8080:8080

helm-port-forward-grpc: ## Port forward gRPC API (50051)
helm-port-forward-grpc:
	@$(LOG_TARGET)
	@echo "$(YELLOW)[INFO]$(NC) Access gRPC API at: localhost:50051"
	@echo "$(YELLOW)[INFO]$(NC) Press Ctrl+C to stop port forwarding"
	@kubectl port-forward -n $(HELM_NAMESPACE) svc/$(HELM_RELEASE_NAME) 50051:50051

helm-port-forward-metrics: ## Port forward Prometheus metrics (9190)
helm-port-forward-metrics:
	@$(LOG_TARGET)
	@echo "$(YELLOW)[INFO]$(NC) Access metrics at: http://localhost:9190/metrics"
	@echo "$(YELLOW)[INFO]$(NC) Press Ctrl+C to stop port forwarding"
	@kubectl port-forward -n $(HELM_NAMESPACE) svc/$(HELM_RELEASE_NAME)-metrics 9190:9190

helm-logs: ## Show semantic-router logs
helm-logs:
	@$(LOG_TARGET)
	@kubectl logs -n $(HELM_NAMESPACE) -l app.kubernetes.io/instance=$(HELM_RELEASE_NAME) -f

helm-setup: helm-dev ## Complete setup: install with dev configuration
helm-setup:
	@echo "$(GREEN)[SUCCESS]$(NC) Helm setup completed!"
	@echo "$(BLUE)[INFO]$(NC) Next steps:"
	@echo "  - Test deployment: make helm-test"
	@echo "  - Port forward API: make helm-port-forward-api"
	@echo "  - View logs: make helm-logs"

helm-cleanup: helm-uninstall ## Complete cleanup: uninstall and delete namespace
helm-cleanup:
	@$(LOG_TARGET)
	@echo "Cleaning up namespace..."
	@kubectl delete namespace $(HELM_NAMESPACE) --ignore-not-found=true
	@echo "$(GREEN)[SUCCESS]$(NC) Complete cleanup finished!"

helm-clean: ## Alias for helm-cleanup
helm-clean: helm-cleanup

helm-reinstall: ## Force reinstall: cleanup and install fresh
helm-reinstall:
	@$(LOG_TARGET)
	@echo "$(YELLOW)[INFO]$(NC) Force reinstalling Helm release..."
	@echo "$(BLUE)[STEP 1/5]$(NC) Uninstalling existing release (if any)..."
	@helm uninstall $(HELM_RELEASE_NAME) --namespace $(HELM_NAMESPACE) 2>/dev/null || echo "No existing release found"
	@echo "$(BLUE)[STEP 2/5]$(NC) Deleting namespace..."
	@kubectl delete namespace $(HELM_NAMESPACE) --ignore-not-found=true --wait=false 2>/dev/null || true
	@echo "$(BLUE)[STEP 3/5]$(NC) Waiting for namespace to be fully deleted..."
	@timeout=30; \
	elapsed=0; \
	while kubectl get namespace $(HELM_NAMESPACE) &>/dev/null && [ $$elapsed -lt $$timeout ]; do \
		echo "  Waiting for namespace $(HELM_NAMESPACE) to terminate... ($$elapsed/$$timeout seconds)"; \
		sleep 2; \
		elapsed=$$((elapsed + 2)); \
	done; \
	if kubectl get namespace $(HELM_NAMESPACE) &>/dev/null; then \
		echo "$(YELLOW)[WARNING]$(NC) Namespace still exists after $$timeout seconds, forcing cleanup..."; \
		kubectl get namespace $(HELM_NAMESPACE) -o json 2>/dev/null | jq '.spec.finalizers = []' | kubectl replace --raw /api/v1/namespaces/$(HELM_NAMESPACE)/finalize -f - 2>/dev/null || true; \
		sleep 2; \
	fi
	@echo "$(GREEN)[✓]$(NC) Namespace deleted successfully"
	@echo "$(BLUE)[STEP 4/5]$(NC) Ensuring namespace is gone..."
	@sleep 3
	@echo "$(BLUE)[STEP 5/5]$(NC) Installing fresh release..."
	@$(MAKE) helm-install
	@echo "$(GREEN)[SUCCESS]$(NC) Helm release reinstalled successfully!"

# Internal helper target to check if Kubernetes is available
_check-k8s:
	@if ! kubectl cluster-info &>/dev/null; then \
		echo "$(RED)[ERROR]$(NC) Kubernetes cluster is not accessible"; \
		echo "$(BLUE)[INFO]$(NC) Please ensure your Kubernetes cluster is running:"; \
		echo "  - For local development: minikube start / kind create cluster / docker desktop"; \
		echo "  - For remote clusters: check your kubeconfig and cluster connection"; \
		echo ""; \
		echo "$(YELLOW)[TIP]$(NC) You can use the following commands to start a local cluster:"; \
		echo "  - minikube: make kube-up"; \
		echo "  - kind: make kind-cluster-create"; \
		exit 1; \
	fi
	@echo "$(GREEN)[✓]$(NC) Kubernetes cluster is accessible"
