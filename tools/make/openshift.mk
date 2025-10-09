# OpenShift deployment targets for semantic-router
# This makefile provides commands for managing OpenShift deployments

# Configuration
OPENSHIFT_SERVER ?=
OPENSHIFT_USER ?= admin
OPENSHIFT_PASSWORD ?=
OPENSHIFT_NAMESPACE ?= vllm-semantic-router-system
OPENSHIFT_DEPLOYMENT_METHOD ?= kustomize
OPENSHIFT_CONTAINER_IMAGE ?= ghcr.io/vllm-project/semantic-router/extproc
OPENSHIFT_CONTAINER_TAG ?= latest
OPENSHIFT_STORAGE_SIZE ?= 10Gi
OPENSHIFT_MEMORY_REQUEST ?= 3Gi
OPENSHIFT_MEMORY_LIMIT ?= 6Gi
OPENSHIFT_CPU_REQUEST ?= 1
OPENSHIFT_CPU_LIMIT ?= 2
OPENSHIFT_LOG_LEVEL ?= info

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: openshift-login openshift-logout openshift-deploy openshift-undeploy openshift-status openshift-logs openshift-routes openshift-test

# Login to OpenShift cluster
openshift-login:
	@echo "$(BLUE)[INFO]$(NC) Logging into OpenShift cluster"
	@if [ -z "$(OPENSHIFT_SERVER)" ]; then \
		echo "$(RED)[ERROR]$(NC) OPENSHIFT_SERVER is required"; \
		exit 1; \
	fi
	@if [ -z "$(OPENSHIFT_PASSWORD)" ]; then \
		echo "$(RED)[ERROR]$(NC) OPENSHIFT_PASSWORD is required"; \
		exit 1; \
	fi
	@oc login -u $(OPENSHIFT_USER) -p $(OPENSHIFT_PASSWORD) $(OPENSHIFT_SERVER) --insecure-skip-tls-verify
	@echo "$(GREEN)[SUCCESS]$(NC) Logged into OpenShift cluster"

# Logout from OpenShift cluster
openshift-logout:
	@echo "$(BLUE)[INFO]$(NC) Logging out from OpenShift cluster"
	@oc logout
	@echo "$(GREEN)[SUCCESS]$(NC) Logged out from OpenShift cluster"

# Deploy semantic-router to OpenShift using Kustomize
openshift-deploy:
	@echo "$(BLUE)[INFO]$(NC) Deploying semantic-router to OpenShift namespace: $(OPENSHIFT_NAMESPACE)"
	@echo "$(BLUE)[INFO]$(NC) Using image: $(OPENSHIFT_CONTAINER_IMAGE):$(OPENSHIFT_CONTAINER_TAG)"
	@oc apply -k deploy/openshift/
	@echo "$(BLUE)[INFO]$(NC) Waiting for deployment to be ready..."
	@oc wait --for=condition=Available deployment/semantic-router -n $(OPENSHIFT_NAMESPACE) --timeout=300s || true
	@echo "$(GREEN)[SUCCESS]$(NC) Deployment completed!"
	@$(MAKE) openshift-status

# Deploy using automated script
openshift-deploy-auto:
	@echo "$(BLUE)[INFO]$(NC) Running automated OpenShift deployment"
	@if [ -z "$(OPENSHIFT_SERVER)" ] || [ -z "$(OPENSHIFT_PASSWORD)" ]; then \
		echo "$(RED)[ERROR]$(NC) OPENSHIFT_SERVER and OPENSHIFT_PASSWORD are required"; \
		echo "$(BLUE)[INFO]$(NC) Usage: make openshift-deploy-auto OPENSHIFT_SERVER=https://... OPENSHIFT_PASSWORD=..."; \
		exit 1; \
	fi
	@./deploy/openshift/deploy-to-openshift.sh \
		--server "$(OPENSHIFT_SERVER)" \
		--user "$(OPENSHIFT_USER)" \
		--password "$(OPENSHIFT_PASSWORD)" \
		--namespace "$(OPENSHIFT_NAMESPACE)" \
		--method "$(OPENSHIFT_DEPLOYMENT_METHOD)" \
		--image "$(OPENSHIFT_CONTAINER_IMAGE)" \
		--tag "$(OPENSHIFT_CONTAINER_TAG)" \
		--storage "$(OPENSHIFT_STORAGE_SIZE)" \
		--memory-request "$(OPENSHIFT_MEMORY_REQUEST)" \
		--memory-limit "$(OPENSHIFT_MEMORY_LIMIT)" \
		--cpu-request "$(OPENSHIFT_CPU_REQUEST)" \
		--cpu-limit "$(OPENSHIFT_CPU_LIMIT)" \
		--log-level "$(OPENSHIFT_LOG_LEVEL)" \
		--skip-models

# Deploy using OpenShift template
openshift-deploy-template:
	@echo "$(BLUE)[INFO]$(NC) Deploying semantic-router using OpenShift template"
	@oc process -f deploy/openshift/template.yaml \
		-p NAMESPACE=$(OPENSHIFT_NAMESPACE) \
		-p CONTAINER_IMAGE=$(OPENSHIFT_CONTAINER_IMAGE) \
		-p CONTAINER_TAG=$(OPENSHIFT_CONTAINER_TAG) \
		-p STORAGE_SIZE=$(OPENSHIFT_STORAGE_SIZE) \
		-p MEMORY_REQUEST=$(OPENSHIFT_MEMORY_REQUEST) \
		-p MEMORY_LIMIT=$(OPENSHIFT_MEMORY_LIMIT) \
		-p CPU_REQUEST=$(OPENSHIFT_CPU_REQUEST) \
		-p CPU_LIMIT=$(OPENSHIFT_CPU_LIMIT) \
		-p LOG_LEVEL=$(OPENSHIFT_LOG_LEVEL) \
		| oc apply -f -
	@echo "$(GREEN)[SUCCESS]$(NC) Template deployment completed!"
	@$(MAKE) openshift-status

# Remove semantic-router from OpenShift
openshift-undeploy:
	@echo "$(BLUE)[INFO]$(NC) Removing semantic-router from OpenShift"
	@oc delete -k deploy/openshift/ --ignore-not-found=true
	@echo "$(GREEN)[SUCCESS]$(NC) Undeployment completed"

# Clean up everything including namespace
openshift-cleanup:
	@echo "$(BLUE)[INFO]$(NC) Cleaning up namespace $(OPENSHIFT_NAMESPACE)"
	@oc delete namespace $(OPENSHIFT_NAMESPACE) --ignore-not-found=true
	@echo "$(GREEN)[SUCCESS]$(NC) Cleanup completed"

# Show deployment status
openshift-status:
	@echo "$(BLUE)[INFO]$(NC) OpenShift deployment status for namespace: $(OPENSHIFT_NAMESPACE)"
	@echo ""
	@echo "$(BLUE)=== Pods ===$(NC)"
	@oc get pods -n $(OPENSHIFT_NAMESPACE) -o wide || echo "$(YELLOW)[WARN]$(NC) Cannot get pods"
	@echo ""
	@echo "$(BLUE)=== Services ===$(NC)"
	@oc get services -n $(OPENSHIFT_NAMESPACE) || echo "$(YELLOW)[WARN]$(NC) Cannot get services"
	@echo ""
	@echo "$(BLUE)=== Routes ===$(NC)"
	@oc get routes -n $(OPENSHIFT_NAMESPACE) || echo "$(YELLOW)[WARN]$(NC) Cannot get routes"
	@echo ""
	@echo "$(BLUE)=== PVCs ===$(NC)"
	@oc get pvc -n $(OPENSHIFT_NAMESPACE) || echo "$(YELLOW)[WARN]$(NC) Cannot get PVCs"

# Show logs
openshift-logs:
	@echo "$(BLUE)[INFO]$(NC) Showing semantic-router logs"
	@oc logs -n $(OPENSHIFT_NAMESPACE) -l app=semantic-router -f

# Show logs from previous pod (for troubleshooting)
openshift-logs-previous:
	@echo "$(BLUE)[INFO]$(NC) Showing previous semantic-router logs"
	@oc logs -n $(OPENSHIFT_NAMESPACE) -l app=semantic-router --previous

# Get route URLs
openshift-routes:
	@echo "$(BLUE)[INFO]$(NC) OpenShift route URLs:"
	@API_ROUTE=$$(oc get route semantic-router-api -n $(OPENSHIFT_NAMESPACE) -o jsonpath='{.spec.host}' 2>/dev/null); \
	GRPC_ROUTE=$$(oc get route semantic-router-grpc -n $(OPENSHIFT_NAMESPACE) -o jsonpath='{.spec.host}' 2>/dev/null); \
	METRICS_ROUTE=$$(oc get route semantic-router-metrics -n $(OPENSHIFT_NAMESPACE) -o jsonpath='{.spec.host}' 2>/dev/null); \
	echo ""; \
	if [ -n "$$API_ROUTE" ]; then \
		echo "$(GREEN)Classification API:$(NC) https://$$API_ROUTE"; \
		echo "$(GREEN)Health Check:$(NC) https://$$API_ROUTE/health"; \
	fi; \
	if [ -n "$$GRPC_ROUTE" ]; then \
		echo "$(GREEN)gRPC API:$(NC) https://$$GRPC_ROUTE"; \
	fi; \
	if [ -n "$$METRICS_ROUTE" ]; then \
		echo "$(GREEN)Metrics:$(NC) https://$$METRICS_ROUTE/metrics"; \
	fi; \
	echo ""

# Test deployment connectivity
openshift-test:
	@echo "$(BLUE)[INFO]$(NC) Testing OpenShift deployment connectivity"
	@API_ROUTE=$$(oc get route semantic-router-api -n $(OPENSHIFT_NAMESPACE) -o jsonpath='{.spec.host}' 2>/dev/null); \
	if [ -n "$$API_ROUTE" ]; then \
		echo "$(BLUE)[INFO]$(NC) Testing API route: https://$$API_ROUTE"; \
		curl -k -f -m 10 "https://$$API_ROUTE/health" 2>/dev/null && \
		echo "$(GREEN)[SUCCESS]$(NC) API route is accessible" || \
		echo "$(YELLOW)[WARN]$(NC) API route test failed (may be expected if models not loaded)"; \
	else \
		echo "$(RED)[ERROR]$(NC) API route not found"; \
	fi

# Port forward services (for testing from local machine)
openshift-port-forward-api:
	@echo "$(BLUE)[INFO]$(NC) Port forwarding Classification API (8080)"
	@echo "$(YELLOW)[INFO]$(NC) Access API at: http://localhost:8080"
	@echo "$(YELLOW)[INFO]$(NC) Press Ctrl+C to stop port forwarding"
	@oc port-forward -n $(OPENSHIFT_NAMESPACE) svc/semantic-router 8080:8080

openshift-port-forward-grpc:
	@echo "$(BLUE)[INFO]$(NC) Port forwarding gRPC API (50051)"
	@echo "$(YELLOW)[INFO]$(NC) Access gRPC API at: localhost:50051"
	@echo "$(YELLOW)[INFO]$(NC) Press Ctrl+C to stop port forwarding"
	@oc port-forward -n $(OPENSHIFT_NAMESPACE) svc/semantic-router 50051:50051

openshift-port-forward-metrics:
	@echo "$(BLUE)[INFO]$(NC) Port forwarding Prometheus metrics (9190)"
	@echo "$(YELLOW)[INFO]$(NC) Access metrics at: http://localhost:9190/metrics"
	@echo "$(YELLOW)[INFO]$(NC) Press Ctrl+C to stop port forwarding"
	@oc port-forward -n $(OPENSHIFT_NAMESPACE) svc/semantic-router-metrics 9190:9190

# Debugging targets
openshift-debug:
	@echo "$(BLUE)[INFO]$(NC) OpenShift debugging information"
	@echo ""
	@echo "$(BLUE)=== Recent Events ===$(NC)"
	@oc get events -n $(OPENSHIFT_NAMESPACE) --sort-by='.lastTimestamp' | tail -10 || echo "$(YELLOW)[WARN]$(NC) Cannot get events"
	@echo ""
	@echo "$(BLUE)=== Pod Description ===$(NC)"
	@oc describe pod -l app=semantic-router -n $(OPENSHIFT_NAMESPACE) | tail -20 || echo "$(YELLOW)[WARN]$(NC) Cannot describe pods"

# Show all available OpenShift targets
openshift-help:
	@echo "$(BLUE)OpenShift deployment targets:$(NC)"
	@echo "  openshift-login              - Login to OpenShift cluster"
	@echo "  openshift-logout             - Logout from OpenShift cluster"
	@echo "  openshift-deploy             - Deploy using Kustomize (basic)"
	@echo "  openshift-deploy-auto        - Deploy using automated script"
	@echo "  openshift-deploy-template    - Deploy using OpenShift template"
	@echo "  openshift-undeploy           - Remove deployment (keep namespace)"
	@echo "  openshift-cleanup            - Remove deployment and namespace"
	@echo "  openshift-status             - Show deployment status"
	@echo "  openshift-logs               - Show application logs (follow)"
	@echo "  openshift-logs-previous      - Show previous pod logs"
	@echo "  openshift-routes             - Show route URLs"
	@echo "  openshift-test               - Test deployment connectivity"
	@echo "  openshift-port-forward-api   - Port forward Classification API"
	@echo "  openshift-port-forward-grpc  - Port forward gRPC API"
	@echo "  openshift-port-forward-metrics - Port forward metrics"
	@echo "  openshift-debug              - Show debugging information"
	@echo ""
	@echo "$(BLUE)Configuration variables:$(NC)"
	@echo "  OPENSHIFT_SERVER             - OpenShift API server URL (required)"
	@echo "  OPENSHIFT_USER               - OpenShift username (default: admin)"
	@echo "  OPENSHIFT_PASSWORD           - OpenShift password (required)"
	@echo "  OPENSHIFT_NAMESPACE          - Deployment namespace (default: $(OPENSHIFT_NAMESPACE))"
	@echo "  OPENSHIFT_DEPLOYMENT_METHOD  - Deployment method: kustomize|template (default: $(OPENSHIFT_DEPLOYMENT_METHOD))"
	@echo "  OPENSHIFT_CONTAINER_IMAGE    - Container image (default: $(OPENSHIFT_CONTAINER_IMAGE))"
	@echo "  OPENSHIFT_CONTAINER_TAG      - Container tag (default: $(OPENSHIFT_CONTAINER_TAG))"
	@echo ""
	@echo "$(BLUE)Example usage:$(NC)"
	@echo "  make openshift-deploy-auto OPENSHIFT_SERVER=https://api.cluster.example.com:6443 OPENSHIFT_PASSWORD=mypass"
	@echo "  make openshift-status OPENSHIFT_NAMESPACE=my-namespace"
	@echo "  make openshift-logs"