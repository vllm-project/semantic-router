# Kubernetes deployment targets for semantic-router
# This makefile provides commands for managing kind clusters and deployments

# Configuration
KIND_CLUSTER_NAME ?= semantic-router-cluster
KIND_CONFIG_FILE ?= tools/kind/kind-config.yaml
KUBE_NAMESPACE ?= vllm-semantic-router-system
DOCKER_IMAGE ?= ghcr.io/vllm-project/semantic-router/extproc:latest

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: create-cluster delete-cluster cluster-info deploy undeploy load-image test-deployment test-api port-forward-api port-forward-grpc

##@ Kubernetes

# Create kind cluster with optimized configuration
create-cluster: ## Create a kind cluster with optimized configuration
	@echo "$(BLUE)[INFO]$(NC) Creating kind cluster: $(KIND_CLUSTER_NAME)"
	@if kind get clusters | grep -q "^$(KIND_CLUSTER_NAME)$$"; then \
		echo "$(YELLOW)[WARNING]$(NC) Cluster $(KIND_CLUSTER_NAME) already exists"; \
		read -p "Delete and recreate? (y/N): " confirm; \
		if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
			$(MAKE) delete-cluster; \
		else \
			echo "$(BLUE)[INFO]$(NC) Using existing cluster"; \
			exit 0; \
		fi; \
	fi
	@if [ ! -f "$(KIND_CONFIG_FILE)" ]; then \
		echo "$(YELLOW)[INFO]$(NC) Kind config not found: $(KIND_CONFIG_FILE). Generating..."; \
		bash tools/kind/generate-kind-config.sh; \
	fi
	@echo "$(BLUE)[INFO]$(NC) Creating cluster with config: $(KIND_CONFIG_FILE)"
	@mkdir -p /tmp/kind-semantic-router
	@kind create cluster --name $(KIND_CLUSTER_NAME) --config $(KIND_CONFIG_FILE)
	@echo "$(GREEN)[SUCCESS]$(NC) Cluster created successfully"
	@echo "$(BLUE)[INFO]$(NC) Waiting for cluster to be ready..."
	@kubectl wait --for=condition=Ready nodes --all --timeout=300s
	@echo "$(GREEN)[SUCCESS]$(NC) Cluster is ready"

# Delete kind cluster
delete-cluster: ## Delete the kind cluster
	@echo "$(BLUE)[INFO]$(NC) Deleting kind cluster: $(KIND_CLUSTER_NAME)"
	@if kind get clusters | grep -q "^$(KIND_CLUSTER_NAME)$$"; then \
		kind delete cluster --name $(KIND_CLUSTER_NAME); \
		echo "$(GREEN)[SUCCESS]$(NC) Cluster deleted"; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) Cluster $(KIND_CLUSTER_NAME) does not exist"; \
	fi

# Show cluster information
cluster-info: ## Show cluster information and resource usage
	@echo "$(BLUE)[INFO]$(NC) Cluster information:"
	@kubectl cluster-info --context kind-$(KIND_CLUSTER_NAME) || echo "$(RED)[ERROR]$(NC) Cluster not accessible"
	@echo "$(BLUE)[INFO]$(NC) Node information:"
	@kubectl get nodes -o wide || echo "$(RED)[ERROR]$(NC) Cannot get nodes"
	@echo "$(BLUE)[INFO]$(NC) Resource usage:"
	@kubectl describe nodes | grep -A 10 "Allocated resources:" || echo "$(YELLOW)[WARNING]$(NC) Cannot get resource info"

# Deploy semantic-router to the cluster
deploy: ## Deploy semantic-router to the cluster
	@echo "$(BLUE)[INFO]$(NC) Deploying semantic-router to cluster"
	@echo "$(BLUE)[INFO]$(NC) Applying Kubernetes manifests..."
	@kubectl apply -k deploy/kubernetes/
	@echo "$(BLUE)[INFO]$(NC) Waiting for namespace to be ready..."
	@kubectl wait --for=condition=Ready namespace/$(KUBE_NAMESPACE) --timeout=60s || true
	@echo "$(BLUE)[INFO]$(NC) Waiting for deployment to be ready..."
	@kubectl wait --for=condition=Available deployment/semantic-router -n $(KUBE_NAMESPACE) --timeout=600s
	@echo "$(GREEN)[SUCCESS]$(NC) Deployment completed successfully!"
	@echo "$(BLUE)[INFO]$(NC) Deployment status:"
	@kubectl get pods -n $(KUBE_NAMESPACE) -o wide
	@kubectl get services -n $(KUBE_NAMESPACE)

# Remove semantic-router from the cluster
undeploy: ## Remove semantic-router from the cluster
	@echo "$(BLUE)[INFO]$(NC) Removing semantic-router from cluster"
	@kubectl delete -k deploy/kubernetes/ --ignore-not-found=true
	@echo "$(GREEN)[SUCCESS]$(NC) Undeployment completed"

# Load Docker image into kind cluster
load-image: ## Load Docker image into kind cluster
	@echo "$(BLUE)[INFO]$(NC) Loading Docker image into kind cluster"
	@if ! kind get clusters | grep -q "^$(KIND_CLUSTER_NAME)$$"; then \
		echo "$(RED)[ERROR]$(NC) Cluster $(KIND_CLUSTER_NAME) does not exist"; \
		echo "$(BLUE)[INFO]$(NC) Run 'make create-cluster' first"; \
		exit 1; \
	fi
	@echo "$(BLUE)[INFO]$(NC) Loading image: $(DOCKER_IMAGE)"
	@kind load docker-image $(DOCKER_IMAGE) --name $(KIND_CLUSTER_NAME)
	@echo "$(GREEN)[SUCCESS]$(NC) Image loaded successfully"

# Test the deployment
test-deployment: ## Test the deployment
	@echo "$(BLUE)[INFO]$(NC) Testing semantic-router deployment"
	@echo "$(BLUE)[INFO]$(NC) Checking pod status..."
	@kubectl get pods -n $(KUBE_NAMESPACE) -o wide
	@echo "$(BLUE)[INFO]$(NC) Checking services..."
	@kubectl get services -n $(KUBE_NAMESPACE)
	@echo "$(BLUE)[INFO]$(NC) Checking PVC..."
	@kubectl get pvc -n $(KUBE_NAMESPACE)
	@echo "$(BLUE)[INFO]$(NC) Checking pod readiness..."
	@kubectl wait --for=condition=Ready pod -l app=semantic-router -n $(KUBE_NAMESPACE) --timeout=60s
	@echo "$(GREEN)[SUCCESS]$(NC) Deployment test completed"

# Test the Classification API
test-api: ## Test the Classification API
	@echo "$(BLUE)[INFO]$(NC) Testing Classification API"
	@echo "$(BLUE)[INFO]$(NC) Testing health endpoint..."
	@curl -s -f http://localhost:8080/health || (echo "$(RED)[ERROR]$(NC) Health check failed. Is port-forward running?" && exit 1)
	@echo "$(GREEN)[SUCCESS]$(NC) Health check passed"
	@echo "$(BLUE)[INFO]$(NC) Testing intent classification..."
	@curl -s -X POST http://localhost:8080/api/v1/classify/intent \
		-H "Content-Type: application/json" \
		-d '{"text": "What is machine learning?"}' | head -c 200
	@echo ""
	@echo "$(GREEN)[SUCCESS]$(NC) API test completed"

# Port forward Classification API (8080)
port-forward-api: ## Port forward Classification API (8080)
	@echo "$(BLUE)[INFO]$(NC) Port forwarding Classification API (8080)"
	@echo "$(YELLOW)[INFO]$(NC) Access API at: http://localhost:8080"
	@echo "$(YELLOW)[INFO]$(NC) Health check: curl http://localhost:8080/health"
	@echo "$(YELLOW)[INFO]$(NC) Press Ctrl+C to stop port forwarding"
	@kubectl port-forward -n $(KUBE_NAMESPACE) svc/semantic-router 8080:8080

# Port forward gRPC API (50051)
port-forward-grpc: ## Port forward gRPC API (50051)
	@echo "$(BLUE)[INFO]$(NC) Port forwarding gRPC API (50051)"
	@echo "$(YELLOW)[INFO]$(NC) Access gRPC API at: localhost:50051"
	@echo "$(YELLOW)[INFO]$(NC) Press Ctrl+C to stop port forwarding"
	@kubectl port-forward -n $(KUBE_NAMESPACE) svc/semantic-router 50051:50051

# Port forward metrics (9190)
port-forward-metrics: ## Port forward Prometheus metrics (9190)
	@echo "$(BLUE)[INFO]$(NC) Port forwarding Prometheus metrics (9190)"
	@echo "$(YELLOW)[INFO]$(NC) Access metrics at: http://localhost:9190/metrics"
	@echo "$(YELLOW)[INFO]$(NC) Press Ctrl+C to stop port forwarding"
	@kubectl port-forward -n $(KUBE_NAMESPACE) svc/semantic-router-metrics 9190:9190

# Show logs
logs: ## Show semantic-router logs
	@echo "$(BLUE)[INFO]$(NC) Showing semantic-router logs"
	@kubectl logs -n $(KUBE_NAMESPACE) -l app=semantic-router -f

# Show deployment status
status: ## Show deployment status
	@echo "$(BLUE)[INFO]$(NC) Semantic Router deployment status"
	@echo "$(BLUE)[INFO]$(NC) Pods:"
	@kubectl get pods -n $(KUBE_NAMESPACE) -o wide || echo "$(RED)[ERROR]$(NC) Cannot get pods"
	@echo "$(BLUE)[INFO]$(NC) Services:"
	@kubectl get services -n $(KUBE_NAMESPACE) || echo "$(RED)[ERROR]$(NC) Cannot get services"
	@echo "$(BLUE)[INFO]$(NC) PVC:"
	@kubectl get pvc -n $(KUBE_NAMESPACE) || echo "$(RED)[ERROR]$(NC) Cannot get PVC"

# Complete setup: create cluster and deploy
setup: create-cluster deploy ## Complete setup: create cluster and deploy
	@echo "$(GREEN)[SUCCESS]$(NC) Complete setup finished!"
	@echo "$(BLUE)[INFO]$(NC) Next steps:"
	@echo "  - Test deployment: make test-deployment"
	@echo "  - Test API: make test-api"
	@echo "  - Port forward API: make port-forward-api"
	@echo "  - View logs: make logs"

# Complete cleanup: undeploy and delete cluster
cleanup: undeploy delete-cluster ## Complete cleanup: undeploy and delete cluster
	@echo "$(GREEN)[SUCCESS]$(NC) Complete cleanup finished!"

##@ LLM Katan Kubernetes

# LLM Katan configuration
LLM_KATAN_NAMESPACE ?= llm-katan-system
LLM_KATAN_BASE_PATH ?= deploy/kubernetes/llm-katan
LLM_KATAN_OVERLAY ?= gpt35
LLM_KATAN_IMAGE ?= $(DOCKER_REGISTRY)/llm-katan:$(DOCKER_TAG)

.PHONY: kube-deploy-llm-katan kube-deploy-llm-katan-gpt35 kube-deploy-llm-katan-claude \
	kube-undeploy-llm-katan kube-status-llm-katan kube-logs-llm-katan \
	kube-port-forward-llm-katan kube-test-llm-katan kube-load-llm-katan-image \
	kube-deploy-llm-katan-multi help-kube-llm-katan

# Deploy llm-katan with specified overlay
kube-deploy-llm-katan: ## Deploy llm-katan to cluster (OVERLAY=gpt35|claude, default: gpt35)
	@echo "$(BLUE)[INFO]$(NC) Deploying llm-katan with overlay: $(LLM_KATAN_OVERLAY)"
	@if ! kubectl cluster-info &>/dev/null; then \
		echo "$(RED)[ERROR]$(NC) Kubernetes cluster is not accessible"; \
		echo "$(BLUE)[INFO]$(NC) Run 'make create-cluster' first"; \
		exit 1; \
	fi
	@echo "$(BLUE)[INFO]$(NC) Applying Kubernetes manifests..."
	@kubectl apply -k $(LLM_KATAN_BASE_PATH)/overlays/$(LLM_KATAN_OVERLAY)
	@echo "$(BLUE)[INFO]$(NC) Waiting for namespace to be ready..."
	@kubectl wait --for=condition=Ready namespace/$(LLM_KATAN_NAMESPACE) --timeout=60s || true
	@echo "$(BLUE)[INFO]$(NC) Waiting for deployment to be ready..."
	@kubectl wait --for=condition=Available deployment/llm-katan-$(LLM_KATAN_OVERLAY) \
		-n $(LLM_KATAN_NAMESPACE) --timeout=600s || echo "$(YELLOW)[WARNING]$(NC) Deployment not ready yet, check status with: make kube-status-llm-katan"
	@echo "$(GREEN)[SUCCESS]$(NC) LLM Katan deployment completed!"
	@echo "$(BLUE)[INFO]$(NC) Deployment status:"
	@kubectl get pods -n $(LLM_KATAN_NAMESPACE) -l app=llm-katan-$(LLM_KATAN_OVERLAY) -o wide

# Deploy llm-katan with gpt35 overlay
kube-deploy-llm-katan-gpt35: ## Deploy llm-katan with GPT-3.5 overlay
	@$(MAKE) kube-deploy-llm-katan LLM_KATAN_OVERLAY=gpt35
	@echo "$(GREEN)[SUCCESS]$(NC) GPT-3.5 simulation deployed!"
	@echo "$(BLUE)[INFO]$(NC) Test with: make kube-test-llm-katan LLM_KATAN_OVERLAY=gpt35"

# Deploy llm-katan with claude overlay
kube-deploy-llm-katan-claude: ## Deploy llm-katan with Claude overlay
	@$(MAKE) kube-deploy-llm-katan LLM_KATAN_OVERLAY=claude
	@echo "$(GREEN)[SUCCESS]$(NC) Claude simulation deployed!"
	@echo "$(BLUE)[INFO]$(NC) Test with: make kube-test-llm-katan LLM_KATAN_OVERLAY=claude"

# Deploy both overlays for multi-model testing
kube-deploy-llm-katan-multi: ## Deploy both gpt35 and claude overlays
	@echo "$(BLUE)[INFO]$(NC) Deploying multiple llm-katan instances..."
	@$(MAKE) kube-deploy-llm-katan-gpt35
	@echo ""
	@$(MAKE) kube-deploy-llm-katan-claude
	@echo ""
	@echo "$(GREEN)[SUCCESS]$(NC) Multi-model deployment completed!"
	@echo "$(BLUE)[INFO]$(NC) Available models:"
	@kubectl get pods -n $(LLM_KATAN_NAMESPACE) -o wide

# Remove llm-katan from the cluster
kube-undeploy-llm-katan: ## Remove llm-katan from cluster (OVERLAY=gpt35|claude|all, default: gpt35)
	@echo "$(BLUE)[INFO]$(NC) Removing llm-katan overlay: $(LLM_KATAN_OVERLAY)"
	@if [ "$(LLM_KATAN_OVERLAY)" = "all" ]; then \
		echo "$(BLUE)[INFO]$(NC) Removing all llm-katan deployments..."; \
		kubectl delete -k $(LLM_KATAN_BASE_PATH)/overlays/gpt35 --ignore-not-found=true; \
		kubectl delete -k $(LLM_KATAN_BASE_PATH)/overlays/claude --ignore-not-found=true; \
	else \
		kubectl delete -k $(LLM_KATAN_BASE_PATH)/overlays/$(LLM_KATAN_OVERLAY) --ignore-not-found=true; \
	fi
	@echo "$(GREEN)[SUCCESS]$(NC) LLM Katan undeployment completed"

# Show llm-katan deployment status
kube-status-llm-katan: ## Show llm-katan deployment status
	@echo "$(BLUE)[INFO]$(NC) LLM Katan deployment status"
	@echo "$(BLUE)[INFO]$(NC) Namespace: $(LLM_KATAN_NAMESPACE)"
	@echo ""
	@echo "$(BLUE)[INFO]$(NC) Pods:"
	@kubectl get pods -n $(LLM_KATAN_NAMESPACE) -o wide || echo "$(RED)[ERROR]$(NC) Cannot get pods"
	@echo ""
	@echo "$(BLUE)[INFO]$(NC) Services:"
	@kubectl get services -n $(LLM_KATAN_NAMESPACE) || echo "$(RED)[ERROR]$(NC) Cannot get services"
	@echo ""
	@echo "$(BLUE)[INFO]$(NC) PVCs:"
	@kubectl get pvc -n $(LLM_KATAN_NAMESPACE) || echo "$(RED)[ERROR]$(NC) Cannot get PVCs"
	@echo ""
	@echo "$(BLUE)[INFO]$(NC) Deployments:"
	@kubectl get deployments -n $(LLM_KATAN_NAMESPACE) || echo "$(RED)[ERROR]$(NC) Cannot get deployments"

# Show llm-katan logs
kube-logs-llm-katan: ## Show llm-katan logs (OVERLAY=gpt35|claude, default: gpt35)
	@echo "$(BLUE)[INFO]$(NC) Showing llm-katan logs for overlay: $(LLM_KATAN_OVERLAY)"
	@kubectl logs -n $(LLM_KATAN_NAMESPACE) -l app=llm-katan-$(LLM_KATAN_OVERLAY) -f

# Port forward llm-katan API
kube-port-forward-llm-katan: ## Port forward llm-katan API (OVERLAY=gpt35|claude, PORT=8000)
	@$(eval PORT ?= 8000)
	@echo "$(BLUE)[INFO]$(NC) Port forwarding llm-katan API (overlay: $(LLM_KATAN_OVERLAY))"
	@echo "$(YELLOW)[INFO]$(NC) Access API at: http://localhost:$(PORT)"
	@echo "$(YELLOW)[INFO]$(NC) Health check: curl http://localhost:$(PORT)/health"
	@echo "$(YELLOW)[INFO]$(NC) Models: curl http://localhost:$(PORT)/v1/models"
	@echo "$(YELLOW)[INFO]$(NC) Press Ctrl+C to stop port forwarding"
	@kubectl port-forward -n $(LLM_KATAN_NAMESPACE) svc/llm-katan-$(LLM_KATAN_OVERLAY) $(PORT):8000

# Test llm-katan deployment
kube-test-llm-katan: ## Test llm-katan deployment (OVERLAY=gpt35|claude, default: gpt35)
	@echo "$(BLUE)[INFO]$(NC) Testing llm-katan deployment (overlay: $(LLM_KATAN_OVERLAY))"
	@echo "$(BLUE)[INFO]$(NC) Checking pod status..."
	@kubectl get pods -n $(LLM_KATAN_NAMESPACE) -l app=llm-katan-$(LLM_KATAN_OVERLAY) -o wide
	@echo ""
	@echo "$(BLUE)[INFO]$(NC) Checking service..."
	@kubectl get svc -n $(LLM_KATAN_NAMESPACE) llm-katan-$(LLM_KATAN_OVERLAY)
	@echo ""
	@echo "$(BLUE)[INFO]$(NC) Checking pod readiness..."
	@kubectl wait --for=condition=Ready pod -l app=llm-katan-$(LLM_KATAN_OVERLAY) \
		-n $(LLM_KATAN_NAMESPACE) --timeout=60s || echo "$(RED)[ERROR]$(NC) Pod not ready"
	@echo ""
	@echo "$(BLUE)[INFO]$(NC) Testing API endpoint (requires port-forward in another terminal)..."
	@echo "$(YELLOW)[INFO]$(NC) Run in another terminal: make kube-port-forward-llm-katan LLM_KATAN_OVERLAY=$(LLM_KATAN_OVERLAY)"
	@echo "$(YELLOW)[INFO]$(NC) Then test with: curl http://localhost:8000/health"
	@echo "$(GREEN)[SUCCESS]$(NC) Deployment test completed"

# Load llm-katan image into kind cluster
kube-load-llm-katan-image: ## Load llm-katan Docker image into kind cluster
	@echo "$(BLUE)[INFO]$(NC) Loading llm-katan Docker image into kind cluster"
	@if ! kind get clusters | grep -q "^$(KIND_CLUSTER_NAME)$$"; then \
		echo "$(RED)[ERROR]$(NC) Cluster $(KIND_CLUSTER_NAME) does not exist"; \
		echo "$(BLUE)[INFO]$(NC) Run 'make create-cluster' first"; \
		exit 1; \
	fi
	@echo "$(BLUE)[INFO]$(NC) Loading image: $(LLM_KATAN_IMAGE)"
	@kind load docker-image $(LLM_KATAN_IMAGE) --name $(KIND_CLUSTER_NAME)
	@echo "$(GREEN)[SUCCESS]$(NC) LLM Katan image loaded successfully"

# Help target
help-kube: ## Show Kubernetes makefile help
	@echo "$(BLUE)Configuration variables:$(NC)"
	@echo "  KIND_CLUSTER_NAME  - Kind cluster name (default: $(KIND_CLUSTER_NAME))"
	@echo "  KIND_CONFIG_FILE   - Kind config file (default: $(KIND_CONFIG_FILE))"
	@echo "  KUBE_NAMESPACE     - Kubernetes namespace (default: $(KUBE_NAMESPACE))"
	@echo "  DOCKER_IMAGE       - Docker image to load (default: $(DOCKER_IMAGE))"
