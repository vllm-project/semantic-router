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

# Help target
help-kube: ## Show Kubernetes makefile help
	@echo "$(BLUE)Configuration variables:$(NC)"
	@echo "  KIND_CLUSTER_NAME  - Kind cluster name (default: $(KIND_CLUSTER_NAME))"
	@echo "  KIND_CONFIG_FILE   - Kind config file (default: $(KIND_CONFIG_FILE))"
	@echo "  KUBE_NAMESPACE     - Kubernetes namespace (default: $(KUBE_NAMESPACE))"
	@echo "  DOCKER_IMAGE       - Docker image to load (default: $(DOCKER_IMAGE))"
