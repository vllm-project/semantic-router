# ======== golang.mk ========
# = Everything For Golang   =
# ======== golang.mk ========

##@ Golang

go-lint: ## Run golangci-lint for src/semantic-router
	@$(LOG_TARGET)
	@echo "Running golangci-lint for src/semantic-router..."
	@cd src/semantic-router/ && golangci-lint run ./... --config ../../tools/linter/go/.golangci.yml
	@echo "✅ src/semantic-router go module lint passed"

go-lint-fix: ## Auto-fix lint issues in src/semantic-router (may need manual fix)
	@$(LOG_TARGET)
	@echo "Running golangci-lint fix for src/semantic-router..."
	@cd src/semantic-router/ && golangci-lint run ./... --fix --config ../../tools/linter/go/.golangci.yml
	@echo "✅ src/semantic-router go module lint fix applied"

vet: $(if $(CI),rust-ci,rust) ## Run go vet for all Go modules (build Rust library first)
	@$(LOG_TARGET)
	@cd candle-binding && go vet ./...
	@cd src/semantic-router && go vet ./...

check-go-mod-tidy: ## Check go mod tidy for all Go modules
	@$(LOG_TARGET)
	@echo "Checking go mod tidy for all Go modules..."
	@echo "Checking candle-binding..."
	@cd candle-binding && go mod tidy && \
		(git diff --exit-code go.mod 2>/dev/null || (echo "ERROR: go.mod file is not tidy in candle-binding. Please run 'go mod tidy' in candle-binding directory and commit the changes." && git diff go.mod && exit 1)) && \
		(test ! -f go.sum || git diff --exit-code go.sum 2>/dev/null || (echo "ERROR: go.sum file is not tidy in candle-binding. Please run 'go mod tidy' in candle-binding directory and commit the changes." && git diff go.sum && exit 1))
	@echo "✅ candle-binding go mod tidy check passed"
	@echo "Checking src/semantic-router..."
	@cd src/semantic-router && go mod tidy && \
		if ! git diff --exit-code go.mod go.sum; then \
			echo "ERROR: go.mod or go.sum files are not tidy in src/semantic-router. Please run 'go mod tidy' in src/semantic-router directory and commit the changes."; \
			git diff go.mod go.sum; \
			exit 1; \
		fi
	@echo "✅ src/semantic-router go mod tidy check passed"
	@echo "✅ All go mod tidy checks passed"

install-controller-gen: ## Install controller-gen for code generation
	@echo "Installing controller-gen..."
	@cd src/semantic-router && go install sigs.k8s.io/controller-tools/cmd/controller-gen@latest

generate-crd: install-controller-gen ## Generate CRD manifests using controller-gen
	@echo "Generating CRD manifests..."
	@cd src/semantic-router && controller-gen crd:crdVersions=v1,allowDangerousTypes=true paths=./pkg/apis/vllm.ai/v1alpha1 output:crd:artifacts:config=../../deploy/kubernetes/crds
	@echo "Copying CRDs to Helm chart..."
	@mkdir -p deploy/helm/semantic-router/crds
	@cp deploy/kubernetes/crds/vllm.ai_intelligentpools.yaml deploy/helm/semantic-router/crds/
	@cp deploy/kubernetes/crds/vllm.ai_intelligentroutes.yaml deploy/helm/semantic-router/crds/
	@echo "✅ CRDs generated and copied to Helm chart"

generate-deepcopy: install-controller-gen ## Generate deepcopy methods using controller-gen
	@echo "Generating deepcopy methods..."
	@cd src/semantic-router && controller-gen object:headerFile=./hack/boilerplate.go.txt paths=./pkg/apis/vllm.ai/v1alpha1

generate-api: generate-deepcopy generate-crd ## Generate all API artifacts (deepcopy, CRDs)
	@echo "Generated all API artifacts"
