# ======== golang.mk ========
# = Everything For Golang   =
# ======== golang.mk ========

# Run go vet for all Go modules
vet:
	@$(LOG_TARGET)
	@cd candle-binding && go vet ./...
	@cd src/semantic-router && go vet ./...

# Check go mod tidy for all Go modules
check-go-mod-tidy:
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

# Controller-gen targets
install-controller-gen:
	@echo "Installing controller-gen..."
	@cd src/semantic-router && go install sigs.k8s.io/controller-tools/cmd/controller-gen@latest

generate-crd: install-controller-gen
	@echo "Generating CRD manifests..."
	@cd src/semantic-router && controller-gen crd:crdVersions=v1,allowDangerousTypes=true paths=./pkg/apis/vllm.ai/v1alpha1 output:crd:artifacts:config=../../deploy/kubernetes/crds

generate-deepcopy: install-controller-gen
	@echo "Generating deepcopy methods..."
	@cd src/semantic-router && controller-gen object:headerFile=./hack/boilerplate.go.txt paths=./pkg/apis/vllm.ai/v1alpha1

generate-api: generate-deepcopy generate-crd
	@echo "Generated all API artifacts"