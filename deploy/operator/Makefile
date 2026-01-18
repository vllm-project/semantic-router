# Image URL to use all building/pushing image targets
IMG ?= ghcr.io/vllm-project/semantic-router-operator:latest
BUNDLE_IMG ?= ghcr.io/vllm-project/semantic-router-operator-bundle:latest

# Platform detection - automatically detects host architecture
# Can be overridden: make docker-build ARCH=arm64
OS ?= $(shell go env GOOS)
ARCH ?= $(shell go env GOARCH)

# Multi-platform build support
# Example: make docker-buildx PLATFORMS=linux/amd64,linux/arm64
PLATFORMS ?= linux/$(ARCH)

# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

# Setting SHELL to bash allows bash commands to be executed by recipes.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

.PHONY: all
all: build

##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "Platform Variables:"
	@echo "  OS=$(OS)"
	@echo "  ARCH=$(ARCH)"
	@echo "  PLATFORMS=$(PLATFORMS)"
	@echo ""
	@echo "Override examples:"
	@echo "  make docker-build ARCH=arm64"
	@echo "  make docker-buildx PLATFORMS=linux/amd64,linux/arm64"

##@ Development

.PHONY: manifests
manifests: controller-gen ## Generate WebhookConfiguration, ClusterRole and CustomResourceDefinition objects.
	$(CONTROLLER_GEN) rbac:roleName=manager-role crd webhook paths="./..." output:crd:artifacts:config=config/crd/bases

.PHONY: generate
generate: controller-gen ## Generate code containing DeepCopy, DeepCopyInto, and DeepCopyObject method implementations.
	$(CONTROLLER_GEN) object:headerFile="hack/boilerplate.go.txt" paths="./..."

.PHONY: fmt
fmt: ## Run go fmt against code.
	go fmt ./...

.PHONY: vet
vet: ## Run go vet against code.
	go vet ./...

.PHONY: test
test: manifests generate fmt vet ## Run tests.
	go test ./... -coverprofile cover.out

##@ Build

.PHONY: build
build: manifests generate fmt vet ## Build manager binary.
	go build -o bin/manager main.go

.PHONY: run
run: manifests generate fmt vet ## Run a controller from your host.
	go run ./main.go

.PHONY: docker-build
docker-build: ## Build docker image with the manager.
	docker build --platform linux/$(ARCH) -t ${IMG} .

.PHONY: docker-push
docker-push: ## Push docker image with the manager.
	docker push ${IMG}

.PHONY: docker-buildx
docker-buildx: ## Build and push multi-platform docker image with the manager.
	docker buildx build --platform $(PLATFORMS) --push -t ${IMG} .

.PHONY: podman-build
podman-build: ## Build podman image with the manager.
	podman build --platform linux/$(ARCH) -t ${IMG} .

.PHONY: podman-push
podman-push: ## Push podman image with the manager.
	podman push ${IMG}

##@ Deployment

ifndef ignore-not-found
  ignore-not-found = false
endif

.PHONY: install
install: manifests kustomize ## Install CRDs into the K8s cluster specified in ~/.kube/config.
	$(KUSTOMIZE) build config/crd | kubectl apply -f -

.PHONY: uninstall
uninstall: manifests kustomize ## Uninstall CRDs from the K8s cluster specified in ~/.kube/config. Call with ignore-not-found=true to ignore resource not found errors during deletion.
	$(KUSTOMIZE) build config/crd | kubectl delete --ignore-not-found=$(ignore-not-found) -f -

.PHONY: deploy
deploy: manifests kustomize ## Deploy controller to the K8s cluster specified in ~/.kube/config.
	cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	$(KUSTOMIZE) build config/default | kubectl apply -f -

.PHONY: undeploy
undeploy: ## Undeploy controller from the K8s cluster specified in ~/.kube/config. Call with ignore-not-found=true to ignore resource not found errors during deletion.
	$(KUSTOMIZE) build config/default | kubectl delete --ignore-not-found=$(ignore-not-found) -f -

##@ Build Dependencies

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

## Tool Binaries
KUSTOMIZE ?= $(LOCALBIN)/kustomize
CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen
ENVTEST ?= $(LOCALBIN)/setup-envtest

## Tool Versions
KUSTOMIZE_VERSION ?= v5.0.0
CONTROLLER_TOOLS_VERSION ?= v0.19.0

KUSTOMIZE_INSTALL_SCRIPT ?= "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh"
.PHONY: kustomize
kustomize: $(KUSTOMIZE) ## Download kustomize locally if necessary.
$(KUSTOMIZE): $(LOCALBIN)
	test -s $(LOCALBIN)/kustomize || { curl -Ss $(KUSTOMIZE_INSTALL_SCRIPT) | bash -s -- $(subst v,,$(KUSTOMIZE_VERSION)) $(LOCALBIN); }

.PHONY: controller-gen
controller-gen: $(CONTROLLER_GEN) ## Download controller-gen locally if necessary.
$(CONTROLLER_GEN): $(LOCALBIN)
	test -s $(LOCALBIN)/controller-gen || GOBIN=$(LOCALBIN) go install sigs.k8s.io/controller-tools/cmd/controller-gen@$(CONTROLLER_TOOLS_VERSION)

.PHONY: envtest
envtest: $(ENVTEST) ## Download envtest-setup locally if necessary.
$(ENVTEST): $(LOCALBIN)
	test -s $(LOCALBIN)/setup-envtest || GOBIN=$(LOCALBIN) go install sigs.k8s.io/controller-runtime/tools/setup-envtest@latest

##@ Bundle

.PHONY: bundle
bundle: manifests kustomize ## Generate bundle manifests and metadata, then validate generated files.
	operator-sdk generate kustomize manifests -q
	cd config/manager && $(KUSTOMIZE) edit set image controller=$(IMG)
	$(KUSTOMIZE) build config/manifests | operator-sdk generate bundle -q --overwrite --version 0.1.0
	operator-sdk bundle validate ./bundle

.PHONY: bundle-build
bundle-build: ## Build the bundle image.
	docker build --platform linux/$(ARCH) -f bundle/Dockerfile -t $(BUNDLE_IMG) .

.PHONY: bundle-push
bundle-push: ## Push the bundle image.
	docker push $(BUNDLE_IMG)

.PHONY: bundle-podman-build
bundle-podman-build: ## Build the bundle image using podman.
	podman build --platform linux/$(ARCH) -f bundle/Dockerfile -t $(BUNDLE_IMG) .

.PHONY: bundle-podman-push
bundle-podman-push: ## Push the bundle image using podman.
	podman push $(BUNDLE_IMG)

.PHONY: opm
OPM = ./bin/opm
opm: ## Download opm locally if necessary.
ifeq (,$(wildcard $(OPM)))
ifeq (,$(shell which opm 2>/dev/null))
	@{ \
	set -e ;\
	mkdir -p $(dir $(OPM)) ;\
	curl -sSLo $(OPM) https://github.com/operator-framework/operator-registry/releases/download/v1.23.0/$(OS)-$(ARCH)-opm ;\
	chmod +x $(OPM) ;\
	}
else
OPM = $(shell which opm)
endif
endif

# A comma-separated list of bundle images (e.g. make catalog-build BUNDLE_IMGS=example.com/operator-bundle:v0.1.0,example.com/operator-bundle:v0.2.0).
# These images MUST exist in a registry and be pull-able.
BUNDLE_IMGS ?= $(BUNDLE_IMG)

# The image tag given to the resulting catalog image (e.g. make catalog-build CATALOG_IMG=example.com/operator-catalog:v0.2.0).
CATALOG_IMG ?= ghcr.io/vllm-project/semantic-router-operator-catalog:latest

# Set CATALOG_BASE_IMG to an existing catalog image tag to add $BUNDLE_IMGS to that image.
ifneq ($(origin CATALOG_BASE_IMG), undefined)
FROM_INDEX_OPT := --from-index $(CATALOG_BASE_IMG)
endif

# Build a catalog image by adding bundle images to an empty catalog using the operator package manager tool, 'opm'.
# This recipe invokes 'opm' in 'semver' bundle add mode. For more information on add modes, see:
# https://github.com/operator-framework/community-operators/blob/7f1438c/docs/packaging-operator.md#updating-your-existing-operator
.PHONY: catalog-build
catalog-build: opm ## Build a catalog image.
	$(OPM) index add --container-tool docker --mode semver --tag $(CATALOG_IMG) --bundles $(BUNDLE_IMGS) $(FROM_INDEX_OPT)

# Push the catalog image.
.PHONY: catalog-push
catalog-push: ## Push a catalog image.
	docker push $(CATALOG_IMG)

##@ OpenShift Specific

.PHONY: openshift-deploy
openshift-deploy: ## Deploy operator on OpenShift using OLM
	@echo "Creating namespace..."
	oc create namespace semantic-router-operator-system --dry-run=client -o yaml | oc apply -f -
	@echo "Creating operator group..."
	@cat <<EOF | oc apply -f - \
	apiVersion: operators.coreos.com/v1 \
	kind: OperatorGroup \
	metadata: \
	  name: semantic-router-operator-group \
	  namespace: semantic-router-operator-system \
	spec: \
	  targetNamespaces: \
	  - semantic-router-operator-system \
	EOF
	@echo "Creating subscription..."
	@cat <<EOF | oc apply -f - \
	apiVersion: operators.coreos.com/v1alpha1 \
	kind: Subscription \
	metadata: \
	  name: semantic-router-operator \
	  namespace: semantic-router-operator-system \
	spec: \
	  channel: stable \
	  name: semantic-router-operator \
	  source: semantic-router-catalog \
	  sourceNamespace: openshift-marketplace \
	EOF

.PHONY: openshift-undeploy
openshift-undeploy: ## Remove operator from OpenShift
	oc delete subscription semantic-router-operator -n semantic-router-operator-system --ignore-not-found=true
	oc delete csv -n semantic-router-operator-system -l operators.coreos.com/semantic-router-operator.semantic-router-operator-system --ignore-not-found=true
	oc delete namespace semantic-router-operator-system --ignore-not-found=true

##@ Quick Development

.PHONY: quick-build
quick-build: fmt vet ## Quick build without code generation
	go build -o bin/manager main.go

.PHONY: quick-test
quick-test: fmt vet ## Quick test without code generation
	go test ./... -v

.PHONY: all-images
all-images: docker-build bundle-build ## Build all images (operator and bundle)
	@echo "Built operator image: $(IMG)"
	@echo "Built bundle image: $(BUNDLE_IMG)"

.PHONY: push-all-images
push-all-images: docker-push bundle-push ## Push all images (operator and bundle)
	@echo "Pushed operator image: $(IMG)"
	@echo "Pushed bundle image: $(BUNDLE_IMG)"
