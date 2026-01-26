/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package mlmodelselection provides the e2e test profile for ML-based model selection.
// This profile demonstrates ML-based model selection using pretrained models downloaded
// from HuggingFace. Supports KNN, KMeans, and SVM algorithms aligned with FusionFactory
// and Avengers-Pro papers.
package mlmodelselection

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"

	// Import testcases package to register all test cases via their init() functions
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

// Profile implements the ml-model-selection e2e test profile
type Profile struct {
	verbose bool
}

// NewProfile creates a new ML Model Selection profile
func NewProfile() *Profile {
	return &Profile{}
}

// Name returns the profile name
func (p *Profile) Name() string {
	return "ml-model-selection"
}

// Description returns the profile description
func (p *Profile) Description() string {
	return `ML-Based Model Selection E2E Profile

This profile demonstrates end-to-end ML-based model selection using:
- KNN (K-Nearest Neighbors) - Quality-weighted voting among similar queries
- KMeans - Cluster-based routing with efficiency optimization (Avengers-Pro)
- SVM (Support Vector Machine) - RBF kernel decision boundaries

Reference Papers:
- FusionFactory (arXiv:2507.10540) - Query-level fusion via LLM routers
- Avengers-Pro (arXiv:2508.12631) - Performance-efficiency optimized routing
`
}

// GetTestCases returns the test cases for this profile
func (p *Profile) GetTestCases() []string {
	return []string{
		"model-selection",
		"domain-classify",
		"chat-completions-request",
	}
}

// GetServiceConfig returns the service configuration for accessing the router via Envoy Gateway
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		LabelSelector: "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router",
		Namespace:     "envoy-gateway-system",
		PortMapping:   "8080:80",
	}
}

// Setup prepares the profile for testing
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose

	p.log("Setting up ML-based model selection profile")
	p.log("")
	p.log("╔════════════════════════════════════════════════════════════════╗")
	p.log("║  ML-BASED MODEL SELECTION E2E DEMO                             ║")
	p.log("║  Aligned with FusionFactory & Avengers-Pro papers              ║")
	p.log("╚════════════════════════════════════════════════════════════════╝")
	p.log("")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Step 1: Train ML models (or verify they exist)
	p.log("Step 1/7: Preparing ML models for model selection")
	if err := p.prepareMLModels(ctx); err != nil {
		return fmt.Errorf("failed to prepare ML models: %w", err)
	}

	// Step 2: Deploy Semantic Router with ML selection
	p.log("Step 2/7: Deploying Semantic Router with ML model selection")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy semantic router: %w", err)
	}

	// Step 3: Deploy Envoy Gateway
	p.log("Step 3/7: Deploying Envoy Gateway")
	if err := p.deployEnvoyGateway(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy envoy gateway: %w", err)
	}

	// Step 4: Deploy Envoy AI Gateway
	p.log("Step 4/7: Deploying Envoy AI Gateway")
	if err := p.deployEnvoyAIGateway(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy envoy ai gateway: %w", err)
	}

	// Step 5: Deploy Gateway API Resources
	p.log("Step 5/7: Deploying Gateway API Resources")
	if err := p.deployGatewayResources(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy gateway resources: %w", err)
	}

	// Step 6: Deploy Mock LLM (to receive routed requests)
	p.log("Step 6/7: Deploying Mock LLM service")
	if err := p.deployMockLLM(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy mock LLM: %w", err)
	}

	// Step 7: Verify deployment
	p.log("Step 7/7: Verifying deployment")
	if err := p.verifyDeployment(ctx, opts); err != nil {
		return fmt.Errorf("failed to verify deployment: %w", err)
	}

	p.log("")
	p.log("✅ ML Model Selection environment setup complete!")
	p.log("")
	p.log("Deployed components:")
	p.log("  • ML Models (KNN, KMeans, SVM) downloaded from HuggingFace and mounted")
	p.log("  • Mock LLM service (receives routed requests)")
	p.log("  • Semantic Router with ML-based model selectors")
	p.log("")

	return nil
}

// Teardown cleans up after testing
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down ML-based model selection profile")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Clean up in reverse order
	p.log("Cleaning up Gateway API resources")
	p.cleanupGatewayResources(ctx, opts)

	p.log("Uninstalling Mock LLM")
	deployer.Uninstall(ctx, "mock-llm", "default")

	p.log("Uninstalling Envoy AI Gateway")
	deployer.Uninstall(ctx, "aieg-crd", "envoy-ai-gateway-system")
	deployer.Uninstall(ctx, "aieg", "envoy-ai-gateway-system")

	p.log("Uninstalling Envoy Gateway")
	deployer.Uninstall(ctx, "eg", "envoy-gateway-system")

	p.log("Uninstalling Semantic Router")
	deployer.Uninstall(ctx, "semantic-router", "vllm-semantic-router-system")

	p.log("ML Model Selection teardown complete")
	return nil
}

func (p *Profile) deployMockLLM(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	// Deploy mock-vllm for testing
	mockOpts := helm.InstallOptions{
		ReleaseName: "mock-llm",
		Chart:       "deploy/helm/mock-vllm",
		Namespace:   "default",
		Set: map[string]string{
			"image.repository": "ghcr.io/vllm-project/semantic-router/mock-vllm",
			"image.tag":        "latest",
			"image.pullPolicy": "Never",
			"service.port":     "8000",
		},
		Wait:    true,
		Timeout: "5m",
	}

	if err := deployer.Install(ctx, mockOpts); err != nil {
		// If mock-vllm chart doesn't exist, create a simple deployment
		p.log("Mock LLM chart not found, creating simple deployment...")
		return p.deploySimpleMockLLM(ctx, opts)
	}

	return deployer.WaitForDeployment(ctx, "default", "mock-llm", 5*time.Minute)
}

func (p *Profile) deploySimpleMockLLM(ctx context.Context, opts *framework.SetupOptions) error {
	// Create a simple mock LLM deployment using kubectl
	manifest := `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mock-llm
  namespace: default
  labels:
    app: mock-llm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mock-llm
  template:
    metadata:
      labels:
        app: mock-llm
    spec:
      containers:
      - name: mock-llm
        image: ghcr.io/vllm-project/semantic-router/mock-vllm:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: "256Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: mock-llm-service
  namespace: default
spec:
  selector:
    app: mock-llm
  ports:
  - port: 8000
    targetPort: 8000
`
	// Apply using kubectl
	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig, "apply", "-f", "-")
	cmd.Stdin = strings.NewReader(manifest)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("kubectl apply failed: %w\nOutput: %s", err, string(output))
	}
	return nil
}

func (p *Profile) deploySemanticRouter(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	chartPath := "deploy/helm/semantic-router"
	valuesFile := "e2e/profiles/ml-model-selection/values.yaml"

	imageRepo := "ghcr.io/vllm-project/semantic-router/extproc"
	imageTag := opts.ImageTag

	installOpts := helm.InstallOptions{
		ReleaseName: "semantic-router",
		Chart:       chartPath,
		Namespace:   "vllm-semantic-router-system",
		ValuesFiles: []string{valuesFile},
		Set: map[string]string{
			"image.repository": imageRepo,
			"image.tag":        imageTag,
			"image.pullPolicy": "Never",
		},
		Wait:    true,
		Timeout: "30m",
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, "vllm-semantic-router-system", "semantic-router", 30*time.Minute)
}

func (p *Profile) verifyDeployment(ctx context.Context, opts *framework.SetupOptions) error {
	p.log("Waiting for all pods to be ready...")

	deployer := helm.NewDeployer(opts.KubeConfig, p.verbose)

	// Wait for semantic-router to be ready
	if err := deployer.WaitForDeployment(ctx, "vllm-semantic-router-system", "semantic-router", 10*time.Minute); err != nil {
		return fmt.Errorf("semantic-router not ready: %w", err)
	}

	// Wait for Envoy Gateway service
	if err := p.verifyGatewayService(ctx, opts); err != nil {
		return fmt.Errorf("envoy gateway not ready: %w", err)
	}

	p.log("✓ All pods are ready")
	return nil
}

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[ml-model-selection] "+format+"\n", args...)
	}
}

// prepareMLModels ensures ML models are downloaded and available for the pod
// Flow: Check local models → Download pretrained models from HuggingFace
func (p *Profile) prepareMLModels(ctx context.Context) error {
	// Source directory where trained models should exist (also used as mount source)
	// Using the project directory ensures Docker Desktop on WSL2 can access it
	sourceDir := "src/semantic-router/pkg/modelselection/data/trained_models"
	trainingDir := "src/training/ml_model_selection"

	// Get absolute path for the source directory (needed for Kind mount)
	absSourceDir, err := filepath.Abs(sourceDir)
	if err != nil {
		return fmt.Errorf("failed to get absolute path: %w", err)
	}
	p.log("ML models directory: %s", absSourceDir)

	modelFiles := []string{"knn_model.json", "kmeans_model.json", "svm_model.json"}

	// Check if models exist in source
	modelsExist := true
	for _, f := range modelFiles {
		if _, err := os.Stat(sourceDir + "/" + f); os.IsNotExist(err) {
			modelsExist = false
			break
		}
	}

	if !modelsExist {
		p.log("ML models not found locally, downloading from HuggingFace...")

		// Ensure huggingface-hub is installed for downloading models
		p.log("Ensuring huggingface-hub is installed...")
		pipInstall := exec.CommandContext(ctx, "pip", "install", "--quiet", "huggingface-hub>=0.20.0")
		pipInstall.Stdout = os.Stdout
		pipInstall.Stderr = os.Stderr
		if err := pipInstall.Run(); err != nil {
			// Try pip3 if pip fails
			pipInstall = exec.CommandContext(ctx, "pip3", "install", "--quiet", "huggingface-hub>=0.20.0")
			pipInstall.Stdout = os.Stdout
			pipInstall.Stderr = os.Stderr
			if err := pipInstall.Run(); err != nil {
				p.log("Warning: could not install huggingface-hub: %v", err)
			}
		}

		// Download pretrained models from HuggingFace
		p.log("Downloading pretrained ML models from HuggingFace...")
		os.MkdirAll(sourceDir, 0755)

		downloadCmd := exec.CommandContext(ctx, "python3", "download_model.py",
			"--output-dir", "../../semantic-router/pkg/modelselection/data/trained_models",
			"--repo-id", "abdallah1008/semantic-router-ml-models",
		)
		downloadCmd.Dir = trainingDir
		downloadCmd.Stdout = os.Stdout
		downloadCmd.Stderr = os.Stderr

		if err := downloadCmd.Run(); err != nil {
			// Try with python if python3 fails
			downloadCmd = exec.CommandContext(ctx, "python", "download_model.py",
				"--output-dir", "../../semantic-router/pkg/modelselection/data/trained_models",
				"--repo-id", "abdallah1008/semantic-router-ml-models",
			)
			downloadCmd.Dir = trainingDir
			downloadCmd.Stdout = os.Stdout
			downloadCmd.Stderr = os.Stderr
			if err := downloadCmd.Run(); err != nil {
				return fmt.Errorf("failed to download ML models from HuggingFace: %w\nPlease ensure models are uploaded to abdallah1008/semantic-router-ml-models", err)
			}
		}
		p.log("✓ ML models downloaded from HuggingFace")
	} else {
		p.log("✓ ML models found in %s", sourceDir)
	}

	// Step 1: Copy models to host directory for Linux CI (where hostPath works)
	// This is the standard approach that works on native Linux
	hostDir := "/tmp/kind-ml-models"
	p.log("Copying models to host directory %s...", hostDir)
	if err := os.MkdirAll(hostDir, 0755); err != nil {
		p.log("  Warning: could not create host directory: %v (may need sudo on some systems)", err)
	} else {
		for _, f := range modelFiles {
			src := sourceDir + "/" + f
			dst := hostDir + "/" + f
			data, err := os.ReadFile(src)
			if err != nil {
				return fmt.Errorf("failed to read %s: %w", src, err)
			}
			if err := os.WriteFile(dst, data, 0644); err != nil {
				p.log("  Warning: could not write %s: %v", dst, err)
			} else {
				p.log("  ✓ Copied %s to host", f)
			}
		}
	}

	// Step 2: Also copy models directly into Kind containers
	// This is required for WSL2/Docker Desktop where hostPath mounts are broken
	// It's harmless on Linux CI (just overwrites the same files)
	p.log("Copying models into Kind node containers...")

	// Get actual Kind node names dynamically
	kindNodes, err := p.getKindNodes(ctx)
	if err != nil {
		return fmt.Errorf("failed to get Kind nodes: %w", err)
	}
	p.log("  Found %d Kind nodes: %v", len(kindNodes), kindNodes)

	for _, node := range kindNodes {
		// First ensure the target directory exists in the container
		mkdirCmd := exec.CommandContext(ctx, "docker", "exec", node, "mkdir", "-p", "/tmp/ml-models")
		if err := mkdirCmd.Run(); err != nil {
			p.log("  Warning: could not create directory in %s: %v", node, err)
			continue
		}

		// Copy each model file by piping content through stdin
		for _, f := range modelFiles {
			src := sourceDir + "/" + f
			dst := "/tmp/ml-models/" + f

			// Read file content
			data, err := os.ReadFile(src)
			if err != nil {
				return fmt.Errorf("failed to read %s: %w", src, err)
			}

			// Pipe content to container using docker exec with stdin
			cpCmd := exec.CommandContext(ctx, "docker", "exec", "-i", node, "tee", dst)
			cpCmd.Stdin = strings.NewReader(string(data))
			// Suppress stdout (tee echoes input)
			cpCmd.Stdout = nil
			if err := cpCmd.Run(); err != nil {
				p.log("  Warning: failed to copy %s to %s: %v", f, node, err)
			}
		}
		p.log("  ✓ Copied models to %s", node)
	}

	p.log("✓ ML models ready in Kind containers")
	return nil
}

// getKindNodes returns the list of node names for the Kind cluster
func (p *Profile) getKindNodes(ctx context.Context) ([]string, error) {
	cmd := exec.CommandContext(ctx, "kind", "get", "nodes", "--name", "semantic-router-e2e")
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("kind get nodes failed: %w", err)
	}

	var nodes []string
	for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		if line != "" {
			nodes = append(nodes, line)
		}
	}

	if len(nodes) == 0 {
		return nil, fmt.Errorf("no Kind nodes found")
	}

	return nodes, nil
}

// Note: We use mock-vllm for E2E testing, so no real LLM models are needed.
// The 4 LLM model names (llama-3.2-1b, etc.) are just identifiers for routing.
// mock-vllm simulates the LLM endpoints without actual model inference.

// Gateway deployment functions

func (p *Profile) deployEnvoyGateway(ctx context.Context, deployer *helm.Deployer, _ *framework.SetupOptions) error {
	installOpts := helm.InstallOptions{
		ReleaseName: "eg",
		Chart:       "oci://docker.io/envoyproxy/gateway-helm",
		Namespace:   "envoy-gateway-system",
		Version:     "v1.6.0",
		ValuesFiles: []string{"https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/manifests/envoy-gateway-values.yaml"},
		Wait:        true,
		Timeout:     "10m",
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, "envoy-gateway-system", "envoy-gateway", 10*time.Minute)
}

func (p *Profile) deployEnvoyAIGateway(ctx context.Context, deployer *helm.Deployer, _ *framework.SetupOptions) error {
	// Install AI Gateway CRDs
	crdOpts := helm.InstallOptions{
		ReleaseName: "aieg-crd",
		Chart:       "oci://docker.io/envoyproxy/ai-gateway-crds-helm",
		Namespace:   "envoy-ai-gateway-system",
		Version:     "v0.4.0",
		Wait:        true,
		Timeout:     "10m",
	}

	if err := deployer.Install(ctx, crdOpts); err != nil {
		return err
	}

	// Install AI Gateway
	installOpts := helm.InstallOptions{
		ReleaseName: "aieg",
		Chart:       "oci://docker.io/envoyproxy/ai-gateway-helm",
		Namespace:   "envoy-ai-gateway-system",
		Version:     "v0.4.0",
		Wait:        true,
		Timeout:     "10m",
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, "envoy-ai-gateway-system", "ai-gateway-controller", 10*time.Minute)
}

func (p *Profile) deployGatewayResources(ctx context.Context, opts *framework.SetupOptions) error {
	// Use profile-specific gateway resources that match our model names
	// These are configured to route x-selected-model header values to mock-llm backend

	// Apply backend resources (AIServiceBackend + Backend for mock-llm)
	if err := p.kubectlApply(ctx, opts.KubeConfig, "e2e/profiles/ml-model-selection/gateway-resources/backend.yaml"); err != nil {
		return fmt.Errorf("failed to apply backend resources: %w", err)
	}

	// Apply gateway API resources (GatewayClass, Gateway, AIGatewayRoute, EnvoyPatchPolicy)
	if err := p.kubectlApply(ctx, opts.KubeConfig, "e2e/profiles/ml-model-selection/gateway-resources/gwapi-resources.yaml"); err != nil {
		return fmt.Errorf("failed to apply gateway API resources: %w", err)
	}

	return nil
}

func (p *Profile) kubectlApply(ctx context.Context, kubeconfig, filepath string) error {
	cmd := exec.CommandContext(ctx, "kubectl", "apply", "-f", filepath)
	if kubeconfig != "" {
		cmd.Env = append(os.Environ(), fmt.Sprintf("KUBECONFIG=%s", kubeconfig))
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (p *Profile) cleanupGatewayResources(ctx context.Context, opts *framework.TeardownOptions) {
	// Delete profile-specific gateway resources (ignore errors during cleanup)
	p.kubectlDelete(ctx, opts.KubeConfig, "e2e/profiles/ml-model-selection/gateway-resources/gwapi-resources.yaml")
	p.kubectlDelete(ctx, opts.KubeConfig, "e2e/profiles/ml-model-selection/gateway-resources/backend.yaml")
}

func (p *Profile) kubectlDelete(ctx context.Context, kubeconfig, filepath string) {
	cmd := exec.CommandContext(ctx, "kubectl", "delete", "-f", filepath, "--ignore-not-found")
	if kubeconfig != "" {
		cmd.Env = append(os.Environ(), fmt.Sprintf("KUBECONFIG=%s", kubeconfig))
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	_ = cmd.Run() // Ignore errors during cleanup
}

func (p *Profile) verifyGatewayService(ctx context.Context, opts *framework.SetupOptions) error {
	// Create Kubernetes client
	config, err := clientcmd.BuildConfigFromFlags("", opts.KubeConfig)
	if err != nil {
		return fmt.Errorf("failed to build kubeconfig: %w", err)
	}

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return fmt.Errorf("failed to create kube client: %w", err)
	}

	// Wait for Envoy Gateway service to be ready with retry
	retryTimeout := 10 * time.Minute
	retryInterval := 5 * time.Second
	startTime := time.Now()

	p.log("Waiting for Envoy Gateway service to be ready...")

	// Label selector for the semantic-router gateway service
	labelSelector := "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router"

	var serviceName string
	for {
		svc, err := helpers.GetServiceByLabelInNamespace(ctx, client, "envoy-gateway-system", labelSelector, p.verbose)
		if err == nil && svc != "" {
			serviceName = svc
			p.log("✓ Envoy Gateway service found: %s", serviceName)
			break
		}

		if time.Since(startTime) > retryTimeout {
			return fmt.Errorf("timeout waiting for Envoy Gateway service (selector: %s)", labelSelector)
		}

		p.log("  Waiting for gateway service... (elapsed: %v)", time.Since(startTime).Round(time.Second))
		time.Sleep(retryInterval)
	}

	// Now wait for the Envoy pods behind the service to be running
	p.log("Waiting for Envoy Gateway pods to be ready...")
	for {
		err := helpers.VerifyServicePodsRunning(ctx, client, "envoy-gateway-system", serviceName, p.verbose)
		if err == nil {
			p.log("✓ Envoy Gateway pods are running")
			return nil
		}

		if time.Since(startTime) > retryTimeout {
			return fmt.Errorf("timeout waiting for Envoy Gateway pods to be ready: %w", err)
		}

		p.log("  Waiting for gateway pods... (elapsed: %v)", time.Since(startTime).Round(time.Second))
		time.Sleep(retryInterval)
	}
}
