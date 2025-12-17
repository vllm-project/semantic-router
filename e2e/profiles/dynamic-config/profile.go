package dynamicconfig

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	// Import testcases package to register all test cases via their init() functions
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

// Profile implements the Dynamic Config test profile
// This profile tests Kubernetes CRD-based dynamic configuration
type Profile struct {
	verbose bool
}

// NewProfile creates a new Dynamic Config profile
func NewProfile() *Profile {
	return &Profile{}
}

// Name returns the profile name
func (p *Profile) Name() string {
	return "dynamic-config"
}

// Description returns a description of what this profile tests
func (p *Profile) Description() string {
	return "Tests Kubernetes CRD-based dynamic configuration with IntelligentPool and IntelligentRoute"
}

// Setup deploys all required components for Dynamic Config testing
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up Dynamic Config test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Step 1: Deploy Semantic Router with Kubernetes config source
	p.log("Step 1/5: Deploying Semantic Router with Kubernetes config source")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy semantic router: %w", err)
	}

	// Step 2: Deploy Envoy Gateway
	p.log("Step 2/5: Deploying Envoy Gateway")
	if err := p.deployEnvoyGateway(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy envoy gateway: %w", err)
	}

	// Step 3: Deploy Envoy AI Gateway
	p.log("Step 3/5: Deploying Envoy AI Gateway")
	if err := p.deployEnvoyAIGateway(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy envoy ai gateway: %w", err)
	}

	// Step 4: Deploy Demo LLM and Gateway API Resources
	p.log("Step 4/5: Deploying Demo LLM and Gateway API Resources")
	if err := p.deployGatewayResources(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy gateway resources: %w", err)
	}

	// Step 5: Deploy CRDs (IntelligentPool and IntelligentRoute)
	p.log("Step 5/5: Deploying IntelligentPool and IntelligentRoute CRDs")
	if err := p.deployCRDs(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy CRDs: %w", err)
	}

	// Step 6: Verify all components are ready
	p.log("Step 6/6: Verifying all components are ready")
	if err := p.verifyEnvironment(ctx, opts); err != nil {
		return fmt.Errorf("failed to verify environment: %w", err)
	}

	p.log("Dynamic Config test environment setup complete")
	return nil
}

// Teardown cleans up resources created during setup
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down Dynamic Config test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Uninstall in reverse order
	_ = deployer.Uninstall(ctx, "envoy-ai-gateway", "envoy-ai-gateway-system")
	_ = deployer.Uninstall(ctx, "eg", "envoy-gateway-system")
	_ = deployer.Uninstall(ctx, "semantic-router", "vllm-semantic-router-system")

	p.log("Dynamic Config test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile
func (p *Profile) GetTestCases() []string {
	return []string{
		// Basic functionality tests
		"chat-completions-request",
		"chat-completions-stress-request",

		// Feature tests
		"semantic-cache",

		// Signal-Decision engine tests (CRD-specific)
		// These tests validate the CRD-based routing approach:
		"decision-priority-selection", // Priority-based routing between signals
		"plugin-chain-execution",      // Plugin ordering and blocking
		"rule-condition-logic",        // AND/OR operators in signal conditions
		"embedding-signal-routing",    // EmbeddingSignal-based semantic similarity routing

		// Load tests
		"chat-completions-progressive-stress",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		LabelSelector: "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router",
		Namespace:     "envoy-gateway-system",
		PortMapping:   "8080:80",
	}
}

func (p *Profile) deploySemanticRouter(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	// Use local Helm chart with dynamic config values
	chartPath := "deploy/helm/semantic-router"
	valuesFile := "e2e/profiles/dynamic-config/values.yaml"

	// Override image to use locally built image
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
			"image.pullPolicy": "Never", // Use local image, don't pull from registry
		},
		Wait:    true,
		Timeout: "30m",
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, "vllm-semantic-router-system", "semantic-router", 10*time.Minute)
}

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

	installOpts := helm.InstallOptions{
		ReleaseName: "envoy-ai-gateway",
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
	// Apply base model
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml"); err != nil {
		return fmt.Errorf("failed to apply base model: %w", err)
	}

	// Apply gateway API resources
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml"); err != nil {
		return fmt.Errorf("failed to apply gateway API resources: %w", err)
	}

	return nil
}

func (p *Profile) deployCRDs(ctx context.Context, opts *framework.SetupOptions) error {
	// Apply IntelligentPool CRD
	if err := p.kubectlApply(ctx, opts.KubeConfig, "e2e/profiles/dynamic-config/crds/intelligentpool.yaml"); err != nil {
		return fmt.Errorf("failed to apply IntelligentPool CRD: %w", err)
	}

	// Apply IntelligentRoute CRD
	if err := p.kubectlApply(ctx, opts.KubeConfig, "e2e/profiles/dynamic-config/crds/intelligentroute.yaml"); err != nil {
		return fmt.Errorf("failed to apply IntelligentRoute CRD: %w", err)
	}

	// Wait for CRDs to be processed by the controller
	time.Sleep(15 * time.Second)

	// Verify CRDs are visible
	if err := p.verifyCRDsExist(ctx, opts.KubeConfig); err != nil {
		return fmt.Errorf("CRD verification failed: %w", err)
	}

	return nil
}

func (p *Profile) kubectlApply(ctx context.Context, kubeconfig, manifestPath string) error {
	cmd := exec.CommandContext(ctx, "kubectl", "apply", "-f", manifestPath, "--kubeconfig", kubeconfig)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (p *Profile) verifyCRDsExist(ctx context.Context, kubeconfig string) error {
	// Verify IntelligentPool exists
	cmd := exec.CommandContext(ctx, "kubectl", "get", "intelligentpool", "ai-gateway-pool", "-n", "default", "--kubeconfig", kubeconfig)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("IntelligentPool 'ai-gateway-pool' not found: %w", err)
	}

	// Verify IntelligentRoute exists
	cmd = exec.CommandContext(ctx, "kubectl", "get", "intelligentroute", "ai-gateway-route", "-n", "default", "--kubeconfig", kubeconfig)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("IntelligentRoute 'ai-gateway-route' not found: %w", err)
	}

	return nil
}

func (p *Profile) verifyEnvironment(ctx context.Context, opts *framework.SetupOptions) error {
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

	var envoyService string
	for {
		// Try to get Envoy service name
		envoyService, err = helpers.GetEnvoyServiceName(ctx, client, labelSelector, p.verbose)
		if err == nil {
			// Verify that the service has exactly 1 pod running with all containers ready
			podErr := helpers.VerifyServicePodsRunning(ctx, client, "envoy-gateway-system", envoyService, p.verbose)
			if podErr == nil {
				p.log("Envoy Gateway service is ready: %s", envoyService)
				break
			}
			if p.verbose {
				p.log("Envoy service found but pods not ready: %v", podErr)
			}
			err = fmt.Errorf("service pods not ready: %w", podErr)
		}

		if time.Since(startTime) >= retryTimeout {
			return fmt.Errorf("failed to get Envoy service with running pods after %v: %w", retryTimeout, err)
		}

		if p.verbose {
			p.log("Envoy service not ready, retrying in %v... (elapsed: %v)",
				retryInterval, time.Since(startTime).Round(time.Second))
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(retryInterval):
			// Continue retry
		}
	}

	// Check all deployments are healthy
	p.log("Verifying all deployments are healthy...")

	// Wait for semantic-router deployment to become ready
	semanticRouterReady := false
	for i := 0; i < 12; i++ { // 12 * 10s = 120 seconds max wait
		if err := helpers.CheckDeployment(ctx, client, "vllm-semantic-router-system", "semantic-router", p.verbose); err == nil {
			break
		}
		if i < 11 { // Don't sleep on last iteration
			time.Sleep(10 * time.Second)
		}
	}

	if !semanticRouterReady {
		// Final check to get the actual error
		if err := helpers.CheckDeployment(ctx, client, "vllm-semantic-router-system", "semantic-router", p.verbose); err != nil {
			return fmt.Errorf("semantic-router deployment not healthy after 120s: %w", err)
		}
	}

	// Check envoy-gateway deployment
	if err := helpers.CheckDeployment(ctx, client, "envoy-gateway-system", "envoy-gateway", p.verbose); err != nil {
		return fmt.Errorf("envoy-gateway deployment not healthy: %w", err)
	}

	// Check ai-gateway-controller deployment
	if err := helpers.CheckDeployment(ctx, client, "envoy-ai-gateway-system", "ai-gateway-controller", p.verbose); err != nil {
		return fmt.Errorf("ai-gateway-controller deployment not healthy: %w", err)
	}

	p.log("All deployments are healthy")

	return nil
}

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[dynamic-config] "+format+"\n", args...)
	}
}
