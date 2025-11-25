package aibrix

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"

	// Import testcases package to register all test cases via their init() functions
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	// Version Configuration
	// AIBrix version - can be overridden via AIBRIX_VERSION environment variable
	defaultAIBrixVersion = "v0.4.1"

	// Kubernetes Namespaces - used frequently throughout
	namespaceSemanticRouter = "vllm-semantic-router-system"
	namespaceEnvoyGateway   = "envoy-gateway-system"
	namespaceAIBrix         = "aibrix-system"

	// Deployment Names - used in multiple verification steps
	deploymentSemanticRouter          = "semantic-router"
	deploymentEnvoyGateway            = "envoy-gateway"
	deploymentAIBrixGatewayPlugins    = "aibrix-gateway-plugins"
	deploymentAIBrixMetadataService   = "aibrix-metadata-service"
	deploymentAIBrixControllerManager = "aibrix-controller-manager"
	deploymentDemoLLM                 = "vllm-llama3-8b-instruct"

	// Label Selectors - complex values
	labelSelectorAIBrixGateway = "gateway.envoyproxy.io/owning-gateway-namespace=aibrix-system,gateway.envoyproxy.io/owning-gateway-name=aibrix-eg"

	// Timeouts - configuration values for tuning
	timeoutSemanticRouterDeploy = 20 * time.Minute
	timeoutComponentDeploy      = 2 * time.Minute // For Envoy Gateway, AIBrix plugins/metadata, Demo LLM
	timeoutWebhookDeploy        = 5 * time.Minute // For webhook-enabled components (controller-manager)
	timeoutEnvoyServiceReady    = 10 * time.Minute
	timeoutStabilization        = 60 * time.Second // Increased for CI environments
	retryInterval               = 5 * time.Second
)

// Profile implements the AIBrix test profile
type Profile struct {
	verbose       bool
	aibrixVersion string
}

// NewProfile creates a new AIBrix profile
func NewProfile() *Profile {
	// Allow version override via environment variable
	version := os.Getenv("AIBRIX_VERSION")
	if version == "" {
		version = defaultAIBrixVersion
	}
	return &Profile{
		aibrixVersion: version,
	}
}

// Name returns the profile name
func (p *Profile) Name() string {
	return "aibrix"
}

// Description returns the profile description
func (p *Profile) Description() string {
	return "Tests Semantic Router with vLLM AIBrix integration"
}

// Setup deploys all required components for AIBrix testing
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up AIBrix test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Track what we've deployed for cleanup on error
	var (
		semanticRouterDeployed   bool
		aibrixDepsDeployed       bool
		aibrixCoreDeployed       bool
		gatewayResourcesDeployed bool
	)

	// Ensure cleanup on error
	defer func() {
		if r := recover(); r != nil {
			p.log("Panic during setup, cleaning up...")
			p.cleanupPartialDeployment(ctx, opts, semanticRouterDeployed, aibrixDepsDeployed, aibrixCoreDeployed, gatewayResourcesDeployed)
			panic(r) // Re-panic after cleanup
		}
	}()

	// Step 1: Deploy Semantic Router
	p.log("Step 1/5: Deploying Semantic Router")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy semantic router: %w", err)
	}
	semanticRouterDeployed = true

	// Step 2: Deploy AIBrix Dependencies
	p.log("Step 2/5: Deploying AIBrix Dependencies")
	if err := p.deployAIBrixDependencies(ctx, opts); err != nil {
		p.cleanupPartialDeployment(ctx, opts, semanticRouterDeployed, false, false, false)
		return fmt.Errorf("failed to deploy AIBrix dependencies: %w", err)
	}
	aibrixDepsDeployed = true

	// Step 3: Deploy AIBrix Core
	p.log("Step 3/5: Deploying AIBrix Core")
	if err := p.deployAIBrixCore(ctx, opts); err != nil {
		p.cleanupPartialDeployment(ctx, opts, semanticRouterDeployed, aibrixDepsDeployed, false, false)
		return fmt.Errorf("failed to deploy AIBrix core: %w", err)
	}
	aibrixCoreDeployed = true

	// Step 4: Deploy Demo LLM and Gateway API Resources
	p.log("Step 4/5: Deploying Demo LLM and Gateway API Resources")
	if err := p.deployGatewayResources(ctx, opts); err != nil {
		p.cleanupPartialDeployment(ctx, opts, semanticRouterDeployed, aibrixDepsDeployed, aibrixCoreDeployed, false)
		return fmt.Errorf("failed to deploy gateway resources: %w", err)
	}
	gatewayResourcesDeployed = true

	// Step 5: Verify all components are ready
	p.log("Step 5/5: Verifying all components are ready")
	if err := p.verifyEnvironment(ctx, opts); err != nil {
		p.log("ERROR: Environment verification failed: %v", err)
		p.cleanupPartialDeployment(ctx, opts, semanticRouterDeployed, aibrixDepsDeployed, aibrixCoreDeployed, gatewayResourcesDeployed)
		return fmt.Errorf("failed to verify environment: %w", err)
	}

	p.log("AIBrix test environment setup complete")
	return nil
}

// Teardown cleans up all deployed resources
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down AIBrix test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Clean up in reverse order
	p.log("Cleaning up Gateway API resources")
	p.cleanupGatewayResources(ctx, opts)

	p.log("Cleaning up AIBrix components")
	p.cleanupAIBrix(ctx, opts)

	p.log("Uninstalling Semantic Router")
	deployer.Uninstall(ctx, deploymentSemanticRouter, namespaceSemanticRouter)

	p.log("AIBrix test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile
func (p *Profile) GetTestCases() []string {
	return []string{
		// Basic functionality tests
		"chat-completions-request",
		"chat-completions-stress-request",

		// Classification and routing tests
		"domain-classify",

		// Feature tests
		"semantic-cache",
		"pii-detection",
		"jailbreak-detection",

		// Signal-Decision engine tests (new architecture)
		"decision-priority-selection", // Priority-based routing
		"plugin-chain-execution",      // Plugin ordering and blocking
		"rule-condition-logic",        // AND/OR operators
		"decision-fallback-behavior",  // Fallback to default
		"plugin-config-variations",    // Plugin configuration testing

		// Load tests
		"chat-completions-progressive-stress",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		LabelSelector: labelSelectorAIBrixGateway,
		Namespace:     namespaceEnvoyGateway,
		PortMapping:   "8080:80",
	}
}

func (p *Profile) deploySemanticRouter(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	// Use local Helm chart instead of remote OCI registry
	installOpts := helm.InstallOptions{
		ReleaseName: deploymentSemanticRouter,
		Chart:       "deploy/helm/semantic-router",
		Namespace:   namespaceSemanticRouter,
		ValuesFiles: []string{"deploy/kubernetes/aibrix/semantic-router-values/values.yaml"},
		Set: map[string]string{
			"image.repository": "ghcr.io/vllm-project/semantic-router/extproc",
			"image.tag":        opts.ImageTag,
			"image.pullPolicy": "Never", // Use local image, don't pull from registry
		},
		Wait:    true,
		Timeout: "20m", // Increased timeout for model downloads
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, namespaceSemanticRouter, deploymentSemanticRouter, timeoutSemanticRouterDeploy)
}

func (p *Profile) deployAIBrixDependencies(ctx context.Context, opts *framework.SetupOptions) error {
	// Apply AIBrix dependency components from GitHub release
	dependencyURL := fmt.Sprintf("https://github.com/vllm-project/aibrix/releases/download/%s/aibrix-dependency-%s.yaml",
		p.aibrixVersion, p.aibrixVersion)

	p.log("Deploying AIBrix dependencies (version: %s)", p.aibrixVersion)
	if err := p.kubectlApply(ctx, opts.KubeConfig, dependencyURL); err != nil {
		return fmt.Errorf("failed to apply AIBrix dependencies: %w", err)
	}

	// Wait for Envoy Gateway to be ready
	return p.waitForDeployment(ctx, opts, namespaceEnvoyGateway, deploymentEnvoyGateway, timeoutComponentDeploy)
}

func (p *Profile) deployAIBrixCore(ctx context.Context, opts *framework.SetupOptions) error {
	// Apply AIBrix core components from GitHub release
	coreURL := fmt.Sprintf("https://github.com/vllm-project/aibrix/releases/download/%s/aibrix-core-%s.yaml",
		p.aibrixVersion, p.aibrixVersion)

	p.log("Deploying AIBrix core (version: %s)", p.aibrixVersion)
	if err := p.kubectlApply(ctx, opts.KubeConfig, coreURL); err != nil {
		return fmt.Errorf("failed to apply AIBrix core: %w", err)
	}

	// Patch aibrix-gateway-plugins to reduce resource requests for CI environments
	// The default requests (2 CPU, 8Gi memory) are too high for GitHub Actions runners
	p.log("Patching aibrix-gateway-plugins resource requests for CI compatibility...")
	patchCmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"patch", "deployment", deploymentAIBrixGatewayPlugins,
		"-n", namespaceAIBrix,
		"--type", "json",
		"-p", `[
			{"op": "replace", "path": "/spec/template/spec/containers/0/resources/requests/cpu", "value": "500m"},
			{"op": "replace", "path": "/spec/template/spec/containers/0/resources/requests/memory", "value": "1Gi"},
			{"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/cpu", "value": "1"},
			{"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/memory", "value": "2Gi"}
		]`)
	if p.verbose {
		patchCmd.Stdout = os.Stdout
		patchCmd.Stderr = os.Stderr
	}
	if err := patchCmd.Run(); err != nil {
		p.log("Warning: Failed to patch resource requests (proceeding anyway): %v", err)
	}

	// Wait for AIBrix core components to be ready
	deployments := []struct {
		namespace string
		name      string
		timeout   time.Duration
	}{
		{namespaceAIBrix, deploymentAIBrixGatewayPlugins, timeoutComponentDeploy},
		{namespaceAIBrix, deploymentAIBrixMetadataService, timeoutComponentDeploy},
		{namespaceAIBrix, deploymentAIBrixControllerManager, timeoutWebhookDeploy}, // Longer timeout for webhook setup
	}

	for _, dep := range deployments {
		p.log("Waiting for %s/%s to be ready (timeout: %v)...", dep.namespace, dep.name, dep.timeout)
		if err := p.waitForDeployment(ctx, opts, dep.namespace, dep.name, dep.timeout); err != nil {
			return fmt.Errorf("deployment %s/%s not ready: %w", dep.namespace, dep.name, err)
		}
	}

	return nil
}

func (p *Profile) deployGatewayResources(ctx context.Context, opts *framework.SetupOptions) error {
	// Apply base model (Demo LLM)
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/aibrix/aigw-resources/base-model.yaml"); err != nil {
		return fmt.Errorf("failed to apply base model: %w", err)
	}

	// Wait for Demo LLM deployment
	if err := p.waitForDeployment(ctx, opts, "default", deploymentDemoLLM, timeoutComponentDeploy); err != nil {
		return fmt.Errorf("demo LLM deployment not ready: %w", err)
	}

	// Apply gateway API resources
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/aibrix/aigw-resources/gwapi-resources.yaml"); err != nil {
		return fmt.Errorf("failed to apply gateway API resources: %w", err)
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

	// Give deployments extra time to stabilize after initial readiness
	p.log("Waiting for deployments to stabilize...")
	time.Sleep(timeoutStabilization)

	// Wait for Envoy Gateway service to be ready with retry
	startTime := time.Now()

	p.log("Waiting for Envoy Gateway service to be ready...")

	var envoyService string
	for {
		// Try to get Envoy service name
		envoyService, err = helpers.GetEnvoyServiceName(ctx, client, labelSelectorAIBrixGateway, p.verbose)
		if err == nil {
			// Verify that the service has exactly 1 pod running with all containers ready
			podErr := helpers.VerifyServicePodsRunning(ctx, client, namespaceEnvoyGateway, envoyService, p.verbose)
			if podErr == nil {
				p.log("Envoy Gateway service is ready: %s", envoyService)
				break
			}
			if p.verbose {
				p.log("Envoy service found but pods not ready: %v", podErr)
			}
			err = fmt.Errorf("service pods not ready: %w", podErr)
		}

		if time.Since(startTime) >= timeoutEnvoyServiceReady {
			return fmt.Errorf("failed to get Envoy service with running pods after %v: %w", timeoutEnvoyServiceReady, err)
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

	// Check semantic-router deployment
	p.log("Checking semantic-router deployment...")
	if err := helpers.CheckDeployment(ctx, client, namespaceSemanticRouter, deploymentSemanticRouter, p.verbose); err != nil {
		return fmt.Errorf("semantic-router deployment not healthy: %w", err)
	}

	// Check AIBrix deployments
	aibrixDeployments := []struct {
		namespace string
		name      string
	}{
		{namespaceAIBrix, deploymentAIBrixGatewayPlugins},
		{namespaceAIBrix, deploymentAIBrixMetadataService},
		{namespaceAIBrix, deploymentAIBrixControllerManager},
	}

	for _, dep := range aibrixDeployments {
		p.log("Checking %s deployment...", dep.name)
		if err := helpers.CheckDeployment(ctx, client, dep.namespace, dep.name, p.verbose); err != nil {
			return fmt.Errorf("%s deployment not healthy: %w", dep.name, err)
		}
	}

	// Check envoy-gateway deployment
	p.log("Checking envoy-gateway deployment...")
	if err := helpers.CheckDeployment(ctx, client, namespaceEnvoyGateway, deploymentEnvoyGateway, p.verbose); err != nil {
		return fmt.Errorf("envoy-gateway deployment not healthy: %w", err)
	}

	// Check demo LLM deployment
	p.log("Checking demo LLM deployment...")
	if err := helpers.CheckDeployment(ctx, client, "default", deploymentDemoLLM, p.verbose); err != nil {
		return fmt.Errorf("demo LLM deployment not healthy: %w", err)
	}

	p.log("All deployments are healthy")

	return nil
}

func (p *Profile) cleanupGatewayResources(ctx context.Context, opts *framework.TeardownOptions) error {
	// Delete in reverse order
	p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/aibrix/aigw-resources/gwapi-resources.yaml")
	p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/aibrix/aigw-resources/base-model.yaml")
	return nil
}

func (p *Profile) cleanupPartialDeployment(ctx context.Context, opts *framework.SetupOptions, semanticRouter, aibrixDeps, aibrixCore, gatewayResources bool) {
	p.log("Cleaning up partial deployment (semanticRouter=%v, aibrixDeps=%v, aibrixCore=%v, gatewayResources=%v)",
		semanticRouter, aibrixDeps, aibrixCore, gatewayResources)

	// Create TeardownOptions from SetupOptions
	teardownOpts := &framework.TeardownOptions{
		KubeClient:  opts.KubeClient,
		KubeConfig:  opts.KubeConfig,
		ClusterName: opts.ClusterName,
		Verbose:     opts.Verbose,
	}

	// Clean up in reverse order
	if gatewayResources {
		p.log("Cleaning up Gateway API resources...")
		p.cleanupGatewayResources(ctx, teardownOpts)
	}

	if aibrixCore || aibrixDeps {
		p.log("Cleaning up AIBrix components...")
		p.cleanupAIBrix(ctx, teardownOpts)
	}

	if semanticRouter {
		p.log("Uninstalling Semantic Router...")
		deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
		deployer.Uninstall(ctx, deploymentSemanticRouter, namespaceSemanticRouter)
	}

	p.log("Partial deployment cleanup complete")
}

func (p *Profile) cleanupAIBrix(ctx context.Context, opts *framework.TeardownOptions) error {
	// Delete AIBrix core and dependencies
	coreURL := fmt.Sprintf("https://github.com/vllm-project/aibrix/releases/download/%s/aibrix-core-%s.yaml",
		p.aibrixVersion, p.aibrixVersion)
	dependencyURL := fmt.Sprintf("https://github.com/vllm-project/aibrix/releases/download/%s/aibrix-dependency-%s.yaml",
		p.aibrixVersion, p.aibrixVersion)

	p.kubectlDelete(ctx, opts.KubeConfig, coreURL)
	p.kubectlDelete(ctx, opts.KubeConfig, dependencyURL)

	return nil
}

func (p *Profile) waitForDeployment(ctx context.Context, opts *framework.SetupOptions, namespace, name string, timeout time.Duration) error {
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	return deployer.WaitForDeployment(ctx, namespace, name, timeout)
}

func (p *Profile) kubectlApply(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "apply", "--server-side", "-f", manifest)
}

func (p *Profile) kubectlDelete(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "delete", "--ignore-not-found", "-f", manifest)
}

func (p *Profile) runKubectl(ctx context.Context, kubeConfig string, args ...string) error {
	args = append([]string{"--kubeconfig", kubeConfig}, args...)
	cmd := exec.CommandContext(ctx, "kubectl", args...)
	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	return cmd.Run()
}

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[AIBrix] "+format+"\n", args...)
	}
}
