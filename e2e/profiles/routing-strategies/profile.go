package routingstrategies

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

// Profile implements the Routing Strategies test profile
type Profile struct {
	verbose         bool
	mcpStdioProcess *exec.Cmd
	mcpHTTPProcess  *exec.Cmd
}

// NewProfile creates a new Routing Strategies profile
func NewProfile() *Profile {
	return &Profile{}
}

// Name returns the profile name
func (p *Profile) Name() string {
	return "routing-strategies"
}

// Description returns the profile description
func (p *Profile) Description() string {
	return "Tests different routing strategies including keyword-based routing"
}

// Setup deploys all required components for Routing Strategies testing
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up Routing Strategies test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Step 1: Deploy Semantic Router with keyword routing configuration
	p.log("Step 1/4: Deploying Semantic Router with keyword routing config")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy semantic router: %w", err)
	}

	// Step 2: Deploy Envoy Gateway
	p.log("Step 2/4: Deploying Envoy Gateway")
	if err := p.deployEnvoyGateway(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy envoy gateway: %w", err)
	}

	// Step 3: Deploy Envoy AI Gateway
	p.log("Step 3/4: Deploying Envoy AI Gateway")
	if err := p.deployEnvoyAIGateway(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy envoy ai gateway: %w", err)
	}

	// Step 4: Deploy Demo LLM and Gateway API Resources
	p.log("Step 4/5: Deploying Demo LLM and Gateway API Resources")
	if err := p.deployGatewayResources(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy gateway resources: %w", err)
	}

	// Step 5: Verify all components are ready
	p.log("Step 5/6: Verifying all components are ready")
	if err := p.verifyEnvironment(ctx, opts); err != nil {
		return fmt.Errorf("failed to verify environment: %w", err)
	}

	// Step 6: Start MCP servers for testing (optional - tests will skip if unavailable)
	p.log("Step 6/6: Starting MCP classification servers (optional)")
	if err := p.startMCPServers(ctx); err != nil {
		p.log("Warning: MCP servers not started: %v", err)
		p.log("MCP-related tests will be skipped")
		// Don't fail setup - MCP tests are optional
	}

	p.log("Routing Strategies test environment setup complete")
	return nil
}

// Teardown cleans up all deployed resources
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down Routing Strategies test environment")

	// Stop MCP servers first
	p.log("Stopping MCP servers")
	if p.mcpStdioProcess != nil {
		p.mcpStdioProcess.Process.Kill()
	}
	if p.mcpHTTPProcess != nil {
		p.mcpHTTPProcess.Process.Kill()
	}

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Clean up in reverse order
	p.log("Cleaning up Gateway API resources")
	p.cleanupGatewayResources(ctx, opts)

	p.log("Uninstalling Envoy AI Gateway")
	deployer.Uninstall(ctx, "aieg-crd", "envoy-ai-gateway-system")
	deployer.Uninstall(ctx, "aieg", "envoy-ai-gateway-system")

	p.log("Uninstalling Envoy Gateway")
	deployer.Uninstall(ctx, "eg", "envoy-gateway-system")

	p.log("Uninstalling Semantic Router")
	deployer.Uninstall(ctx, "semantic-router", "vllm-semantic-router-system")

	p.log("Routing Strategies test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile
func (p *Profile) GetTestCases() []string {
	return []string{
		"keyword-routing",
		"entropy-routing",
		"routing-fallback", // Test sequential fallback: Keyword → Embedding → BERT → MCP
		// MCP tests are registered but not run by default
		// To run MCP tests, use: E2E_TESTS="mcp-stdio-classification,mcp-http-classification,..."
		// "mcp-stdio-classification",
		// "mcp-http-classification",
		// "mcp-model-reasoning",
		// "mcp-probability-distribution",
		// "mcp-fallback-behavior",
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
	// Use local Helm chart with keyword routing configuration
	chartPath := "deploy/helm/semantic-router"
	valuesFile := "e2e/profiles/routing-strategies/values.yaml"

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

	return deployer.WaitForDeployment(ctx, "vllm-semantic-router-system", "semantic-router", 30*time.Minute)
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
	// Apply base model
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/routing-strategies/aigw-resources/base-model.yaml"); err != nil {
		return fmt.Errorf("failed to apply base model: %w", err)
	}

	// Apply gateway API resources
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/routing-strategies/aigw-resources/gwapi-resources.yaml"); err != nil {
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

	// Check semantic-router deployment
	if err := helpers.CheckDeployment(ctx, client, "vllm-semantic-router-system", "semantic-router", p.verbose); err != nil {
		return fmt.Errorf("semantic-router deployment not healthy: %w", err)
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

func (p *Profile) cleanupGatewayResources(ctx context.Context, opts *framework.TeardownOptions) error {
	// Delete in reverse order
	p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/routing-strategies/aigw-resources/gwapi-resources.yaml")
	p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/routing-strategies/aigw-resources/base-model.yaml")
	return nil
}

func (p *Profile) kubectlApply(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "apply", "-f", manifest)
}

func (p *Profile) kubectlDelete(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "delete", "-f", manifest)
}

func (p *Profile) runKubectl(ctx context.Context, kubeConfig string, args ...string) error {
	args = append(args, "--kubeconfig", kubeConfig)
	cmd := exec.CommandContext(ctx, "kubectl", args...)
	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	return cmd.Run()
}

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[Routing-Strategies] "+format+"\n", args...)
	}
}

func (p *Profile) startMCPServers(ctx context.Context) error {
	p.log("Starting MCP classification servers")

	// Check if Python 3 is available
	if _, err := exec.LookPath("python3"); err != nil {
		p.log("Warning: python3 not found, skipping MCP server startup")
		p.log("MCP tests will be skipped or may fail")
		return nil
	}

	// Start stdio MCP server (keyword-based classifier)
	p.log("Starting stdio MCP server (keyword-based)")
	p.mcpStdioProcess = exec.CommandContext(ctx,
		"python3",
		"deploy/examples/mcp-classifier-server/server_keyword.py")

	// Capture output for debugging
	if p.verbose {
		p.mcpStdioProcess.Stdout = os.Stdout
		p.mcpStdioProcess.Stderr = os.Stderr
	}

	if err := p.mcpStdioProcess.Start(); err != nil {
		p.log("Warning: failed to start stdio MCP server: %v", err)
		// Continue without stdio server - tests may skip or fail gracefully
	} else {
		p.log("Stdio MCP server started (PID: %d)", p.mcpStdioProcess.Process.Pid)
	}

	// Start HTTP MCP server (embedding-based classifier)
	p.log("Starting HTTP MCP server (embedding-based)")
	p.mcpHTTPProcess = exec.CommandContext(ctx,
		"python3",
		"deploy/examples/mcp-classifier-server/server_embedding.py",
		"--port", "8090")

	// Capture output for debugging
	if p.verbose {
		p.mcpHTTPProcess.Stdout = os.Stdout
		p.mcpHTTPProcess.Stderr = os.Stderr
	}

	if err := p.mcpHTTPProcess.Start(); err != nil {
		p.log("Warning: failed to start HTTP MCP server: %v", err)
		// If stdio server failed too, return error
		if p.mcpStdioProcess == nil {
			return fmt.Errorf("failed to start any MCP servers: %w", err)
		}
		p.log("Continuing with only stdio MCP server")
	} else {
		p.log("HTTP MCP server started (PID: %d)", p.mcpHTTPProcess.Process.Pid)
	}

	// Wait for servers to be ready
	p.log("Waiting for MCP servers to initialize...")
	time.Sleep(3 * time.Second)

	p.log("MCP servers started successfully")
	return nil
}
