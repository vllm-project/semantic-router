package nginx

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
	// Nginx Ingress Configuration
	nginxNamespace      = "ingress-nginx"
	nginxControllerName = "ingress-nginx-controller"
	nginxIngressVersion = "4.12.2"

	// Semantic Router Configuration
	semanticRouterNamespace  = "vllm-semantic-router-system"
	semanticRouterDeployment = "semantic-router"

	// LLM Backend Configuration
	// Using mock-llm for fast E2E testing (instant responses)
	mockLLMNamespace = "mock-llm"

	// Auth Service Configuration
	// Using mock-auth to demonstrate auth_request + proxy mode working together
	mockAuthNamespace = "mock-auth"

	// Timeouts
	timeoutNginxInstall         = 5 * time.Minute
	timeoutSemanticRouterDeploy = 30 * time.Minute
	timeoutLLMBackendDeploy     = 2 * time.Minute
	timeoutAuthServiceDeploy    = 1 * time.Minute
)

// Profile implements the nginx ingress test profile
type Profile struct {
	verbose bool
}

// NewProfile creates a new nginx profile
func NewProfile() *Profile {
	return &Profile{}
}

// Name returns the profile name
func (p *Profile) Name() string {
	return "nginx"
}

// Description returns the profile description
func (p *Profile) Description() string {
	return "Tests Semantic Router with nginx Ingress Controller - REAL end-to-end flow through nginx"
}

// Setup deploys all required components for nginx testing
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up nginx ingress test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Step 1: Deploy nginx Ingress Controller
	p.log("Step 1/5: Deploying nginx Ingress Controller")
	if err := p.deployNginxIngress(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy nginx ingress: %w", err)
	}

	// Step 2: Deploy Auth Service (for auth_request demonstration)
	p.log("Step 2/5: Deploying Auth Service (mock-auth)")
	if err := p.deployAuthService(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy auth service: %w", err)
	}

	// Step 3: Deploy LLM backend (backends must be ready before vSR routes to them)
	p.log("Step 3/5: Deploying LLM backend")
	if err := p.deployLLMBackend(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy LLM backend: %w", err)
	}

	// Step 4: Deploy Semantic Router (after backends are ready)
	p.log("Step 4/5: Deploying Semantic Router")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy semantic router: %w", err)
	}

	// Step 5: Deploy Ingress resources (with auth_request + proxy to vSR)
	p.log("Step 5/5: Deploying Ingress resources")
	if err := p.deployIngressResources(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy ingress resources: %w", err)
	}

	// Verify environment
	p.log("Verifying environment")
	if err := p.verifyEnvironment(ctx, opts); err != nil {
		return fmt.Errorf("failed to verify environment: %w", err)
	}

	p.log("nginx ingress test environment setup complete")
	return nil
}

// Teardown cleans up all deployed resources
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Starting teardown of nginx ingress test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Clean up in reverse order of deployment (Ingress → vSR → LLM backend → Auth → nginx)
	p.log("Cleaning up Ingress resources")
	p.cleanupIngressResources(ctx, opts)

	p.log("Uninstalling Semantic Router")
	deployer.Uninstall(ctx, semanticRouterDeployment, semanticRouterNamespace)

	p.log("Cleaning up LLM backend (mock-llm)")
	p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/nginx/mock-llm.yaml")

	p.log("Cleaning up Auth Service (mock-auth)")
	p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/nginx/mock-auth.yaml")

	p.log("Uninstalling nginx Ingress Controller")
	deployer.Uninstall(ctx, "ingress-nginx", nginxNamespace)

	p.log("nginx ingress test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile
func (p *Profile) GetTestCases() []string {
	return []string{
		// nginx proxy mode tests (production-ready)
		// vSR receives full requests, classifies, blocks or forwards to LLM
		// This provides FULL classification and blocking capabilities

		// Health check
		"nginx-proxy-health",

		// Authentication tests (auth_request + proxy mode)
		// Demonstrates nginx auth_request (headers only) works alongside
		// vSR proxy mode (full body classification)
		"nginx-proxy-auth-valid-token",
		"nginx-proxy-auth-invalid-token",

		// Proxy mode tests - full blocking works!
		"nginx-proxy-normal-request",
		"nginx-proxy-jailbreak-block",
		"nginx-proxy-pii-block",
		"nginx-proxy-classification",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	// Point to nginx ingress controller to test the REAL end-to-end flow:
	// Client → nginx → vSR → mock-llm
	// This tests the actual ingress routing, not just vSR directly
	return framework.ServiceConfig{
		Name:        nginxControllerName, // "ingress-nginx-controller"
		Namespace:   nginxNamespace,      // "ingress-nginx"
		PortMapping: "8080:80",           // nginx listens on port 80
	}
}

func (p *Profile) deployNginxIngress(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	p.log("Installing nginx Ingress Controller via Helm (version: %s)", nginxIngressVersion)

	// Add ingress-nginx helm repo
	p.log("Adding ingress-nginx Helm repository...")
	repoCmd := exec.CommandContext(ctx, "helm", "repo", "add", "ingress-nginx",
		"https://kubernetes.github.io/ingress-nginx", "--force-update")
	if p.verbose {
		repoCmd.Stdout = os.Stdout
		repoCmd.Stderr = os.Stderr
	}
	if err := repoCmd.Run(); err != nil {
		return fmt.Errorf("failed to add ingress-nginx helm repo: %w", err)
	}

	// Update helm repos
	updateCmd := exec.CommandContext(ctx, "helm", "repo", "update")
	if p.verbose {
		updateCmd.Stdout = os.Stdout
		updateCmd.Stderr = os.Stderr
	}
	if err := updateCmd.Run(); err != nil {
		p.log("Warning: helm repo update failed: %v", err)
	}

	// Use the nginx-ingress values.yaml for consistency with user deployment
	nginxValuesFile := "deploy/kubernetes/nginx/nginx-ingress/values.yaml"

	installOpts := helm.InstallOptions{
		ReleaseName: "ingress-nginx",
		Chart:       "ingress-nginx/ingress-nginx",
		Namespace:   nginxNamespace,
		Version:     nginxIngressVersion,
		ValuesFiles: []string{nginxValuesFile},
		Set: map[string]string{
			// Override for Kind cluster (NodePort instead of LoadBalancer)
			"controller.service.type":            "NodePort",
			"controller.service.nodePorts.http":  "30080",
			"controller.service.nodePorts.https": "30443",
		},
		Wait:    true,
		Timeout: "10m",
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	// Wait for nginx controller to be ready
	p.log("Waiting for nginx ingress controller to be ready...")
	return deployer.WaitForDeployment(ctx, nginxNamespace, nginxControllerName, timeoutNginxInstall)
}

func (p *Profile) deploySemanticRouter(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	chartPath := "deploy/helm/semantic-router"
	valuesFile := "deploy/kubernetes/nginx/semantic-router-values/values.yaml"

	imageRepo := "ghcr.io/vllm-project/semantic-router/extproc"
	imageTag := opts.ImageTag

	installOpts := helm.InstallOptions{
		ReleaseName: semanticRouterDeployment,
		Chart:       chartPath,
		Namespace:   semanticRouterNamespace,
		ValuesFiles: []string{valuesFile},
		Set: map[string]string{
			"image.repository": imageRepo,
			"image.tag":        imageTag,
			"image.pullPolicy": "Never", // Use local image
		},
		Wait:    true,
		Timeout: "30m",
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, semanticRouterNamespace, semanticRouterDeployment, timeoutSemanticRouterDeploy)
}

func (p *Profile) deployAuthService(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	// Deploy mock-auth - simple auth service for auth_request demonstration
	// This validates Authorization header tokens before requests reach vSR
	p.log("Deploying mock-auth (auth_request demonstration)")
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/nginx/mock-auth.yaml"); err != nil {
		return fmt.Errorf("failed to apply mock-auth: %w", err)
	}

	// Wait for mock-auth to be ready
	p.log("Waiting for mock-auth to be ready...")
	if err := deployer.WaitForDeployment(ctx, mockAuthNamespace, "mock-auth", timeoutAuthServiceDeploy); err != nil {
		return fmt.Errorf("mock-auth deployment not ready: %w", err)
	}

	p.log("mock-auth is ready!")
	return nil
}

func (p *Profile) deployLLMBackend(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	// Deploy mock-llm - lightweight simulator for fast E2E testing
	p.log("Using mock-llm (llm-d-inference-sim for fast testing)")
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/nginx/mock-llm.yaml"); err != nil {
		return fmt.Errorf("failed to apply mock-llm: %w", err)
	}

	// Wait for mock-llm to be ready
	p.log("Waiting for mock-llm to be ready...")
	if err := deployer.WaitForDeployment(ctx, mockLLMNamespace, "mock-llm", timeoutLLMBackendDeploy); err != nil {
		return fmt.Errorf("mock-llm deployment not ready: %w", err)
	}

	p.log("mock-llm is ready!")
	return nil
}

func (p *Profile) deployIngressResources(ctx context.Context, opts *framework.SetupOptions) error {
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/nginx/ingress.yaml"); err != nil {
		return fmt.Errorf("failed to apply ingress: %w", err)
	}
	// Wait for nginx ingress controller to sync the ingress rules
	p.log("Waiting for ingress rules to sync...")
	time.Sleep(5 * time.Second)
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

	// Check nginx ingress controller deployment
	p.log("Verifying nginx ingress controller...")
	if err := helpers.CheckDeployment(ctx, client, nginxNamespace, nginxControllerName, p.verbose); err != nil {
		return fmt.Errorf("nginx ingress controller not healthy: %w", err)
	}

	// Check mock-auth deployment - auth_request target
	p.log("Verifying mock-auth...")
	if err := helpers.CheckDeployment(ctx, client, mockAuthNamespace, "mock-auth", p.verbose); err != nil {
		return fmt.Errorf("mock-auth deployment not healthy: %w", err)
	}

	// Check semantic-router deployment
	p.log("Verifying semantic router...")
	if err := helpers.CheckDeployment(ctx, client, semanticRouterNamespace, semanticRouterDeployment, p.verbose); err != nil {
		return fmt.Errorf("semantic-router deployment not healthy: %w", err)
	}

	// Check mock-llm deployment - required for nginx flow tests
	p.log("Verifying mock-llm...")
	if err := helpers.CheckDeployment(ctx, client, mockLLMNamespace, "mock-llm", p.verbose); err != nil {
		return fmt.Errorf("mock-llm deployment not healthy: %w", err)
	}

	p.log("All deployments are healthy")
	return nil
}

func (p *Profile) cleanupIngressResources(ctx context.Context, opts *framework.TeardownOptions) error {
	p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/nginx/ingress.yaml")
	return nil
}

func (p *Profile) kubectlApply(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "apply", "-f", manifest)
}

func (p *Profile) kubectlDelete(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "delete", "--ignore-not-found", "-f", manifest)
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
		fmt.Printf("[nginx] "+format+"\n", args...)
	}
}
