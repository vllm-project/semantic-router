package responseapi

import (
	"context"
	"fmt"
	"time"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"

	// Import testcases package to register all test cases via their init() functions
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

// Profile implements the Response API test profile
type Profile struct {
	verbose    bool
	kubeConfig string
}

// NewProfile creates a new Response API profile
func NewProfile() *Profile {
	return &Profile{}
}

// Name returns the profile name
func (p *Profile) Name() string {
	return "response-api"
}

// Description returns the profile description
func (p *Profile) Description() string {
	return "Tests Response API endpoints (POST/GET/DELETE /v1/responses)"
}

// Setup deploys all required components for Response API testing
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.kubeConfig = opts.KubeConfig
	p.log("Setting up Response API test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Step 1: Deploy Semantic Router with Response API enabled
	p.log("Step 1/3: Deploying Semantic Router with Response API")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy semantic router: %w", err)
	}

	// Step 2: Deploy Envoy Gateway
	p.log("Step 2/3: Deploying Envoy Gateway")
	if err := p.deployEnvoyGateway(ctx, deployer); err != nil {
		return fmt.Errorf("failed to deploy envoy gateway: %w", err)
	}

	// Step 3: Verify all components are ready
	p.log("Step 3/3: Verifying all components are ready")
	if err := p.verifyEnvironment(ctx, opts); err != nil {
		return fmt.Errorf("failed to verify environment: %w", err)
	}

	p.log("Response API test environment setup complete")
	return nil
}

// Teardown cleans up all deployed resources
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down Response API test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	p.log("Uninstalling Envoy Gateway")
	_ = deployer.Uninstall(ctx, "eg", "envoy-gateway-system")

	p.log("Uninstalling Semantic Router")
	_ = deployer.Uninstall(ctx, "semantic-router", "vllm-semantic-router-system")

	p.log("Response API test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile
func (p *Profile) GetTestCases() []string {
	return []string{
		// Response API basic operations
		"response-api-create",
		"response-api-get",
		"response-api-delete",
		"response-api-input-items",
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
	imageTag := opts.ImageTag
	if imageTag == "" {
		imageTag = "latest"
	}

	return deployer.Install(ctx, helm.InstallOptions{
		ReleaseName: "semantic-router",
		Chart:       "deploy/helm/semantic-router",
		Namespace:   "vllm-semantic-router-system",
		ValuesFiles: []string{"e2e/profiles/response-api/values.yaml"},
		Set: map[string]string{
			"image.repository": "ghcr.io/vllm-project/semantic-router/extproc",
			"image.tag":        imageTag,
		},
		Wait:    true,
		Timeout: "300s",
	})
}

func (p *Profile) deployEnvoyGateway(ctx context.Context, deployer *helm.Deployer) error {
	return deployer.Install(ctx, helm.InstallOptions{
		ReleaseName: "eg",
		Chart:       "oci://docker.io/envoyproxy/gateway-helm",
		Namespace:   "envoy-gateway-system",
		Wait:        true,
		Timeout:     "300s",
	})
}

func (p *Profile) verifyEnvironment(ctx context.Context, opts *framework.SetupOptions) error {
	config, err := clientcmd.BuildConfigFromFlags("", opts.KubeConfig)
	if err != nil {
		return fmt.Errorf("failed to build kubeconfig: %w", err)
	}

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	// Wait for semantic router deployment
	p.log("Waiting for Semantic Router deployment...")
	if err := p.waitForDeployment(ctx, client, "vllm-semantic-router-system", "semantic-router"); err != nil {
		return fmt.Errorf("semantic router deployment not ready: %w", err)
	}

	p.log("All components are ready")
	return nil
}

func (p *Profile) waitForDeployment(ctx context.Context, client *kubernetes.Clientset, namespace, name string) error {
	timeout := 5 * time.Minute
	interval := 5 * time.Second
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		if err := helpers.CheckDeployment(ctx, client, namespace, name, p.verbose); err == nil {
			return nil
		}
		time.Sleep(interval)
	}

	return fmt.Errorf("timeout waiting for deployment %s/%s", namespace, name)
}

func (p *Profile) log(msg string) {
	if p.verbose {
		fmt.Printf("[response-api] %s\n", msg)
	}
}
