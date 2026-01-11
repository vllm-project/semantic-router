package responseapi

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
	p.log("Step 1/5: Deploying Semantic Router with Response API")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy semantic router: %w", err)
	}

	// Step 2: Deploy Envoy Gateway
	p.log("Step 2/5: Deploying Envoy Gateway")
	if err := p.deployEnvoyGateway(ctx, deployer); err != nil {
		return fmt.Errorf("failed to deploy envoy gateway: %w", err)
	}

	// Step 3: Deploy mock vLLM backend
	p.log("Step 3/5: Deploying mock vLLM backend")
	if err := p.deployMockVLLM(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy mock vLLM: %w", err)
	}

	// Step 4: Deploy Gateway API resources (Gateway/Route/ExtProc patch policy)
	p.log("Step 4/5: Deploying Gateway API resources")
	if err := p.deployGatewayResources(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy gateway resources: %w", err)
	}

	// Step 5: Verify all components are ready
	p.log("Step 5/5: Verifying all components are ready")
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

	p.log("Cleaning up Gateway API resources")
	p.cleanupGatewayResources(ctx, opts)

	p.log("Uninstalling mock vLLM backend")
	p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/response-api/mock-vllm.yaml")

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

		// Response API conversation chaining
		"response-api-conversation-chaining",
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
		Timeout: "30m",
	})
}

func (p *Profile) deployEnvoyGateway(ctx context.Context, deployer *helm.Deployer) error {
	return deployer.Install(ctx, helm.InstallOptions{
		ReleaseName: "eg",
		Chart:       "oci://docker.io/envoyproxy/gateway-helm",
		Namespace:   "envoy-gateway-system",
		Version:     "v1.6.0",
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

	// Wait for Envoy Gateway deployment
	p.log("Waiting for Envoy Gateway deployment...")
	if err := p.waitForDeployment(ctx, client, "envoy-gateway-system", "envoy-gateway"); err != nil {
		return fmt.Errorf("envoy gateway deployment not ready: %w", err)
	}

	// Wait for mock vLLM deployment
	p.log("Waiting for mock vLLM deployment...")
	if err := p.waitForDeployment(ctx, client, "default", "mock-vllm"); err != nil {
		return fmt.Errorf("mock vLLM deployment not ready: %w", err)
	}

	// Wait for Envoy service to show up and be ready
	p.log("Waiting for Envoy Gateway service to be ready...")
	labelSelector := "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router"
	retryTimeout := 10 * time.Minute
	retryInterval := 5 * time.Second
	startTime := time.Now()

	for {
		envoyService, svcErr := helpers.GetEnvoyServiceName(ctx, client, labelSelector, p.verbose)
		if svcErr == nil {
			if podErr := helpers.VerifyServicePodsRunning(ctx, client, "envoy-gateway-system", envoyService, p.verbose); podErr == nil {
				break
			}
		}

		if time.Since(startTime) >= retryTimeout {
			if svcErr != nil {
				return fmt.Errorf("envoy gateway service not ready after %v: %w", retryTimeout, svcErr)
			}
			return fmt.Errorf("envoy gateway service pods not ready after %v", retryTimeout)
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(retryInterval):
		}
	}

	p.log("All components are ready")
	return nil
}

func (p *Profile) waitForDeployment(ctx context.Context, client *kubernetes.Clientset, namespace, name string) error {
	timeout := 30 * time.Minute
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

func (p *Profile) deployMockVLLM(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/response-api/mock-vllm.yaml"); err != nil {
		return err
	}
	return deployer.WaitForDeployment(ctx, "default", "mock-vllm", 5*time.Minute)
}

func (p *Profile) deployGatewayResources(ctx context.Context, opts *framework.SetupOptions) error {
	return p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/response-api/gwapi-resources.yaml")
}

func (p *Profile) cleanupGatewayResources(ctx context.Context, opts *framework.TeardownOptions) error {
	p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/response-api/gwapi-resources.yaml")
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
