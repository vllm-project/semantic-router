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
)

const (
	// Namespace constants
	namespaceSemanticRouter = "vllm-semantic-router-system"
	namespaceEnvoyGateway   = "envoy-gateway-system"
	namespaceAIGateway      = "envoy-ai-gateway-system"

	// Release name constants
	releaseSemanticRouter = "semantic-router"
	releaseEnvoyGateway   = "eg"
	releaseAIGatewayCRD   = "aieg-crd"
	releaseAIGateway      = "aieg"

	// Deployment name constants
	deploymentSemanticRouter = "semantic-router"
	deploymentEnvoyGateway   = "envoy-gateway"
	deploymentAIGateway      = "ai-gateway-controller"

	// Chart and URL constants
	chartPathSemanticRouter = "deploy/helm/semantic-router"
	chartEnvoyGateway       = "oci://docker.io/envoyproxy/gateway-helm"
	chartAIGatewayCRD       = "oci://docker.io/envoyproxy/ai-gateway-crds-helm"
	chartAIGateway          = "oci://docker.io/envoyproxy/ai-gateway-helm"
	envoyGatewayValuesURL   = "https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/manifests/envoy-gateway-values.yaml"

	// Shared manifest paths
	mockVLLMManifest   = "deploy/kubernetes/response-api/mock-vllm.yaml"
	gatewayAPIManifest = "deploy/kubernetes/response-api/gwapi-resources.yaml"

	// Timeout constants
	timeoutSemanticRouterInstall = "30m"
	timeoutHelmInstall           = "10m"
	timeoutDeploymentWait        = 30 * time.Minute
	timeoutServiceRetry          = 10 * time.Minute
	intervalServiceRetry         = 5 * time.Second

	// Image constants
	imageRepository = "ghcr.io/vllm-project/semantic-router/extproc"
	imagePullPolicy = "Never"

	// Label selector constants
	labelSelectorGateway = "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router"
)

// RedisProfile implements the shared setup for Redis-backed Response API profiles.
type RedisProfile struct {
	verbose       bool
	name          string
	description   string
	valuesFile    string
	redisManifest string
}

// NewRedisProfile constructs a shared Redis-backed Response API profile.
func NewRedisProfile(name, description, valuesFile, redisManifest string) *RedisProfile {
	return &RedisProfile{
		name:          name,
		description:   description,
		valuesFile:    valuesFile,
		redisManifest: redisManifest,
	}
}

// Name returns the profile name.
func (p *RedisProfile) Name() string {
	return p.name
}

// Description returns the profile description.
func (p *RedisProfile) Description() string {
	return p.description
}

// Setup deploys all required components for Response API testing.
func (p *RedisProfile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up Response API Redis test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Step 1: Deploy Redis (required before semantic-router starts)
	p.log("Step 1/6: Deploying Redis")
	if err := p.deployRedis(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy redis: %w", err)
	}

	// Step 2: Deploy Semantic Router with Response API (Redis) enabled
	p.log("Step 2/6: Deploying Semantic Router with Response API (Redis)")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy semantic router: %w", err)
	}

	// Step 3: Deploy Envoy Gateway
	p.log("Step 3/6: Deploying Envoy Gateway")
	if err := p.deployEnvoyGateway(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy envoy gateway: %w", err)
	}

	// Step 4: Deploy Envoy AI Gateway
	p.log("Step 4/6: Deploying Envoy AI Gateway")
	if err := p.deployEnvoyAIGateway(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy envoy ai gateway: %w", err)
	}

	// Step 5: Deploy Gateway API resources
	p.log("Step 5/6: Deploying Gateway API resources")
	if err := p.deployGatewayResources(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy gateway resources: %w", err)
	}

	// Step 6: Verify all components are ready
	p.log("Step 6/6: Verifying all components are ready")
	if err := p.verifyEnvironment(ctx, opts); err != nil {
		return fmt.Errorf("failed to verify environment: %w", err)
	}

	p.log("Response API Redis test environment setup complete")
	return nil
}

// Teardown cleans up all deployed resources.
func (p *RedisProfile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down Response API Redis test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	p.log("Cleaning up Gateway API resources and Redis")
	if err := p.cleanupGatewayResources(ctx, opts); err != nil {
		p.log("Warning: Failed to cleanup Gateway API resources: %v", err)
	}

	p.log("Uninstalling Envoy AI Gateway")
	deployer.Uninstall(ctx, releaseAIGatewayCRD, namespaceAIGateway)
	deployer.Uninstall(ctx, releaseAIGateway, namespaceAIGateway)

	p.log("Uninstalling Envoy Gateway")
	deployer.Uninstall(ctx, releaseEnvoyGateway, namespaceEnvoyGateway)

	p.log("Uninstalling Semantic Router")
	deployer.Uninstall(ctx, releaseSemanticRouter, namespaceSemanticRouter)

	p.log("Response API Redis test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile.
func (p *RedisProfile) GetTestCases() []string {
	return []string{
		"response-api-create",
		"response-api-get",
		"response-api-delete",
		"response-api-input-items",
		"response-api-conversation-chaining",
		"response-api-ttl-expiry",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *RedisProfile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		LabelSelector: labelSelectorGateway,
		Namespace:     namespaceEnvoyGateway,
		PortMapping:   "8080:80",
	}
}

func (p *RedisProfile) deploySemanticRouter(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	installOpts := helm.InstallOptions{
		ReleaseName: releaseSemanticRouter,
		Chart:       chartPathSemanticRouter,
		Namespace:   namespaceSemanticRouter,
		ValuesFiles: []string{p.valuesFile},
		Set: map[string]string{
			"image.repository": imageRepository,
			"image.tag":        opts.ImageTag,
			"image.pullPolicy": imagePullPolicy,
		},
		Wait:    true,
		Timeout: timeoutSemanticRouterInstall,
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	if err := deployer.WaitForDeployment(ctx, namespaceSemanticRouter, deploymentSemanticRouter, timeoutDeploymentWait); err != nil {
		return err
	}

	return nil
}

func (p *RedisProfile) deployRedis(ctx context.Context, opts *framework.SetupOptions) error {
	if err := p.kubectlApply(ctx, opts.KubeConfig, p.redisManifest); err != nil {
		return fmt.Errorf("failed to apply redis: %w", err)
	}
	return nil
}

func (p *RedisProfile) deployEnvoyGateway(ctx context.Context, deployer *helm.Deployer, _ *framework.SetupOptions) error {
	installOpts := helm.InstallOptions{
		ReleaseName: releaseEnvoyGateway,
		Chart:       chartEnvoyGateway,
		Namespace:   namespaceEnvoyGateway,
		Version:     "v1.6.0",
		ValuesFiles: []string{envoyGatewayValuesURL},
		Wait:        true,
		Timeout:     timeoutHelmInstall,
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, namespaceEnvoyGateway, deploymentEnvoyGateway, timeoutDeploymentWait)
}

func (p *RedisProfile) deployEnvoyAIGateway(ctx context.Context, deployer *helm.Deployer, _ *framework.SetupOptions) error {
	// Install AI Gateway CRDs
	crdOpts := helm.InstallOptions{
		ReleaseName: releaseAIGatewayCRD,
		Chart:       chartAIGatewayCRD,
		Namespace:   namespaceAIGateway,
		Version:     "v0.4.0",
		Wait:        true,
		Timeout:     timeoutHelmInstall,
	}

	if err := deployer.Install(ctx, crdOpts); err != nil {
		return err
	}

	// Install AI Gateway
	installOpts := helm.InstallOptions{
		ReleaseName: releaseAIGateway,
		Chart:       chartAIGateway,
		Namespace:   namespaceAIGateway,
		Version:     "v0.4.0",
		Wait:        true,
		Timeout:     timeoutHelmInstall,
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, namespaceAIGateway, deploymentAIGateway, timeoutDeploymentWait)
}

func (p *RedisProfile) deployGatewayResources(ctx context.Context, opts *framework.SetupOptions) error {
	if err := p.kubectlApply(ctx, opts.KubeConfig, mockVLLMManifest); err != nil {
		return fmt.Errorf("failed to apply mock-vllm: %w", err)
	}

	if err := p.kubectlApply(ctx, opts.KubeConfig, gatewayAPIManifest); err != nil {
		return fmt.Errorf("failed to apply gateway API resources: %w", err)
	}

	return nil
}

func (p *RedisProfile) verifyEnvironment(ctx context.Context, opts *framework.SetupOptions) error {
	// Create Kubernetes client
	config, err := clientcmd.BuildConfigFromFlags("", opts.KubeConfig)
	if err != nil {
		return fmt.Errorf("failed to build kubeconfig: %w", err)
	}

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return fmt.Errorf("failed to create kube client: %w", err)
	}

	startTime := time.Now()
	p.log("Waiting for Envoy Gateway service to be ready...")

	var envoyService string
	for {
		envoyService, err = helpers.GetEnvoyServiceName(ctx, client, labelSelectorGateway, p.verbose)
		if err == nil {
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

		if time.Since(startTime) >= timeoutServiceRetry {
			return fmt.Errorf("failed to get Envoy service with running pods after %v: %w", timeoutServiceRetry, err)
		}

		if p.verbose {
			p.log("Envoy service not ready, retrying in %v... (elapsed: %v)",
				intervalServiceRetry, time.Since(startTime).Round(time.Second))
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(intervalServiceRetry):
		}
	}

	p.log("Verifying all deployments are healthy...")

	if err := helpers.CheckDeployment(ctx, client, namespaceSemanticRouter, deploymentSemanticRouter, p.verbose); err != nil {
		return fmt.Errorf("semantic-router deployment not healthy: %w", err)
	}

	if err := helpers.CheckDeployment(ctx, client, namespaceEnvoyGateway, deploymentEnvoyGateway, p.verbose); err != nil {
		return fmt.Errorf("envoy-gateway deployment not healthy: %w", err)
	}

	p.log("All components are ready")
	return nil
}

func (p *RedisProfile) cleanupGatewayResources(ctx context.Context, opts *framework.TeardownOptions) error {
	if err := p.kubectlDelete(ctx, opts.KubeConfig, gatewayAPIManifest); err != nil {
		return fmt.Errorf("failed to delete gateway API resources: %w", err)
	}
	if err := p.kubectlDelete(ctx, opts.KubeConfig, mockVLLMManifest); err != nil {
		return fmt.Errorf("failed to delete mock-vllm: %w", err)
	}
	if err := p.kubectlDelete(ctx, opts.KubeConfig, p.redisManifest); err != nil {
		return fmt.Errorf("failed to delete redis: %w", err)
	}
	return nil
}

func (p *RedisProfile) kubectlApply(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "apply", "-f", manifest)
}

func (p *RedisProfile) kubectlDelete(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "delete", "-f", manifest, "--ignore-not-found=true")
}

func (p *RedisProfile) runKubectl(ctx context.Context, kubeConfig string, args ...string) error {
	args = append(args, "--kubeconfig", kubeConfig)
	cmd := exec.CommandContext(ctx, "kubectl", args...)
	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	return cmd.Run()
}

func (p *RedisProfile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[%s] "+format+"\n", append([]interface{}{p.name}, args...)...)
	}
}
