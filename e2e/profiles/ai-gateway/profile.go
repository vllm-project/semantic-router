package aigateway

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
)

// Profile implements the AI Gateway test profile
type Profile struct {
	verbose bool
}

// NewProfile creates a new AI Gateway profile
func NewProfile() *Profile {
	return &Profile{}
}

// Name returns the profile name
func (p *Profile) Name() string {
	return "ai-gateway"
}

// Description returns the profile description
func (p *Profile) Description() string {
	return "Tests Semantic Router with Envoy AI Gateway integration"
}

// Setup deploys all required components for AI Gateway testing
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up AI Gateway test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Step 1: Deploy Semantic Router
	p.log("Step 1/4: Deploying Semantic Router")
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
	p.log("Step 4/4: Deploying Demo LLM and Gateway API Resources")
	if err := p.deployGatewayResources(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy gateway resources: %w", err)
	}

	p.log("AI Gateway test environment setup complete")
	return nil
}

// Teardown cleans up all deployed resources
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down AI Gateway test environment")

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

	p.log("AI Gateway test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile
func (p *Profile) GetTestCases() []string {
	return []string{
		"basic-health-check",
		"chat-completions-request",
	}
}

func (p *Profile) deploySemanticRouter(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	valuesFile := "deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml"

	installOpts := helm.InstallOptions{
		ReleaseName: "semantic-router",
		Chart:       "oci://ghcr.io/vllm-project/charts/semantic-router",
		Namespace:   "vllm-semantic-router-system",
		Version:     "v0.0.0-latest",
		ValuesFiles: []string{valuesFile},
		Wait:        true,
		Timeout:     "10m",
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, "vllm-semantic-router-system", "semantic-router", 10*time.Minute)
}

func (p *Profile) deployEnvoyGateway(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	installOpts := helm.InstallOptions{
		ReleaseName: "eg",
		Chart:       "oci://docker.io/envoyproxy/gateway-helm",
		Namespace:   "envoy-gateway-system",
		Version:     "v0.0.0-latest",
		ValuesFiles: []string{"https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/manifests/envoy-gateway-values.yaml"},
		Wait:        true,
		Timeout:     "5m",
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, "envoy-gateway-system", "envoy-gateway", 5*time.Minute)
}

func (p *Profile) deployEnvoyAIGateway(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	// Install AI Gateway CRDs
	crdOpts := helm.InstallOptions{
		ReleaseName: "aieg-crd",
		Chart:       "oci://docker.io/envoyproxy/ai-gateway-crds-helm",
		Namespace:   "envoy-ai-gateway-system",
		Version:     "v0.0.0-latest",
		Wait:        true,
		Timeout:     "5m",
	}

	if err := deployer.Install(ctx, crdOpts); err != nil {
		return err
	}

	// Install AI Gateway
	installOpts := helm.InstallOptions{
		ReleaseName: "aieg",
		Chart:       "oci://docker.io/envoyproxy/ai-gateway-helm",
		Namespace:   "envoy-ai-gateway-system",
		Version:     "v0.0.0-latest",
		Wait:        true,
		Timeout:     "5m",
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, "envoy-ai-gateway-system", "ai-gateway-controller", 5*time.Minute)
}

func (p *Profile) deployGatewayResources(ctx context.Context, opts *framework.SetupOptions) error {
	// Apply base model
	if err := p.kubectlApply(ctx, opts.KubeConfig, "https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml"); err != nil {
		return fmt.Errorf("failed to apply base model: %w", err)
	}

	// Apply gateway API resources
	if err := p.kubectlApply(ctx, opts.KubeConfig, "https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml"); err != nil {
		return fmt.Errorf("failed to apply gateway API resources: %w", err)
	}

	return nil
}

func (p *Profile) cleanupGatewayResources(ctx context.Context, opts *framework.TeardownOptions) error {
	// Delete in reverse order
	p.kubectlDelete(ctx, opts.KubeConfig, "https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml")
	p.kubectlDelete(ctx, opts.KubeConfig, "https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml")
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
		fmt.Printf("[AI-Gateway] "+format+"\n", args...)
	}
}
