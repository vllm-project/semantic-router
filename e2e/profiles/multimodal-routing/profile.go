package multimodalrouting

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

	// Import testcases package to register all test cases via their init() functions
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

// Profile implements the Multimodal Routing test profile.
// This profile exercises image-modality EmbeddingSignal routing wired through
// PR #1867 (API surface) and PR #1868 (request-path wire-through). It is
// forked from the dynamic-config profile and differs only in the embedding
// model configuration (multimodal vs text-only) and the IntelligentRoute
// rules + decisions (image-modality medical_imagery_phi vs the text
// embedding rules in dynamic-config).
type Profile struct {
	verbose bool
}

// NewProfile creates a new Multimodal Routing profile.
func NewProfile() *Profile {
	return &Profile{}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "multimodal-routing"
}

// Description returns a description of what this profile tests.
func (p *Profile) Description() string {
	return "Tests image-modality EmbeddingSignal routing via IntelligentRoute with the multi-modal-embed-small model"
}

// Setup deploys all required components for Multimodal Routing testing.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up Multimodal Routing test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Step 1: Deploy Semantic Router with multimodal embedding model configured
	p.log("Step 1/5: Deploying Semantic Router with multimodal embedding model")
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

	// Step 5: Deploy CRDs (IntelligentPool and IntelligentRoute with image-modality rules)
	p.log("Step 5/5: Deploying IntelligentPool and IntelligentRoute CRDs")
	if err := p.deployCRDs(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy CRDs: %w", err)
	}

	// Step 6: Verify all components are ready
	p.log("Step 6/6: Verifying all components are ready")
	if err := p.verifyEnvironment(ctx, opts); err != nil {
		return fmt.Errorf("failed to verify environment: %w", err)
	}

	p.log("Multimodal Routing test environment setup complete")
	return nil
}

// Teardown cleans up resources created during setup.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down Multimodal Routing test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Uninstall in reverse order
	_ = deployer.Uninstall(ctx, "envoy-ai-gateway", "envoy-ai-gateway-system")
	_ = deployer.Uninstall(ctx, "eg", "envoy-gateway-system")
	_ = deployer.Uninstall(ctx, "semantic-router", "vllm-semantic-router-system")

	p.log("Multimodal Routing test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return testmatrix.Combine(
		testmatrix.RouterSmoke,
		[]string{
			"embedding-signal-image-routing",
			"apiserver-embedding-input-limits",
			"apiserver-malformed-image-input",
		},
	)
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		LabelSelector: "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router",
		Namespace:     "envoy-gateway-system",
		PortMapping:   "8080:80",
	}
}

func (p *Profile) deploySemanticRouter(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	chartPath := "deploy/helm/semantic-router"
	valuesFile := "e2e/profiles/multimodal-routing/values.yaml"

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
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml"); err != nil {
		return fmt.Errorf("failed to apply base model: %w", err)
	}

	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml"); err != nil {
		return fmt.Errorf("failed to apply gateway API resources: %w", err)
	}

	return nil
}

func (p *Profile) deployCRDs(ctx context.Context, opts *framework.SetupOptions) error {
	if err := p.kubectlApply(ctx, opts.KubeConfig, "e2e/profiles/multimodal-routing/crds/intelligentpool.yaml"); err != nil {
		return fmt.Errorf("failed to apply IntelligentPool CRD: %w", err)
	}

	if err := p.kubectlApply(ctx, opts.KubeConfig, "e2e/profiles/multimodal-routing/crds/intelligentroute.yaml"); err != nil {
		return fmt.Errorf("failed to apply IntelligentRoute CRD: %w", err)
	}

	if err := p.waitForCRDReady(ctx, opts.KubeConfig); err != nil {
		return fmt.Errorf("CRD readiness check failed: %w", err)
	}

	return nil
}

func (p *Profile) kubectlApply(ctx context.Context, kubeconfig, manifestPath string) error {
	cmd := exec.CommandContext(ctx, "kubectl", "apply", "-f", manifestPath, "--kubeconfig", kubeconfig)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (p *Profile) waitForCRDReady(ctx context.Context, kubeconfig string) error {
	p.log("Waiting for IntelligentRoute to be reconciled (Ready condition)...")

	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		cmd := exec.CommandContext(ctx, "kubectl", "get", "intelligentroute", "ai-gateway-route",
			"-n", "default", "--kubeconfig", kubeconfig,
			"-o", "jsonpath={.status.conditions[?(@.type==\"Ready\")].status}")
		output, err := cmd.Output()
		if err == nil && strings.TrimSpace(string(output)) == "True" {
			p.log("IntelligentRoute is Ready")
			return nil
		}

		p.log("IntelligentRoute not yet Ready, retrying in 5s...")
		time.Sleep(5 * time.Second)
	}

	return fmt.Errorf("timed out waiting for IntelligentRoute Ready condition (2m)")
}

func (p *Profile) verifyEnvironment(ctx context.Context, opts *framework.SetupOptions) error {
	client, err := helpers.NewKubeClient(opts.KubeConfig)
	if err != nil {
		return err
	}

	p.log("Waiting for Envoy Gateway service to be ready...")
	envoyService, err := helpers.WaitForServiceByLabelWithReadyPods(
		ctx,
		client,
		"envoy-gateway-system",
		"gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router",
		10*time.Minute,
		5*time.Second,
		p.verbose,
		p.log,
	)
	if err != nil {
		return err
	}
	p.log("Envoy Gateway service is ready: %s", envoyService)

	p.log("Verifying all deployments are healthy...")
	if err := helpers.WaitForDeploymentReady(
		ctx,
		client,
		"vllm-semantic-router-system",
		"semantic-router",
		120*time.Second,
		10*time.Second,
		p.verbose,
	); err != nil {
		return err
	}
	if err := helpers.VerifyDeployments(ctx, client, []helpers.DeploymentRef{
		{Namespace: "vllm-semantic-router-system", Name: "semantic-router"},
		{Namespace: "envoy-gateway-system", Name: "envoy-gateway"},
		{Namespace: "envoy-ai-gateway-system", Name: "ai-gateway-controller"},
	}, p.verbose); err != nil {
		return fmt.Errorf("required deployment not healthy: %w", err)
	}

	p.log("All deployments are healthy")

	return nil
}

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[multimodal-routing] "+format+"\n", args...)
	}
}
