package istio

import (
	"context"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

	// Import testcases package to register all test cases via their init() functions
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	// Istio Configuration
	istioVersionDefault = "1.28.0"       // Default Istio version to install
	istioNamespace      = "istio-system" // Istio control plane namespace
	istioIngressGateway = "istio-ingressgateway"

	// Semantic Router Configuration
	semanticRouterNamespace  = "vllm-semantic-router-system" // Namespace for semantic router
	semanticRouterDeployment = "semantic-router"
	semanticRouterService    = "semantic-router"
)

// Profile implements the Istio test profile
type Profile struct {
	verbose      bool
	istioVersion string
}

type setupState struct {
	istioInstalled         bool
	namespaceConfigured    bool
	semanticRouterDeployed bool
	envoyGatewayDeployed   bool
	envoyAIGatewayDeployed bool
	demoLLMDeployed        bool
	gatewayResources       bool
}

// NewProfile creates a new Istio profile
func NewProfile() *Profile {
	istioVersion := os.Getenv("ISTIO_VERSION")
	if istioVersion == "" {
		istioVersion = istioVersionDefault
	}

	return &Profile{
		istioVersion: istioVersion,
	}
}

// Name returns the profile name
func (p *Profile) Name() string {
	return "istio"
}

// Description returns the profile description
func (p *Profile) Description() string {
	return fmt.Sprintf("Tests Semantic Router with Istio service mesh (version: %s)", p.istioVersion)
}

// Setup deploys all required components for Istio testing
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up Istio test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	state := &setupState{}

	// Ensure cleanup on error
	defer func() {
		if r := recover(); r != nil {
			p.log("Panic during setup, cleaning up...")
			p.cleanupPartialDeployment(ctx, opts, state)
			panic(r) // Re-panic after cleanup
		}
	}()

	if err := p.setupMeshControlPlane(ctx, opts, deployer, state); err != nil {
		return p.failSetup(ctx, opts, state, err)
	}
	if err := p.setupGatewayPlane(ctx, opts, deployer, state); err != nil {
		return p.failSetup(ctx, opts, state, err)
	}
	if err := p.finishSetup(ctx, opts, state); err != nil {
		return p.failSetup(ctx, opts, state, err)
	}

	p.log("Istio test environment setup complete")
	return nil
}

// Teardown cleans up all deployed resources
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down Istio test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Clean up in reverse order
	p.log("Cleaning up Gateway API resources")
	p.cleanupGatewayResources(ctx, opts)

	p.log("Cleaning up Demo LLM")
	_ = p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml")

	p.log("Uninstalling Envoy AI Gateway")
	_ = deployer.Uninstall(ctx, "aieg", "envoy-ai-gateway-system")
	_ = deployer.Uninstall(ctx, "aieg-crd", "envoy-ai-gateway-system")

	p.log("Uninstalling Envoy Gateway")
	_ = deployer.Uninstall(ctx, "eg", "envoy-gateway-system")

	p.log("Uninstalling Semantic Router")
	_ = deployer.Uninstall(ctx, semanticRouterDeployment, semanticRouterNamespace)

	p.log("Removing sidecar injection label from namespace")
	p.removeSidecarInjection(ctx, opts)

	p.log("Uninstalling Istio")
	p.uninstallIstio(ctx, opts)

	p.log("Istio test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile
func (p *Profile) GetTestCases() []string {
	return testmatrix.Combine(
		testmatrix.RouterSmoke,
		[]string{
			"istio-sidecar-health-check",
			"istio-traffic-routing",
			"istio-mtls-verification",
			"istio-tracing-observability",
		},
	)
}

// GetServiceConfig returns the service configuration for accessing the deployed service
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		LabelSelector: "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router",
		Namespace:     "envoy-gateway-system",
		PortMapping:   "8080:80",
	}
}

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[istio] "+format+"\n", args...)
	}
}
