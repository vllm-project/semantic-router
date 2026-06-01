package agentgateway

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

	// Import testcases package to register all test cases via their init() functions
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	agentGatewayVersion      = "v1.3.0-alpha.1"
	agentGatewayNamespace    = "agentgateway-system"
	agentGatewayProxyService = "agentgateway-proxy"
	semanticRouterDeployment = "semantic-router"
	semanticRouterValuesFile = "deploy/kubernetes/agentgateway/semantic-router-values/values.yaml"
)

// Profile implements the agentgateway test profile.
type Profile struct {
	verbose bool
}

type setupState struct {
	gatewayAPICRDsInstalled bool
	agentGatewayInstalled   bool
	gatewayProxyCreated     bool
	demoLLMDeployed         bool
	semanticRouterDeployed  bool
	routingResourcesApplied bool
	extProcPolicyApplied    bool
}

// NewProfile creates a new agentgateway profile.
func NewProfile() *Profile {
	return &Profile{}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "agentgateway"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return fmt.Sprintf("Tests Semantic Router through agentgateway (version: %s)", agentGatewayVersion)
}

// Setup deploys all required components for agentgateway testing.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up agentgateway test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	state := &setupState{}

	defer func() {
		if r := recover(); r != nil {
			p.log("Panic during setup, cleaning up...")
			p.cleanupPartialDeployment(ctx, opts, state)
			panic(r)
		}
	}()

	p.log("Step 1/7: Installing Kubernetes Gateway API CRDs")
	if err := p.installGatewayAPICRDs(ctx, opts); err != nil {
		return p.failSetup(ctx, opts, state, fmt.Errorf("install Gateway API CRDs: %w", err))
	}
	state.gatewayAPICRDsInstalled = true

	p.log("Step 2/7: Installing agentgateway CRDs and controller")
	if err := p.installAgentGateway(ctx, deployer, opts); err != nil {
		return p.failSetup(ctx, opts, state, fmt.Errorf("install agentgateway: %w", err))
	}
	state.agentGatewayInstalled = true

	p.log("Step 3/7: Creating agentgateway proxy")
	if err := p.applyManifest(ctx, opts.KubeConfig, "deploy/kubernetes/agentgateway/gateway.yaml"); err != nil {
		return p.failSetup(ctx, opts, state, fmt.Errorf("create Gateway proxy: %w", err))
	}
	state.gatewayProxyCreated = true

	p.log("Step 4/7: Deploying Demo LLM backend")
	if err := p.deployDemoLLM(ctx, deployer, opts); err != nil {
		return p.failSetup(ctx, opts, state, fmt.Errorf("deploy demo LLM: %w", err))
	}
	state.demoLLMDeployed = true

	p.log("Step 5/7: Deploying Semantic Router")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		return p.failSetup(ctx, opts, state, fmt.Errorf("deploy Semantic Router: %w", err))
	}
	state.semanticRouterDeployed = true

	p.log("Step 6/7: Applying agentgateway routing resources")
	if err := p.applyManifest(ctx, opts.KubeConfig, "deploy/kubernetes/agentgateway/routing-resources.yaml"); err != nil {
		return p.failSetup(ctx, opts, state, fmt.Errorf("apply routing resources: %w", err))
	}
	state.routingResourcesApplied = true

	p.log("Step 7/7: Attaching Semantic Router as ExtProc")
	if err := p.applyManifest(ctx, opts.KubeConfig, "deploy/kubernetes/agentgateway/extproc-policy.yaml"); err != nil {
		return p.failSetup(ctx, opts, state, fmt.Errorf("apply ExtProc policy: %w", err))
	}
	state.extProcPolicyApplied = true

	p.log("agentgateway test environment setup complete")
	return nil
}

// Teardown cleans up all deployed resources.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down agentgateway test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	_ = p.deleteManifest(ctx, opts.KubeConfig, "deploy/kubernetes/agentgateway/extproc-policy.yaml")
	_ = p.deleteManifest(ctx, opts.KubeConfig, "deploy/kubernetes/agentgateway/routing-resources.yaml")
	_ = p.deleteManifest(ctx, opts.KubeConfig, "deploy/kubernetes/agentgateway/gateway.yaml")
	_ = p.deleteManifest(ctx, opts.KubeConfig, "deploy/kubernetes/agentgateway/demo-llm.yaml")
	_ = deployer.Uninstall(ctx, semanticRouterDeployment, agentGatewayNamespace)
	_ = deployer.Uninstall(ctx, "agentgateway", agentGatewayNamespace)
	_ = deployer.Uninstall(ctx, "agentgateway-crds", agentGatewayNamespace)

	p.log("agentgateway test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return testmatrix.Combine(
		testmatrix.RouterSmoke,
		[]string{"agentgateway-traffic-routing"},
	)
}

// GetServiceConfig returns the service configuration for accessing the agentgateway proxy.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		Name:        agentGatewayProxyService,
		Namespace:   agentGatewayNamespace,
		ServicePort: "80",
	}
}

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[agentgateway] "+format+"\n", args...)
	}
}
