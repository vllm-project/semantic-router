package routingstrategies

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/routing-strategies/values.yaml"

var resourceManifests = []string{
	"deploy/kubernetes/routing-strategies/aigw-resources/base-model.yaml",
	"deploy/kubernetes/routing-strategies/aigw-resources/gwapi-resources.yaml",
	// Serves the System One fixture the jailbreak and PII rules ask.
	"deploy/kubernetes/hallucination/provider-mocker.yaml",
}

// Profile implements the Routing Strategies test profile.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new Routing Strategies profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "routing-strategies",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          []helpers.DeploymentRef{{Namespace: "default", Name: "provider-mocker"}},
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "routing-strategies"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests different routing strategies including keyword-based routing"
}

// Setup deploys the shared gateway stack.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown tears down the shared gateway stack.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return []string{
		"decision-scoped-multi-factor",
		"decision-scoped-random",
		"metadata-routing",
		"input-modality-routing",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
