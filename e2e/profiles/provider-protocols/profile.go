package providerprotocols

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/provider-protocols/values.yaml"

var (
	resourceManifests = []string{
		"e2e/profiles/provider-protocols/gateway-resources/backend.yaml",
		"e2e/profiles/provider-protocols/gateway-resources/gwapi-resources.yaml",
	}
	waitDeployments = []helpers.DeploymentRef{
		{Namespace: "provider-protocols-system", Name: "provider-mocker"},
	}
)

// Profile tests native Anthropic provider boundaries through the Router and Envoy.
// The unified provider-mocker serves the backend protocol directly.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new provider-protocols profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "provider-protocols",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "provider-protocols"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests native Anthropic Messages, protocol translation, and cache-cycle behavior"
}

// Setup deploys the shared gateway stack and provider-protocols backend.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and provider-protocols backend.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return testmatrix.ProviderProtocolsContract
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
