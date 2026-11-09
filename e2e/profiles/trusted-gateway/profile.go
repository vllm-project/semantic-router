package trustedgateway

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/trusted-gateway/values.yaml"

var (
	resourceManifests = []string{
		"e2e/profiles/trusted-gateway/gateway-resources/backend.yaml",
		"e2e/profiles/trusted-gateway/gateway-resources/gwapi-resources.yaml",
	}
	waitDeployments = []helpers.DeploymentRef{
		{Namespace: "trusted-gateway-system", Name: "trusted-gateway"},
		{Namespace: "trusted-gateway-system", Name: "static-backend"},
	}
)

// Profile proves the trusted gateway credential contract (issue #2286) end
// to end: a deployment authenticator, stood in for by an Envoy Lua filter,
// strips client-supplied authentication outputs and attaches a dedicated
// gateway credential to authenticated requests; the Router forwards that
// credential only to the backend marked trusted_gateway, rejects requests
// without one before dispatch, and keeps every other backend on its static
// key. Both backends run the provider-mocker, which refuses a request whose
// Authorization is not the credential it expects or that still carries the
// gateway credential header.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new trusted-gateway profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "trusted-gateway",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "trusted-gateway"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests the deployment-injected trusted gateway credential: forwarded only to trusted_gateway backends, rejected when missing, never leaked"
}

// Setup deploys the shared gateway stack and the two mock backends.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and the mock backends.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return []string{"trusted-gateway-credential"}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
