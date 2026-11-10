package delegatedcredential

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/delegated-credential/values.yaml"

var (
	resourceManifests = []string{
		"e2e/profiles/delegated-credential/gateway-resources/backend.yaml",
		"e2e/profiles/delegated-credential/gateway-resources/gwapi-resources.yaml",
	}
	waitDeployments = []helpers.DeploymentRef{
		{Namespace: "delegated-credential-system", Name: "delegated-gateway"},
		{Namespace: "delegated-credential-system", Name: "static-backend"},
	}
)

// Profile proves the delegated credential contract (issue #2286) end to end:
// a deployment authenticator, stood in for by an Envoy Lua filter, strips
// client-supplied authentication outputs and attaches the credential it
// issued an authenticated user; the Router presents that credential only to
// upstreams whose auth is credential: delegated, a provider backend and a
// decision-model deployment alike, answers 401 before a call when a request
// carries none, refuses its own calls to them, and keeps every other upstream
// on its static key. Both backends run the provider-mocker, which refuses a
// request whose Authorization is not the credential it expects or that still
// carries the delegated credential header.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new delegated-credential profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "delegated-credential",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "delegated-credential"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests the delegated credential: presented only to delegated upstreams by calls made for a request, rejected when missing, refused to Router-own calls, never leaked"
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
	return []string{"delegated-credential"}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
