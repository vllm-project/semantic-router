// Package structurerouting validates the structure signal's rule matching
// end-to-end, isolated in its own profile so its decisions never have to
// compete for priority space with a shared profile's other signals. See
// #3178.
package structurerouting

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/structure-routing/values.yaml"

var resourceManifests = []string{
	"e2e/profiles/ai-gateway/gateway-resources/backend.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/responses-route.yaml",
}

// Profile validates the structure signal's rule matching in isolation.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new structure-routing profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "structure-routing",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string { return "structure-routing" }

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests the structure signal's rule matching end-to-end"
}

// Setup deploys the shared gateway stack and this profile's resources.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and this profile's resources.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return []string{"structure-routing"}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
