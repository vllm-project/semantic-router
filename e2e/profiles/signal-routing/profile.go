// Package signalrouting validates the structure, context, conversation,
// complexity and preference signals' rule matching end to end. Each signal has
// its own recipe behind its own entrypoint, so its decisions never compete for
// priority space with another signal's (#3178), and one deployment serves all
// five cases.
package signalrouting

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/signal-routing/values.yaml"

var resourceManifests = []string{
	"e2e/profiles/ai-gateway/gateway-resources/backend.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/responses-route.yaml",
	// Serves the System One fixture the complexity and preference rules ask.
	"deploy/kubernetes/hallucination/provider-mocker.yaml",
}

// Profile validates five signals' rule matching, one recipe per signal.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new signal-routing profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "signal-routing",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          []helpers.DeploymentRef{{Namespace: "default", Name: "provider-mocker"}},
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string { return "signal-routing" }

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests the structure, context, conversation, complexity and preference signals' rule matching end-to-end, one recipe each"
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
	return []string{"structure-routing", "context-routing", "conversation-routing", "complexity-routing", "preference-routing"}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
