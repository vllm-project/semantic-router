// Package velahalu verifies the published Halu pair-input detector in a deployed router.
package velahalu

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/vela-halu/values.yaml"

var resourceManifests = []string{
	"e2e/profiles/ai-gateway/gateway-resources/backend.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/responses-route.yaml",
}

// Profile validates the published Halu detector in isolation.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new vela-halu profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "vela-halu",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments: []helpers.DeploymentRef{
				{Namespace: "default", Name: "vllm-llama3-8b-instruct"},
			},
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string { return "vela-halu" }

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests real Vela Halu grounding, Unicode spans, and task input limits through the plugin API"
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
	return []string{"vela-halu-grounding"}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
