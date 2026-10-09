// Package routingerrors exercises unroutable requests through the real Envoy
// transport without a default provider hiding an unmatched recipe.
package routingerrors

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

// Profile owns an isolated Router and the shared Envoy AI Gateway stack.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates the error-contract deployment with no learned signals.
func NewProfile() *Profile {
	return &Profile{stack: gatewaystack.New(gatewaystack.Config{
		Name:                     "routing-errors",
		SemanticRouterValuesFile: "e2e/profiles/routing-errors/values.yaml",
		ResourceManifests: []string{
			"e2e/profiles/ai-gateway/gateway-resources/backend.yaml",
			"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
		},
		WaitDeployments: []helpers.DeploymentRef{
			{Namespace: "default", Name: "vllm-llama3-8b-instruct"},
		},
	})}
}

// Name returns the profile name.
func (p *Profile) Name() string { return "routing-errors" }

// Description explains why this policy is separate from the baseline.
func (p *Profile) Description() string {
	return "Router error codes through Envoy, including no_route without a default backend"
}

// Setup deploys the existing gateway stack with the isolated policy.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes only this profile's deployed resources.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases retains all four error probes and a routed positive control.
func (p *Profile) GetTestCases() []string {
	return []string{"chat-completions-request", "routing-error-codes"}
}

// GetServiceConfig returns the same Envoy listener used by the baseline.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
