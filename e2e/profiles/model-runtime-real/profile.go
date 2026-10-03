// Package modelruntimereal provides the opt-in E2E profile that runs real
// models on CPU through the Router's managed model runtime: Decision 2.0
// Kai-0.6B and the Vela Domain, PII and Guard defaults, downloaded from the
// Hugging Face Hub at start-up.
package modelruntimereal

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/model-runtime-real/values.yaml"

var resourceManifests = []string{
	"e2e/profiles/ai-gateway/gateway-resources/backend.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/responses-route.yaml",
}

// Profile validates the Router with real models in managed runtimes.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates the model-runtime-real profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "model-runtime-real",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments: []helpers.DeploymentRef{
				{Namespace: "default", Name: "vllm-llama3-8b-instruct"},
			},
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string { return "model-runtime-real" }

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests real Decision 2.0 Kai-0.6B and Vela Domain, PII and Guard on CPU in the Router's managed model runtime"
}

// Setup deploys the gateway stack.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the gateway stack.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the profile's contracts.
func (p *Profile) GetTestCases() []string {
	return []string{"model-runtime-real-routing"}
}

// GetServiceConfig returns the gateway access configuration.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
