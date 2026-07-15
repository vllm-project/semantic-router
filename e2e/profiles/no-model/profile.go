package nomodel

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/no-model/values.yaml"

var resourceManifests = []string{
	"deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
}

// Profile implements the no-model readiness test profile.
// This profile deploys the router without any embedding or classifier models
// so that all classification and embedding endpoints return 503.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new no-model profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "no-model",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "no-model"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests 503 readiness responses when no embedding or classifier models are loaded"
}

// Setup deploys the shared gateway stack with no-model configuration.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the readiness-only test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return testmatrix.Combine(testmatrix.ReadinessContract)
}

// GetServiceConfig returns the service configuration.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}

func init() {
	_ = fmt.Sprintf("no-model profile registered")
}
