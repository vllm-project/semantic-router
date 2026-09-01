package externalgatewayresponses

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "deploy/kubernetes/ai-gateway/semantic-router-values/responses-state.yaml"

var resourceManifests = []string{
	"e2e/profiles/ai-gateway/gateway-resources/backend.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/responses-route.yaml",
}

// Profile verifies Responses state when an external gateway owns dispatch.
type Profile struct {
	stack *gatewaystack.Stack
}

func NewProfile() *Profile {
	return &Profile{stack: gatewaystack.New(gatewaystack.Config{
		Name:                     "external-gateway-responses",
		SemanticRouterValuesFile: valuesFile,
		ResourceManifests:        resourceManifests,
	})}
}

func (p *Profile) Name() string {
	return "external-gateway-responses"
}

func (p *Profile) Description() string {
	return "Tests Responses state and conversion with external gateway-owned dispatch"
}

func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

func (p *Profile) GetTestCases() []string {
	return []string{
		"response-api-create",
		"response-api-get",
		"response-api-conversation-chaining",
	}
}

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
