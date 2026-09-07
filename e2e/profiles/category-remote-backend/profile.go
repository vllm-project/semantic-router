package categoryremotebackend

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/category-remote-backend/values.yaml"

var resourceManifests = []string{
	"e2e/profiles/category-remote-backend/manifests/mock-category-classifier.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/backend.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/responses-route.yaml",
}

// Profile validates the shared remote category http_classify backend in isolation.
type Profile struct {
	stack *gatewaystack.Stack
}

func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "category-remote-backend",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments: []helpers.DeploymentRef{
				{Namespace: "default", Name: "mock-category-classifier"},
			},
		}),
	}
}

func (p *Profile) Name() string { return "category-remote-backend" }

func (p *Profile) Description() string {
	return "Tests the shared remote category http_classify backend end-to-end"
}

func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

func (p *Profile) GetTestCases() []string { return []string{"category-backend-routing"} }

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
