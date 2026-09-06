package selectoralgorithms

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/selector-algorithms/values.yaml"

var resourceManifests = []string{
	"e2e/profiles/selector-algorithms/mock-vllm.yaml",
	"e2e/profiles/selector-algorithms/gateway-resources.yaml",
}

// Profile runs deterministic contracts for selector-executed algorithms.
type Profile struct {
	stack *gatewaystack.Stack
}

func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "selector-algorithms",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments: []helpers.DeploymentRef{
				{Namespace: "default", Name: "selector-mock-vllm"},
			},
		}),
	}
}

func (p *Profile) Name() string { return "selector-algorithms" }

func (p *Profile) Description() string {
	return "Runs deterministic end-to-end contracts for selector algorithms"
}

func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

func (p *Profile) GetTestCases() []string { return []string{"selector-static"} }

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
