// Package decisionruntime provides the e2e profile for the built-in model
// runtime: the Router attaches to a runtime serving a tiny random-weight
// Decision 2.0 fixture on CPU, routes on its decision signals, lets it choose
// among a decision's models, and fails open for a deployment that never
// answers.
package decisionruntime

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/decision-runtime/values.yaml"

var resourceManifests = []string{
	"e2e/profiles/decision-runtime/manifests/decision-runtime.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/backend.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/responses-route.yaml",
}

// Profile validates Router ↔ model runtime integration end to end.
type Profile struct {
	stack *gatewaystack.Stack
}

func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "decision-runtime",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments: []helpers.DeploymentRef{
				{Namespace: "default", Name: "decision-runtime"},
			},
		}),
	}
}

func (p *Profile) Name() string { return "decision-runtime" }

func (p *Profile) Description() string {
	return "Tests decision signals and the decision selector against the built-in model runtime"
}

func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

func (p *Profile) GetTestCases() []string {
	return []string{
		"decision-runtime-routing",
	}
}

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
