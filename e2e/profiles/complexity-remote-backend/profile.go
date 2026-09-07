// Package complexityremotebackend provides the e2e profile for #2921's
// complexity backend. It drives the complexity signal through a remote
// score.v1 scorer with no local candidates at all, so a verdict can only have
// come from the remote call.
package complexityremotebackend

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/complexity-remote-backend/values.yaml"

var resourceManifests = []string{
	"e2e/profiles/complexity-remote-backend/manifests/mock-difficulty-scorer.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/backend.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/responses-route.yaml",
}

// Profile validates the shared remote complexity score.v1 backend in isolation.
type Profile struct {
	stack *gatewaystack.Stack
}

func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "complexity-remote-backend",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments: []helpers.DeploymentRef{
				{Namespace: "default", Name: "mock-difficulty-scorer"},
			},
		}),
	}
}

func (p *Profile) Name() string { return "complexity-remote-backend" }

func (p *Profile) Description() string {
	return "Tests the shared remote complexity score.v1 backend end-to-end"
}

func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

func (p *Profile) GetTestCases() []string {
	return []string{
		"complexity-backend-routing",
	}
}

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
