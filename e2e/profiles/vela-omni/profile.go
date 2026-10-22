// Package velaomni verifies both maintained Omni models through a deployed router.
package velaomni

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

type Profile struct{ stack *gatewaystack.Stack }

func NewProfile() *Profile {
	return &Profile{stack: gatewaystack.New(gatewaystack.Config{
		Name: "vela-omni", SemanticRouterValuesFile: "e2e/profiles/vela-omni/values.yaml",
		ResourceManifests: []string{"e2e/profiles/ai-gateway/gateway-resources/backend.yaml", "deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml"},
		WaitDeployments: []helpers.DeploymentRef{
			{Namespace: "default", Name: "vllm-llama3-8b-instruct"},
		},
	})}
}
func (p *Profile) Name() string { return "vela-omni" }
func (p *Profile) Description() string {
	return "Real Nano and Mini text, image, original audio, representation constraints and recipe isolation"
}

func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}
func (p *Profile) GetServiceConfig() framework.ServiceConfig { return p.stack.ServiceConfig() }
func (p *Profile) GetTestCases() []string                    { return []string{"vela-omni-contract"} }
