package progressgate

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

type Profile struct {
	stack *gatewaystack.Stack
}

func NewProfile() *Profile {
	return &Profile{stack: gatewaystack.New(gatewaystack.Config{
		Name:                     "progress-gate",
		SemanticRouterValuesFile: "e2e/profiles/progress-gate/values.yaml",
		ResourceManifests: []string{
			"deploy/kubernetes/response-api/mock-vllm.yaml",
			"deploy/kubernetes/response-api/gwapi-resources.yaml",
		},
	})}
}

func (p *Profile) Name() string { return "progress-gate" }

func (p *Profile) Description() string {
	return "Tests session evidence capture, outcome ingest, enforced switches, cooldown and Replay"
}

func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

func (p *Profile) GetTestCases() []string {
	return []string{"progress-gate-evidence-to-switch"}
}

func (p *Profile) GetServiceConfig() framework.ServiceConfig { return p.stack.ServiceConfig() }
