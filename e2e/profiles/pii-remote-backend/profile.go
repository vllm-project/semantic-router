// Package piiremotebackend provides the e2e profile for #2922's PII backend.
// It drives the PII signal and the PII API through a remote systemone
// deployment, provider-mocker's System One fixture, with no local PII model
// at all, so a detection can only have come from the remote call, and it
// exercises the fail-closed decision against a call that fails.
package piiremotebackend

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/pii-remote-backend/values.yaml"

// The System One fixture runs in its own provider-mocker, apart from the chat
// backend, so the PII deployment and the model it guards are separate peers.
var resourceManifests = []string{
	"deploy/kubernetes/hallucination/provider-mocker.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/backend.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/responses-route.yaml",
}

// Profile validates a remote PII deployment in isolation.
type Profile struct {
	stack *gatewaystack.Stack
}

func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "pii-remote-backend",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments: []helpers.DeploymentRef{
				{Namespace: "default", Name: "provider-mocker"},
			},
		}),
	}
}

func (p *Profile) Name() string { return "pii-remote-backend" }

func (p *Profile) Description() string {
	return "Tests a remote systemone PII deployment and the fail-closed decision end-to-end"
}

func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

func (p *Profile) GetTestCases() []string {
	return []string{
		"pii-backend-routing",
		"pii-api-masking",
		"pii-repeated-value-masking",
	}
}

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
