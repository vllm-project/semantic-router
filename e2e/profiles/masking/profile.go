// Package masking provides the e2e profile for the masking decision plugin
// (#3566). It has no PII model to host, so it drives detection through
// provider-mocker's System One fixture, and verifies the provider-bound bytes
// through provider-mocker's deterministic request echo rather than Router
// Replay, which is deliberately blind on a masking route (D3).
package masking

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/masking/values.yaml"

// resourceManifests reuses the plain provider-mocker manifest unchanged: it
// is both the chat backend and the systemone deployment that answers the pii
// rule. It is a generic fixture, not specific to the profile that first
// added it -- route-action reuses the same manifest.
var resourceManifests = []string{
	"deploy/kubernetes/hallucination/provider-mocker.yaml",
	"deploy/kubernetes/masking/gwapi-resources.yaml",
}

var waitDeployments = []helpers.DeploymentRef{
	{Namespace: "default", Name: "provider-mocker"},
}

// Profile validates the masking plugin end to end across Chat Completions,
// Responses and Anthropic Messages.
type Profile struct {
	stack *gatewaystack.Stack
}

func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "masking",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

func (p *Profile) Name() string { return "masking" }

func (p *Profile) Description() string {
	return "Tests the masking decision plugin across Chat, Responses and Anthropic, and its fail-closed behavior when the PII deployment errors"
}

func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

func (p *Profile) GetTestCases() []string {
	return []string{
		"masking-cross-protocol",
		"masking-keeps-echoed-placeholder",
		"masking-classifier-unavailable-fails-closed",
	}
}

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
