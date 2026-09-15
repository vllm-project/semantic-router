// Package masking provides the e2e profile for the masking decision plugin
// (#3566). It has no PII model to host, so it drives detection through the
// same remote token_spans.v1 stub the pii-remote-backend profile uses, and
// verifies the provider-bound bytes through mock-vllm's deterministic
// request echo rather than Router Replay, which is deliberately blind on a
// masking route (D3).
package masking

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	valuesFile           = "e2e/profiles/masking/values.yaml"
	mappingConfigMapYAML = "deploy/kubernetes/pii-remote-backend/pii-mapping-configmap.yaml"
)

// resourceManifests reuses two other profiles' manifests unchanged: the
// plain mock-vllm chat backend (deploy/kubernetes/hallucination/mock-vllm.yaml)
// and the remote PII stub server (pii-remote-backend's mock-pii-spans). Both
// are generic fixtures, not specific to the profile that first added them --
// pii-remote-backend itself reuses ai-gateway's backend manifest the same way.
var resourceManifests = []string{
	"deploy/kubernetes/hallucination/mock-vllm.yaml",
	"deploy/kubernetes/masking/gwapi-resources.yaml",
	"e2e/profiles/pii-remote-backend/manifests/mock-pii-spans.yaml",
}

var waitDeployments = []helpers.DeploymentRef{
	{Namespace: "default", Name: "mock-vllm"},
	{Namespace: "default", Name: "mock-pii-spans"},
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
			PrerequisiteManifests:    []string{mappingConfigMapYAML},
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

func (p *Profile) Name() string { return "masking" }

func (p *Profile) Description() string {
	return "Tests the masking decision plugin across Chat, Responses and Anthropic, and its fail-closed behavior when the PII classifier errors"
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
		"masking-classifier-unavailable-fails-closed",
	}
}

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
