// Package piiremotebackend provides the e2e profile for #2922's PII backend.
// It drives the PII signal through a remote token_spans.v1 service with no
// local PII model at all, so a detection can only have come from the remote
// call, and it exercises on_error: block against a response the contract
// rejects.
package piiremotebackend

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	valuesFile           = "e2e/profiles/pii-remote-backend/values.yaml"
	mappingConfigMapYAML = "deploy/kubernetes/pii-remote-backend/pii-mapping-configmap.yaml"
)

var resourceManifests = []string{
	"e2e/profiles/pii-remote-backend/manifests/mock-pii-spans.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/backend.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/responses-route.yaml",
}

// Profile validates the remote PII token_spans.v1 backend in isolation.
type Profile struct {
	stack *gatewaystack.Stack
}

func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "pii-remote-backend",
			SemanticRouterValuesFile: valuesFile,
			PrerequisiteManifests:    []string{mappingConfigMapYAML},
			ResourceManifests:        resourceManifests,
			WaitDeployments: []helpers.DeploymentRef{
				{Namespace: "default", Name: "mock-pii-spans"},
			},
		}),
	}
}

func (p *Profile) Name() string { return "pii-remote-backend" }

func (p *Profile) Description() string {
	return "Tests the remote PII token_spans.v1 backend and its on_error policy end-to-end"
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
	}
}

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
