package looper

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	valuesFile   = "e2e/profiles/looper/values.yaml"
	fakeManifest = "e2e/profiles/looper/manifests/fake-backend.yaml"
)

var gatewayResources = []string{
	"deploy/kubernetes/routing-strategies/aigw-resources/base-model.yaml",
	"deploy/kubernetes/routing-strategies/aigw-resources/gwapi-resources.yaml",
}

// Profile validates deterministic Looper algorithm contracts.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates the Looper E2E profile.
func NewProfile() *Profile {
	return &Profile{stack: gatewaystack.New(gatewaystack.Config{
		Name:                     "looper",
		SemanticRouterValuesFile: valuesFile,
		PrerequisiteManifests:    []string{fakeManifest},
		ResourceManifests:        gatewayResources,
		WaitDeployments: []helpers.DeploymentRef{
			{Namespace: "default", Name: "looper-fake-backend"},
			{Namespace: "default", Name: "vllm-llama3-8b-instruct"},
		},
	})}
}

// Name returns the profile name.
func (p *Profile) Name() string { return "looper" }

// Description returns the profile contract.
func (p *Profile) Description() string {
	return "Tests deterministic Looper algorithm behavior through Envoy AI Gateway"
}

// Setup deploys the shared gateway stack and deterministic backend.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and deterministic backend.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the focused Looper contract tests.
func (p *Profile) GetTestCases() []string { return []string{"looper-ratings-happy-path"} }

// GetServiceConfig returns the shared gateway service configuration.
func (p *Profile) GetServiceConfig() framework.ServiceConfig { return p.stack.ServiceConfig() }
