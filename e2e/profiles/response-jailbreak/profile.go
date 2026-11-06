// Package responsejailbreak provides the e2e test profile for the
// response-direction jailbreak signal and the response_guard that acts on it. It
// deploys the router with its jailbreak rule answered by provider-mocker's
// System One fixture, then verifies that jailbreak content the model emits at
// the end of a long answer is still caught in the buffered response and
// reported through the selected decision's configured action.
package responsejailbreak

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/response-jailbreak/values.yaml"

var resourceManifests = []string{
	"deploy/kubernetes/hallucination/provider-mocker.yaml",
	"deploy/kubernetes/response-jailbreak/gwapi-resources.yaml",
}

// waitDeployments are the mock backends the profile must wait on. provider-mocker
// is both the chat backend and, on /v1/systemone, the deployment that answers
// the jailbreak rule.
var waitDeployments = []helpers.DeploymentRef{
	{Namespace: "default", Name: "provider-mocker"},
}

// Profile implements the response-jailbreak test profile.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new response-jailbreak profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "response-jailbreak",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
			WaitDeployments:          waitDeployments,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "response-jailbreak"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests the response-direction jailbreak signal on the end of a long LLM output and the response_guard that acts on it"
}

// Setup deploys the shared gateway stack and this profile's resources.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and this profile's resources.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return []string{
		"response-jailbreak-window-block",
		"response-jailbreak-window-warning",
		"response-jailbreak-streaming-passthrough",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
