// Package responsejailbreak provides the e2e test profile for the
// response-direction jailbreak signal and the response_jailbreak plugin. It
// deploys the router with prompt_guard pointed at a window-limited
// http_classify stand-in, then verifies that jailbreak content the model emits
// past that window is still caught in the buffered response and reported
// through the selected decision's configured action.
package responsejailbreak

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	valuesFile           = "e2e/profiles/response-jailbreak/values.yaml"
	mappingConfigMapYAML = "deploy/kubernetes/response-jailbreak/jailbreak-mapping-configmap.yaml"
)

var resourceManifests = []string{
	"deploy/kubernetes/hallucination/mock-vllm.yaml",
	"deploy/kubernetes/response-jailbreak/gwapi-resources.yaml",
}

// waitDeployments are the mock backends the profile must wait on. mock-vllm
// is both the chat backend and, on /classify, the http_classify guardrail
// this profile thresholds against.
var waitDeployments = []helpers.DeploymentRef{
	{Namespace: "default", Name: "mock-vllm"},
}

// Profile implements the response_jailbreak test profile.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new response-jailbreak profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "response-jailbreak",
			SemanticRouterValuesFile: valuesFile,
			PrerequisiteManifests:    []string{mappingConfigMapYAML},
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
	return "Tests the response-direction jailbreak signal on LLM output past the classifier's sequence window and the response_jailbreak plugin that enforces on it"
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
