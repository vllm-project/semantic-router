package responseapi

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	valuesFile = "e2e/profiles/response-api/values.yaml"
	// optInValuesFile configures a second Router that accepts the
	// tool-result failure loss the main Router rejects.
	optInValuesFile = "e2e/profiles/response-api/tool-result-opt-in-values.yaml"
)

var resourceManifests = []string{
	"deploy/kubernetes/response-api/provider-mocker.yaml",
	"deploy/kubernetes/response-api/gwapi-resources.yaml",
}

// Profile implements the Response API test profile.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new Response API profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "response-api",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "response-api"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests Response API endpoints (POST/GET/DELETE /v1/responses)"
}

// Setup deploys the shared gateway stack and Response API resources, then the
// opt-in Router next to them.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	if err := p.stack.Setup(ctx, opts); err != nil {
		return err
	}
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	release := optInRelease()
	release.Set = map[string]string{
		"image.repository": helm.RouterImageRepository,
		"image.tag":        opts.ImageTag,
		"image.pullPolicy": "Never",
	}
	if err := deployer.Install(ctx, release); err != nil {
		return fmt.Errorf("install the opt-in Router: %w", err)
	}
	return deployer.WaitForDeployment(ctx, release.Namespace, release.ReleaseName, 10*time.Minute)
}

// Teardown removes the opt-in Router, then the shared gateway stack and
// Response API resources.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	release := optInRelease()
	if err := helm.NewDeployer(opts.KubeConfig, opts.Verbose).Uninstall(ctx, release.ReleaseName, release.Namespace); err != nil && opts.Verbose {
		fmt.Printf("[response-api] Warning: failed to uninstall the opt-in Router: %v\n", err)
	}
	return p.stack.Teardown(ctx, opts)
}

func optInRelease() helm.InstallOptions {
	release := helm.SemanticRouterRelease.Clone()
	release.ReleaseName = testcases.ToolResultOptInRelease
	release.Namespace = testcases.ToolResultOptInNamespace
	release.ValuesFiles = []string{optInValuesFile}
	return release
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return []string{
		"provider-native-image-generation",
		"response-api-create",
		"response-api-get",
		"response-api-delete",
		"response-api-input-items",
		"response-api-streaming-sse",
		"response-api-conversation-chaining",
		"response-api-error-missing-input",
		"response-api-error-nonexistent-previous-response-id",
		"response-api-error-nonexistent-response-id-get",
		"response-api-error-nonexistent-response-id-delete",
		"response-api-edge-empty-input",
		"response-api-edge-large-input",
		"response-api-edge-special-characters",
		"response-api-edge-concurrent-requests",
		"response-api-image-file-id",
		"response-api-image-generation",
		"response-api-image-generation-stream",
		"input-modality-cross-protocol",
		"model-catalog-astra",
		"protocol-codec-chat-backend-buffered-matrix",
		"protocol-codec-chat-backend-streaming-matrix",
		"protocol-codec-openrouter-reply",
		"protocol-codec-responses-backend-buffered-matrix",
		"protocol-codec-responses-backend-streaming-matrix",
		"protocol-codec-responses-provider-decorations",
		"protocol-codec-chat-backend-tool-lifecycle",
		"protocol-codec-anthropic-cache-tool-loop-chat",
		"protocol-codec-anthropic-adaptive-chat",
		"protocol-codec-chat-custom-tool-kind-switch",
		"protocol-codec-responses-custom-tool-loop",
		"protocol-codec-responses-backend-tool-lifecycle",
		"protocol-codec-tool-result-error-opt-in",
		"protocol-codec-anthropic-system-messages-opt-in",
		"protocol-codec-anthropic-cache-tool-loop-responses",
		"protocol-codec-anthropic-adaptive-responses",
		"protocol-codec-anthropic-per-message-effort",
		"protocol-codec-chat-backend-structured-output",
		"protocol-codec-responses-backend-structured-output",
		"protocol-codec-chat-backend-error-matrix",
		"protocol-codec-responses-backend-error-matrix",
		"protocol-codec-chat-backend-incomplete-stream-matrix",
		"protocol-codec-incomplete-stream-terminal",
		"protocol-codec-chat-backend-midstream-error-matrix",
		"protocol-codec-responses-backend-midstream-error-matrix",
		"protocol-codec-agent-client-fields",
		"prompt-cache-key-affinity",
		"protocol-codec-responses-input-compat",
		"protocol-codec-azure-ingress",
		"protocol-codec-reasoning-summary-responses-backend",
		"protocol-codec-reasoning-summary-template-responses-backend",
		"protocol-codec-ollama-output-limit",
		"protocol-codec-ollama-empty-content",
		"protocol-codec-vllm-stop-sequence",
		"protocol-codec-vllm-reasoning-order",
		"protocol-codec-vllm-stream-error",
		"protocol-codec-auto-unsupported-capability",
		"protocol-codec-chat-backend-agent-client-replay",
		"protocol-codec-responses-backend-agent-client-replay",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
