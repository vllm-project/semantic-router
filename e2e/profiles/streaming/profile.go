package streaming

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/streaming/values.yaml"

var resourceManifests = []string{
	"deploy/kubernetes/routing-strategies/aigw-resources/base-model.yaml",
	"deploy/kubernetes/streaming/aigw-resources/gwapi-resources.yaml",
}

// Profile implements the Streaming Body test profile.
// It deploys the semantic router with Envoy's ext_proc request_body_mode set
// to STREAMED (instead of BUFFERED) so that request bodies arrive as multiple
// gRPC chunks.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates a new streaming profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "streaming",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
		}),
	}
}

func (p *Profile) Name() string {
	return "streaming"
}

func (p *Profile) Description() string {
	return "Tests streamed request body mode: chunked delivery, cache round-trip, large payloads, multimodal, SSE streaming responses"
}

func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

func (p *Profile) GetTestCases() []string {
	return []string{
		"streaming-keyword-routing",
		"streaming-cache-roundtrip",
		"streaming-large-body",
		"streaming-sse-cache",
	}
}

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
