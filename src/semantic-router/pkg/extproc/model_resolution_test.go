package extproc

import (
	"context"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// testRouterConfigYAML is a minimal canonical config with a single backed model.
const testRouterConfigYAML = `
version: v0.3
listeners: []
providers:
  defaults:
    default_model: known-model
  models:
    - name: known-model
      backend_refs:
        - endpoint: 127.0.0.1:8000
          api_key: test-secret
routing:
  modelCards:
    - name: known-model
  decisions:
    - name: default_route
      priority: 1
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: known-model}]
`

func newModelResolutionTestRouter(t *testing.T) (*OpenAIRouter, *config.RouterConfig) {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(testRouterConfigYAML))
	require.NoError(t, err, "parse test config")
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: buildDefaultCredentialResolver(cfg, false),
	}
	return router, cfg
}

func newModelResolutionTestContext(t *testing.T) *RequestContext {
	t.Helper()
	return &RequestContext{
		RequestID:           "test-req",
		Headers:             map[string]string{},
		TraceContext:        context.Background(),
		SourceFormat:        llmprotocol.OpenAIChatV1,
		ProcessingStartTime: time.Now(),
		SemanticRequest:     modelResolutionRequest("x"),
	}
}

func modelResolutionRequest(model string) *llmprotocol.Request {
	return &llmprotocol.Request{
		Generation: 1,
		Model:      model,
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hi"}},
		}},
	}
}

func selectedModelHeader(resp *ext_proc.ProcessingResponse) (string, bool) {
	body := resp.GetRequestBody()
	if body == nil || body.GetResponse() == nil || body.GetResponse().GetHeaderMutation() == nil {
		return "", false
	}
	for _, h := range body.GetResponse().GetHeaderMutation().GetSetHeaders() {
		if h.GetHeader().GetKey() == headers.SelectedModel {
			return string(h.GetHeader().GetRawValue()), true
		}
	}
	return "", false
}

// B1: a client-specified model that is not in the router config must be
// rejected with a clear 400, not silently forwarded (which produces a
// misleading upstream "401 No api key" because no credential is resolvable).
func TestSpecifiedUnknownModelReturns400(t *testing.T) {
	router, _ := newModelResolutionTestRouter(t)
	ctx := newModelResolutionTestContext(t)
	req := modelResolutionRequest("no-such-model")
	ctx.SemanticRequest = req

	resp, err := router.handleSpecifiedModelRouting(req, "no-such-model", "", ctx)
	require.NoError(t, err)
	require.NotNil(t, resp.GetImmediateResponse(), "unknown model should produce an immediate error response, not a passthrough")
	assert.Equal(t, 400, int(resp.GetImmediateResponse().GetStatus().GetCode()))
}

// B1: a known, configured model must still route normally.
func TestSpecifiedKnownModelRoutes(t *testing.T) {
	router, _ := newModelResolutionTestRouter(t)
	ctx := newModelResolutionTestContext(t)
	req := modelResolutionRequest("known-model")
	ctx.SemanticRequest = req

	resp, err := router.handleModelRouting(req, "known-model", "", entropy.ReasoningDecision{}, "", ctx)
	require.NoError(t, err)
	assert.Nil(t, resp.GetImmediateResponse(), "known model should not be rejected")
	model, ok := selectedModelHeader(resp)
	require.True(t, ok, "known model should set x-selected-model header")
	assert.Equal(t, "known-model", model)
}

// B2: an Entrypoint that yields no selection fails closed. A global default
// Model cannot bypass the Entrypoint's Recipe and assignment contract.
func TestEntrypointNoSelectionReturns400(t *testing.T) {
	router, _ := newModelResolutionTestRouter(t)
	ctx := newModelResolutionTestContext(t)
	req := modelResolutionRequest("router/default")
	ctx.SemanticRequest = req

	resp, err := router.handleModelRouting(req, "router/default", "", entropy.ReasoningDecision{}, "", ctx)
	require.NoError(t, err)
	require.NotNil(t, resp.GetImmediateResponse(), "no selection should produce an immediate error response")
	assert.Equal(t, 400, int(resp.GetImmediateResponse().GetStatus().GetCode()))
}

// The Entrypoint branch that dispatches to the looper (algorithm.type
// confidence/ratings matched by normal decision evaluation, as opposed to a
// client requesting the "fusion"/"remom"/"workflows" pseudo-model directly)
// must set ctx.VSRSelectedDecisionName before calling handleLooperExecution,
// the same way handleEntrypointModelRouting's trackVSRDecision call does for
// non-looper decisions. Without it,
// x-vsr-selected-decision comes back empty for any looper decision reached via
// normal auto-matching (caught by the #2694 e2e test, which is the first e2e
// coverage to exercise a confidence-algorithm decision through this path).
func TestLooperEntrypointRoutingSetsSelectedDecisionName(t *testing.T) {
	router, cfg := newModelResolutionTestRouter(t)
	cfg.Looper.Endpoint = "http://127.0.0.1:0/v1/chat/completions" // unreachable; only ctx state before dispatch is under test
	ctx := newModelResolutionTestContext(t)
	decision := &config.Decision{
		Name: "looper_test_decision",
		Algorithm: &config.AlgorithmConfig{
			Type:       "confidence",
			Confidence: &config.ConfidenceAlgorithmConfig{ConfidenceMethod: "hybrid", Threshold: 0.5},
		},
		ModelRefs: []config.ModelRef{
			{Model: "known-model"},
			{Model: "known-model"},
		},
	}
	ctx.VSRSelectedDecision = decision
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
	req := modelResolutionRequest("router/default")
	ctx.SemanticRequest = req

	_, err := router.handleModelRouting(req, "router/default", decision.Name, entropy.ReasoningDecision{}, "known-model", ctx)
	require.NoError(t, err)
	assert.Equal(t, "looper_test_decision", ctx.VSRSelectedDecisionName)
}
