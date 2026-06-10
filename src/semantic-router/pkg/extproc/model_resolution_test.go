package extproc

import (
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
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
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: known-model
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

func newModelResolutionTestContext() *RequestContext {
	return &RequestContext{
		RequestID:           "test-req",
		Headers:             map[string]string{},
		ProcessingStartTime: time.Now(),
		OriginalRequestBody: []byte(`{"model":"x","messages":[{"role":"user","content":"hi"}]}`),
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
	ctx := newModelResolutionTestContext()
	req := &openai.ChatCompletionNewParams{Model: "no-such-model"}

	resp, err := router.handleSpecifiedModelRouting(req, "no-such-model", "", ctx)
	require.NoError(t, err)
	require.NotNil(t, resp.GetImmediateResponse(), "unknown model should produce an immediate error response, not a passthrough")
	assert.Equal(t, 400, int(resp.GetImmediateResponse().GetStatus().GetCode()))
}

// B1: a known, configured model must still route normally.
func TestSpecifiedKnownModelRoutes(t *testing.T) {
	router, _ := newModelResolutionTestRouter(t)
	ctx := newModelResolutionTestContext()
	req := &openai.ChatCompletionNewParams{Model: "known-model"}

	resp, err := router.handleSpecifiedModelRouting(req, "known-model", "", ctx)
	require.NoError(t, err)
	assert.Nil(t, resp.GetImmediateResponse(), "known model should not be rejected")
	model, ok := selectedModelHeader(resp)
	require.True(t, ok, "known model should set x-selected-model header")
	assert.Equal(t, "known-model", model)
}

// B2: an auto model that yields no selection (e.g. empty/contentless messages)
// must fall back to the configured default model instead of forwarding the
// unresolved auto-model name without credentials.
func TestAutoModelNoSelectionFallsBackToDefault(t *testing.T) {
	router, _ := newModelResolutionTestRouter(t)
	ctx := newModelResolutionTestContext()
	req := &openai.ChatCompletionNewParams{Model: "MoM"}

	resp, err := router.handleModelRouting(req, "MoM", "", entropy.ReasoningDecision{}, "", ctx)
	require.NoError(t, err)
	assert.Nil(t, resp.GetImmediateResponse(), "auto model with a configured default should route, not error")
	model, ok := selectedModelHeader(resp)
	require.True(t, ok, "fallback should route to the default model and set x-selected-model")
	assert.Equal(t, "known-model", model)
}

// B2: when auto routing yields no selection AND no default model is configured,
// the request must be rejected with a clear 400 rather than silently forwarded.
func TestAutoModelNoSelectionNoDefaultReturns400(t *testing.T) {
	router, cfg := newModelResolutionTestRouter(t)
	cfg.DefaultModel = ""
	ctx := newModelResolutionTestContext()
	req := &openai.ChatCompletionNewParams{Model: "MoM"}

	resp, err := router.handleModelRouting(req, "MoM", "", entropy.ReasoningDecision{}, "", ctx)
	require.NoError(t, err)
	require.NotNil(t, resp.GetImmediateResponse(), "no selection and no default should produce an immediate error response")
	assert.Equal(t, 400, int(resp.GetImmediateResponse().GetStatus().GetCode()))
}
