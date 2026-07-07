package extproc

import (
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backend"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	replaystore "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const backendSelectionConfigYAML = `
version: v0.3
providers:
  defaults:
    default_model: qwen
  models:
    - name: qwen
      backend_refs:
        - name: primary
          backend_id: qwen-primary
          engine_kind: vllm
          endpoint: 127.0.0.1:8000
          api_key: test-secret
          weight: 10
        - name: secondary
          backend_id: qwen-secondary
          engine_kind: vllm
          endpoint: 127.0.0.1:8001
          api_key: test-secret
          weight: 1
routing:
  modelCards:
    - name: qwen
  decisions:
    - name: default_route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: qwen
`

func newBackendSelectionTestRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(backendSelectionConfigYAML))
	require.NoError(t, err)
	return &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: buildDefaultCredentialResolver(cfg, false),
	}
}

func TestSpecifiedRoutingEmitsSelectedBackendWhenTelemetryIsFresh(t *testing.T) {
	backend.DefaultStore().Reset()
	t.Cleanup(backend.DefaultStore().Reset)
	require.NoError(t, backend.Upsert(backendTelemetry("qwen-primary", "qwen", 8)))
	require.NoError(t, backend.Upsert(backendTelemetry("qwen-secondary", "qwen", 1)))

	router := newBackendSelectionTestRouter(t)
	ctx := newModelResolutionTestContext()
	resp, err := router.handleSpecifiedModelRouting(&openai.ChatCompletionNewParams{Model: "qwen"}, "qwen", "", ctx)
	require.NoError(t, err)

	mutation := resp.GetRequestBody().GetResponse().GetHeaderMutation()
	require.Equal(t, "qwen-secondary", backendSelectionSetHeaderValue(mutation, headers.VSRSelectedBackend))
	require.Equal(t, "qwen-secondary", ctx.RequestedBackendID)
	require.Empty(t, ctx.BackendFallbackReason)
}

func TestSpecifiedRoutingFailsOpenWhenTelemetryIsMissing(t *testing.T) {
	backend.DefaultStore().Reset()
	t.Cleanup(backend.DefaultStore().Reset)

	router := newBackendSelectionTestRouter(t)
	ctx := newModelResolutionTestContext()
	resp, err := router.handleSpecifiedModelRouting(&openai.ChatCompletionNewParams{Model: "qwen"}, "qwen", "", ctx)
	require.NoError(t, err)

	mutation := resp.GetRequestBody().GetResponse().GetHeaderMutation()
	require.Empty(t, backendSelectionSetHeaderValue(mutation, headers.VSRSelectedBackend))
	require.Empty(t, ctx.RequestedBackendID)
	require.Equal(t, backend.FallbackReasonMissingTelemetry, ctx.BackendFallbackReason)
}

func TestInternalBackendHeadersAreRemovedFromInboundRequests(t *testing.T) {
	mutation := buildIdentityEncodingRequestMutation()
	require.Contains(t, mutation.RemoveHeaders, headers.VSRSelectedBackend)
	require.Contains(t, mutation.RemoveHeaders, headers.VSRActualBackend)

	ctx := &RequestContext{Headers: map[string]string{
		headers.VSRSelectedBackend: "spoofed",
		headers.VSRActualBackend:   "spoofed",
	}}
	applyHeaderPassThroughPolicy(ctx)
	require.NotContains(t, ctx.Headers, headers.VSRSelectedBackend)
	require.NotContains(t, ctx.Headers, headers.VSRActualBackend)
}

func TestBackendFeedbackUpdatesReplayDiagnostics(t *testing.T) {
	recorder := routerreplay.NewRecorder(replaystore.NewMemoryStore(10, 0))
	replayID, err := recorder.AddRecord(routerreplay.RoutingRecord{
		RequestID: "req-1",
		RouteDiagnostics: &routerreplay.RouteDiagnostics{
			RequestedBackendID: "qwen-primary",
		},
	})
	require.NoError(t, err)

	router := &OpenAIRouter{ReplayRecorder: recorder}
	ctx := &RequestContext{
		RequestID:            "req-1",
		RouterReplayID:       replayID,
		RouterReplayRecorder: recorder,
		RequestedBackendID:   "qwen-primary",
	}
	router.captureBackendFeedback(&core.HeaderMap{Headers: []*core.HeaderValue{
		{Key: headers.VSRActualBackend, RawValue: []byte("qwen-secondary")},
		{Key: headers.VSRActualReplica, RawValue: []byte("engine-0")},
		{Key: headers.VSRActualUpstream, RawValue: []byte("10.0.0.2:8000")},
	}}, ctx)

	record, found := recorder.GetRecord(replayID)
	require.True(t, found)
	require.NotNil(t, record.RouteDiagnostics)
	require.Equal(t, "qwen-secondary", record.RouteDiagnostics.ActualBackendID)
	require.Equal(t, "engine-0", record.RouteDiagnostics.ActualReplicaID)
	require.Equal(t, "10.0.0.2:8000", record.RouteDiagnostics.ActualUpstream)
}

func backendTelemetry(backendID string, model string, queueDepth int) backend.BackendTelemetry {
	return backend.BackendTelemetry{
		Identity: backend.BackendIdentity{
			BackendID:  backendID,
			ModelName:  model,
			EngineKind: backend.EngineKindVLLM,
		},
		QueueDepth:  &queueDepth,
		Health:      backend.HealthStateHealthy,
		CollectedAt: time.Now(),
		TTL:         backend.DefaultTelemetryTTL,
	}
}

func backendSelectionSetHeaderValue(mutation *ext_proc.HeaderMutation, key string) string {
	if mutation == nil {
		return ""
	}
	for _, header := range mutation.SetHeaders {
		if header.GetHeader().GetKey() == key {
			return string(header.GetHeader().GetRawValue())
		}
	}
	return ""
}
