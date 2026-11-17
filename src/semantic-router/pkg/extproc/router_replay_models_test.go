package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// The models route labels every configured model as a person reads it: a
// catalog-backed model by its card, a provider model id the catalog maps by
// that card, any other by its provider model id.
func TestReplayModelsLabelTheConfiguredModels(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.5
listeners: []
providers:
  defaults:
    model: qwen-msg
  models:
    - name: qwen-msg
      catalog: qwen/qwen3.6-35b-a3b
      provider_model_id: qwen36
      backend_refs:
        - name: local
          provider: vllm
          endpoint: http://127.0.0.1:18211/v1
    - name: known-id
      provider_model_id: Qwen/Qwen3.6-35B-A3B
      backend_refs:
        - name: local
          provider: vllm
          endpoint: http://127.0.0.1:18211/v1
    - name: premium-ref
      provider_model_id: qwen36
      backend_refs:
        - name: local
          provider: vllm
          endpoint: http://127.0.0.1:18211/v1
recipes:
  - name: default
global: {}
`))
	require.NoError(t, err)
	router := &OpenAIRouter{Config: cfg, ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(1, 0))}
	var models routerreplay.ModelListResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/models", &models))
	require.Equal(t, "router_replay.models", models.Object)
	require.Equal(t, []routerreplay.ModelLabel{
		{Name: "known-id", Label: "Qwen3.6 35B A3B"},
		{Name: "premium-ref", Label: "qwen36"},
		{Name: "qwen-msg", Label: "Qwen3.6 35B A3B"},
	}, models.Data)
}
