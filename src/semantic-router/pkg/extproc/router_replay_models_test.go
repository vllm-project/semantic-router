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

// Every current Claude and GPT model ID Claude Code and Codex call has a card,
// so Insights names it rather than showing its ID.
func TestReplayModelsNameTheVendorModelIDsAgentsCall(t *testing.T) {
	document := "version: v0.5\nlisteners: []\nproviders:\n  models:\n"
	ids := map[string]string{
		"claude-haiku-5-5": "anthropic", "claude-sonnet-5-5": "anthropic",
		"gpt-6.1-sol": "openai", "gpt-6.1-terra": "openai", "gpt-6-sol": "openai", "gpt-6-luna": "openai",
	}
	for id, provider := range ids {
		document += "    - name: " + id + "\n      provider_model_id: " + id + "\n      backend_refs:\n        - {name: vendor, provider: " + provider + "}\n"
	}
	cfg, err := config.ParseYAMLBytes([]byte(document + "recipes:\n  - name: default\n"))
	require.NoError(t, err)
	router := &OpenAIRouter{Config: cfg, ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(1, 0))}
	var models routerreplay.ModelListResponse
	require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/models", &models))
	require.ElementsMatch(t, []routerreplay.ModelLabel{
		{Name: "claude-haiku-5-5", Label: "Claude Haiku 5.5"},
		{Name: "claude-sonnet-5-5", Label: "Claude Sonnet 5.5"},
		{Name: "gpt-6.1-sol", Label: "GPT-6.1 Sol"},
		{Name: "gpt-6.1-terra", Label: "GPT-6.1 Terra"},
		{Name: "gpt-6-sol", Label: "GPT-6 Sol"},
		{Name: "gpt-6-luna", Label: "GPT-6 Luna"},
	}, models.Data)
}
