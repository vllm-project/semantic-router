//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestNativePreviewHTTPUsesRendererAndHonorsDeadline(t *testing.T) {
	for _, timeout := range []bool{false, true} {
		name := "selection"
		if timeout {
			name = "deadline"
		}
		t.Run(name, func(t *testing.T) {
			var renders atomic.Int32
			stopped := make(chan struct{})
			release := make(chan struct{})
			provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				renders.Add(1)
				require.Equal(t, "/v1/chat/completions/render", r.URL.Path, "Preview must never call generation")
				var body map[string]any
				require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
				_, err := io.Copy(io.Discard, r.Body)
				require.NoError(t, err)
				require.Nil(t, body["max_tokens"])
				if timeout {
					select {
					case <-r.Context().Done():
						close(stopped)
					case <-release:
					}
					return
				}
				require.NoError(t, json.NewEncoder(w).Encode(map[string]any{"model": body["model"], "token_ids": []int{1, 2, 3}, "sampling_params": map[string]any{"max_tokens": 32765}}))
			}))
			t.Cleanup(provider.Close)
			t.Cleanup(func() { close(release) })
			cfg := contextEvalRouterConfig()
			cfg.ContextRules = cfg.ContextRules[:1]
			cfg.Decisions[0].Rules.Name = "small-request-context"
			cfg.Decisions[0].ModelRefs = []config.ModelRef{{Model: "native-model"}}
			cfg.Decisions[0].Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmStatic}
			payload, err := config.NewStructuredPayload(map[string]any{"default_max_tokens": "auto"})
			require.NoError(t, err)
			cfg.Decisions[0].Plugins = []config.DecisionPlugin{{Type: "request_params", Configuration: payload}}
			cfg.ModelConfig = map[string]config.ModelParams{"native-model": {PreferredEndpoints: []string{"backend"}, APIFormat: config.APIFormatOpenAI, Capabilities: []string{"chat"}, ContextWindowSize: 32768, MaxOutputTokens: 32768}}
			cfg.VLLMEndpoints = []config.VLLMEndpoint{{Name: "backend", Address: "127.0.0.1", Port: 8000, ProviderProfileName: "local-render"}}
			cfg.ProviderProfiles = map[string]config.ProviderProfile{"local-render": {Type: "vllm", BaseURL: provider.URL + "/v1"}}
			cfg.CandidateRequirements = &config.CandidateRequirements{Capabilities: config.CandidateCapabilitiesDeclared, Context: config.CandidateContextKnownLimits}
			cfg.DocumentHash = strings.Repeat("a", 64)
			seconds := 1
			cfg.API.RoutingPreview.RequestTimeoutSeconds = &seconds
			api := newContextEvalServer(t, cfg)
			api.config = cfg
			service := api.classificationSvc.(*services.ClassificationService)
			service.SetEvalModelSelector(&extproc.OpenAIRouter{Config: cfg})
			t.Cleanup(func() { require.NoError(t, service.Close()) })
			server := httptest.NewServer(api.setupRoutes())
			t.Cleanup(server.Close)
			request, err := http.NewRequest(http.MethodPost, server.URL+apiRoutingPreviewPath, strings.NewReader(`{"text":"hello"}`))
			require.NoError(t, err)
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set(headers.SRBenchExpectedConfigHash, cfg.DocumentHash)
			response, err := server.Client().Do(request)
			require.NoError(t, err)
			defer response.Body.Close()
			body, err := io.ReadAll(response.Body)
			require.NoError(t, err)
			if timeout {
				require.Equal(t, http.StatusGatewayTimeout, response.StatusCode, string(body))
				require.Contains(t, string(body), "REQUEST_TIMEOUT")
				select {
				case <-stopped:
				case <-time.After(3 * time.Second):
					t.Fatal("HTTP Preview deadline did not cancel provider rendering")
				}
			} else {
				require.Equal(t, http.StatusOK, response.StatusCode, string(body))
				var result services.EvalResponse
				require.NoError(t, json.Unmarshal(body, &result))
				require.Equal(t, services.EvalSelectionSelected, result.SelectionStatus)
				require.Equal(t, "native-model", result.SelectedModel)
				require.Equal(t, cfg.DocumentHash, result.ConfigHash)
			}
			require.EqualValues(t, 1, renders.Load())
		})
	}
}
