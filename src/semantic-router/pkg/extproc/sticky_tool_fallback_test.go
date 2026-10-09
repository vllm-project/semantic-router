package extproc

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const anthropicFallbackAnswer = `{"id":"msg_fallback","type":"message","role":"assistant","model":"model-fallback-2",` +
	`"content":[{"type":"text","text":"From the candidate."}],"stop_reason":"end_turn",` +
	`"usage":{"input_tokens":10,"output_tokens":4}}`

// Provider fallback re-sends the finalized request: a candidate on another
// protocol receives exactly the sticky selection, re-encoded for its wire,
// and fallback never re-runs selection or touches session state.
func TestStickyFinalToolsReachProviderFallback(t *testing.T) {
	h := newStickyHarness(t, llmprotocol.OpenAIChatV1, stickyAddDecision(t, &config.StickyToolSelectionConfig{MaxTools: intPtr(3)}), nil)
	h.run(stickyTurn{query: stickyTestQueries["weather"]})
	final := h.run(stickyTurn{query: stickyTestQueries["calendar"]})
	require.Equal(t, []string{"weather", "calendar"}, final.tools)
	operations := h.store.operations()

	router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
	ctx := withRoutingFacts(testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-2"}))
	request, err := cloneSemanticRequestForReplay(final.ctx.SemanticRequest)
	require.NoError(t, err)
	ctx.SemanticRequest = request
	ctx.FallbackRequest, err = cloneSemanticRequestForReplay(request)
	require.NoError(t, err)
	var sent []byte
	router.fallbackCaller = func(_ context.Context, model string, body []byte, _ map[string]string) ([]byte, int, error) {
		require.Equal(t, "model-fallback-2", model)
		sent = append([]byte(nil), body...)
		return []byte(anthropicFallbackAnswer), http.StatusOK, nil
	}

	response, err := router.handleResponseHeaders(upstreamStatus("503"), ctx)
	require.NoError(t, err)
	require.NotNil(t, response.GetImmediateResponse(), "the candidate answers for the failed primary")
	decoded, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, sent)
	require.NoError(t, err, "fallback body: %s", sent)
	names := make([]string, 0, len(decoded.Tools))
	for _, tool := range decoded.Tools {
		names = append(names, tool.Name)
	}
	require.Equal(t, []string{"weather", "calendar"}, names)
	require.Contains(t, string(sent), "9007199254740993")
	require.Equal(t, operations, h.store.operations(), "fallback must not read or write session state")
}
