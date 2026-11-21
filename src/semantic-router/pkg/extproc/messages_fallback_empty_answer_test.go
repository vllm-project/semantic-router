package extproc

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// emptyAnthropicUsage is the usage Claude Haiku 5.5 reported for a prompt
// suggestion it answered with nothing.
var emptyAnthropicUsage = map[string]any{
	"input_tokens": 506, "cache_creation_input_tokens": 14, "cache_read_input_tokens": 3000, "output_tokens": 2,
}

// writeEmptyAnthropicAnswer answers with an end_turn message and no content
// block: message_start, message_delta and message_stop when streamed.
func writeEmptyAnthropicAnswer(w http.ResponseWriter, stream bool) {
	message := map[string]any{
		"id": "msg-empty", "type": "message", "role": "assistant", "model": "empty-model",
		"content": []any{}, "stop_reason": "end_turn", "stop_sequence": nil, "usage": emptyAnthropicUsage,
	}
	if !stream {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(message)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	start := map[string]any{}
	for key, value := range message {
		start[key] = value
	}
	start["stop_reason"] = nil
	for _, event := range []map[string]any{
		{"type": "message_start", "message": start},
		{"type": "message_delta", "delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil}, "usage": emptyAnthropicUsage},
		{"type": "message_stop"},
	} {
		data, _ := json.Marshal(event)
		_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event["type"], data)
	}
}

// claudeCodeThinking is the thinking block Claude Code 2.1.325 sends.
const claudeCodeThinking = `"thinking":{"type":"adaptive","display":"updates"}`

// A fallback candidate whose model has no reasoning family gets the client's
// thinking as the first choice for that model gets it, so it answers instead
// of being skipped for a display left without its mode.
func TestMessagesFallbackKeepsTheClientsThinkingForAModelWithoutAFamily(t *testing.T) {
	h := newServedModelHarness(t)
	body := func(text string) string {
		return `{"model":"vllm-sr/auto","max_tokens":50,` + claudeCodeThinking +
			`,"messages":[{"role":"user","content":"` + text + `"}]}`
	}
	first := h.send(t, "/v1/messages", body("PLAINMSG question"))
	require.Equal(t, []string{"plain-msg"}, first.called)

	handoff := h.send(t, "/v1/messages", body("HANDOFFMSG question"))
	require.Equal(t, http.StatusOK, handoff.status, "%s", handoff.body)
	require.Equal(t, []string{"fail-model", "plain-msg"}, handoff.called)
	require.Equal(t, first.bodies[0]["thinking"], handoff.bodies[1]["thinking"],
		"the candidate's thinking differs from what the same model gets as first choice")
	require.Equal(t, "plain-msg", handoff.record.SelectedModel)
	outcomes := fallbackOutcomes(handoff.record)
	require.Len(t, outcomes, 1)
	require.Equal(t, "completed", outcomes[0].Verdict)
}

// Anthropic answers some requests with end_turn and no content block. That is
// an empty completion, not an upstream failure: every client protocol gets a
// successful empty answer, and the record completes with its usage and cost.
func TestAnEmptyEndTurnAnswerIsAnEmptyCompletion(t *testing.T) {
	h := newServedModelHarness(t)
	for _, client := range []struct{ path, body string }{
		{"/v1/messages", `{"model":"vllm-sr/auto","max_tokens":64,"messages":[{"role":"user","content":"SILENTANSWER"}]}`},
		{"/v1/chat/completions", `{"model":"vllm-sr/auto","max_tokens":64,"messages":[{"role":"user","content":"SILENTANSWER"}]}`},
		{"/v1/responses", `{"model":"vllm-sr/auto","max_output_tokens":64,"input":"SILENTANSWER"}`},
	} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%v", client.path, stream), func(t *testing.T) {
				request := client.body
				if stream {
					request = strings.Replace(request, `"max`, `"stream":true,"max`, 1)
				}
				call := h.send(t, client.path, request)
				require.Equal(t, []string{"empty-model"}, call.called)
				require.Equal(t, http.StatusOK, call.status, "%s", call.body)
				require.NotContains(t, string(call.body), `"error":{`)
				require.NotContains(t, string(call.body), "event: error")
				requireEmptyCompletion(t, client.path, stream, call.body)
				require.Equal(t, routerreplay.LifecycleCompleted, call.record.LifecycleState, call.record.TerminalReason)
				require.NotNil(t, call.record.CompletionTokens)
				require.Equal(t, 2, *call.record.CompletionTokens)
				require.NotNil(t, call.record.ActualCost)
				require.Positive(t, *call.record.ActualCost)
			})
		}
	}
}

// requireEmptyCompletion checks that the client got a finished answer with no
// text in its own protocol: an empty Chat message, as OpenAI sends one, and a
// completed Responses object with no output.
func requireEmptyCompletion(t *testing.T, path string, stream bool, body []byte) {
	t.Helper()
	text := string(body)
	switch {
	case path == "/v1/messages" && stream:
		require.Contains(t, text, "event: message_stop")
		require.Contains(t, text, `"stop_reason":"end_turn"`)
	case path == "/v1/messages":
		require.Contains(t, text, `"stop_reason":"end_turn"`)
		require.Contains(t, text, `"content":[]`)
	case path == "/v1/chat/completions" && stream:
		require.Contains(t, text, `"finish_reason":"stop"`)
		require.Contains(t, text, "data: [DONE]")
	case path == "/v1/chat/completions":
		var answer struct {
			Choices []struct {
				FinishReason string `json:"finish_reason"`
				Message      struct {
					Content *string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		require.NoError(t, json.Unmarshal(body, &answer), text)
		require.Len(t, answer.Choices, 1, text)
		require.Equal(t, "stop", answer.Choices[0].FinishReason)
		require.NotNil(t, answer.Choices[0].Message.Content, text)
		require.Empty(t, *answer.Choices[0].Message.Content)
	case stream:
		require.Contains(t, text, "event: response.completed")
	default:
		require.Contains(t, text, `"status":"completed"`)
		require.Contains(t, text, `"output":[]`)
	}
}
