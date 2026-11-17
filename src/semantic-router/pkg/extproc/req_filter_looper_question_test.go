package extproc

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A user turn sent as content blocks still reaches ReMoM's final round as the
// question: the user's own words, not a <system-reminder> block a coding
// agent puts next to them. A turn that holds only a reminder, as a Claude
// Code tool-result turn does, leaves the question to the turn before it.
func TestReMoMFinalRoundAsksTheQuestionOfAContentBlockTurn(t *testing.T) {
	for _, tc := range []struct {
		name               string
		format             llmprotocol.WireFormat
		body               string
		question, reminder string
	}{
		{
			"messages", llmprotocol.AnthropicMessagesV1, `{"model":"vllm-sr/remom","max_tokens":64,"messages":[{"role":"user","content":[` +
				`{"type":"text","text":"<system-reminder>Keep answers short.</system-reminder>"},{"type":"text","text":"What is the capital of Australia?"}]}]}`,
			"What is the capital of Australia?", "Keep answers short.",
		},
		{
			"chat", llmprotocol.OpenAIChatV1, `{"model":"vllm-sr/remom","messages":[{"role":"user","content":[` +
				`{"type":"text","text":"What is the capital of Australia?"},{"type":"text","text":"Answer in one word."}]}]}`,
			"What is the capital of Australia?\nAnswer in one word.", "",
		},
		{
			"messages_tool_loop", llmprotocol.AnthropicMessagesV1, `{"model":"vllm-sr/remom","max_tokens":64,` +
				`"tools":[{"name":"Bash","input_schema":{"type":"object"}}],"messages":[` +
				`{"role":"user","content":[{"type":"text","text":"<system-reminder>Keep answers short.</system-reminder>"},{"type":"text","text":"Fix the failing test"}]},` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"tu1","name":"Bash","input":{"cmd":"ls"}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu1","content":"a.go b.go"},` +
				`{"type":"text","text":"<system-reminder>The task tools have not been used recently.</system-reminder>"}]}]}`,
			"Fix the failing test", "The task tools have not been used recently.",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var bodies []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				mu.Lock()
				bodies = append(bodies, string(raw))
				mu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"c","object":"chat.completion","created":0,"model":"model-a","choices":[{"index":0,` +
					`"message":{"role":"assistant","content":"Canberra"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`))
			}))
			defer server.Close()
			router := &OpenAIRouter{Config: &config.RouterConfig{Looper: config.LooperConfig{Endpoint: server.URL}}}
			ctx := &RequestContext{SourceFormat: tc.format, TraceContext: context.Background(), Headers: map[string]string{}}
			request, rejected := router.prepareProtocolRequest([]byte(tc.body), ctx)
			require.Nil(t, rejected)
			decision := &config.Decision{Name: "remom", ModelRefs: []config.ModelRef{{Model: "model-a"}}, Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmReMoM, ReMoM: &config.ReMoMAlgorithmConfig{BreadthSchedule: []int{1}, ModelDistribution: "first_only"},
			}}

			_, err := router.handleLooperExecution(context.Background(), request, decision, ctx)
			require.NoError(t, err)
			require.Len(t, bodies, 2, "one round-1 call and the final synthesis")
			var final struct {
				Messages []struct {
					Content json.RawMessage `json:"content"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal([]byte(bodies[1]), &final))
			var prompt string
			require.NoError(t, json.Unmarshal(final.Messages[len(final.Messages)-1].Content, &prompt))
			require.Contains(t, prompt, "Original Problem:\n"+tc.question+"\n")
			if tc.reminder != "" {
				require.NotContains(t, prompt, tc.reminder)
			}
		})
	}
}
