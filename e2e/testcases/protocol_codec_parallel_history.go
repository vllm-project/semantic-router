package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

// Check the actual provider-bound payload: a permissive simulator returning
// 200 alone would not catch the invalid assistant/tool ordering from #4561.
func runResponsesParallelHistory(ctx context.Context, router, provider *fixtures.ServiceSession, model string) error {
	for _, stream := range []bool{false, true} {
		sessionID := "parallel-history-" + uuid.NewString()
		body := map[string]any{
			"model": model, "store": false, "stream": stream,
			"input": []map[string]any{
				{"role": "user", "content": "Summarize both weather results"},
				{"type": "function_call", "call_id": "call_a", "name": "lookup", "arguments": `{"query":"Paris"}`},
				{"type": "function_call", "call_id": "call_b", "name": "lookup", "arguments": `{"query":"London"}`},
				{"type": "function_call_output", "call_id": "call_a", "output": "sunny"},
				{"type": "function_call_output", "call_id": "call_b", "output": "rainy"},
			},
		}
		result, err := sendProtocolMatrixRaw(ctx, router, "/v1/responses", body, stream,
			map[string]string{"x-vsr-test-session-id": sessionID})
		if err != nil {
			return fmt.Errorf("parallel history stream=%t: %w", stream, err)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("parallel history stream=%t: HTTP %d: %s", stream, result.StatusCode, truncateString(string(result.Body), 500))
		}
		if stream {
			if streamErr := validateResponseAPIStreamingSSEBody(string(result.Body)); streamErr != nil {
				return streamErr
			}
		}
		raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
		if err != nil {
			return err
		}
		if historyErr := assertParallelHistoryProviderRequest(raw); historyErr != nil {
			return historyErr
		}
	}
	return nil
}

func assertParallelHistoryProviderRequest(raw []byte) error {
	var observation struct {
		Body struct {
			Messages []struct {
				Role       string          `json:"role"`
				Content    json.RawMessage `json:"content"`
				ToolCallID string          `json:"tool_call_id"`
				ToolCalls  []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"messages"`
		} `json:"body"`
	}
	if err := json.Unmarshal(raw, &observation); err != nil {
		return err
	}
	messages := observation.Body.Messages
	if len(messages) != 4 || messages[0].Role != "user" || messages[1].Role != "assistant" || len(messages[1].ToolCalls) != 2 {
		return fmt.Errorf("parallel history did not reach Chat as user/assistant(two calls)/tool/tool: %s", truncateString(string(raw), 1200))
	}
	for i, id := range []string{"call_a", "call_b"} {
		call, result := messages[1].ToolCalls[i], messages[i+2]
		arguments := []string{`{"query":"Paris"}`, `{"query":"London"}`}[i]
		output := []string{`"sunny"`, `"rainy"`}[i]
		if call.ID != id || call.Type != "function" || call.Function.Name != "lookup" || call.Function.Arguments != arguments ||
			result.Role != "tool" || result.ToolCallID != id || string(result.Content) != output {
			return fmt.Errorf("parallel history call/result %d changed: %s", i, truncateString(string(raw), 1200))
		}
	}
	return nil
}
