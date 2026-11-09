package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("history-reset-protocols", pkgtestcases.TestCase{
		Description: "An enabled history_reset removes or keeps prior turns, tool exchanges included, for Chat, Anthropic Messages, and Responses clients",
		Tags:        []string{"plugin", "context", "history-reset", "protocol-codec"},
		Fn:          testHistoryResetProtocols,
	})
}

const (
	historyResetProtocolProbe  = "__HISTORY_RESET_PROTOCOL_PROBE__"
	historyResetProtocolSystem = "Answer briefly."
	historyResetProtocolCallID = "call_weather"

	// The provider-protocols profile's Anthropic Messages provider mocker.
	historyResetProtocolBackendNamespace = "provider-protocols-system"
	historyResetProtocolBackendService   = "provider-mocker"
	historyResetProtocolBackendPort      = "8000"
)

var (
	historyResetChangeTurn       = "Unrelated question: how do I renew a passport? " + historyResetProtocolProbe
	historyResetContinuationTurn = "As you said earlier, it is sunny in Paris; should I bring a jacket tonight? " + historyResetProtocolProbe
)

// historyResetProtocolRequest builds one client request: a system prompt, a
// prior turn that contains a complete tool exchange, and the live turn.
type historyResetProtocolRequest struct {
	name string
	path string
	body func(live string) map[string]any
}

func historyResetProtocolRequests() []historyResetProtocolRequest {
	return []historyResetProtocolRequest{
		{name: "chat", path: "/v1/chat/completions", body: historyResetChatBody},
		{name: "anthropic", path: "/v1/messages", body: historyResetAnthropicBody},
		{name: "responses", path: "/v1/responses", body: historyResetResponsesBody},
	}
}

func historyResetChatBody(live string) map[string]any {
	return map[string]any{
		"model": "vllm-sr/auto",
		"tools": []any{map[string]any{"type": "function", "function": map[string]any{
			"name": "lookup", "description": "Look up a value",
			"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
		}}},
		"messages": []any{
			map[string]any{"role": "system", "content": historyResetProtocolSystem},
			map[string]any{"role": "user", "content": "What is the weather in Paris today?"},
			map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{map[string]any{
				"id": historyResetProtocolCallID, "type": "function",
				"function": map[string]any{"name": "lookup", "arguments": `{"city":"Paris"}`},
			}}},
			map[string]any{"role": "tool", "tool_call_id": historyResetProtocolCallID, "content": "Sunny, 21C"},
			map[string]any{"role": "assistant", "content": "It is sunny in Paris, 21C."},
			map[string]any{"role": "user", "content": live},
		},
	}
}

func historyResetAnthropicBody(live string) map[string]any {
	return map[string]any{
		"model": "vllm-sr/auto", "max_tokens": 64, "system": historyResetProtocolSystem,
		"tools": []any{map[string]any{
			"name": "lookup", "description": "Look up a value",
			"input_schema": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
		}},
		"messages": []any{
			map[string]any{"role": "user", "content": "What is the weather in Paris today?"},
			map[string]any{"role": "assistant", "content": []any{map[string]any{
				"type": "tool_use", "id": historyResetProtocolCallID, "name": "lookup",
				"input": map[string]any{"city": "Paris"},
			}}},
			map[string]any{"role": "user", "content": []any{map[string]any{
				"type": "tool_result", "tool_use_id": historyResetProtocolCallID, "content": "Sunny, 21C",
			}}},
			map[string]any{"role": "assistant", "content": "It is sunny in Paris, 21C."},
			map[string]any{"role": "user", "content": live},
		},
	}
}

func historyResetResponsesBody(live string) map[string]any {
	return map[string]any{
		"model": "vllm-sr/auto", "store": false, "instructions": historyResetProtocolSystem,
		"tools": []any{map[string]any{
			"type": "function", "name": "lookup", "description": "Look up a value",
			"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
		}},
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "What is the weather in Paris today?"},
			map[string]any{
				"type": "function_call", "id": "item_weather", "call_id": historyResetProtocolCallID,
				"name": "lookup", "arguments": `{"city":"Paris"}`,
			},
			map[string]any{"type": "function_call_output", "call_id": historyResetProtocolCallID, "output": "Sunny, 21C"},
			map[string]any{"type": "message", "role": "assistant", "content": "It is sunny in Paris, 21C."},
			map[string]any{"type": "message", "role": "user", "content": live},
		},
	}
}

func testHistoryResetProtocols(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	backend, err := fixtures.OpenServiceEndpointSession(ctx, client, opts,
		historyResetProtocolBackendNamespace, historyResetProtocolBackendService, historyResetProtocolBackendPort)
	if err != nil {
		return err
	}
	defer backend.Close()

	details := map[string]interface{}{}
	for _, protocol := range historyResetProtocolRequests() {
		for _, change := range []bool{true, false} {
			live, name := historyResetContinuationTurn, protocol.name+" continuation"
			if change {
				live, name = historyResetChangeTurn, protocol.name+" change"
			}
			body, dispatchErr := historyResetProtocolDispatch(ctx, session, backend, name, protocol.path, protocol.body(live))
			if dispatchErr != nil {
				return dispatchErr
			}
			if assertErr := assertHistoryResetAnthropicBody(body, live, change); assertErr != nil {
				return fmt.Errorf("%s: %w", name, assertErr)
			}
			details[name] = len(historyResetMessages(body))
		}
	}

	// Responses history retained behind previous_response_id is resolved
	// before the reset, so the reset removes it from the provider request
	// and dispatch does not prepend it again.
	first, err := historyResetPost(ctx, session, "/v1/responses", map[string]any{
		"model": "vllm-sr/auto", "store": true, "instructions": historyResetProtocolSystem,
		"input": "What is the weather in Paris today?",
	}, map[string]string{"anthropic-version": "2023-06-01"})
	if err != nil {
		return fmt.Errorf("responses chain first turn: %w", err)
	}
	if first.StatusCode != http.StatusOK {
		return fmt.Errorf("responses chain first turn: status %d", first.StatusCode)
	}
	var created struct {
		ID string `json:"id"`
	}
	if json.Unmarshal(first.Body, &created) != nil || created.ID == "" {
		return fmt.Errorf("responses chain first turn returned no response id")
	}
	body, err := historyResetProtocolDispatch(ctx, session, backend, "responses previous_response_id change",
		"/v1/responses", map[string]any{
			"model": "vllm-sr/auto", "store": false, "instructions": historyResetProtocolSystem,
			"previous_response_id": created.ID, "input": historyResetChangeTurn,
		})
	if err != nil {
		return err
	}
	upstream := historyResetMessages(body)
	if len(upstream) != 1 || !strings.Contains(historyResetMessageText(upstream[0]), historyResetChangeTurn) {
		return fmt.Errorf("responses previous_response_id change: provider received %d messages, want only the live turn", len(upstream))
	}
	details["responses previous_response_id change"] = len(upstream)

	if opts.SetDetails != nil {
		opts.SetDetails(details)
	}
	return nil
}

// historyResetProtocolDispatch sends one request and returns the native body
// the provider received for it.
func historyResetProtocolDispatch(
	ctx context.Context,
	session *fixtures.ServiceSession,
	backend *fixtures.ServiceSession,
	name string,
	path string,
	payload map[string]any,
) (map[string]any, error) {
	observationID := historyResetObservationID(strings.ReplaceAll(name, " ", "-"))
	response, err := historyResetPost(ctx, session, path, payload, map[string]string{
		historyResetObservationHeader: observationID,
		"anthropic-version":           "2023-06-01",
	})
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: status %d", name, response.StatusCode)
	}
	body, dispatched, err := historyResetObserve(ctx, backend, observationID)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	if !dispatched {
		return nil, fmt.Errorf("%s: the request never reached the provider", name)
	}
	return body, nil
}

// assertHistoryResetAnthropicBody checks the native Anthropic Messages body.
// The system prompt always survives. A change leaves only the live turn; a
// continuation keeps the prior turn with its tool_use and tool_result linked.
func assertHistoryResetAnthropicBody(body map[string]any, live string, change bool) error {
	if system := fmt.Sprint(body["system"]); !strings.Contains(system, historyResetProtocolSystem) {
		return fmt.Errorf("the system prompt did not survive")
	}
	messages := historyResetMessages(body)
	if len(messages) == 0 || !strings.Contains(historyResetMessageText(messages[len(messages)-1]), live) {
		return fmt.Errorf("the live turn is not the last provider-bound message")
	}
	if change {
		if len(messages) != 1 {
			return fmt.Errorf("provider received %d messages, want only the live turn", len(messages))
		}
		return nil
	}
	if len(messages) != 5 {
		return fmt.Errorf("provider received %d messages, want the prior turn and the live turn (5)", len(messages))
	}
	use, result := historyResetAnthropicBlock(messages[1], "tool_use"), historyResetAnthropicBlock(messages[2], "tool_result")
	if use == nil || result == nil || use["id"] != historyResetProtocolCallID ||
		result["tool_use_id"] != historyResetProtocolCallID {
		return fmt.Errorf("the prior tool exchange was not kept linked")
	}
	return nil
}

func historyResetAnthropicBlock(message map[string]any, kind string) map[string]any {
	blocks, _ := message["content"].([]any)
	for _, block := range blocks {
		if part, ok := block.(map[string]any); ok && part["type"] == kind {
			return part
		}
	}
	return nil
}
