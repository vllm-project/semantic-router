package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	responsesCustomFixtureMarker = "__mock_responses_custom_tool__"
	responsesCustomHistoryInput  = "*** Begin Patch\n+hello\n*** End Patch"
	responsesCustomProviderInput = "*** Begin Patch\n+provider\n*** End Patch"
	responsesCustomResult        = "Success"
)

func init() {
	pkgtestcases.Register("protocol-codec-responses-custom-tool-loop", pkgtestcases.TestCase{
		Description: "Responses custom tool calls, results, grammar and verbosity reach Chat and Responses providers in buffered and streaming requests",
		Tags:        []string{"protocol-codec", "response-api", "agents", "tools", "streaming"},
		Fn:          testProtocolCodecResponsesCustomToolLoop,
	})
	pkgtestcases.Register("protocol-codec-responses-verbosity-anthropic", pkgtestcases.TestCase{
		Description: "Responses verbosity is dropped with a warning when routed to Messages in buffered and streaming requests",
		Tags:        []string{"protocol-codec", "response-api", "anthropic", "streaming"},
		Fn:          testProtocolCodecResponsesVerbosityAnthropic,
	})
}

func testProtocolCodecResponsesCustomToolLoop(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	for _, backend := range []struct {
		name, model, format string
	}{
		{"chat", chatBackendModel, "openai.chat.v1"},
		{"responses", nativeResponsesBackendModel, "openai.responses.v1"},
	} {
		provider, err := openProtocolCodecProviderSession(ctx, client, opts, backend.format)
		if err != nil {
			return err
		}
		for _, stream := range []bool{false, true} {
			marker := responsesCustomFixtureMarker + " Responses custom tool loop " + backend.name
			sessionID := "responses-custom-" + uuid.NewString()
			body := map[string]any{
				"model": backend.model, "stream": stream, "store": false,
				"text": map[string]any{"verbosity": "low"},
				"tools": []any{map[string]any{
					"type": "custom", "name": "apply_patch", "description": "Apply a patch",
					"format": map[string]any{"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
				}},
				"tool_choice": map[string]any{"type": "custom", "name": "apply_patch"},
				"input": []any{
					map[string]any{"role": "user", "content": marker},
					map[string]any{
						"type": "custom_tool_call", "id": "item_1", "call_id": "call_1",
						"name": "apply_patch", "input": responsesCustomHistoryInput, "status": "completed",
					},
					map[string]any{"type": "custom_tool_call_output", "call_id": "call_1", "output": responsesCustomResult, "status": "completed"},
				},
			}
			result, requestErr := sendProtocolMatrixRaw(ctx, session, "/v1/responses", body, stream,
				map[string]string{"x-vsr-test-session-id": sessionID})
			if requestErr != nil {
				provider.Close()
				return fmt.Errorf("%s stream=%t request: %w", backend.name, stream, requestErr)
			}
			if result.StatusCode != http.StatusOK {
				provider.Close()
				return fmt.Errorf("%s stream=%t HTTP %d: %s", backend.name, stream, result.StatusCode, truncateString(string(result.Body), 700))
			}
			if stream {
				err = assertResponsesCustomToolStream(result.Body)
			} else {
				err = assertResponsesCustomToolBody(result.Body)
			}
			if err != nil {
				provider.Close()
				return fmt.Errorf("%s stream=%t response: %w", backend.name, stream, err)
			}
			raw, observationErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
			if observationErr != nil {
				provider.Close()
				return observationErr
			}
			if err := assertResponsesCustomProviderRequest(raw, backend.name, marker, stream); err != nil {
				provider.Close()
				return fmt.Errorf("%s stream=%t provider request: %w", backend.name, stream, err)
			}
		}
		provider.Close()
	}
	return nil
}

func testProtocolCodecResponsesVerbosityAnthropic(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, "anthropic.messages.v1")
	if err != nil {
		return err
	}
	defer provider.Close()

	for _, stream := range []bool{false, true} {
		marker := protocolCodecAnthropicProbe + " Responses verbosity probe"
		sessionID := "responses-verbosity-anthropic-" + uuid.NewString()
		result, requestErr := sendProtocolMatrixRaw(ctx, session, "/v1/responses", map[string]any{
			"model": "MoM", "input": marker, "stream": stream,
			"text": map[string]any{"verbosity": "low"},
		}, stream, map[string]string{"x-vsr-test-session-id": sessionID})
		if requestErr != nil {
			return fmt.Errorf("messages stream=%t request: %w", stream, requestErr)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("messages stream=%t HTTP %d: %s", stream, result.StatusCode, truncateString(string(result.Body), 500))
		}
		if stream {
			err = validateResponsesProtocolStream(result.Body, protocolCodecAnthropicReply)
		} else {
			err = assertResponsesBody(result.Body, protocolCodecAnthropicReply)
		}
		if err != nil {
			return fmt.Errorf("messages stream=%t response: %w", stream, err)
		}
		if warnings := result.Headers.Get("x-vsr-protocol-warnings"); !hasProtocolFieldDiagnostic(warnings, "dropped", "text.verbosity") {
			return fmt.Errorf("messages stream=%t did not report dropped text.verbosity: %q", stream, warnings)
		}
		if err := verifyProviderSimulatorRequest(ctx, provider, sessionID, "anthropic.messages.v1", marker); err != nil {
			return fmt.Errorf("messages stream=%t provider request: %w", stream, err)
		}
		raw, observationErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
		if observationErr != nil {
			return observationErr
		}
		var receipt struct {
			Body map[string]json.RawMessage `json:"body"`
		}
		if json.Unmarshal(raw, &receipt) != nil {
			return fmt.Errorf("messages stream=%t provider receipt is invalid: %s", stream, truncateString(string(raw), 500))
		}
		if _, leaked := receipt.Body["verbosity"]; leaked {
			return fmt.Errorf("messages stream=%t received OpenAI verbosity: %s", stream, truncateString(string(raw), 500))
		}
	}
	return nil
}

func assertResponsesCustomProviderRequest(raw []byte, backend, marker string, stream bool) error {
	var receipt struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(raw, &receipt); err != nil {
		return err
	}
	if len(receipt.Body) == 0 || !jsonContainsText(receipt.Body, marker) {
		return fmt.Errorf("provider receipt does not belong to this turn: %s", truncateString(string(raw), 650))
	}
	// Buffered requests omit stream:false in both upstream wire formats.
	providerStream := false
	if rawStream, present := receipt.Body["stream"]; present {
		if err := json.Unmarshal(rawStream, &providerStream); err != nil {
			return fmt.Errorf("provider request stream is invalid: %w", err)
		}
	}
	if providerStream != stream {
		return fmt.Errorf("provider request stream mismatch: %s", truncateString(string(raw), 650))
	}
	var tools []map[string]json.RawMessage
	if err := json.Unmarshal(receipt.Body["tools"], &tools); err != nil || len(tools) != 1 {
		return fmt.Errorf("provider lost custom tool: %s", receipt.Body["tools"])
	}
	if backend == "chat" {
		if string(tools[0]["type"]) != `"custom"` {
			return fmt.Errorf("chat provider lost custom tool kind: %s", receipt.Body["tools"])
		}
		var nested struct {
			Name   string `json:"name"`
			Format struct {
				Grammar struct {
					Syntax string `json:"syntax"`
				} `json:"grammar"`
			} `json:"format"`
		}
		if json.Unmarshal(tools[0]["custom"], &nested) != nil || nested.Name != "apply_patch" || nested.Format.Grammar.Syntax != "lark" {
			return fmt.Errorf("chat provider lost custom grammar: %s", receipt.Body["tools"])
		}
		var choice struct {
			Type   string `json:"type"`
			Custom struct {
				Name string `json:"name"`
			} `json:"custom"`
		}
		if json.Unmarshal(receipt.Body["tool_choice"], &choice) != nil || choice.Type != "custom" || choice.Custom.Name != "apply_patch" ||
			string(receipt.Body["verbosity"]) != `"low"` {
			return fmt.Errorf("chat provider lost custom call, result or verbosity: %s", truncateString(string(raw), 700))
		}
		var messages []struct {
			Role       string          `json:"role"`
			Content    json.RawMessage `json:"content"`
			ToolCallID string          `json:"tool_call_id"`
			ToolCalls  []struct {
				ID     string `json:"id"`
				Type   string `json:"type"`
				Custom struct {
					Name  string `json:"name"`
					Input string `json:"input"`
				} `json:"custom"`
			} `json:"tool_calls"`
		}
		if json.Unmarshal(receipt.Body["messages"], &messages) != nil || len(messages) != 3 ||
			messages[1].Role != "assistant" || len(messages[1].ToolCalls) != 1 ||
			messages[1].ToolCalls[0].ID != "call_1" || messages[1].ToolCalls[0].Type != "custom" ||
			messages[1].ToolCalls[0].Custom.Name != "apply_patch" ||
			messages[1].ToolCalls[0].Custom.Input != responsesCustomHistoryInput ||
			messages[2].Role != "tool" || messages[2].ToolCallID != "call_1" ||
			decodeCustomToolResultText(messages[2].Content) != responsesCustomResult {
			return fmt.Errorf("chat provider lost exact custom input or output: %s", truncateString(string(raw), 900))
		}
		return nil
	}
	if string(tools[0]["type"]) != `"custom"` {
		return fmt.Errorf("responses provider lost custom tool kind: %s", receipt.Body["tools"])
	}
	var format struct{ Type, Syntax, Definition string }
	if json.Unmarshal(tools[0]["format"], &format) != nil || format.Type != "grammar" || format.Syntax != "lark" || format.Definition == "" {
		return fmt.Errorf("responses provider lost flattened grammar: %s", receipt.Body["tools"])
	}
	var choice struct{ Type, Name string }
	if json.Unmarshal(receipt.Body["tool_choice"], &choice) != nil || choice.Type != "custom" || choice.Name != "apply_patch" {
		return fmt.Errorf("responses provider lost named custom choice: %s", receipt.Body["tool_choice"])
	}
	var items []map[string]json.RawMessage
	if json.Unmarshal(receipt.Body["input"], &items) != nil || len(items) != 3 ||
		string(items[1]["type"]) != `"custom_tool_call"` || string(items[1]["call_id"]) != `"call_1"` ||
		string(items[1]["name"]) != `"apply_patch"` || decodeCustomToolResultText(items[1]["input"]) != responsesCustomHistoryInput ||
		string(items[2]["type"]) != `"custom_tool_call_output"` || string(items[2]["call_id"]) != `"call_1"` ||
		decodeCustomToolResultText(items[2]["output"]) != responsesCustomResult ||
		!jsonContainsText(receipt.Body, `"verbosity":"low"`) {
		return fmt.Errorf("responses provider lost custom call, result or verbosity: %s", truncateString(string(raw), 700))
	}
	return nil
}

type responsesCustomOutputItem struct {
	Type   string `json:"type"`
	ID     string `json:"id"`
	CallID string `json:"call_id"`
	Name   string `json:"name"`
	Input  string `json:"input"`
}

func assertResponsesCustomItem(item responsesCustomOutputItem) error {
	if item.Type != "custom_tool_call" || item.ID == "" || item.CallID != "call_mock_patch" ||
		item.Name != "apply_patch" || item.Input != responsesCustomProviderInput {
		return fmt.Errorf("public custom tool item lost exact call or freeform input: %+v", item)
	}
	return nil
}

func assertResponsesCustomToolBody(body []byte) error {
	var response struct {
		Object string                      `json:"object"`
		Status string                      `json:"status"`
		Output []responsesCustomOutputItem `json:"output"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return err
	}
	if response.Object != "response" || response.Status != "completed" || len(response.Output) != 1 {
		return fmt.Errorf("public Responses body lost custom tool output: %s", truncateString(string(body), 750))
	}
	return assertResponsesCustomItem(response.Output[0])
}

func assertResponsesCustomToolStream(body []byte) error {
	stream := string(body)
	if err := validateOrderedStreamMarkers(stream, []string{
		"event: response.created", "event: response.in_progress", "event: response.output_item.added",
		"event: response.custom_tool_call_input.delta", "event: response.custom_tool_call_input.done",
		"event: response.output_item.done", "event: response.completed",
	}); err != nil {
		return err
	}
	if err := rejectStreamFragments(stream, "Responses", []string{"chat.completion.chunk", "data: [DONE]"}); err != nil {
		return err
	}
	if err := validateResponsesStreamEventShapes(stream); err != nil {
		return err
	}
	var itemID, callID string
	var input strings.Builder
	var deltas, completions, outputDone, responseDone int
	for _, data := range protocolSSEDataFrames(body) {
		var event struct {
			Type        string                    `json:"type"`
			ItemID      string                    `json:"item_id"`
			OutputIndex *int                      `json:"output_index"`
			Delta       string                    `json:"delta"`
			Input       string                    `json:"input"`
			Item        responsesCustomOutputItem `json:"item"`
			Response    struct {
				Status string                      `json:"status"`
				Output []responsesCustomOutputItem `json:"output"`
			} `json:"response"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return fmt.Errorf("decode public custom tool SSE event: %w", err)
		}
		switch event.Type {
		case "response.output_item.added":
			if itemID != "" || event.OutputIndex == nil || *event.OutputIndex != 0 ||
				event.Item.Type != "custom_tool_call" || event.Item.ID == "" ||
				event.Item.CallID != "call_mock_patch" || event.Item.Name != "apply_patch" || event.Item.Input != "" {
				return fmt.Errorf("invalid public custom tool added event: %s", data)
			}
			itemID, callID = event.Item.ID, event.Item.CallID
		case "response.custom_tool_call_input.delta":
			if itemID == "" || event.ItemID != itemID || event.OutputIndex == nil || *event.OutputIndex != 0 || event.Delta == "" {
				return fmt.Errorf("invalid public custom tool delta event: %s", data)
			}
			input.WriteString(event.Delta)
			deltas++
		case "response.custom_tool_call_input.done":
			if event.ItemID != itemID || event.OutputIndex == nil || *event.OutputIndex != 0 ||
				event.Input != responsesCustomProviderInput || input.String() != event.Input {
				return fmt.Errorf("public custom tool completion differs from streamed input: %s", data)
			}
			completions++
		case "response.output_item.done":
			if event.OutputIndex == nil || *event.OutputIndex != 0 || event.Item.ID != itemID || event.Item.CallID != callID {
				return fmt.Errorf("public custom tool completed item changed identity: %s", data)
			}
			if err := assertResponsesCustomItem(event.Item); err != nil {
				return err
			}
			outputDone++
		case "response.completed":
			if event.Response.Status != "completed" || len(event.Response.Output) != 1 ||
				event.Response.Output[0].ID != itemID {
				return fmt.Errorf("public Responses completion lost custom tool: %s", data)
			}
			if err := assertResponsesCustomItem(event.Response.Output[0]); err != nil {
				return err
			}
			responseDone++
		}
	}
	if deltas < 2 || completions != 1 || outputDone != 1 || responseDone != 1 {
		return fmt.Errorf("public custom tool stream incomplete: deltas=%d input_done=%d item_done=%d response_done=%d: %s",
			deltas, completions, outputDone, responseDone, truncateString(stream, 1200))
	}
	return nil
}

func decodeCustomToolResultText(raw json.RawMessage) string {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	var output strings.Builder
	for _, part := range parts {
		output.WriteString(part.Text)
	}
	return output.String()
}

func jsonContainsText(body map[string]json.RawMessage, substring string) bool {
	encoded, _ := json.Marshal(body)
	return strings.Contains(string(encoded), substring)
}
