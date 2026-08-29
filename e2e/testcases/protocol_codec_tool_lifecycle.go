package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func runProtocolCodecToolLifecycle(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	model string,
	backendFormat string,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	tool := map[string]any{
		"type": "function", "name": "lookup", "description": "Look up a value",
		"parameters": map[string]any{
			"type":       "object",
			"properties": map[string]any{"query": map[string]any{"type": "string"}},
			"required":   []string{"query"},
		},
	}
	firstBody := map[string]any{
		"model": model, "input": "__mock_tool_call__", "store": false,
		"tools":       []any{tool},
		"tool_choice": map[string]any{"type": "function", "name": "lookup"},
	}
	firstBody["stream"] = true
	streamedFirst, err := sendProtocolMatrixRequest(ctx, session, "/v1/responses", firstBody, true)
	if err != nil {
		return fmt.Errorf("streaming tool-call turn: %w", err)
	}
	streamCall, err := decodeResponsesFunctionCallStream(streamedFirst)
	if err != nil {
		return fmt.Errorf("streaming tool-call turn: %w", err)
	}
	delete(firstBody, "stream")
	first, err := sendProtocolMatrixRequest(ctx, session, "/v1/responses", firstBody, false)
	if err != nil {
		return fmt.Errorf("tool-call turn: %w", err)
	}
	call, err := decodeResponsesFunctionCall(first)
	if err != nil {
		return err
	}
	if call != streamCall {
		return fmt.Errorf("buffered and streamed calls differ: buffered=%+v streamed=%+v", call, streamCall)
	}

	secondBody := map[string]any{
		"model": model, "store": false, "tools": []any{tool},
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "__mock_tool_call__"},
			map[string]any{
				"type": "function_call", "id": "item_mock_lookup", "call_id": call.CallID,
				"name": call.Name, "arguments": call.Arguments,
			},
			map[string]any{"type": "function_call_output", "call_id": call.CallID, "output": "sunny"},
		},
	}
	secondBody["stream"] = true
	streamedSecond, err := sendProtocolMatrixRequest(ctx, session, "/v1/responses", secondBody, true)
	if err != nil {
		return fmt.Errorf("streaming tool-result turn: %w", err)
	}
	if err := validateResponseAPIStreamingSSEBody(string(streamedSecond)); err != nil {
		return fmt.Errorf("streaming tool-result turn: %w", err)
	}
	if !strings.Contains(string(streamedSecond), "tool result accepted") {
		return fmt.Errorf("streaming tool-result turn lost output: %s", truncateString(string(streamedSecond), 1200))
	}
	delete(secondBody, "stream")
	second, err := sendProtocolMatrixRequest(ctx, session, "/v1/responses", secondBody, false)
	if err != nil {
		return fmt.Errorf("tool-result turn: %w", err)
	}
	if err := assertResponsesText(second, "tool result accepted"); err != nil {
		return fmt.Errorf("tool-result turn: %w", err)
	}
	if err := runChatClientToolLifecycle(ctx, session, model); err != nil {
		return fmt.Errorf("Chat Completions client: %w", err)
	}
	if err := runAnthropicClientToolLifecycle(ctx, session, model); err != nil {
		return fmt.Errorf("Anthropic Messages client: %w", err)
	}
	for _, stream := range []bool{false, true} {
		if err := runStoredResponsesToolContinuation(ctx, session, model, stream); err != nil {
			return fmt.Errorf("Responses stored tool continuation (stream=%t): %w", stream, err)
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"backend_format":            backendFormat,
			"client_formats":            3,
			"tool":                      call.Name,
			"inline_turns":              6,
			"stored_continuation_turns": 4,
			"streaming_turns":           8,
		})
	}
	return nil
}

func runStoredResponsesToolContinuation(
	ctx context.Context,
	session *fixtures.ServiceSession,
	model string,
	stream bool,
) error {
	store := true
	tool := map[string]any{
		"type": "function", "name": "lookup", "description": "Look up a value",
		"parameters": map[string]any{
			"type":       "object",
			"properties": map[string]any{"query": map[string]any{"type": "string"}},
			"required":   []string{"query"},
		},
	}
	firstBody := map[string]any{
		"model": model, "input": "__mock_tool_call__", "store": store,
		"tools":       []any{tool},
		"tool_choice": map[string]any{"type": "function", "name": "lookup"},
		"stream":      stream,
	}
	first, err := sendProtocolMatrixRequest(ctx, session, "/v1/responses", firstBody, stream)
	if err != nil {
		return fmt.Errorf("stored tool-call turn: %w", err)
	}
	var call responsesFunctionCall
	if stream {
		call, err = decodeResponsesFunctionCallStream(first)
	} else {
		call, err = decodeResponsesFunctionCall(first)
	}
	if err != nil {
		return fmt.Errorf("decode stored tool-call turn: %w", err)
	}
	responseID, err := decodeResponsesResponseID(first, stream)
	if err != nil {
		return fmt.Errorf("decode stored tool-call response ID: %w", err)
	}

	secondBody := map[string]any{
		"model": model, "store": false, "stream": stream,
		"previous_response_id": responseID,
		"tools":                []any{tool},
		"input": []any{map[string]any{
			"type": "function_call_output", "call_id": call.CallID, "output": "sunny",
		}},
	}
	second, err := sendProtocolMatrixRequest(ctx, session, "/v1/responses", secondBody, stream)
	if err != nil {
		return fmt.Errorf("stored tool-result turn: %w", err)
	}
	if stream {
		if err := validateResponseAPIStreamingSSEBody(string(second)); err != nil {
			return fmt.Errorf("stored tool-result stream: %w", err)
		}
		if !strings.Contains(string(second), "tool result accepted") {
			return fmt.Errorf("stored tool-result stream lost output: %s", truncateString(string(second), 1200))
		}
		return nil
	}
	if err := assertResponsesText(second, "tool result accepted"); err != nil {
		return fmt.Errorf("stored tool-result response: %w", err)
	}
	return nil
}

func decodeResponsesResponseID(body []byte, stream bool) (string, error) {
	if !stream {
		var response struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(body, &response); err != nil {
			return "", err
		}
		if response.ID == "" {
			return "", fmt.Errorf("buffered response has no id: %s", truncateString(string(body), 800))
		}
		return response.ID, nil
	}

	for _, frame := range strings.Split(string(body), "\n\n") {
		var data string
		for _, line := range strings.Split(frame, "\n") {
			if strings.HasPrefix(line, "data: ") {
				data = strings.TrimPrefix(line, "data: ")
				break
			}
		}
		if data == "" {
			continue
		}
		var event struct {
			Type     string `json:"type"`
			Response struct {
				ID string `json:"id"`
			} `json:"response"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return "", err
		}
		if event.Type == "response.completed" && event.Response.ID != "" {
			return event.Response.ID, nil
		}
	}
	return "", fmt.Errorf("stream has no completed response id: %s", truncateString(string(body), 1200))
}

type responsesFunctionCall struct {
	CallID    string
	Name      string
	Arguments string
}

func decodeResponsesFunctionCallStream(body []byte) (responsesFunctionCall, error) {
	stream := string(body)
	requiredEvents := []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added",
		"response.function_call_arguments.delta",
		"response.function_call_arguments.done",
		"response.output_item.done",
		"response.completed",
	}
	previousIndex := -1
	for _, event := range requiredEvents {
		index := strings.Index(stream, "event: "+event)
		if index < 0 {
			return responsesFunctionCall{}, fmt.Errorf("missing Responses tool event %q: %s", event, truncateString(stream, 1200))
		}
		if index < previousIndex {
			return responsesFunctionCall{}, fmt.Errorf("Responses tool event %q is out of order: %s", event, truncateString(stream, 1200))
		}
		previousIndex = index
	}
	for _, leaked := range []string{"chat.completion.chunk", "data: [DONE]", "event: message_start", `"type":"message_start"`} {
		if strings.Contains(stream, leaked) {
			return responsesFunctionCall{}, fmt.Errorf("backend stream format leaked %q: %s", leaked, truncateString(stream, 1200))
		}
	}
	if err := validateResponsesStreamEventShapes(stream); err != nil {
		return responsesFunctionCall{}, err
	}

	var call responsesFunctionCall
	var deltas strings.Builder
	deltaCount := 0
	doneCount := 0
	for _, frame := range strings.Split(stream, "\n\n") {
		var data string
		for _, line := range strings.Split(frame, "\n") {
			if strings.HasPrefix(line, "data: ") {
				data = strings.TrimPrefix(line, "data: ")
				break
			}
		}
		if data == "" {
			continue
		}
		var event struct {
			Type      string `json:"type"`
			ItemID    string `json:"item_id"`
			Delta     string `json:"delta"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
			Item      struct {
				Type      string `json:"type"`
				CallID    string `json:"call_id"`
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"item"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return responsesFunctionCall{}, fmt.Errorf("decode Responses tool stream event: %w", err)
		}
		switch event.Type {
		case "response.output_item.added":
			if event.Item.Type != "function_call" || event.Item.CallID == "" || event.Item.Name == "" {
				return responsesFunctionCall{}, fmt.Errorf("malformed Responses function_call item: %s", data)
			}
			call.CallID = event.Item.CallID
			call.Name = event.Item.Name
		case "response.function_call_arguments.delta":
			deltaCount++
			deltas.WriteString(event.Delta)
		case "response.function_call_arguments.done":
			doneCount++
			if event.Name != call.Name || event.ItemID == "" {
				return responsesFunctionCall{}, fmt.Errorf("Responses function arguments completion lost identity: %s", data)
			}
			call.Arguments = event.Arguments
		case "response.output_item.done":
			if event.Item.CallID != call.CallID || event.Item.Name != call.Name || event.Item.Arguments != call.Arguments {
				return responsesFunctionCall{}, fmt.Errorf("Responses completed function item changed identity or arguments: %s", data)
			}
		}
	}
	if call.CallID != "call_mock_lookup" || call.Name != "lookup" ||
		call.Arguments != `{"query":"weather"}` || deltaCount < 2 || doneCount != 1 {
		return responsesFunctionCall{}, fmt.Errorf("Responses tool stream lifecycle is incomplete: %s", truncateString(stream, 1200))
	}
	if deltas.String() != call.Arguments || !json.Valid([]byte(call.Arguments)) {
		return responsesFunctionCall{}, fmt.Errorf("Responses tool argument deltas do not reconstruct valid JSON: deltas=%q done=%q", deltas.String(), call.Arguments)
	}
	return call, nil
}

func decodeResponsesFunctionCall(body []byte) (responsesFunctionCall, error) {
	var response struct {
		Output []struct {
			Type      string `json:"type"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return responsesFunctionCall{}, fmt.Errorf("decode Responses tool call: %w", err)
	}
	for _, output := range response.Output {
		if output.Type == "function_call" && output.CallID == "call_mock_lookup" && output.Name == "lookup" &&
			output.Arguments == `{"query":"weather"}` && json.Valid([]byte(output.Arguments)) {
			return responsesFunctionCall{CallID: output.CallID, Name: output.Name, Arguments: output.Arguments}, nil
		}
	}
	return responsesFunctionCall{}, fmt.Errorf("Responses tool call is missing or malformed: %s", truncateString(string(body), 800))
}

func assertResponsesText(body []byte, contains string) error {
	var response struct {
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return err
	}
	for _, output := range response.Output {
		for _, content := range output.Content {
			if output.Type == "message" && content.Type == "output_text" && strings.Contains(content.Text, contains) {
				return nil
			}
		}
	}
	return fmt.Errorf("Responses output text does not contain %q: %s", contains, truncateString(string(body), 800))
}
