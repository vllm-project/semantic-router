package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

func runEmptyChatToolResults(ctx context.Context, router, provider *fixtures.ServiceSession, model string) error {
	for _, source := range protocolCodecE2EClients {
		if err := runEmptyChatToolResultsForClient(ctx, router, provider, source, model, model); err != nil {
			return err
		}
		if source.path == "/v1/chat/completions" {
			// The response-api profile resolves vllm-sr/auto to the Chat backend model.
			// Unlike direct model dispatch, this must exercise re-encoding.
			if err := runEmptyChatToolResultsForClient(ctx, router, provider, source, "vllm-sr/auto", model); err != nil {
				return err
			}
		}
	}
	return nil
}

func runEmptyChatToolResultsForClient(ctx context.Context, router, provider *fixtures.ServiceSession, source protocolCodecE2EClient, requestModel, providerModel string) error {
	for _, stream := range []bool{false, true} {
		for _, value := range []struct{ wire, want string }{{`[]`, `""`}, {`""`, `""`}, {`"result"`, `"result"`}} {
			body, err := emptyChatToolHistory(source.path, value.wire)
			if err != nil {
				return err
			}
			body["model"], body["stream"] = requestModel, stream
			sessionID := "empty-tool-" + uuid.NewString()
			result, err := sendProtocolMatrixRaw(ctx, router, source.path, body, stream, map[string]string{"x-vsr-test-session-id": sessionID})
			if err != nil {
				return err
			}
			if result.StatusCode != http.StatusOK {
				return fmt.Errorf("tool result (%s stream=%t content=%s): HTTP %d: %s", source.name, stream, value.wire, result.StatusCode, truncateString(string(result.Body), 500))
			}
			if stream {
				err = source.validateStream(result.Body, "tool result accepted")
			} else {
				err = source.validateBuffered(result.Body, "tool result accepted")
			}
			if err != nil {
				return err
			}
			raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
			if err != nil {
				return err
			}
			// The mock can accept invalid wire, so HTTP 200 alone is insufficient.
			want := value.want
			if source.path == "/v1/chat/completions" && requestModel == providerModel && !stream {
				// Buffered native Chat preserves valid content arrays. Streaming
				// fixtures omit include_usage, so dispatch adds it and re-encodes
				// even when the model is unchanged.
				want = value.wire
			}
			var observation struct {
				Body struct {
					Model string `json:"model"`
				} `json:"body"`
			}
			if decodeErr := json.Unmarshal(raw, &observation); decodeErr != nil {
				return decodeErr
			}
			if observation.Body.Model != providerModel {
				return fmt.Errorf("tool result model: got %q want %q (ingress %q)", observation.Body.Model, providerModel, requestModel)
			}
			if contractErr := assertChatToolResultContent(raw, want); contractErr != nil {
				return fmt.Errorf("tool result (%s model=%s stream=%t content=%s): %w", source.name, requestModel, stream, value.wire, contractErr)
			}
		}
	}
	return nil
}

func emptyChatToolHistory(path, content string) (map[string]any, error) {
	var fixture string
	switch path {
	case "/v1/chat/completions":
		fixture = `{"messages":[{"role":"user","content":"lookup"},{"role":"assistant","tool_calls":[{"id":"call_a","type":"function","function":{"name":"lookup","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_a","content":%s}]}`
	case "/v1/responses":
		fixture = `{"input":[{"role":"user","content":"lookup"},{"type":"function_call","call_id":"call_a","name":"lookup","arguments":"{}"},{"type":"function_call_output","call_id":"call_a","output":%s}]}`
	case "/v1/messages":
		fixture = `{"max_tokens":32,"messages":[{"role":"user","content":"lookup"},{"role":"assistant","content":[{"type":"tool_use","id":"call_a","name":"lookup","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_a","content":%s}]}]}`
	default:
		return nil, fmt.Errorf("unsupported tool history ingress %s", path)
	}
	var body map[string]any
	err := json.Unmarshal([]byte(fmt.Sprintf(fixture, content)), &body)
	return body, err
}

func assertChatToolResultContent(raw []byte, want string) error {
	var observation struct {
		Body struct {
			Messages []struct {
				Role    string          `json:"role"`
				CallID  string          `json:"tool_call_id"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		} `json:"body"`
	}
	if err := json.Unmarshal(raw, &observation); err != nil {
		return err
	}
	count := 0
	for _, message := range observation.Body.Messages {
		if message.Role != "tool" {
			continue
		}
		count++
		if message.CallID != "call_a" || string(message.Content) != want {
			return fmt.Errorf("tool result lost call ID or explicit content: call_id=%q content=%s want=%s", message.CallID, message.Content, want)
		}
	}
	if count != 1 {
		return fmt.Errorf("expected one tool result, got %d", count)
	}
	return nil
}
