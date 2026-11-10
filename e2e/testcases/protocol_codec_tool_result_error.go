package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

func runToolResultErrorPolicy(ctx context.Context, session, provider *fixtures.ServiceSession, model, backendFormat string) error {
	for _, stream := range []bool{false, true} {
		for _, isError := range []bool{false, true} {
			body := map[string]any{
				"model": model, "max_tokens": 64, "stream": stream,
				"messages": []any{
					map[string]any{"role": "user", "content": "Read the record"},
					map[string]any{"role": "assistant", "content": []any{
						map[string]any{"type": "tool_use", "id": "call_1", "name": "lookup", "input": map[string]any{"query": "record"}},
					}},
					map[string]any{"role": "user", "content": []any{
						map[string]any{"type": "tool_result", "tool_use_id": "call_1", "content": "[]", "is_error": isError},
					}},
				},
			}
			sessionID := "tool-result-error-" + uuid.NewString()
			result, err := sendProtocolMatrixRaw(ctx, session, "/v1/messages", body, stream,
				map[string]string{"x-vsr-test-session-id": sessionID})
			if err != nil {
				return fmt.Errorf("tool-result error policy request: %w", err)
			}
			dispatched, _, observationErr := lookupShortCircuitDispatch(ctx, provider, sessionID)
			if observationErr != nil {
				return observationErr
			}
			if isError && backendFormat != "anthropic.messages.v1" {
				if rejectionErr := assertToolResultErrorRejection(result, dispatched); rejectionErr != nil {
					return fmt.Errorf("tool-result rejection (backend=%s stream=%t): %w", backendFormat, stream, rejectionErr)
				}
				continue
			}
			if result.StatusCode != http.StatusOK || !dispatched {
				return fmt.Errorf("representable tool result was not dispatched (backend=%s stream=%t is_error=%t): status=%d body=%s",
					backendFormat, stream, isError, result.StatusCode, truncateString(string(result.Body), 500))
			}
			if backendFormat == "anthropic.messages.v1" {
				raw, captureErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
				if captureErr != nil {
					return captureErr
				}
				var observation struct {
					Body struct {
						Messages []struct {
							Content json.RawMessage `json:"content"`
						} `json:"messages"`
					} `json:"body"`
				}
				if decodeErr := json.Unmarshal(raw, &observation); decodeErr != nil {
					return decodeErr
				}
				if len(observation.Body.Messages) != 3 {
					return fmt.Errorf("provider history changed for Anthropic: %s", truncateString(string(raw), 500))
				}
				var blocks []struct {
					Type    string `json:"type"`
					CallID  string `json:"tool_use_id"`
					IsError *bool  `json:"is_error"`
				}
				if decodeErr := json.Unmarshal(observation.Body.Messages[2].Content, &blocks); decodeErr != nil {
					return decodeErr
				}
				if len(blocks) != 1 || blocks[0].Type != "tool_result" || blocks[0].CallID != "call_1" || blocks[0].IsError == nil || *blocks[0].IsError != isError {
					return fmt.Errorf("provider lost tool-result error flag for Anthropic: %s", truncateString(string(raw), 500))
				}
			}
			if stream {
				if streamErr := assertAnthropicTextStream(result.Body, "tool result accepted"); streamErr != nil {
					return streamErr
				}
			} else if responseErr := assertAnthropicText(result.Body, "tool result accepted"); responseErr != nil {
				return responseErr
			}
		}
	}
	return nil
}

func assertToolResultErrorRejection(result protocolMatrixHTTPResult, dispatched bool) error {
	// Messages renders unsupported-feature errors using its native error type
	// and message, not the internal neutral ProtocolError.Code field.
	var wire struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		return fmt.Errorf("invalid Messages error response: %w", err)
	}
	if result.StatusCode != http.StatusBadRequest || dispatched || wire.Type != "error" ||
		wire.Error.Type != "invalid_request_error" || wire.Error.Message != "translation would lose tool_result.is_error" {
		return fmt.Errorf("failed tool result must be rejected before dispatch: status=%d dispatched=%t body=%s",
			result.StatusCode, dispatched, truncateString(string(result.Body), 500))
	}
	return nil
}
