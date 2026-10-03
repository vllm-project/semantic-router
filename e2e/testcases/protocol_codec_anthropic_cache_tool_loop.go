package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("protocol-codec-anthropic-cache-tool-loop-chat", pkgtestcases.TestCase{
		Description: "Anthropic tool loops preserve cache boundaries through a Chat backend",
		Tags:        []string{"protocol-codec", "anthropic", "cache", "agents", "tools", "streaming"},
		Fn:          anthropicCacheToolLoopTest(chatBackendModel, "openai.chat.v1"),
	})
	pkgtestcases.Register("protocol-codec-anthropic-cache-tool-loop-responses", pkgtestcases.TestCase{
		Description: "Anthropic tool loops drop unsupported cache boundaries with a warning through a Responses backend",
		Tags:        []string{"protocol-codec", "anthropic", "cache", "agents", "tools", "streaming"},
		Fn:          anthropicCacheToolLoopTest(nativeResponsesBackendModel, "openai.responses.v1"),
	})
}

func anthropicCacheToolLoopTest(model, backendFormat string) func(context.Context, *kubernetes.Clientset, pkgtestcases.TestCaseOptions) error {
	return func(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
		session, err := fixtures.OpenServiceSession(ctx, client, opts)
		if err != nil {
			return err
		}
		defer session.Close()
		provider, err := openProtocolCodecProviderSession(ctx, client, opts, backendFormat)
		if err != nil {
			return err
		}
		defer provider.Close()

		tool := map[string]any{
			"name": "lookup", "description": "Look up a value",
			"input_schema": map[string]any{"type": "object", "properties": map[string]any{"query": map[string]any{"type": "string"}}, "required": []string{"query"}},
		}
		for _, stream := range []bool{false, true} {
			first := map[string]any{
				"model": model, "max_tokens": 64, "stream": stream,
				"messages": []any{map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "text", "text": "Run lookup now"},
					map[string]any{"type": "text", "text": "__mock_tool_call__", "cache_control": map[string]any{"type": "ephemeral"}},
				}}},
				"tools": []any{tool},
			}
			firstResult, err := sendAnthropicCacheToolTurn(ctx, session, provider, first, backendFormat, stream, "call")
			if err != nil {
				return err
			}
			var call responsesFunctionCall
			if stream {
				call, err = decodeAnthropicToolUseStream(firstResult.Body, false)
			} else {
				call, err = decodeAnthropicToolUse(firstResult.Body, false)
			}
			if err != nil {
				return fmt.Errorf("%s tool-call turn: %w", backendFormat, err)
			}
			var input map[string]any
			if err = json.Unmarshal([]byte(call.Arguments), &input); err != nil {
				return err
			}
			second := map[string]any{
				"model": model, "max_tokens": 64, "stream": stream,
				"messages": []any{
					map[string]any{"role": "user", "content": "__mock_tool_call__"},
					map[string]any{"role": "assistant", "content": []any{map[string]any{
						"type": "tool_use", "id": call.CallID, "name": call.Name, "input": input,
					}}},
					map[string]any{"role": "user", "content": []any{map[string]any{
						"type": "tool_result", "tool_use_id": call.CallID, "content": "sunny",
						"cache_control": map[string]any{"type": "ephemeral"},
					}}},
				},
				"tools": []any{tool},
			}
			secondResult, err := sendAnthropicCacheToolTurn(ctx, session, provider, second, backendFormat, stream, "result")
			if err != nil {
				return err
			}
			if stream {
				err = assertAnthropicTextStream(secondResult.Body, "tool result accepted")
			} else {
				err = assertAnthropicText(secondResult.Body, "tool result accepted")
			}
			if err != nil {
				return fmt.Errorf("%s tool-result turn: %w", backendFormat, err)
			}
		}
		return nil
	}
}

func sendAnthropicCacheToolTurn(
	ctx context.Context,
	session, provider *fixtures.ServiceSession,
	body map[string]any,
	backendFormat string,
	stream bool,
	turn string,
) (protocolMatrixHTTPResult, error) {
	sessionID := fmt.Sprintf("anthropic-cache-tool-%s-%t-%s", backendFormat, stream, turn)
	result, err := sendProtocolMatrixRaw(ctx, session, "/v1/messages", body, stream,
		map[string]string{"x-vsr-test-session-id": sessionID})
	if err != nil {
		return result, err
	}
	if result.StatusCode != http.StatusOK {
		return result, fmt.Errorf("%s returned HTTP %d: %s", sessionID, result.StatusCode, truncateString(string(result.Body), 800))
	}
	upstream, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if err != nil {
		return result, err
	}
	if backendFormat == "openai.responses.v1" {
		if warnings := result.Headers.Get("x-vsr-protocol-warnings"); !hasProtocolFieldDiagnostic(warnings, "dropped", "cache_control") {
			return result, fmt.Errorf("%s omitted the cache drop warning: %q", sessionID, warnings)
		}
		if bytes.Contains(upstream, []byte("cache_control")) {
			return result, fmt.Errorf("%s forwarded an unsupported cache directive: %s", sessionID, truncateString(string(upstream), 800))
		}
	} else if !bytes.Contains(upstream, []byte("cache_control")) {
		return result, fmt.Errorf("%s lost a Chat backend cache boundary: %s", sessionID, truncateString(string(upstream), 800))
	}
	if turn == "result" && backendFormat == "openai.chat.v1" {
		if err := assertChatToolResultCacheBoundary(upstream); err != nil {
			return result, fmt.Errorf("%s: %w", sessionID, err)
		}
	}
	return result, nil
}

func assertChatToolResultCacheBoundary(upstream []byte) error {
	var observation struct {
		Body struct {
			Messages []struct {
				Role    string `json:"role"`
				Content any    `json:"content"`
			} `json:"messages"`
		} `json:"body"`
	}
	if err := json.Unmarshal(upstream, &observation); err != nil {
		return err
	}
	for _, message := range observation.Body.Messages {
		if message.Role != "tool" {
			continue
		}
		parts, ok := message.Content.([]any)
		if !ok || len(parts) == 0 {
			return fmt.Errorf("chat tool result has no content parts: %s", truncateString(string(upstream), 800))
		}
		last, ok := parts[len(parts)-1].(map[string]any)
		if !ok || last["text"] != "sunny" || last["cache_control"] == nil {
			return fmt.Errorf("chat tool result lost its cache boundary: %s", truncateString(string(upstream), 800))
		}
		return nil
	}
	return fmt.Errorf("chat backend received no tool result: %s", truncateString(string(upstream), 800))
}
