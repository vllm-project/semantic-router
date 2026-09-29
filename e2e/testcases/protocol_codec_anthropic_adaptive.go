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
	pkgtestcases.Register("protocol-codec-anthropic-adaptive-chat", pkgtestcases.TestCase{
		Description: "Claude Code adaptive thinking, omitted display, and no-op context edits reach a Chat backend",
		Tags:        []string{"protocol-codec", "anthropic", "response-api", "agents", "streaming"},
		Fn:          anthropicAdaptiveProjectionTest(chatBackendModel, "openai.chat.v1", protocolCodecChatReply),
	})
	pkgtestcases.Register("protocol-codec-anthropic-adaptive-responses", pkgtestcases.TestCase{
		Description: "Claude Code adaptive thinking, omitted display, and no-op context edits reach a Responses backend",
		Tags:        []string{"protocol-codec", "anthropic", "response-api", "agents", "streaming"},
		Fn:          anthropicAdaptiveProjectionTest(nativeResponsesBackendModel, "openai.responses.v1", protocolCodecResponsesReply),
	})
}

func anthropicAdaptiveProjectionTest(model, backendFormat, reply string) func(context.Context, *kubernetes.Clientset, pkgtestcases.TestCaseOptions) error {
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

		const prompt = "Claude Code adaptive projection probe"
		for _, stream := range []bool{false, true} {
			sessionID := fmt.Sprintf("anthropic-adaptive-%s-%t", backendFormat, stream)
			result, requestErr := sendProtocolMatrixRaw(ctx, session, "/v1/messages", map[string]any{
				"model": model, "max_tokens": 64, "stream": stream,
				"messages":      []map[string]string{{"role": "user", "content": prompt}},
				"thinking":      map[string]string{"type": "adaptive", "display": "omitted"},
				"output_config": map[string]string{"effort": "medium"},
				"context_management": map[string]any{"edits": []map[string]string{{
					"type": "clear_thinking_20251015", "keep": "all",
				}}},
			}, stream, map[string]string{"x-vsr-test-session-id": sessionID})
			if requestErr != nil {
				return fmt.Errorf("%s request: %w", sessionID, requestErr)
			}
			if result.StatusCode != http.StatusOK {
				return fmt.Errorf("%s returned HTTP %d: %s", sessionID, result.StatusCode,
					truncateString(string(result.Body), 800))
			}
			if stream {
				err = assertAnthropicTextStream(result.Body, reply)
			} else {
				err = assertAnthropicText(result.Body, reply)
			}
			if err != nil {
				return fmt.Errorf("%s client response: %w", sessionID, err)
			}
			upstream, observationErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
			if observationErr != nil {
				return fmt.Errorf("%s provider observation: %w", sessionID, observationErr)
			}
			if projectionErr := assertAnthropicAdaptiveProjection(upstream, model, backendFormat, prompt); projectionErr != nil {
				return fmt.Errorf("%s provider projection: %w", sessionID, projectionErr)
			}
		}
		return nil
	}
}

func assertAnthropicAdaptiveProjection(observation []byte, model, backendFormat, prompt string) error {
	var receipt struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(observation, &receipt); err != nil {
		return fmt.Errorf("decode provider observation: %w", err)
	}
	if string(receipt.Body["model"]) != `"`+model+`"` || !strings.Contains(string(observation), prompt) {
		return fmt.Errorf("backend model or prompt was lost: %s", truncateString(string(observation), 800))
	}
	for _, field := range []string{"thinking", "context_management", "output_config"} {
		if _, found := receipt.Body[field]; found {
			return fmt.Errorf("anthropic-only %s reached %s: %s", field, backendFormat,
				truncateString(string(observation), 800))
		}
	}
	switch backendFormat {
	case "openai.chat.v1":
		if len(receipt.Body["messages"]) == 0 || string(receipt.Body["reasoning_effort"]) != `"medium"` {
			return fmt.Errorf("chat backend lost messages or medium reasoning effort: %s",
				truncateString(string(observation), 800))
		}
	case "openai.responses.v1":
		var reasoning struct {
			Effort string `json:"effort"`
		}
		if len(receipt.Body["input"]) == 0 || json.Unmarshal(receipt.Body["reasoning"], &reasoning) != nil || reasoning.Effort != "medium" {
			return fmt.Errorf("responses backend lost input or medium reasoning effort: %s",
				truncateString(string(observation), 800))
		}
	default:
		return fmt.Errorf("unsupported backend format %q", backendFormat)
	}
	return nil
}
