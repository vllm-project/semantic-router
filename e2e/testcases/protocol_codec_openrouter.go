package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	openRouterReplyMarker     = "__mock_openrouter_reply__"
	openRouterReplyAnswer     = "OpenRouter fixture answer"
	openRouterReasoningMarker = "__mock_openrouter_reply_reasoning__"
	openRouterReasoningText   = "Look up the result. Use the lookup tool."
	openRouterOpaqueData      = "opaque-continuity-fixture"
)

func init() {
	pkgtestcases.Register("protocol-codec-openrouter-reply", pkgtestcases.TestCase{
		Description: "OpenRouter Chat response decorations, repeated finish chunks, and reasoning details translate to every client format",
		Tags:        []string{"protocol-codec", "response-api", "provider", "streaming"},
		Fn:          testProtocolCodecOpenRouterReply,
	})
}

func testProtocolCodecOpenRouterReply(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, "openai.chat.v1")
	if err != nil {
		return err
	}
	defer provider.Close()

	results := make(map[string]string, 2*len(protocolCodecE2EClients))
	for _, clientContract := range protocolCodecE2EClients {
		for _, streamed := range []bool{false, true} {
			mode := "buffered"
			if streamed {
				mode = "streamed"
			}
			cell := clientContract.name + "/" + mode
			sessionID := "openrouter-reply-" + clientContract.name + "-" + mode
			result, requestErr := sendProtocolMatrixRaw(ctx, session, clientContract.path,
				clientContract.request(chatBackendModel, openRouterReplyMarker, streamed), streamed,
				map[string]string{"x-vsr-test-session-id": sessionID})
			if requestErr != nil {
				return fmt.Errorf("%s request: %w", cell, requestErr)
			}
			if result.StatusCode != http.StatusOK {
				return fmt.Errorf("%s returned HTTP %d: %s", cell, result.StatusCode, truncateString(string(result.Body), 500))
			}
			if streamed {
				if err := clientContract.validateStream(result.Body, openRouterReplyAnswer); err != nil {
					return fmt.Errorf("%s stream: %w", cell, err)
				}
				text, err := extractProtocolStructuredOutputStreamText(clientContract.path, result.Body)
				if err != nil {
					return fmt.Errorf("%s stream text: %w", cell, err)
				}
				if text != openRouterReplyAnswer {
					return fmt.Errorf("%s duplicated or changed output after repeated terminal chunks: %q", cell, text)
				}
				if clientContract.path == "/v1/chat/completions" &&
					(strings.Count(string(result.Body), `"finish_reason":"stop"`) != 1 ||
						strings.Count(string(result.Body), "data: [DONE]") != 1) {
					return fmt.Errorf("%s emitted repeated Chat terminal chunks: %s", cell, truncateString(string(result.Body), 600))
				}
			} else {
				if err := clientContract.validateBuffered(result.Body, openRouterReplyAnswer); err != nil {
					return fmt.Errorf("%s response: %w", cell, err)
				}
				warnings := result.Headers.Get("x-vsr-protocol-warnings")
				for _, field := range []string{
					"provider", "choices.native_finish_reason", "usage.cost", "usage.is_byok",
					"usage.cost_details", "usage.server_tool_use", "usage.prompt_tokens_details.video_tokens",
					"usage.completion_tokens_details.image_tokens",
				} {
					if !hasProtocolFieldDiagnostic(warnings, "dropped", field) {
						return fmt.Errorf("%s omitted %s without a diagnostic: %q", cell, field, warnings)
					}
				}
			}
			forbiddenFields := []string{
				`"provider":`, `"native_finish_reason":`, `"cost":`, `"cost_details":`, `"is_byok":`,
				`"video_tokens":`, `"image_tokens":`,
			}
			if clientContract.path != "/v1/messages" {
				forbiddenFields = append(forbiddenFields, `"server_tool_use":`)
			}
			for _, field := range forbiddenFields {
				if strings.Contains(string(result.Body), field) {
					return fmt.Errorf("%s leaked OpenRouter-only field %s: %s", cell, field, truncateString(string(result.Body), 600))
				}
			}
			if err := verifyProviderSimulatorRequest(ctx, provider, sessionID, "openai.chat.v1", openRouterReplyMarker); err != nil {
				return fmt.Errorf("%s provider dispatch: %w", cell, err)
			}
			results[cell] = "passed"
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"matrix_cells": results, "backend_format": "openai.chat.v1"})
	}
	if err := testOpenRouterReasoningToolContinuity(ctx, session, provider); err != nil {
		return err
	}
	return nil
}

// The mock provider refuses a follow-up unless it receives the original blocks
// and tool call. This tests continuity through dispatch, not just response text.
func testOpenRouterReasoningToolContinuity(ctx context.Context, session, provider *fixtures.ServiceSession) error {
	for _, streamed := range []bool{false, true} {
		sessionID := fmt.Sprintf("openrouter-reasoning-%t", streamed)
		headers := map[string]string{"x-vsr-test-session-id": sessionID}
		request := map[string]any{"model": chatBackendModel, "messages": []map[string]any{{"role": "user", "content": openRouterReasoningMarker}}, "stream": streamed}
		first, err := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions", request, streamed, headers)
		if err != nil {
			return err
		}
		if first.StatusCode != http.StatusOK {
			return fmt.Errorf("OpenRouter reasoning stream=%t returned %d: %s", streamed, first.StatusCode, truncateString(string(first.Body), 500))
		}
		assistant, reasoning, err := openRouterReasoningAssistant(first.Body, streamed)
		if err != nil {
			return err
		}
		if reasoning != openRouterReasoningText {
			return fmt.Errorf("OpenRouter reasoning stream=%t lost the display text: %q", streamed, reasoning)
		}
		if details, ok := assistant["reasoning_details"].([]any); !ok || len(details) == 0 {
			return fmt.Errorf("OpenRouter reasoning stream=%t lost typed blocks", streamed)
		}
		request["messages"] = []map[string]any{{"role": "user", "content": openRouterReasoningMarker}, assistant, {"role": "tool", "tool_call_id": "openrouter-call-1", "content": "result"}}
		second, err := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions", request, streamed, headers)
		if err != nil {
			return err
		}
		if second.StatusCode != http.StatusOK || !strings.Contains(string(second.Body), openRouterReplyAnswer) {
			return fmt.Errorf("OpenRouter reasoning follow-up stream=%t failed: HTTP %d %s", streamed, second.StatusCode, truncateString(string(second.Body), 500))
		}
		observed, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
		if err != nil {
			return err
		}
		var wire struct {
			Body struct {
				Messages []map[string]any `json:"messages"`
			} `json:"body"`
		}
		if err := json.Unmarshal(observed, &wire); err != nil {
			return err
		}
		if len(wire.Body.Messages) != 3 {
			return fmt.Errorf("OpenRouter follow-up lost conversation messages")
		}
		forwarded := wire.Body.Messages[1]
		want, _ := json.Marshal(assistant["reasoning_details"])
		got, _ := json.Marshal(forwarded["reasoning_details"])
		if string(want) != string(got) {
			return fmt.Errorf("OpenRouter follow-up changed reasoning details: want %s got %s", want, got)
		}
		if !reflect.DeepEqual(assistant["tool_calls"], forwarded["tool_calls"]) {
			return fmt.Errorf("OpenRouter follow-up changed the tool call")
		}
		for _, alias := range []string{"reasoning", "reasoning_content"} {
			if _, found := forwarded[alias]; found {
				return fmt.Errorf("OpenRouter follow-up stream=%t forwarded the reasoning text twice as %s", streamed, alias)
			}
		}
	}
	return testOpenRouterReasoningForMessagesClient(ctx, session)
}

// A Messages client gets the reasoning text and a visible omission, not an
// error, for a reply the provider has already produced.
func testOpenRouterReasoningForMessagesClient(ctx context.Context, session *fixtures.ServiceSession) error {
	client := protocolCodecE2EClient{name: "anthropic_messages", path: "/v1/messages"}
	for _, streamed := range []bool{false, true} {
		headers := map[string]string{"x-vsr-test-session-id": fmt.Sprintf("openrouter-reasoning-messages-%t", streamed)}
		result, err := sendProtocolMatrixRaw(ctx, session, client.path, client.request(chatBackendModel, openRouterReasoningMarker, streamed), streamed, headers)
		if err != nil {
			return err
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("a Messages client of OpenRouter reasoning stream=%t returned %d: %s", streamed, result.StatusCode, truncateString(string(result.Body), 500))
		}
		body := string(result.Body)
		if !strings.Contains(body, "Use the lookup tool.") || strings.Contains(body, openRouterOpaqueData) || strings.Contains(body, "reasoning_details") {
			return fmt.Errorf("a Messages client of OpenRouter reasoning stream=%t did not degrade to the reasoning text: %s", streamed, truncateString(body, 600))
		}
		// A stream has sent its headers before the drop is known, so only a
		// buffered reply can report it in x-vsr-protocol-warnings.
		if !streamed && !hasProtocolFieldDiagnostic(result.Headers.Get("x-vsr-protocol-warnings"), "dropped", "reasoning_details") {
			return fmt.Errorf("a Messages client dropped continuity blocks without a diagnostic: %q", result.Headers.Get("x-vsr-protocol-warnings"))
		}
	}
	return nil
}

// openRouterReasoningAssistant rebuilds the assistant message a client returns
// with tool results, and the reasoning text it renders. Stream fragments of one
// block share a type and index and are concatenated in order.
func openRouterReasoningAssistant(body []byte, streamed bool) (map[string]any, string, error) {
	if !streamed {
		var wire struct {
			Choices []struct {
				Message map[string]any `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(body, &wire); err != nil {
			return nil, "", err
		}
		if len(wire.Choices) != 1 {
			return nil, "", fmt.Errorf("OpenRouter reasoning response must contain one choice")
		}
		message := wire.Choices[0].Message
		reasoning, _ := message["reasoning_content"].(string)
		return message, reasoning, nil
	}
	var reasoning strings.Builder
	details := make([]any, 0, 2)
	calls := make([]any, 0, 1)
	finished, done := false, false
	for _, line := range strings.Split(string(body), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			done = true
			continue
		}
		var chunk struct {
			Error   json.RawMessage `json:"error"`
			Choices []struct {
				Delta struct {
					Reasoning        *string          `json:"reasoning"`
					ReasoningContent *string          `json:"reasoning_content"`
					Details          []map[string]any `json:"reasoning_details"`
					Calls            []map[string]any `json:"tool_calls"`
				} `json:"delta"`
				Finish *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, "", err
		}
		if len(chunk.Error) > 0 {
			return nil, "", fmt.Errorf("OpenRouter reasoning stream emitted an error: %s", chunk.Error)
		}
		for _, choice := range chunk.Choices {
			// A same-format Chat stream passes the provider's deltas through, so
			// the text keeps OpenRouter's field name.
			if choice.Delta.ReasoningContent != nil {
				return nil, "", fmt.Errorf("a Chat stream renamed the provider's reasoning field")
			}
			if choice.Delta.Reasoning != nil {
				reasoning.WriteString(*choice.Delta.Reasoning)
			}
			for _, fragment := range choice.Delta.Details {
				details = appendReasoningFragment(details, fragment)
			}
			for _, call := range choice.Delta.Calls {
				delete(call, "index")
				calls = append(calls, call)
			}
			if choice.Finish != nil {
				if *choice.Finish != "tool_calls" || finished {
					return nil, "", fmt.Errorf("OpenRouter reasoning stream has an invalid terminal")
				}
				finished = true
			}
		}
	}
	if !finished || !done || len(calls) != 1 {
		return nil, "", fmt.Errorf("OpenRouter reasoning stream did not complete one tool call")
	}
	assistant := map[string]any{"role": "assistant", "content": nil, "reasoning": reasoning.String(), "reasoning_details": details, "tool_calls": calls}
	return assistant, reasoning.String(), nil
}

func appendReasoningFragment(details []any, fragment map[string]any) []any {
	for _, existing := range details {
		block := existing.(map[string]any)
		if block["type"] != fragment["type"] || block["index"] != fragment["index"] {
			continue
		}
		for _, field := range []string{"text", "signature", "summary", "data"} {
			if value, ok := fragment[field].(string); ok {
				previous, _ := block[field].(string)
				block[field] = previous + value
			}
		}
		return details
	}
	return append(details, fragment)
}
