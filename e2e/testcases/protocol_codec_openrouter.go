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
	openRouterReplyMarker = "__mock_openrouter_reply__"
	openRouterReplyAnswer = "OpenRouter fixture answer"
)

func init() {
	pkgtestcases.Register("protocol-codec-openrouter-reply", pkgtestcases.TestCase{
		Description: "OpenRouter Chat response decorations and repeated finish chunks translate to every client format",
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
	const marker = "__mock_openrouter_reply_reasoning__"
	for _, streamed := range []bool{false, true} {
		sessionID := fmt.Sprintf("openrouter-reasoning-%t", streamed)
		headers := map[string]string{"x-vsr-test-session-id": sessionID}
		request := map[string]any{"model": chatBackendModel, "messages": []map[string]any{{"role": "user", "content": marker}}, "stream": streamed}
		body, err := json.Marshal(request)
		if err != nil {
			return err
		}
		first, err := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions", body, streamed, headers)
		if err != nil {
			return err
		}
		if first.StatusCode != http.StatusOK {
			return fmt.Errorf("OpenRouter reasoning stream=%t returned %d: %s", streamed, first.StatusCode, truncateString(string(first.Body), 500))
		}
		assistant, err := openRouterReasoningAssistant(first.Body, streamed)
		if err != nil {
			return err
		}
		details, ok := assistant["reasoning_details"].([]any)
		if !ok || len(details) != 3 {
			return fmt.Errorf("OpenRouter reasoning response lost typed blocks")
		}
		request["messages"] = []map[string]any{{"role": "user", "content": marker}, assistant, {"role": "tool", "tool_call_id": "openrouter-call-1", "content": "result"}}
		body, err = json.Marshal(request)
		if err != nil {
			return err
		}
		second, err := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions", body, streamed, headers)
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
		want, _ := json.Marshal(assistant["reasoning_details"])
		got, _ := json.Marshal(wire.Body.Messages[1]["reasoning_details"])
		if string(want) != string(got) {
			return fmt.Errorf("OpenRouter follow-up changed reasoning details: want %s got %s", want, got)
		}
		if !reflect.DeepEqual(assistant["tool_calls"], wire.Body.Messages[1]["tool_calls"]) {
			return fmt.Errorf("OpenRouter follow-up changed the tool call")
		}
	}
	return nil
}

func openRouterReasoningAssistant(body []byte, streamed bool) (map[string]any, error) {
	if !streamed {
		var wire struct {
			Choices []struct {
				Message map[string]any `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(body, &wire); err != nil {
			return nil, err
		}
		if len(wire.Choices) != 1 {
			return nil, fmt.Errorf("OpenRouter reasoning response must contain one choice")
		}
		return wire.Choices[0].Message, nil
	}
	assistant := map[string]any{"role": "assistant", "content": nil}
	details := make([]any, 0, 3)
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
					Details []any            `json:"reasoning_details"`
					Calls   []map[string]any `json:"tool_calls"`
				} `json:"delta"`
				Finish *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, err
		}
		if len(chunk.Error) > 0 {
			return nil, fmt.Errorf("OpenRouter reasoning stream emitted an error: %s", chunk.Error)
		}
		for _, choice := range chunk.Choices {
			details = append(details, choice.Delta.Details...)
			for _, call := range choice.Delta.Calls {
				delete(call, "index")
				calls = append(calls, call)
			}
			if choice.Finish != nil {
				if *choice.Finish != "tool_calls" || finished {
					return nil, fmt.Errorf("OpenRouter reasoning stream has an invalid terminal")
				}
				finished = true
			}
		}
	}
	if !finished || !done || len(calls) != 1 {
		return nil, fmt.Errorf("OpenRouter reasoning stream did not complete one tool call")
	}
	assistant["reasoning_details"], assistant["tool_calls"] = details, calls
	return assistant, nil
}
