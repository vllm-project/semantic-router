package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("protocol-codec-chat-custom-tool-kind-switch", pkgtestcases.TestCase{
		Description: "Chat backend rejects custom/function kind switches while preserving custom stream deltas",
		Tags:        []string{"protocol-codec", "response-api", "tools", "streaming", "failure"},
		Fn:          testProtocolCodecChatCustomToolKindSwitch,
	})
}

type customToolKindStreamChunk struct {
	Choices []struct {
		Delta struct {
			ToolCalls []struct {
				Index  int    `json:"index"`
				ID     string `json:"id"`
				Type   string `json:"type"`
				Custom *struct {
					Name  string `json:"name"`
					Input string `json:"input"`
				} `json:"custom"`
				Function json.RawMessage `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Code string `json:"code"`
	} `json:"error"`
}

func testProtocolCodecChatCustomToolKindSwitch(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, sessionErr := fixtures.OpenServiceSession(ctx, client, opts)
	if sessionErr != nil {
		return sessionErr
	}
	defer session.Close()
	provider, providerErr := openProtocolCodecProviderSession(ctx, client, opts, "openai.chat.v1")
	if providerErr != nil {
		return providerErr
	}
	defer provider.Close()

	cases := []struct {
		name, marker, firstKind string
		valid                   bool
	}{
		{"custom_to_function", "__mock_custom_kind_custom_to_function__", "custom", false},
		{"custom_to_untyped_function", "__mock_custom_kind_custom_to_untyped_function__", "custom", false},
		{"function_to_custom", "__mock_custom_kind_function_to_custom__", "function", false},
		{"valid_custom", "__mock_custom_kind_valid_custom__", "custom", true},
	}
	for index, check := range cases {
		sessionID := fmt.Sprintf("custom-kind-%d-%d", time.Now().UnixNano(), index)
		result, requestErr := sendProtocolMatrixRaw(
			ctx, session, "/v1/chat/completions", customToolKindRequest(check.marker), true,
			map[string]string{"x-vsr-test-session-id": sessionID},
		)
		if requestErr != nil {
			return fmt.Errorf("%s: %w", check.name, requestErr)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("%s: HTTP %d, want streamed 200: %s", check.name,
				result.StatusCode, truncateString(string(result.Body), 600))
		}
		if !strings.HasPrefix(result.Headers.Get("Content-Type"), "text/event-stream") {
			return fmt.Errorf("%s: content type = %q, want text/event-stream", check.name,
				result.Headers.Get("Content-Type"))
		}
		if err := verifyCustomToolReachedProvider(ctx, provider, sessionID); err != nil {
			return fmt.Errorf("%s: %w", check.name, err)
		}
		var checkErr error
		if check.valid {
			checkErr = assertValidCustomToolStream(result.Body)
		} else {
			checkErr = assertRejectedToolKindSwitch(result.Body, check.firstKind)
		}
		if checkErr != nil {
			return fmt.Errorf("%s: %w", check.name, checkErr)
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"kind_switches_rejected": 3, "valid_custom_streams": 1})
	}
	return nil
}

func customToolKindRequest(marker string) map[string]any {
	return map[string]any{
		"model": chatBackendModel, "stream": true,
		"messages": []any{map[string]any{"role": "user", "content": marker}},
		"tools": []any{map[string]any{
			"type": "custom",
			"custom": map[string]any{
				"name": "apply_patch", "description": "Apply a patch",
				"format": map[string]any{
					"type":    "grammar",
					"grammar": map[string]any{"syntax": "lark", "definition": `start: "ok"`},
				},
			},
		}},
	}
}

func verifyCustomToolReachedProvider(ctx context.Context, provider *fixtures.ServiceSession, sessionID string) error {
	observed, observationErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if observationErr != nil {
		return observationErr
	}
	var receipt struct {
		Body struct {
			Stream bool `json:"stream"`
			Tools  []struct {
				Type   string `json:"type"`
				Custom struct {
					Name   string `json:"name"`
					Format struct {
						Type    string `json:"type"`
						Grammar struct {
							Syntax     string `json:"syntax"`
							Definition string `json:"definition"`
						} `json:"grammar"`
					} `json:"format"`
				} `json:"custom"`
			} `json:"tools"`
		} `json:"body"`
	}
	if err := json.Unmarshal(observed, &receipt); err != nil {
		return fmt.Errorf("decode provider request receipt: %w", err)
	}
	if !receipt.Body.Stream || len(receipt.Body.Tools) != 1 || receipt.Body.Tools[0].Type != "custom" ||
		receipt.Body.Tools[0].Custom.Name != "apply_patch" ||
		receipt.Body.Tools[0].Custom.Format.Type != "grammar" ||
		receipt.Body.Tools[0].Custom.Format.Grammar.Syntax != "lark" ||
		receipt.Body.Tools[0].Custom.Format.Grammar.Definition != `start: "ok"` {
		return fmt.Errorf("custom tool or grammar did not reach the Chat provider: %s", truncateString(string(observed), 700))
	}
	return nil
}

func assertRejectedToolKindSwitch(body []byte, firstKind string) error {
	frames := protocolSSEDataFrames(body)
	partialCalls, failures := 0, 0
	for index, data := range frames {
		if data == "[DONE]" {
			return fmt.Errorf("kind switch ended as success: %s", truncateString(string(body), 900))
		}
		var chunk customToolKindStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("decode kind-switch SSE frame: %w", err)
		}
		if chunk.Error != nil {
			failures++
			if chunk.Error.Code != "stream_tool_identity_mismatch" || partialCalls != 1 || index != len(frames)-1 {
				return fmt.Errorf("kind switch failed without the accepted prefix or exact code: %s", truncateString(string(body), 900))
			}
			continue
		}
		if failures != 0 {
			return fmt.Errorf("kind switch emitted output after the failure: %s", truncateString(string(body), 900))
		}
		for _, choice := range chunk.Choices {
			if choice.FinishReason != nil {
				return fmt.Errorf("kind switch emitted a success terminal: %s", truncateString(string(body), 900))
			}
			for _, call := range choice.Delta.ToolCalls {
				partialCalls++
				if partialCalls != 1 || call.Index != 0 || call.ID != "call_mock_custom_kind" ||
					call.Type != firstKind || (firstKind == "custom" && len(call.Function) != 0) ||
					(firstKind == "function" && call.Custom != nil) {
					return fmt.Errorf("kind switch leaked the changed tool delta: %s", truncateString(string(body), 900))
				}
				if firstKind == "custom" && (call.Custom == nil || call.Custom.Name != "apply_patch" || call.Custom.Input != "abc") {
					return fmt.Errorf("kind switch lost the accepted custom prefix: %s", truncateString(string(body), 900))
				}
				if firstKind == "function" {
					var function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					}
					if err := json.Unmarshal(call.Function, &function); err != nil ||
						function.Name != "lookup" || function.Arguments != "{}" {
						return fmt.Errorf("kind switch lost the accepted function prefix: %s", truncateString(string(body), 900))
					}
				}
			}
		}
	}
	if partialCalls != 1 || failures != 1 {
		return fmt.Errorf("kind switch produced %d partial calls and %d errors, want one each: %s",
			partialCalls, failures, truncateString(string(body), 900))
	}
	return nil
}

func assertValidCustomToolStream(body []byte) error {
	frames := protocolSSEDataFrames(body)
	input, toolDeltas, terminals, done := "", 0, 0, 0
	for _, data := range frames {
		if data == "[DONE]" {
			done++
			continue
		}
		var chunk customToolKindStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("decode valid custom SSE frame: %w", err)
		}
		if chunk.Error != nil {
			return fmt.Errorf("valid custom stream failed with %s", chunk.Error.Code)
		}
		for _, choice := range chunk.Choices {
			if choice.FinishReason != nil {
				if *choice.FinishReason != "tool_calls" {
					return fmt.Errorf("valid custom stream finished with %q", *choice.FinishReason)
				}
				terminals++
			}
			for _, call := range choice.Delta.ToolCalls {
				toolDeltas++
				if call.Index != 0 || call.ID != "call_mock_custom_kind" || call.Type != "custom" ||
					call.Custom == nil || len(call.Function) != 0 ||
					call.Custom.Name != "apply_patch" {
					return fmt.Errorf("valid stream changed custom identity: %s", truncateString(string(body), 900))
				}
				input += call.Custom.Input
			}
		}
	}
	if toolDeltas != 4 || input != "abcdefghi" || terminals != 1 || done != 1 ||
		len(frames) == 0 || frames[len(frames)-1] != "[DONE]" {
		return fmt.Errorf("valid custom stream incomplete: deltas=%d input=%q terminals=%d done=%d: %s",
			toolDeltas, input, terminals, done, truncateString(string(body), 900))
	}
	return nil
}
