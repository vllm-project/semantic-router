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

const (
	contextCompressionDecision = "context_compression_decision"
	contextCompressionMarker   = "__CONTEXT_COMPRESSION__"
	contextCompressionRelevant = "authentication token validator failed"
	contextCompressionOmission = "[... context omitted by route compression ...]"
)

func init() {
	pkgtestcases.Register("context-compression-provider-boundary", pkgtestcases.TestCase{
		Description: "Verify context_compression shortens provider-bound tool output and honors authorized bypass",
		Tags:        []string{"plugin", "context-compression", "provider-boundary"},
		Fn:          testContextCompressionProviderBoundary,
	})
}

func testContextCompressionProviderBoundary(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	backendOpts := opts
	backendOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Namespace:   "default",
		Name:        "vllm-llama3-8b-instruct",
		ServicePort: "8000",
	}
	backendSession, err := fixtures.OpenServiceSession(ctx, client, backendOpts)
	if err != nil {
		return err
	}
	defer backendSession.Close()

	originalToolOutput := contextCompressionToolOutput()
	cases := []struct {
		name       string
		control    string
		compressed bool
	}{
		{name: "extractive", compressed: true},
		{name: "bypass", control: "bypass"},
	}
	details := map[string]interface{}{
		"cases_total":                len(cases),
		"plugins_verified":           []string{"context_compression"},
		"provider_boundary_verified": true,
	}
	for _, testCase := range cases {
		sessionID := fmt.Sprintf("context-compression-%s-%d", testCase.name, time.Now().UnixNano())
		headers := map[string]string{
			"x-vsr-debug":           "true",
			"x-vsr-test-session-id": sessionID,
		}
		if testCase.control != "" {
			headers["x-vsr-compression-control"] = testCase.control
		}
		response, requestErr := sendProtocolMatrixRaw(
			ctx,
			session,
			"/v1/chat/completions",
			contextCompressionChatRequest(originalToolOutput),
			false,
			headers,
		)
		if requestErr != nil {
			return fmt.Errorf("%s request: %w", testCase.name, requestErr)
		}
		if response.StatusCode != http.StatusOK ||
			response.Headers.Get("x-vsr-response-path") != "upstream" ||
			response.Headers.Get("x-vsr-selected-recipe") != "e2e-plugins" ||
			response.Headers.Get("x-vsr-selected-decision") != contextCompressionDecision {
			return fmt.Errorf("%s did not dispatch through context compression: status=%d headers=%v",
				testCase.name, response.StatusCode, response.Headers)
		}
		if responseErr := validateContextCompressionResponse(response.Body); responseErr != nil {
			return fmt.Errorf("%s response: %w", testCase.name, responseErr)
		}

		observed, observeErr := lastProviderSimulatorRequest(ctx, backendSession, sessionID)
		if observeErr != nil {
			return fmt.Errorf("%s provider observation: %w", testCase.name, observeErr)
		}
		if contractErr := validateContextCompressionProviderRequest(
			observed,
			originalToolOutput,
			testCase.compressed,
		); contractErr != nil {
			return fmt.Errorf("%s provider contract: %w", testCase.name, contractErr)
		}
	}
	details["cases_passed"] = len(cases)
	details["bypass_verified"] = true
	details["protocol_preserved"] = true
	if opts.SetDetails != nil {
		opts.SetDetails(details)
	}
	return nil
}

func contextCompressionToolOutput() string {
	return strings.Join([]string{
		"diagnostic source header",
		strings.Repeat("irrelevant inventory values ", 180),
		contextCompressionRelevant,
		strings.Repeat("irrelevant billing values ", 180),
		"diagnostic source footer",
	}, "\n")
}

func contextCompressionChatRequest(toolOutput string) map[string]any {
	return map[string]any{
		"model": "e2e-plugins",
		"messages": []any{
			map[string]any{"role": "user", "content": "Collect authentication diagnostics."},
			map[string]any{
				"role": "assistant",
				"tool_calls": []any{map[string]any{
					"id":   "call_context_compression",
					"type": "function",
					"function": map[string]any{
						"name":      "diagnostics",
						"arguments": `{"query":"authentication token validator failed"}`,
					},
				}},
			},
			map[string]any{
				"role":         "tool",
				"tool_call_id": "call_context_compression",
				"content":      toolOutput,
			},
			map[string]any{"role": "assistant", "content": "I reviewed the diagnostic output."},
			map[string]any{
				"role":    "user",
				"content": contextCompressionMarker + " Fix the authentication token validator.",
			},
		},
	}
}

func validateContextCompressionResponse(body []byte) error {
	var response struct {
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return fmt.Errorf("decode chat-completions response: %w", err)
	}
	if response.Object != "chat.completion" || len(response.Choices) != 1 ||
		response.Choices[0].Message.Content == "" {
		return fmt.Errorf("compression changed the Chat Completions response contract: %s",
			truncateString(string(body), 600))
	}
	return nil
}

func validateContextCompressionProviderRequest(
	observed []byte,
	originalToolOutput string,
	expectCompressed bool,
) error {
	var request struct {
		Body struct {
			Messages []contextCompressionProviderMessage `json:"messages"`
		} `json:"body"`
	}
	if err := json.Unmarshal(observed, &request); err != nil {
		return fmt.Errorf("decode provider-bound request: %w", err)
	}
	messages := request.Body.Messages
	if len(messages) != 5 {
		return fmt.Errorf("provider messages = %d, want 5: %s", len(messages), truncateString(string(observed), 800))
	}
	if err := validateContextCompressionMessageEnvelope(messages); err != nil {
		return err
	}

	toolOutput := messages[2].Content
	if expectCompressed {
		if toolOutput == originalToolOutput || len(toolOutput) >= len(originalToolOutput) {
			return fmt.Errorf("tool output was not compressed: original=%d provider=%d",
				len(originalToolOutput), len(toolOutput))
		}
		if !strings.Contains(toolOutput, contextCompressionRelevant) {
			return fmt.Errorf("compressed output removed query-relevant evidence: %s",
				truncateString(toolOutput, 800))
		}
		if !strings.Contains(toolOutput, contextCompressionOmission) {
			return fmt.Errorf("compressed output lacks the omission marker: %s",
				truncateString(toolOutput, 800))
		}
		if strings.Contains(toolOutput, strings.Repeat("irrelevant inventory values ", 20)) {
			return fmt.Errorf("compressed output retained an irrelevant run: %s",
				truncateString(toolOutput, 800))
		}
		return nil
	}
	if toolOutput != originalToolOutput {
		return fmt.Errorf("bypass changed tool output: got %d bytes, want %d",
			len(toolOutput), len(originalToolOutput))
	}
	return nil
}

type contextCompressionProviderMessage struct {
	Role       string `json:"role"`
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id"`
	ToolCalls  []struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
}

func validateContextCompressionMessageEnvelope(messages []contextCompressionProviderMessage) error {
	wantRoles := []string{"user", "assistant", "tool", "assistant", "user"}
	for index, role := range wantRoles {
		if messages[index].Role != role {
			return fmt.Errorf("provider message %d role = %q, want %q", index, messages[index].Role, role)
		}
	}
	call := messages[1]
	if len(call.ToolCalls) != 1 ||
		call.ToolCalls[0].ID != "call_context_compression" ||
		call.ToolCalls[0].Type != "function" ||
		call.ToolCalls[0].Function.Name != "diagnostics" ||
		call.ToolCalls[0].Function.Arguments != `{"query":"authentication token validator failed"}` {
		return fmt.Errorf("assistant tool call changed: %#v", call.ToolCalls)
	}
	if messages[2].ToolCallID != "call_context_compression" {
		return fmt.Errorf("tool result correlation changed: %q", messages[2].ToolCallID)
	}
	if messages[3].Content != "I reviewed the diagnostic output." {
		return fmt.Errorf("recent assistant message changed: %q", messages[3].Content)
	}
	wantUser := contextCompressionMarker + " Fix the authentication token validator."
	if messages[4].Content != wantUser {
		return fmt.Errorf("current user message changed: %q", messages[4].Content)
	}
	return nil
}
