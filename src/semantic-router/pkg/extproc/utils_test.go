package extproc

import (
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestStatusCodeToEnumIncludesClientAndUpstreamErrors(t *testing.T) {
	tests := []struct {
		statusCode int
		want       typev3.StatusCode
	}{
		{statusCode: 400, want: typev3.StatusCode_BadRequest},
		{statusCode: 201, want: typev3.StatusCode_Created},
		{statusCode: 401, want: typev3.StatusCode_Unauthorized},
		{statusCode: 403, want: typev3.StatusCode_Forbidden},
		{statusCode: 404, want: typev3.StatusCode_NotFound},
		{statusCode: 405, want: typev3.StatusCode_MethodNotAllowed},
		{statusCode: 409, want: typev3.StatusCode_Conflict},
		{statusCode: 413, want: typev3.StatusCode_PayloadTooLarge},
		{statusCode: 422, want: typev3.StatusCode_UnprocessableEntity},
		{statusCode: 429, want: typev3.StatusCode_TooManyRequests},
		{statusCode: 500, want: typev3.StatusCode_InternalServerError},
		{statusCode: 502, want: typev3.StatusCode_BadGateway},
		{statusCode: 503, want: typev3.StatusCode_ServiceUnavailable},
	}

	for _, tt := range tests {
		t.Run(tt.want.String(), func(t *testing.T) {
			if got := statusCodeToImmediateResponseCode(tt.statusCode); got != tt.want {
				t.Fatalf("statusCodeToImmediateResponseCode(%d) = %v, want %v", tt.statusCode, got, tt.want)
			}
		})
	}
}

func TestExtractUserAndNonUserContentUsesLastUserAndJoinsTextParts(t *testing.T) {
	req := &llmprotocol.Request{
		Instructions: []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleSystem, Content: textBlocks("System", "Context")}},
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleAssistant, Content: textBlocks("Assistant", "Reply")},
			{Role: llmprotocol.RoleUser, Content: textBlocks("first user message")},
			{Role: llmprotocol.RoleUser, Content: textBlocks("latest", "question")},
		},
	}

	fast := extractSemanticRequestSignals(req)
	if fast.UserContent != "latest question" {
		t.Fatalf("expected latest user content, got %q", fast.UserContent)
	}
	if len(fast.NonUserMessages) != 2 {
		t.Fatalf("expected two non-user messages, got %d", len(fast.NonUserMessages))
	}
	if fast.NonUserMessages[0] != "System Context" {
		t.Fatalf("expected joined system content, got %q", fast.NonUserMessages[0])
	}
	if fast.NonUserMessages[1] != "Assistant Reply" {
		t.Fatalf("expected joined assistant content, got %q", fast.NonUserMessages[1])
	}
}

func TestExtractUserAndNonUserContentIgnoresToolMessages(t *testing.T) {
	req := &llmprotocol.Request{Messages: []llmprotocol.Message{
		neutralToolResult("tool-call-id", "tool output"),
		{Role: llmprotocol.RoleUser, Content: textBlocks("hello")},
	}}

	fast := extractSemanticRequestSignals(req)
	if fast.UserContent != "hello" {
		t.Fatalf("expected user content hello, got %q", fast.UserContent)
	}
	if len(fast.NonUserMessages) != 0 {
		t.Fatalf("expected tool messages to be ignored, got %#v", fast.NonUserMessages)
	}
}
