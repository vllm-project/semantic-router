package extproc

import (
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/openai/openai-go"
)

func TestStatusCodeToEnumIncludesClientAndUpstreamErrors(t *testing.T) {
	tests := []struct {
		statusCode int
		want       typev3.StatusCode
	}{
		{statusCode: 400, want: typev3.StatusCode_BadRequest},
		{statusCode: 401, want: typev3.StatusCode_Unauthorized},
		{statusCode: 403, want: typev3.StatusCode_Forbidden},
		{statusCode: 404, want: typev3.StatusCode_NotFound},
		{statusCode: 405, want: typev3.StatusCode_MethodNotAllowed},
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
	req := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage([]openai.ChatCompletionContentPartTextParam{
				{Text: "System"},
				{Text: "Context"},
			}),
			openai.AssistantMessage(
				[]openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
					{OfText: &openai.ChatCompletionContentPartTextParam{Text: "Assistant"}},
					{OfText: &openai.ChatCompletionContentPartTextParam{Text: "Reply"}},
				},
			),
			openai.UserMessage("first user message"),
			openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
				openai.TextContentPart("latest"),
				openai.TextContentPart("question"),
			}),
		},
	}

	userContent, nonUser := extractUserAndNonUserContent(req)
	if userContent != "latest question" {
		t.Fatalf("expected latest user content, got %q", userContent)
	}
	if len(nonUser) != 2 {
		t.Fatalf("expected two non-user messages, got %d", len(nonUser))
	}
	if nonUser[0] != "System Context" {
		t.Fatalf("expected joined system content, got %q", nonUser[0])
	}
	if nonUser[1] != "Assistant Reply" {
		t.Fatalf("expected joined assistant content, got %q", nonUser[1])
	}
}

func TestExtractUserAndNonUserContentIgnoresToolMessages(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.ToolMessage("tool output", "tool-call-id"),
			openai.UserMessage("hello"),
		},
	}

	userContent, nonUser := extractUserAndNonUserContent(req)
	if userContent != "hello" {
		t.Fatalf("expected user content hello, got %q", userContent)
	}
	if len(nonUser) != 0 {
		t.Fatalf("expected tool messages to be ignored, got %#v", nonUser)
	}
}
