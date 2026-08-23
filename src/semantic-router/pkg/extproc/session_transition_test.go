package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func TestPopulateSessionTransitionFieldsUsesRetainedObjectHistory(t *testing.T) {
	ctx := &RequestContext{ResponseObjectState: &ResponseObjectState{
		ConversationID:     "conv-123",
		PreviousResponseID: "resp-abc",
		ConversationHistory: []*responseapi.StoredResponse{
			{Model: "model-a", Usage: &responseapi.Usage{InputTokens: 100, OutputTokens: 50}},
			{Model: "model-b", Usage: &responseapi.Usage{InputTokens: 20, OutputTokens: 10}},
		},
	}}
	populateSessionTransitionFields(ctx)
	if ctx.SessionID != "conv-123" || ctx.PreviousResponseID != "resp-abc" ||
		ctx.PreviousModel != "model-b" || ctx.TurnIndex != 2 || ctx.HistoryTokenCount != 180 {
		t.Fatalf("object session state = %+v", ctx)
	}
}

func TestPopulateSessionTransitionFieldsUsesNeutralMessages(t *testing.T) {
	ctx := &RequestContext{
		InferenceAccess: testInferenceRequestAccess("user-123", ""),
		SemanticRequest: &llmprotocol.Request{Generation: 1, Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "first"}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "answer"}}},
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "second"}}},
		}},
	}
	populateSessionTransitionFields(ctx)
	if ctx.SessionID == "" || ctx.TurnIndex != 1 || ctx.HistoryTokenCount != 2 {
		t.Fatalf("neutral session state = %+v", ctx)
	}
}

func TestSessionIdentityPriority(t *testing.T) {
	request := &llmprotocol.Request{Generation: 1, Metadata: map[string]string{"user_id": "semantic-user"}}
	ctx := &RequestContext{
		Headers:         map[string]string{headers.XClaudeCodeSessionID: "client-session"},
		SemanticRequest: request,
	}
	if got := deriveSessionIDFromAnthropicSignals(ctx); got != "client-session" {
		t.Fatalf("transport session = %q", got)
	}
	delete(ctx.Headers, headers.XClaudeCodeSessionID)
	if got := deriveSessionIDFromAnthropicSignals(ctx); got != "ant-md-semantic-user" {
		t.Fatalf("semantic metadata session = %q", got)
	}
	ctx.Headers[headers.XSessionID] = "pinned"
	populateSessionTransitionFields(ctx)
	if ctx.SessionID != "pinned" {
		t.Fatalf("pinned session = %q", ctx.SessionID)
	}
}

func TestHistoryTokenFallbackUsesInputAndOutput(t *testing.T) {
	stored := &responseapi.StoredResponse{
		Input: []responseapi.InputItem{{
			Type: responseapi.ItemTypeMessage, Role: responseapi.RoleUser,
			Content: json.RawMessage(`"abcdefgh"`),
		}},
		OutputText: "abcdefgh",
	}
	if got := estimateStoredResponseTokens(stored); got != 4 {
		t.Fatalf("estimated tokens = %d", got)
	}
	if got := historyTokensFromStoredResponses([]*responseapi.StoredResponse{nil, stored}); got != 4 {
		t.Fatalf("history tokens = %d", got)
	}
}

func TestEmptyNeutralThreadUsesRequestFallback(t *testing.T) {
	ctx := &RequestContext{RequestID: "request-123"}
	populateSessionTransitionFields(ctx)
	if !strings.HasPrefix(ctx.SessionID, "rid-") {
		t.Fatalf("session id = %q", ctx.SessionID)
	}
}
