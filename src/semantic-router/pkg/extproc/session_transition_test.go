package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func TestPopulateSessionTransitionFieldsUsesRetainedObjectHistory(t *testing.T) {
	ctx := &RequestContext{ResponseObjectState: &ResponseObjectState{
		ConversationID:     "conv-123",
		SessionTrackingID:  "respapi:conversation:conv-123",
		PreviousResponseID: "resp-abc",
		ConversationHistory: []*responseapi.StoredResponse{
			{Model: "model-a", Usage: &responseapi.Usage{InputTokens: 100, OutputTokens: 50}},
			{Model: "model-b", Usage: &responseapi.Usage{InputTokens: 20, OutputTokens: 10}},
		},
	}}
	populateSessionTransitionFields(ctx)
	if ctx.SessionID != "respapi:conversation:conv-123" || ctx.PreviousResponseID != "resp-abc" ||
		ctx.PreviousModel != "model-b" || ctx.TurnIndex != 2 || ctx.HistoryTokenCount != 180 {
		t.Fatalf("object session state = %+v", ctx)
	}
}

// TestPopulateSessionTransitionFieldsUsesTrackingIDWhenConversationEmpty
// covers a previous_response_id-only continuation: ConversationID is empty
// (strict external membership), but ctx.SessionID must still come from the
// non-empty internal SessionTrackingID rather than going blank.
func TestPopulateSessionTransitionFieldsUsesTrackingIDWhenConversationEmpty(t *testing.T) {
	ctx := &RequestContext{ResponseObjectState: &ResponseObjectState{
		ConversationID:      "",
		SessionTrackingID:   "respapi:lineage:resp-root",
		PreviousResponseID:  "resp-abc",
		ConversationHistory: []*responseapi.StoredResponse{{Model: "model-a"}},
	}}
	populateSessionTransitionFields(ctx)
	if ctx.SessionID != "respapi:lineage:resp-root" {
		t.Fatalf("session id = %q, want internal tracking id despite empty ConversationID", ctx.SessionID)
	}
}

func TestPopulateSessionTransitionFieldsUsesNeutralMessages(t *testing.T) {
	ctx := &RequestContext{
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

func TestSessionIdentityUsesTypedIngressIdentity(t *testing.T) {
	request := &llmprotocol.Request{
		Generation: 1,
		Metadata:   map[string]string{"user_id": "spoofed-body-user"},
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}},
		}},
	}
	ctx := &RequestContext{
		Headers:         map[string]string{"x-session-id": "spoofed-session", "x-claude-code-session-id": "spoofed-anthropic"},
		TrustedIdentity: authz.TrustedIdentity{SessionID: "client-session"},
		SemanticRequest: request,
	}
	populateSessionTransitionFields(ctx)
	if ctx.SessionID != "client-session" {
		t.Fatalf("typed ingress session = %q", ctx.SessionID)
	}

	ctx.TrustedIdentity.SessionID = ""
	ctx.SessionID = ""
	populateSessionTransitionFields(ctx)
	if strings.Contains(ctx.SessionID, "spoofed") || strings.Contains(ctx.SessionID, "body-user") {
		t.Fatalf("untrusted identity influenced session = %q", ctx.SessionID)
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
