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
	if ctx.SessionProvenance != SessionProvenanceResponseAPI {
		t.Fatalf("provenance = %q, want %q", ctx.SessionProvenance, SessionProvenanceResponseAPI)
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
	if ctx.SessionProvenance != SessionProvenanceResponseAPI {
		t.Fatalf("provenance = %q, want %q", ctx.SessionProvenance, SessionProvenanceResponseAPI)
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
	// No AuthzUserID header, so extractUserID returns "" and the session
	// falls to the message-structure fingerprint, not the userID+message
	// hash — both are SessionProvenanceMessageHash regardless.
	if ctx.SessionProvenance != SessionProvenanceMessageHash {
		t.Fatalf("provenance = %q, want %q", ctx.SessionProvenance, SessionProvenanceMessageHash)
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
	if ctx.SessionProvenance != SessionProvenanceHeader {
		t.Fatalf("provenance = %q, want %q", ctx.SessionProvenance, SessionProvenanceHeader)
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
	if ctx.SessionProvenance != SessionProvenanceRequestID {
		t.Fatalf("provenance = %q, want %q", ctx.SessionProvenance, SessionProvenanceRequestID)
	}
}

func TestPopulateSessionTransitionFieldsAnthropicProvenance(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{headers.XClaudeCodeSessionID: "client-session"},
	}
	populateSessionTransitionFields(ctx)
	if ctx.SessionID != "client-session" {
		t.Fatalf("session id = %q, want the raw Claude Code session id", ctx.SessionID)
	}
	if ctx.SessionProvenance != SessionProvenanceAnthropicPromptCache {
		t.Fatalf("provenance = %q, want %q", ctx.SessionProvenance, SessionProvenanceAnthropicPromptCache)
	}
}

func TestPopulateSessionTransitionFieldsMessageHashWithUserIDProvenance(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{headers.AuthzUserID: "user-42"},
		SemanticRequest: &llmprotocol.Request{Generation: 1, Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}}},
		}},
	}
	populateSessionTransitionFields(ctx)
	if ctx.SessionID == "" {
		t.Fatal("expected a non-empty session id")
	}
	if ctx.SessionProvenance != SessionProvenanceMessageHash {
		t.Fatalf("provenance = %q, want %q", ctx.SessionProvenance, SessionProvenanceMessageHash)
	}
}

func TestPopulateSessionTransitionFieldsAuthenticatedPrincipal(t *testing.T) {
	t.Run("present auth header", func(t *testing.T) {
		ctx := &RequestContext{Headers: map[string]string{headers.AuthzUserID: "user-42"}}
		populateSessionTransitionFields(ctx)
		if ctx.AuthenticatedPrincipal != "user-42" {
			t.Fatalf("AuthenticatedPrincipal = %q, want %q", ctx.AuthenticatedPrincipal, "user-42")
		}
	})
	t.Run("absent auth header", func(t *testing.T) {
		ctx := &RequestContext{}
		populateSessionTransitionFields(ctx)
		if ctx.AuthenticatedPrincipal != "" {
			t.Fatalf("AuthenticatedPrincipal = %q, want empty with no auth header", ctx.AuthenticatedPrincipal)
		}
	})
}

func TestPopulatePinnedSessionFromHeadersProvenanceAndPrincipal(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		headers.XSessionID:  "pinned-session",
		headers.AuthzUserID: "user-42",
	}}
	populatePinnedSessionFromHeaders(ctx)
	if ctx.SessionID != "pinned-session" {
		t.Fatalf("session id = %q", ctx.SessionID)
	}
	if ctx.SessionProvenance != SessionProvenanceHeader {
		t.Fatalf("provenance = %q, want %q", ctx.SessionProvenance, SessionProvenanceHeader)
	}
	if ctx.AuthenticatedPrincipal != "user-42" {
		t.Fatalf("AuthenticatedPrincipal = %q, want %q", ctx.AuthenticatedPrincipal, "user-42")
	}
}

func TestPopulatePinnedSessionFromHeaders_NoSessionHeader_ProvenanceStaysNone(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{headers.AuthzUserID: "user-42"}}
	populatePinnedSessionFromHeaders(ctx)
	if ctx.SessionID != "" {
		t.Fatalf("session id = %q, want empty with no x-session-id header", ctx.SessionID)
	}
	if ctx.SessionProvenance != SessionProvenanceNone {
		t.Fatalf("provenance = %q, want %q (unset)", ctx.SessionProvenance, SessionProvenanceNone)
	}
	// AuthenticatedPrincipal is populated regardless of whether a session
	// was pinned — the fast-extract path needs it available as early as
	// the session header itself.
	if ctx.AuthenticatedPrincipal != "user-42" {
		t.Fatalf("AuthenticatedPrincipal = %q, want %q", ctx.AuthenticatedPrincipal, "user-42")
	}
}
