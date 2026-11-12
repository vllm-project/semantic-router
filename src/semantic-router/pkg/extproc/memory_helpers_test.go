package extproc

import (
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func TestExtractMemoryInfoUsesNeutralConversationAndAuthenticatedIdentity(t *testing.T) {
	request := &llmprotocol.Request{Generation: 1, Messages: []llmprotocol.Message{
		neutralTextMessage(llmprotocol.RoleUser, "What is Go?"),
		neutralTextMessage(llmprotocol.RoleAssistant, "A language."),
	}}
	ctx := &RequestContext{
		SemanticRequest: request,
		SessionID:       "session-7",
		Headers:         map[string]string{"x-authz-user-id": "user-7"},
	}
	sessionID, userID, history, err := extractMemoryInfo(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if sessionID != "session-7" || userID != "user-7" || len(history) != 2 {
		t.Fatalf("session=%q user=%q history=%+v", sessionID, userID, history)
	}
	history[0].Content[0].Text = "changed"
	if request.Messages[0].Content[0].Text != "What is Go?" {
		t.Fatal("returned history aliases the live neutral request")
	}
}

func TestExtractMemoryInfoPrefixesRetainedObjectHistory(t *testing.T) {
	ctx := &RequestContext{
		SemanticRequest: &llmprotocol.Request{Generation: 1, Messages: []llmprotocol.Message{
			neutralTextMessage(llmprotocol.RoleUser, "current"),
		}},
		Headers: map[string]string{"x-authz-user-id": "user-7"},
		ResponseObjectState: &ResponseObjectState{ConversationHistory: []*responseapi.StoredResponse{{
			Input: []responseapi.InputItem{{
				Type: responseapi.ItemTypeMessage, Role: responseapi.RoleUser,
				Content: json.RawMessage(`"previous"`),
			}},
			OutputText: "answer",
		}}},
	}
	_, _, history, err := extractMemoryInfo(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(history) != 3 || semanticText(history[0].Content) != "previous" ||
		semanticText(history[1].Content) != "answer" || semanticText(history[2].Content) != "current" {
		t.Fatalf("history=%+v", history)
	}
}

func TestExtractMemoryInfoRejectsMissingAuthenticatedUser(t *testing.T) {
	ctx := &RequestContext{SemanticRequest: &llmprotocol.Request{Generation: 1, Messages: []llmprotocol.Message{
		neutralTextMessage(llmprotocol.RoleUser, "hello"),
	}}}
	if _, _, history, err := extractMemoryInfo(ctx); err == nil || len(history) != 0 {
		t.Fatalf("history=%+v err=%v", history, err)
	}
}

func TestNeutralMemoryMessageAndCurrentUserExtraction(t *testing.T) {
	ctx := &RequestContext{SemanticRequest: &llmprotocol.Request{Generation: 1, Messages: []llmprotocol.Message{
		neutralTextMessage(llmprotocol.RoleUser, "first"),
		neutralTextMessage(llmprotocol.RoleAssistant, "answer"),
		neutralTextMessage(llmprotocol.RoleUser, "latest"),
	}}}
	if got := extractCurrentUserMessage(ctx); got != "latest" {
		t.Fatalf("current user message=%q", got)
	}
	if got := neutralMemoryMessage("not-a-role", "text"); got.Role != llmprotocol.RoleUser {
		t.Fatalf("unknown role=%q", got.Role)
	}
}

func neutralTextMessage(role llmprotocol.Role, text string) llmprotocol.Message {
	return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}}
}

func conversationStartRequest(role llmprotocol.Role, system string, messages ...llmprotocol.Message) *llmprotocol.Request {
	return &llmprotocol.Request{
		Instructions: []llmprotocol.InstructionBlock{{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: system}}}},
		Messages:     messages,
	}
}

func TestDeriveConversationStartIDReadsOnlyTheConversationStart(t *testing.T) {
	system := "You are a coding agent." + strings.Repeat(" Follow the repository's conventions.", 20)
	task := "Fix the failing test in parser.py." + strings.Repeat(" The traceback follows.", 40)
	base := deriveConversationStartID(conversationStartRequest(llmprotocol.RoleSystem, system,
		neutralTextMessage(llmprotocol.RoleUser, task)), "")
	if !strings.HasPrefix(base, "cc-start-") || len(base) != len("cc-start-")+16 {
		t.Fatalf("fingerprint %q", base)
	}
	for name, test := range map[string]struct {
		request *llmprotocol.Request
		tenant  string
		same    bool
	}{
		"later turns": {request: conversationStartRequest(llmprotocol.RoleSystem, system,
			neutralTextMessage(llmprotocol.RoleUser, task),
			neutralTextMessage(llmprotocol.RoleAssistant, "Running the tests."),
			neutralTextMessage(llmprotocol.RoleUser, "Now update the changelog.")), same: true},
		"developer role": {request: conversationStartRequest(llmprotocol.RoleDeveloper, system,
			neutralTextMessage(llmprotocol.RoleUser, task)), same: true},
		"instructions changed past the bound": {request: conversationStartRequest(llmprotocol.RoleSystem, system+" Today is Friday.",
			neutralTextMessage(llmprotocol.RoleUser, task)), same: true},
		"task changed past the bound": {request: conversationStartRequest(llmprotocol.RoleSystem, system,
			neutralTextMessage(llmprotocol.RoleUser, task+" One more line.")), same: true},
		"other instructions": {request: conversationStartRequest(llmprotocol.RoleSystem, "Write a five-word title.",
			neutralTextMessage(llmprotocol.RoleUser, task))},
		"other task": {request: conversationStartRequest(llmprotocol.RoleSystem, system,
			neutralTextMessage(llmprotocol.RoleUser, "Add a --dry-run flag to the release script."))},
		"tenant": {request: conversationStartRequest(llmprotocol.RoleSystem, system,
			neutralTextMessage(llmprotocol.RoleUser, task)), tenant: "tenant-a"},
	} {
		if got := deriveConversationStartID(test.request, test.tenant); (got == base) != test.same || got == "" {
			t.Errorf("%s: %q against %q, same=%t", name, got, base, test.same)
		}
	}
}

func TestDeriveConversationStartIDBoundaries(t *testing.T) {
	if got := deriveConversationStartID(&llmprotocol.Request{Messages: []llmprotocol.Message{
		neutralTextMessage(llmprotocol.RoleAssistant, "No user turn yet."),
	}}, "tenant-a"); got != "" {
		t.Fatalf("a request without instructions or user text got %q", got)
	}
	// Each part is length-prefixed, so text cannot move between parts.
	split := deriveConversationStartID(conversationStartRequest(llmprotocol.RoleSystem, "b", neutralTextMessage(llmprotocol.RoleUser, "c")), "a")
	joined := deriveConversationStartID(conversationStartRequest(llmprotocol.RoleSystem, "", neutralTextMessage(llmprotocol.RoleUser, "c")), "ab")
	if split == joined {
		t.Fatalf("tenant and instructions collide: %q", split)
	}
	wide := strings.Repeat("会", conversationStartRunes+8)
	if got := leadingRunes(wide, conversationStartRunes); !utf8.ValidString(got) || utf8.RuneCountInString(got) != conversationStartRunes {
		t.Fatalf("leadingRunes kept %d runes, valid=%t", utf8.RuneCountInString(got), utf8.ValidString(got))
	}
}

func TestConversationStartIsTakenOnceFromTheClientRequest(t *testing.T) {
	ctx := &RequestContext{SemanticRequest: conversationStartRequest(llmprotocol.RoleSystem, "You are a coding agent.",
		neutralTextMessage(llmprotocol.RoleUser, "Fix the failing test."))}
	populateSessionTransitionFields(ctx)
	taken := ctx.SessionID
	ctx.SemanticRequest.Instructions[0].Content[0].Text = "A system prompt the Router inserted."
	if got := ctx.conversationStart(); got != taken || got == "" {
		t.Fatalf("conversation start moved after the request changed: %q, was %q", got, taken)
	}
	responses := &RequestContext{
		SemanticRequest:     ctx.SemanticRequest,
		ResponseObjectState: &ResponseObjectState{SessionTrackingID: "respapi:lineage:resp_root"},
	}
	if got := responses.conversationStart(); got != "" {
		t.Fatalf("a Responses request got a conversation start %q; its lineage identifies it", got)
	}
}
