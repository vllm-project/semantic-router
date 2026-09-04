package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	if _, _, history, err := extractMemoryInfo(ctx); err == nil || len(history) != 1 {
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

func TestExtractAutoStoreUsesDecisionPolicy(t *testing.T) {
	autoStore := true
	payload, err := config.NewStructuredPayload(config.MemoryPluginConfig{Enabled: true, AutoStore: &autoStore})
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{VSRSelectedDecision: &config.Decision{
		Name: "memory", Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginMemory, Configuration: payload}},
	}}
	if !extractAutoStore(ctx) {
		t.Fatal("decision auto-store policy was not used")
	}
}

func neutralTextMessage(role llmprotocol.Role, text string) llmprotocol.Message {
	return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}}
}
