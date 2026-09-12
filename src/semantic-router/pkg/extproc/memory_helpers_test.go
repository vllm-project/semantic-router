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

func TestExtractAutoStoreUsesDecisionPolicy(t *testing.T) {
	on, off := true, false
	for _, tc := range []struct {
		name      string
		autoStore *bool
		wantValue bool
		wantSet   bool
	}{
		{name: "no_decision"},
		{name: "no_memory_plugin"},
		{name: "omitted"},
		{name: "explicit_false", autoStore: &off, wantSet: true},
		{name: "explicit_true", autoStore: &on, wantValue: true, wantSet: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := &RequestContext{}
			if tc.name != "no_decision" {
				ctx.VSRSelectedDecision = &config.Decision{Name: "memory"}
				if tc.name != "no_memory_plugin" {
					payload, err := config.NewStructuredPayload(config.MemoryPluginConfig{Enabled: true, AutoStore: tc.autoStore})
					if err != nil {
						t.Fatal(err)
					}
					ctx.VSRSelectedDecision.Plugins = []config.DecisionPlugin{{Type: config.DecisionPluginMemory, Configuration: payload}}
				}
			}
			value, set := extractAutoStore(ctx)
			if value != tc.wantValue || set != tc.wantSet {
				t.Fatalf("auto_store=(%v, %v), want (%v, %v)", value, set, tc.wantValue, tc.wantSet)
			}
		})
	}
}

func neutralTextMessage(role llmprotocol.Role, text string) llmprotocol.Message {
	return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}}
}
