package extproc

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/redaction"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestReplayCapturesConfiguredConversationIdentity(t *testing.T) {
	for _, mode := range []string{"apply", "observe", "bypass", "disabled"} {
		for _, stored := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stored=%v", mode, stored), func(t *testing.T) {
				sessionHeader, conversationHeader := "x-client-session", "x-client-conversation"
				cfg := &config.RouterConfig{}
				cfg.RouterLearning.Enabled = mode != "disabled"
				cfg.RouterLearning.Protection.Identity.Headers.Session = &sessionHeader
				cfg.RouterLearning.Protection.Identity.Headers.Conversation = &conversationHeader
				recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
				r := &OpenAIRouter{Config: cfg, ReplayRecorder: recorder, ReplayStoreShared: true}
				ctx := &RequestContext{
					Headers:                  map[string]string{"X-Client-Session": " session-one ", "X-Client-Conversation": " conversation-one ", "x-conversation-id": "unconfigured"},
					SemanticRequest:          &llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "A neutral turn."}}}}},
					RouterReplayPluginConfig: &config.RouterReplayPluginConfig{Enabled: true},
					VSRSelectedDecision:      &config.Decision{Name: "route", Adaptations: config.DecisionAdaptationsConfig{Mode: mode}},
				}
				if stored {
					ctx.ResponseObjectState = &ResponseObjectState{ConversationID: "stored-object-conversation", SessionTrackingID: "stored-object-session"}
				}
				r.startRouterReplay(ctx, "public", "backend", "route")
				if ctx.RouterReplayID == "" {
					t.Fatal("record not started")
				}
				response := r.handleRouterReplayRecordAPI("GET", ctx.RouterReplayID).GetImmediateResponse()
				var record routerreplay.RoutingRecord
				if err := json.Unmarshal(response.Body, &record); err != nil {
					t.Fatal(err)
				}
				if record.SessionID != "session-one" || record.ConversationID != "conversation-one" {
					t.Fatalf("configured identity missing: session=%q conversation=%q", record.SessionID, record.ConversationID)
				}
				if ctx.VSRLearningConversationID != "" {
					t.Fatal("replay must not activate protection")
				}
			})
		}
	}
}

func TestReplayIdentityFallbackAndSessionScope(t *testing.T) {
	for _, scope := range []string{config.RouterLearningScopeConversation, config.RouterLearningScopeSession} {
		for _, supplied := range []bool{false, true} {
			cfg := &config.RouterConfig{}
			cfg.RouterLearning.Protection.Scope = scope
			r := &OpenAIRouter{Config: cfg}
			ctx := &RequestContext{Headers: map[string]string{"x-session-id": "client-session"}}
			if supplied {
				ctx.Headers["x-conversation-id"] = "client-conversation"
			}
			record := routerreplay.RoutingRecord{SessionID: "legacy-session", ConversationID: "stored-conversation"}
			r.populateReplayIdentity(&record, ctx)
			want := "stored-conversation"
			if supplied {
				want = "client-conversation"
			}
			if record.ConversationID != want {
				t.Fatalf("scope %s changed identity fallback: %q", scope, record.ConversationID)
			}
		}
	}
}

func TestReplayTrajectorySeparatesConversationTurns(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	r := &OpenAIRouter{ReplayRecorder: recorder, ReplayStoreShared: true}
	first := []routerreplay.ToolTraceStep{
		{Type: replayToolStepUserInput, Text: "First task"},
		{Type: replayToolStepAssistantToolCall, ToolCallID: "call-a", ToolName: "read_state", Arguments: `{}`},
	}
	complete := append(append([]routerreplay.ToolTraceStep{}, first...), routerreplay.ToolTraceStep{Type: replayToolStepClientToolResult, ToolCallID: "call-a", Text: "ready"}, routerreplay.ToolTraceStep{Type: replayToolStepAssistantFinalResponse, Text: "First complete"})
	records := []routerreplay.RoutingRecord{
		{ConversationID: "conversation-a", TurnIndex: 0, ToolTrace: &routerreplay.ToolTrace{Steps: first}},
		{ConversationID: "conversation-a", TurnIndex: 0, ToolTrace: &routerreplay.ToolTrace{Steps: complete}},
		{ConversationID: "conversation-a", TurnIndex: 1, ToolTrace: &routerreplay.ToolTrace{Steps: []routerreplay.ToolTraceStep{{Type: replayToolStepUserInput, Text: "Next task"}, {Type: replayToolStepAssistantFinalResponse, Text: "Next complete"}}}},
		{ConversationID: "conversation-b", TurnIndex: 0, ToolTrace: &routerreplay.ToolTrace{Steps: []routerreplay.ToolTraceStep{{Type: replayToolStepUserInput, Text: "Separate task"}, {Type: replayToolStepAssistantFinalResponse, Text: "Separate complete"}}}},
	}
	for i, record := range records {
		record.SessionID = "shared-session"
		record.Recipe = "same-recipe"
		record.Timestamp = time.Unix(int64(i+1), 0)
		if _, err := recorder.AddRecord(record); err != nil {
			t.Fatal(err)
		}
	}
	response := r.handleRouterReplayTrajectoryAPI("GET", "session_id=shared-session&recipe=same-recipe").GetImmediateResponse()
	if int(response.Status.Code) != 200 {
		t.Fatal("trajectory failed")
	}
	var trajectory struct {
		RecordCount int              `json:"record_count"`
		TurnCount   int              `json:"turn_count"`
		Messages    []map[string]any `json:"messages"`
		Routes      []map[string]any `json:"routes"`
	}
	if err := json.Unmarshal(response.Body, &trajectory); err != nil {
		t.Fatal(err)
	}
	if trajectory.RecordCount != 4 || trajectory.TurnCount != 3 || len(trajectory.Messages) != 8 || len(trajectory.Routes) != 4 {
		t.Fatalf("conversation turns collapsed: records=%d turns=%d messages=%d routes=%d", trajectory.RecordCount, trajectory.TurnCount, len(trajectory.Messages), len(trajectory.Routes))
	}
	for i, message := range trajectory.Messages {
		want := "conversation-a"
		if i >= 6 {
			want = "conversation-b"
		}
		if message["conversation_id"] != want {
			t.Fatalf("message %d identity=%v", i, message["conversation_id"])
		}
	}
	for i, route := range trajectory.Routes {
		if route["conversation_id"] != records[i].ConversationID {
			t.Fatalf("route %d lost identity", i)
		}
	}
	redacted, _, err := redaction.RedactResponseBody(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(redacted, &trajectory); err != nil {
		t.Fatal(err)
	}
	if trajectory.Messages[6]["conversation_id"] != "conversation-b" || trajectory.Messages[6]["content"] != "" {
		t.Fatal("redaction lost identity or retained content")
	}
}

func TestReplayToolCallsDoNotCoalesceAcrossConversations(t *testing.T) {
	records := []routerreplay.RoutingRecord{
		{ConversationID: "first", TurnIndex: 0, ToolTrace: &routerreplay.ToolTrace{Steps: []routerreplay.ToolTraceStep{{Type: replayToolStepAssistantToolCall, ToolCallID: "call-first"}}}},
		{ConversationID: "second", TurnIndex: 0, ToolTrace: &routerreplay.ToolTrace{Steps: []routerreplay.ToolTraceStep{{Type: replayToolStepAssistantToolCall, ToolCallID: "call-second"}}}},
	}
	messages := buildTrajectoryMessages(buildTrajectoryTurns(records))
	if len(messages) != 2 || len(messages[0].ToolCalls) != 1 || len(messages[1].ToolCalls) != 1 || messages[0].ToolCalls[0].ID != "call-first" || messages[1].ToolCalls[0].ID != "call-second" {
		t.Fatalf("calls from separate conversations merged: %+v", messages)
	}
}
