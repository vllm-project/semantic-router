package extproc

import (
	"encoding/json"
	"fmt"
	"net/url"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// agentConversation replays one conversation the way a Chat client sends it:
// every request repeats the system prompt and the whole history so far.
type agentConversation struct {
	system   string
	messages []llmprotocol.Message
}

func newAgentConversation(system string, task string) *agentConversation {
	return &agentConversation{system: system, messages: []llmprotocol.Message{neutralTextMessage(llmprotocol.RoleUser, task)}}
}

func (c *agentConversation) request() *llmprotocol.Request {
	return &llmprotocol.Request{
		Generation: 1,
		Instructions: []llmprotocol.InstructionBlock{{
			Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: c.system}},
		}},
		Messages: append([]llmprotocol.Message(nil), c.messages...),
	}
}

// advance adds the answer to turn and what the client sends next: a tool call
// and its result on every third turn, else a reply and the next instruction.
func (c *agentConversation) advance(turn int) {
	if turn%3 == 0 {
		callID := fmt.Sprintf("call-%d", turn)
		c.messages = append(c.messages,
			llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: callID, Name: "run_tests", Arguments: `{}`},
			}}},
			llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
					CallID: callID, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "2 passed"}},
				},
			}}},
		)
		return
	}
	c.messages = append(c.messages,
		neutralTextMessage(llmprotocol.RoleAssistant, fmt.Sprintf("Step %d is done.", turn)),
		neutralTextMessage(llmprotocol.RoleUser, fmt.Sprintf("Continue with step %d.", turn+1)),
	)
}

func headerlessTurn(conversation *agentConversation, requestHeaders map[string]string, turn int, model string) *RequestContext {
	ctx := &RequestContext{
		RequestID: fmt.Sprintf("request-%d", turn), RequestModel: model,
		Headers: requestHeaders, SemanticRequest: conversation.request(),
	}
	populateSessionTransitionFields(ctx)
	return ctx
}

func TestHeaderlessConversationKeepsOneSessionForTwentyTurns(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	router := &OpenAIRouter{
		Config:         &config.RouterConfig{},
		ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(64, 0)), ReplayStoreShared: true,
	}
	conversation := newAgentConversation("You are a coding agent working in a repository.", "Fix the failing test in parser.py.")
	models := []string{"model-a", "model-b"}
	sessionID := ""
	for turn := 1; turn <= 20; turn++ {
		ctx := headerlessTurn(conversation, map[string]string{}, turn, models[turn%2])
		if turn == 1 {
			sessionID = ctx.SessionID
		}
		if ctx.SessionID == "" || ctx.SessionID != sessionID || ctx.SessionProvenance != SessionProvenanceMessageHash {
			t.Fatalf("turn %d: session %q (%s), want turn 1's %q", turn, ctx.SessionID, ctx.SessionProvenance, sessionID)
		}
		if previous := models[(turn-1)%2]; turn > 1 && ctx.PreviousModel != previous {
			t.Fatalf("turn %d: previous model %q, want the previous turn's %q", turn, ctx.PreviousModel, previous)
		}
		ctx.RouterReplayPluginConfig = &config.RouterReplayPluginConfig{Enabled: true}
		ctx.VSRSelectedDecision = &config.Decision{Name: "route"}
		router.startRouterReplay(ctx, "public", ctx.RequestModel, "route")
		recordSessionTurn(ctx, responseUsageMetrics{promptTokens: 100, completionTokens: 10}, sessiontelemetry.TurnPricing{})
		conversation.advance(turn)
	}

	usage, ok := sessiontelemetry.GetRouterSessionSnapshot(sessiontelemetry.RoutingSessionKey("", sessionID), time.Now())
	if !ok || usage.CumulativePromptTokens != 2000 || usage.CumulativeCompletionTokens != 200 {
		t.Fatalf("session usage did not accumulate over the 20 turns: %+v found=%t", usage, ok)
	}
	response := router.handleRouterReplayTrajectoryAPI("GET", "session_id="+url.QueryEscape(sessionID)).GetImmediateResponse()
	var trajectory struct {
		RecordCount int `json:"record_count"`
	}
	if err := json.Unmarshal(response.GetBody(), &trajectory); err != nil || trajectory.RecordCount != 20 {
		t.Fatalf("Replay shows %d of the conversation's 20 requests in its session (err=%v)", trajectory.RecordCount, err)
	}
}

func TestParallelHeaderlessConversationsStayDistinct(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	const system = "You are a coding agent working in a repository."
	first := newAgentConversation(system, "Fix the failing test in parser.py.")
	second := newAgentConversation(system, "Add a --dry-run flag to the release script.")
	sessions := map[*agentConversation]string{}
	for turn := 1; turn <= 6; turn++ {
		for _, run := range []struct {
			conversation *agentConversation
			model        string
		}{{first, "model-a"}, {second, "model-b"}} {
			ctx := headerlessTurn(run.conversation, map[string]string{}, turn, run.model)
			if turn == 1 {
				sessions[run.conversation] = ctx.SessionID
			}
			if ctx.SessionID == "" || ctx.SessionID != sessions[run.conversation] {
				t.Fatalf("turn %d: session %q, want %q", turn, ctx.SessionID, sessions[run.conversation])
			}
			if turn > 1 && ctx.PreviousModel != run.model {
				t.Fatalf("turn %d: previous model %q came from the other conversation, want %q", turn, ctx.PreviousModel, run.model)
			}
			recordSessionTurn(ctx, responseUsageMetrics{promptTokens: 100, completionTokens: 10}, sessiontelemetry.TurnPricing{})
			run.conversation.advance(turn)
		}
	}
	if sessions[first] == sessions[second] {
		t.Fatalf("two conversations share session %q", sessions[first])
	}

	// The same opening in two tenants is two conversations.
	same := newAgentConversation(system, "Fix the failing test in parser.py.")
	tenantA := headerlessTurn(same, map[string]string{headers.AuthzTenantID: "tenant-a"}, 1, "model-a")
	tenantB := headerlessTurn(same, map[string]string{headers.AuthzTenantID: "tenant-b"}, 1, "model-a")
	if tenantA.SessionID == "" || tenantA.SessionID == tenantB.SessionID || tenantA.SessionID == sessions[first] {
		t.Fatalf("tenants share a session: %q, %q, untenanted %q", tenantA.SessionID, tenantB.SessionID, sessions[first])
	}

	// A session header stays authoritative.
	pinned := headerlessTurn(first, map[string]string{headers.XSessionID: "client-session"}, 7, "model-a")
	if pinned.SessionID != "client-session" || pinned.SessionProvenance != SessionProvenanceHeader {
		t.Fatalf("session header lost to the fingerprint: %q (%s)", pinned.SessionID, pinned.SessionProvenance)
	}
}

func TestSessionHeaderOnlyClientGetsProtection(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	registry := selection.NewRegistry()
	// The protection tutorial's settings: scope conversation, identity x-session-id and x-conversation-id.
	router := &OpenAIRouter{Config: routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation), ModelSelector: registry}
	both := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
	route := func(conversation *agentConversation, requestHeaders map[string]string, proposal string, toolLoop bool, candidates []config.ModelRef) (string, *RequestContext) {
		t.Helper()
		ctx := &RequestContext{Headers: requestHeaders, SemanticRequest: conversation.request()}
		populateSessionTransitionFields(ctx)
		ctx.VSRSelectedDecision = &config.Decision{Name: "choice"}
		ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: toolLoop}
		registry.Register(selection.MethodStatic, selectionResultSelector{result: &selection.SelectionResult{
			SelectedModel: proposal, Score: 1, Confidence: 1, Method: selection.MethodStatic,
			AllScores: map[string]float64{proposal: 1},
		}})
		selCtx := router.buildSelectionContext(candidates, "choice", "", "", nil, ctx)
		selected, _, err := router.selectModelFromCandidates(selCtx, nil, ctx)
		if err != nil || selected == nil {
			t.Fatalf("selection failed: selected=%+v err=%v", selected, err)
		}
		if err := commitAgenticSessionDecision(ctx); err != nil {
			t.Fatal(err)
		}
		return selected.Model, ctx
	}

	session := map[string]string{headers.XSessionID: "agent-run-1"}
	thread := newAgentConversation("You are a coding agent working in a repository.", "Fix the failing test in parser.py.")
	if model, ctx := route(thread, session, "frontier", false, both); model != "frontier" ||
		ctx.VSRLearningPolicy.String("reason") == protectionReasonMissingIdentity {
		t.Fatalf("first turn: model %q, protection %#v", model, ctx.VSRLearningPolicy.ToMap())
	}
	thread.advance(3)
	model, ctx := route(thread, session, "cheap", true, both)
	if model != "frontier" {
		t.Fatalf("an x-session-id client lost its model in a tool loop: got %q, protection %#v", model, ctx.VSRLearningPolicy.ToMap())
	}
	if part := learningIdentityPart(t, ctx.VSRLearningPolicy, "conversation"); part["source"] != "message_hash" || part["status"] != "present" {
		t.Fatalf("conversation identity diagnostics %#v", part)
	}

	// A side call in the same session, such as a title request, is its own
	// conversation: it neither inherits nor replaces the agent thread's model.
	title := newAgentConversation("Write a five-word title for this conversation.", "Fix the failing test in parser.py.")
	if model, _ := route(title, session, "cheap", false, []config.ModelRef{{Model: "cheap"}}); model != "cheap" {
		t.Fatalf("side call: got %q, want its only candidate", model)
	}
	thread.advance(6)
	if model, ctx := route(thread, session, "cheap", true, both); model != "frontier" {
		t.Fatalf("the side call took over the agent thread: got %q, protection %#v", model, ctx.VSRLearningPolicy.ToMap())
	}

	// Without a session header the derived fingerprint never enables protection.
	anonymous := newAgentConversation("You are a coding agent working in a repository.", "Rename natural_list to natural_join.")
	route(anonymous, map[string]string{}, "frontier", false, both)
	anonymous.advance(3)
	if model, ctx := route(anonymous, map[string]string{}, "cheap", true, both); model != "cheap" ||
		ctx.VSRLearningPolicy.String("reason") != protectionReasonMissingIdentity {
		t.Fatalf("header-less request was protected: got %q, protection %#v", model, ctx.VSRLearningPolicy.ToMap())
	}
}
