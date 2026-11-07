package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// Codex CLI sends flat session-id and thread-id headers on every request. A
// root thread sends its own id as both; an inline review child keeps the root's
// session-id and sends its own thread-id. Only turn ids in the turn metadata
// change between turns. Without an agent header, session scope holds the child
// on the root's model; with thread-id as the agent header, the root and the
// child are each held on their own.
func TestProtectionCodexThreadIdentityHoldsRootAndReviewChildSeparately(t *testing.T) {
	const (
		sessionID   = "root-thread"
		childThread = "review-thread"
	)
	for _, tc := range []struct {
		name        string
		agentHeader string
		wantChild   string
	}{
		{name: "without_agent_header", wantChild: "frontier"},
		{name: "with_thread_id_agent_header", agentHeader: "thread-id", wantChild: "cheap"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			registry := selection.NewRegistry()
			cfg := routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeSession)
			cfg.RouterLearning.Protection.Identity.Headers.Session = stringPtr("session-id")
			if tc.agentHeader != "" {
				cfg.RouterLearning.Protection.Identity.Headers.Agent = stringPtr(tc.agentHeader)
			}
			router := &OpenAIRouter{Config: cfg, ModelSelector: registry}
			both := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
			type outcome struct{ model, keyHash string }
			selectRequest := func(thread, turn, proposal string, tool bool) outcome {
				t.Helper()
				ctx := &RequestContext{Headers: codexRequestHeaders(t, sessionID, thread, turn), SessionID: sessionID}
				ctx.VSRSelectedDecision = &config.Decision{Name: "choice"}
				ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: tool}
				registry.Register(selection.MethodStatic, selectionResultSelector{result: &selection.SelectionResult{
					SelectedModel: proposal, Score: 1, Confidence: 1, Method: selection.MethodStatic,
					AllScores: map[string]float64{proposal: 1},
				}})
				selCtx := router.buildSelectionContext(both, "choice", "", nil, "", nil, ctx)
				selected, _, err := router.selectModelFromCandidates(selCtx, nil, ctx)
				if err != nil || selected == nil {
					t.Fatalf("selection failed for thread=%q turn=%q: selected=%+v err=%v", thread, turn, selected, err)
				}
				if err := commitAgenticSessionDecision(ctx); err != nil {
					t.Fatal(err)
				}
				identity, _ := ctx.VSRLearningPolicy.ToMap()["identity"].(map[string]interface{})
				_, hasAgent := identity["agent"]
				if hasAgent != (tc.agentHeader != "") {
					t.Fatalf("thread=%q agent identity diagnostics present=%v with agent header %q: %#v",
						thread, hasAgent, tc.agentHeader, identity)
				}
				keyHash, _ := identity["memory_key_hash"].(string)
				return outcome{model: selected.Model, keyHash: keyHash}
			}

			root := selectRequest(sessionID, "turn-1", "frontier", false)
			if root.model != "frontier" {
				t.Fatalf("root did not establish frontier: %q", root.model)
			}
			rootRequests := []outcome{
				selectRequest(sessionID, "turn-1", "cheap", true),
				selectRequest(sessionID, "turn-2", "cheap", false),
				selectRequest(sessionID, "turn-2", "cheap", true),
			}
			child := selectRequest(childThread, "review-turn", "cheap", false)
			if child.model != tc.wantChild {
				t.Fatalf("review child first call got %q, want %q", child.model, tc.wantChild)
			}
			if got := selectRequest(childThread, "review-turn", "frontier", true); got.model != tc.wantChild || got.keyHash != child.keyHash {
				t.Errorf("review child tool continuation got %+v, want it held on %q under key %q", got, tc.wantChild, child.keyHash)
			}
			rootRequests = append(rootRequests,
				selectRequest(sessionID, "turn-3", "cheap", false),
				selectRequest(sessionID, "turn-3", "cheap", true),
			)
			for i, got := range rootRequests {
				if got.model != "frontier" || got.keyHash != root.keyHash {
					t.Errorf("root request %d got %+v, want it held on frontier under key %q", i+2, got, root.keyHash)
				}
			}
			if separate := child.keyHash != root.keyHash; separate != (tc.agentHeader != "") {
				t.Errorf("root and review child keys separate=%v with agent header %q", separate, tc.agentHeader)
			}
		})
	}
}

// codexRequestHeaders returns the identity headers of a Codex CLI 0.167.0
// request on thread inside sessionID. A review child names its parent thread
// and its subagent kind.
func codexRequestHeaders(t *testing.T, sessionID, thread, turn string) map[string]string {
	t.Helper()
	metadata, err := json.Marshal(map[string]string{"session_id": sessionID, "thread_id": thread, "turn_id": turn})
	if err != nil {
		t.Fatal(err)
	}
	headers := map[string]string{
		"session-id":            sessionID,
		"thread-id":             thread,
		"x-codex-turn-metadata": string(metadata),
	}
	if thread != sessionID {
		headers["x-codex-parent-thread-id"] = sessionID
		headers["x-openai-subagent"] = "review"
	}
	return headers
}
