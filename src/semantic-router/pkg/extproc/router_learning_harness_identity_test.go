package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/codexreview"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// The identity Claude Code 2.1.325 and Codex CLI 0.166.0 send, as captured:
// Claude Code sends x-claude-code-session-id on every request and
// x-claude-code-agent-id on a subagent's; Codex sends session-id and thread-id
// on every request, a root thread its session id as both, and the same ids in
// its turn metadata. With no identity headers configured, protection, Replay
// and Insights read them all.
func TestProtectionIdentityReadsSupportedHarnessesWithoutConfig(t *testing.T) {
	for _, tc := range []struct {
		name         string
		headers      map[string]string
		turn         codexreview.Turn
		session      string
		agent        string
		agentHeader  string
		insightAgent bool
	}{
		{
			name:    "claude code main conversation",
			headers: map[string]string{"x-claude-code-session-id": "cc-session"},
			session: "cc-session",
		},
		{
			name:    "claude code subagent",
			headers: map[string]string{"x-claude-code-session-id": "cc-session", "x-claude-code-agent-id": "a9a0d9cc"},
			session: "cc-session", agent: "a9a0d9cc", agentHeader: "x-claude-code-agent-id", insightAgent: true,
		},
		{
			name:    "codex root thread",
			headers: map[string]string{"session-id": "cx-session", "thread-id": "cx-session"},
			session: "cx-session", agent: "cx-session", agentHeader: "thread-id",
		},
		{
			name:    "codex subagent thread",
			headers: map[string]string{"session-id": "cx-session", "thread-id": "cx-child"},
			session: "cx-session", agent: "cx-child", agentHeader: "thread-id", insightAgent: true,
		},
		{
			name:    "codex turn metadata alone",
			headers: map[string]string{},
			turn:    codexreview.Turn{SessionID: "cx-session", ThreadID: "cx-child", TurnID: "turn"},
			session: "cx-session", agent: "cx-child", agentHeader: "x-codex-turn-metadata", insightAgent: true,
		},
		{
			name:    "a pinned session comes first",
			headers: map[string]string{"x-vsr-session-id": "pinned", "session-id": "cx-session", "thread-id": "cx-child"},
			session: "pinned", agent: "cx-child", agentHeader: "thread-id", insightAgent: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &config.RouterConfig{}
			cfg.RouterLearning.Protection.Scope = config.RouterLearningScopeSession
			r := &OpenAIRouter{Config: cfg}
			ctx := &RequestContext{Headers: tc.headers, AgentTurn: tc.turn}
			identity, ok := r.protectionIdentity(ctx, cfg.RouterLearning.Protection)
			if !ok || identity.sessionID != tc.session {
				t.Fatalf("session = %q (resolved %v), want %q", identity.sessionID, ok, tc.session)
			}
			if identity.agentID != tc.agent || identity.agentHeader != tc.agentHeader {
				t.Fatalf("agent = %q from %q, want %q from %q", identity.agentID, identity.agentHeader, tc.agent, tc.agentHeader)
			}
			record := routerreplay.RoutingRecord{SessionID: "derived-session"}
			r.populateReplayIdentity(&record, ctx)
			if record.SessionID != tc.session {
				t.Fatalf("replay filed the call under %q, want %q", record.SessionID, tc.session)
			}
			record.SessionPolicy = map[string]interface{}{"identity": newRouterLearningIdentityDiagnostics(identity).toPolicyMap()}
			if got := agentOf(record); (got != "") != tc.insightAgent {
				t.Fatalf("Insights agent = %q, want a subagent %v", got, tc.insightAgent)
			}
		})
	}
}

// Configured identity headers replace the defaults: a request is read only
// from the headers the configuration names, whatever harness sent it.
func TestProtectionIdentityConfiguredHeadersReplaceTheDefaults(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.RouterLearning.Protection.Scope = config.RouterLearningScopeSession
	cfg.RouterLearning.Protection.Identity.Headers.Session = stringPtr("x-tenant-session")
	cfg.RouterLearning.Protection.Identity.Headers.Agent = stringPtr("x-tenant-agent")
	r := &OpenAIRouter{Config: cfg}
	harness := map[string]string{
		"x-claude-code-session-id": "cc-session", "x-claude-code-agent-id": "cc-agent",
		"session-id": "cx-session", "thread-id": "cx-child",
	}
	turn := codexreview.Turn{SessionID: "cx-session", ThreadID: "cx-child", TurnID: "turn"}
	if identity, ok := r.protectionIdentity(&RequestContext{Headers: harness, AgentTurn: turn}, cfg.RouterLearning.Protection); ok {
		t.Fatalf("a request without x-tenant-session resolved %+v", identity)
	}
	harness["x-tenant-session"] = "tenant-session"
	identity, ok := r.protectionIdentity(&RequestContext{Headers: harness, AgentTurn: turn}, cfg.RouterLearning.Protection)
	if !ok || identity.sessionID != "tenant-session" || identity.agentID != "" || identity.agentHeader != "x-tenant-agent" {
		t.Fatalf("identity = %+v (resolved %v), want the tenant session and no agent", identity, ok)
	}
}
