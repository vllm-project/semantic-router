package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/codexreview"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The headers Claude Code 2.1.325 and Codex CLI 0.166.0 send, as captured.
// Claude Code names a subagent's type only with
// CLAUDE_CODE_GATEWAY_HINT_HEADERS set, and every custom agent as "custom";
// Codex names a spawn_agent child collab_spawn whatever its role, and a
// review child review. The role is the identity protection keys on.
func TestAgentFactsReadCapturedHarnessHeaders(t *testing.T) {
	const (
		ccSession = "555d064b-48e8-49b3-a9d8-80299c76d8db"
		cxSession = "01a1e34e-5e5b-7f52-bf87-13658a168d14"
	)
	subagent, main := config.AgentRoleSubagent, config.AgentRoleMain
	cc, cx := config.AgentHarnessClaudeCode, config.AgentHarnessCodex
	for _, tc := range []struct {
		name    string
		headers map[string]string
		turn    codexreview.Turn
		agent   *string
		want    classification.AgentFacts
	}{
		{
			name:    "claude code main agent",
			headers: map[string]string{"x-claude-code-session-id": ccSession},
			want:    classification.AgentFacts{Role: main, Harness: cc},
		},
		{
			name:    "claude code subagent without hint headers",
			headers: map[string]string{"x-claude-code-session-id": ccSession, "x-claude-code-agent-id": "a6f5e4d69556090f8"},
			want:    classification.AgentFacts{Role: subagent, Harness: cc},
		},
		{
			name: "claude code explore subagent",
			headers: map[string]string{
				"x-claude-code-session-id": ccSession, "x-claude-code-agent-id": "a63564960b8c70c16",
				"x-claude-code-agent-type": "Explore", "x-claude-code-request-class": "subagent",
			},
			want: classification.AgentFacts{Role: subagent, Harness: cc, Type: "Explore", RequestClass: "subagent"},
		},
		{
			name: "claude code compaction",
			headers: map[string]string{
				"x-claude-code-session-id": ccSession, "x-claude-code-request-class": "compaction", "x-claude-code-compaction": "manual",
			},
			want: classification.AgentFacts{Role: main, Harness: cc, RequestClass: "compaction"},
		},
		{
			name:    "a request class from another client is not read",
			headers: map[string]string{"session-id": cxSession, "thread-id": cxSession, "x-claude-code-request-class": "auxiliary"},
			want:    classification.AgentFacts{Role: main, Harness: cx, RequestClass: config.AgentRequestClassMain},
		},
		{
			name: "claude code custom subagent",
			headers: map[string]string{
				"x-claude-code-session-id": ccSession, "x-claude-code-agent-id": "a314db6ae39d828de", "x-claude-code-agent-type": "custom",
			},
			want: classification.AgentFacts{Role: subagent, Harness: cc, Type: "custom"},
		},
		{
			name:    "codex root thread",
			headers: map[string]string{"session-id": cxSession, "thread-id": cxSession},
			want:    classification.AgentFacts{Role: main, Harness: cx, RequestClass: config.AgentRequestClassMain},
		},
		{
			name: "codex spawned agent",
			headers: map[string]string{
				"session-id": cxSession, "thread-id": "01a1e34e-685e-70f1-9d78-dccc14907d6e",
				"x-codex-parent-thread-id": cxSession, "x-openai-subagent": "collab_spawn",
			},
			want: classification.AgentFacts{Role: subagent, Harness: cx, Type: "collab_spawn", RequestClass: config.AgentRequestClassSubagent},
		},
		{
			name: "codex review child",
			headers: map[string]string{
				"session-id": "01a1e352-9a79-7921-8b06-dc8c3c2170c9", "thread-id": "01a1e352-9acf-7f60-900e-d91726ad5675",
				"x-openai-subagent": "review",
			},
			want: classification.AgentFacts{Role: subagent, Harness: cx, Type: "review", RequestClass: config.AgentRequestClassSubagent},
		},
		{
			name: "codex turn metadata alone",
			turn: codexreview.Turn{SessionID: cxSession, ThreadID: "child", TurnID: "turn"},
			want: classification.AgentFacts{Role: subagent, Harness: cx, RequestClass: config.AgentRequestClassSubagent},
		},
		{
			name: "a type header without a subagent names no type",
			headers: map[string]string{
				"x-claude-code-session-id": ccSession, "x-claude-code-agent-type": "Explore",
			},
			want: classification.AgentFacts{Role: main, Harness: cc},
		},
		{
			name:    "configured agent header",
			headers: map[string]string{"x-session-id": "s", "x-tenant-agent": "worker-1"},
			agent:   stringPtr("x-tenant-agent"),
			want:    classification.AgentFacts{Role: subagent},
		},
		{
			name: "plain request",
			want: classification.AgentFacts{},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := config.RouterLearningProtectionConfig{}
			cfg.Identity.Headers.Agent = tc.agent
			ctx := &RequestContext{Headers: tc.headers, AgentTurn: tc.turn}
			if got := agentFacts(ctx, cfg); got != tc.want {
				t.Fatalf("agentFacts() = %+v, want %+v", got, tc.want)
			}
		})
	}
}

// Codex is known by its session-id. A thread-id alone, which any client may
// send, names the harness Router Replay records for the request, not Codex.
func TestAgentHarnessNeedsCodexSessionID(t *testing.T) {
	for _, tc := range []struct {
		userAgent, want string
	}{
		{"", ""},
		{"opencode/1.18.10", config.AgentHarnessOpenCode},
		{"Cursor/2.0", config.AgentHarnessCursor},
	} {
		ctx := &RequestContext{Headers: map[string]string{"thread-id": "t1", "user-agent": tc.userAgent}}
		if got := agentHarness(ctx); got != tc.want {
			t.Fatalf("user agent %q: agent signal harness %q, want %q as Replay records it", tc.userAgent, got, tc.want)
		}
	}
}
