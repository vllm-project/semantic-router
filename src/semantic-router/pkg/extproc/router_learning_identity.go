package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// protectionSessionID is the session id protection keys on and the header it
// came from: identity.headers.session when configured, else the session id
// the request pins, else the session a supported harness sends.
func protectionSessionID(ctx *RequestContext, cfg config.RouterLearningProtectionConfig) (string, string) {
	if cfg.HeaderName("session") == "" {
		if sessionID, header := pinnedSessionID(ctx); sessionID != "" {
			return sessionID, header
		}
	}
	return protectionIdentityHeader(ctx, cfg, "session", ctx.AgentTurn.SessionID)
}

// protectionAgentID is the agent id protection keys on and the header it came
// from: identity.headers.agent when configured, else the agent a supported
// harness sends, a Claude Code subagent's id or a Codex thread's. A Codex
// thread whose id is its session-id is the root thread, the main
// conversation, and names no agent, also when the request pins a session.
func protectionAgentID(ctx *RequestContext, cfg config.RouterLearningProtectionConfig) (string, string) {
	agentID, header := protectionIdentityHeader(ctx, cfg, "agent", ctx.AgentTurn.ThreadID)
	if header == headers.CodexThreadID || header == headers.XCodexTurnMetadata {
		codexSessionID := strings.TrimSpace(headerValueCI(ctx, headers.CodexSessionID))
		if agentID == codexSessionID || agentID == ctx.AgentTurn.SessionID {
			return "", header
		}
	}
	return agentID, header
}

// protectionIdentityHeader reads key's id from the first identity header that
// carries one. A configured header is reported even when the request lacks
// it. With none configured, the id Codex's turn metadata carries comes last:
// Codex sends the same ids in its headers.
func protectionIdentityHeader(ctx *RequestContext, cfg config.RouterLearningProtectionConfig, key, codexTurnID string) (string, string) {
	for _, header := range cfg.IdentityHeaderNames(key) {
		if id := strings.TrimSpace(headerValueCI(ctx, header)); id != "" {
			return id, header
		}
	}
	if configured := cfg.HeaderName(key); configured != "" {
		return "", configured
	}
	if codexTurnID != "" {
		return codexTurnID, headers.XCodexTurnMetadata
	}
	return "", ""
}
