package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// maxAgentTypeBytes bounds the subagent type or request class a header may
// name; a longer value names none.
const maxAgentTypeBytes = 128

// agentFacts is the agent a request belongs to, for the agent signal. A
// request is a subagent's exactly when protection reads an agent id for it,
// so a decision on the role holds under the key protection keeps, and the
// main agent's when it names a session and no agent. The type is the one the
// harness sends with a subagent's requests: Claude Code's
// x-claude-code-agent-type, Codex's x-openai-subagent. The request class is
// Claude Code's x-claude-code-request-class.
func agentFacts(ctx *RequestContext, cfg config.RouterLearningProtectionConfig) classification.AgentFacts {
	facts := classification.AgentFacts{Harness: agentHarness(ctx)}
	if facts.Harness == config.AgentHarnessClaudeCode {
		facts.RequestClass = boundedHeader(ctx, headers.XClaudeCodeRequestClass)
	}
	if agentID, _ := protectionAgentID(ctx, cfg); agentID != "" {
		facts.Role = config.AgentRoleSubagent
		facts.Type = subagentType(ctx, facts.Harness)
	} else if sessionID, _ := protectionSessionID(ctx, cfg); sessionID != "" {
		facts.Role = config.AgentRoleMain
	}
	return facts
}

// agentHarness names the harness whose identity headers the request carries.
func agentHarness(ctx *RequestContext) string {
	switch {
	case headerValueCI(ctx, headers.XClaudeCodeSessionID) != "" || headerValueCI(ctx, headers.XClaudeCodeAgentID) != "":
		return config.AgentHarnessClaudeCode
	case headerValueCI(ctx, headers.CodexSessionID) != "" || headerValueCI(ctx, headers.CodexThreadID) != "" ||
		ctx.AgentTurn.SessionID != "" || ctx.AgentTurn.ThreadID != "":
		return config.AgentHarnessCodex
	default:
		return ""
	}
}

func subagentType(ctx *RequestContext, harness string) string {
	header := ""
	switch harness {
	case config.AgentHarnessClaudeCode:
		header = headers.XClaudeCodeAgentType
	case config.AgentHarnessCodex:
		header = headers.XOpenAISubagent
	}
	return boundedHeader(ctx, header)
}

// boundedHeader is a header's trimmed value, empty when it is longer than a
// type or class may be.
func boundedHeader(ctx *RequestContext, header string) string {
	value := strings.TrimSpace(headerValueCI(ctx, header))
	if len(value) > maxAgentTypeBytes {
		return ""
	}
	return value
}
