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
// Claude Code's x-claude-code-request-class, or the class an OpenCode, Codex
// or OpenClaw call's built-in prompts show.
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
	subagent := facts.Role == config.AgentRoleSubagent
	switch facts.Harness {
	case config.AgentHarnessOpenCode:
		facts.RequestClass = openCodeRequestClass(ctx.SemanticRequest, subagent)
	case config.AgentHarnessCodex:
		facts.RequestClass = codexRequestClass(ctx, subagent)
	case config.AgentHarnessOpenClaw:
		facts.RequestClass = openClawRequestClass(ctx.SemanticRequest, subagent)
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
	case isOpenCode(ctx):
		return config.AgentHarnessOpenCode
	case isDeepSeekHarness(ctx):
		return config.AgentHarnessDeepSeekHarness
	case isCursor(ctx):
		return config.AgentHarnessCursor
	case isOpenClaw(ctx):
		return config.AgentHarnessOpenClaw
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
