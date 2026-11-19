package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// OpenCode names itself in its User-Agent and its session in x-session-id on
// every call. A subagent runs in a child session of its own: its calls carry
// the child's id in x-session-id and the session that spawned it in
// x-parent-session-id. OpenCode marks neither a request class nor a subagent
// type, so its session-title and compaction calls are known only by the
// built-in instructions they open with; both carry no tools.
const (
	openCodeUserAgentPrefix       = "opencode/"
	openCodeTitleInstruction      = "You are a title generator."
	openCodeCompactionInstruction = "You are an anchored context summarization assistant"
	// OpenCode 1.19 opens its compaction with this instead.
	openCodeCompactionAgentInstruction = "You are a context summarization agent."
)

func isOpenCode(ctx *RequestContext) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(headerValueCI(ctx, headers.UserAgent))), openCodeUserAgentPrefix)
}

// openCodeSubagentSessions are the session that spawned an OpenCode
// subagent's session and that child session, both empty for a call of the
// main session or of any other client.
func openCodeSubagentSessions(ctx *RequestContext) (parent string, child string) {
	if !isOpenCode(ctx) {
		return "", ""
	}
	parent = strings.TrimSpace(headerValueCI(ctx, headers.XParentSessionID))
	child = strings.TrimSpace(headerValueCI(ctx, headers.XSessionID))
	if parent == "" || child == "" || parent == child {
		return "", ""
	}
	return parent, child
}

// openCodeRequestClass is the class of an OpenCode call: auxiliary for a
// session title, compaction for a summary of the session, else subagent or
// main by the session it belongs to.
func openCodeRequestClass(request *llmprotocol.Request, subagent bool) string {
	if request != nil && len(request.Tools) == 0 {
		switch instructions := leadingInstructionText(request); {
		case strings.HasPrefix(instructions, openCodeTitleInstruction):
			return config.AgentRequestClassAuxiliary
		case strings.HasPrefix(instructions, openCodeCompactionInstruction),
			strings.HasPrefix(instructions, openCodeCompactionAgentInstruction):
			return config.AgentRequestClassCompaction
		}
	}
	if subagent {
		return config.AgentRequestClassSubagent
	}
	return config.AgentRequestClassMain
}

// leadingInstructionText is the text a request's first system or developer
// instruction opens with, from its instructions or else from a system message
// at the head of the conversation.
func leadingInstructionText(request *llmprotocol.Request) string {
	for _, block := range request.Instructions {
		if text := strings.TrimSpace(semanticText(block.Content)); text != "" {
			return text
		}
	}
	for _, message := range request.Messages {
		if message.Role != llmprotocol.RoleSystem && message.Role != llmprotocol.RoleDeveloper {
			break
		}
		if text := strings.TrimSpace(semanticText(message.Content)); text != "" {
			return text
		}
	}
	return ""
}

// isOpenCodeTitleRequest reports an OpenCode call asking for its session's
// title.
func isOpenCodeTitleRequest(ctx *RequestContext, request *llmprotocol.Request) bool {
	return isOpenCode(ctx) && openCodeRequestClass(request, false) == config.AgentRequestClassAuxiliary
}
