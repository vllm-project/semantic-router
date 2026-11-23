package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Codex and OpenClaw mark no request class either. Codex compacts with a
// tool-less call whose last user message is its built-in checkpoint prompt,
// and names its title thread in its turn metadata; OpenClaw compacts with a
// tool-less call whose system prompt is its built-in summarization prompt.
const (
	codexCompactionPrompt    = "You are performing a CONTEXT CHECKPOINT COMPACTION."
	openClawCompactionPrompt = "You are a context summarization assistant."
)

// codexRequestClass is the class of a Codex call: auxiliary for its title
// thread, compaction for its checkpoint, else subagent or main by its thread.
func codexRequestClass(ctx *RequestContext, subagent bool) string {
	if ctx.AgentTurn.ThreadSource == codexThreadSourceTitle {
		return config.AgentRequestClassAuxiliary
	}
	if request := ctx.SemanticRequest; request != nil && len(request.Tools) == 0 &&
		strings.HasPrefix(lastUserText(request), codexCompactionPrompt) {
		return config.AgentRequestClassCompaction
	}
	return mainOrSubagent(subagent)
}

// openClawRequestClass is the class of an OpenClaw call: compaction for its
// summary of the session, else subagent or main.
func openClawRequestClass(request *llmprotocol.Request, subagent bool) string {
	if request != nil && len(request.Tools) == 0 && strings.HasPrefix(leadingInstructionText(request), openClawCompactionPrompt) {
		return config.AgentRequestClassCompaction
	}
	return mainOrSubagent(subagent)
}

func mainOrSubagent(subagent bool) string {
	if subagent {
		return config.AgentRequestClassSubagent
	}
	return config.AgentRequestClassMain
}

// lastUserText is the text of a request's last message when a user wrote it.
func lastUserText(request *llmprotocol.Request) string {
	if len(request.Messages) == 0 {
		return ""
	}
	last := request.Messages[len(request.Messages)-1]
	if last.Role != llmprotocol.RoleUser {
		return ""
	}
	return strings.TrimSpace(semanticText(last.Content))
}
