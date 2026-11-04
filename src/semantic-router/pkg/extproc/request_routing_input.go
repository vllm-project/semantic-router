package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routinginput"
)

func (r *OpenAIRouter) routingInputSelector() *routinginput.Selector {
	if r == nil || r.Config == nil {
		return nil
	}
	return routinginput.For(r.Config.RoutingInput)
}

func observeRoutingInput(outcome routinginput.Outcome) {
	metrics.RecordRoutingInputRemovals(outcome.SkippedMessages, outcome.StrippedBlocks)
}

// replayPrompt is the prompt a Replay record keeps and its privacy check reads:
// the routing text when routing_input changed the request, otherwise the last
// user message.
func replayPrompt(ctx *RequestContext, request *llmprotocol.Request) string {
	if ctx != nil && ctx.RoutingInput.Changed() {
		return ctx.UserContent
	}
	prompt, _ := extractSemanticPromptAndTools(request)
	return prompt
}

func textOnlyMessage(message llmprotocol.Message) bool {
	for _, content := range message.Content {
		if content.Kind != llmprotocol.ContentText {
			return false
		}
	}
	return true
}
