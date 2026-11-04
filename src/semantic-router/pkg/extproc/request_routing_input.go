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

// routedUserMessage returns a user message as routing reads it, with its text
// replaced by the routing text and its other content unchanged, or reports
// that routing skips it.
func routedUserMessage(routing *routinginput.Walk, index int, message llmprotocol.Message) (llmprotocol.Message, bool) {
	text := semanticText(message.Content)
	routed, skipped := routing.UserMessage(index, text, textOnlyMessage(message))
	if skipped || routed == text {
		return message, skipped
	}
	view := message
	view.Content = make([]llmprotocol.Content, 0, len(message.Content)+1)
	if routed != "" {
		view.Content = append(view.Content, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: routed})
	}
	for _, content := range message.Content {
		if content.Kind != llmprotocol.ContentText {
			view.Content = append(view.Content, content)
		}
	}
	return view, false
}

func textOnlyMessage(message llmprotocol.Message) bool {
	for _, content := range message.Content {
		if content.Kind != llmprotocol.ContentText {
			return false
		}
	}
	return true
}
