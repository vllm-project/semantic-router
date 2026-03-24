package extproc

import (
	"strings"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Conversational Routing Momentum (CRM)
//
// Prevents model bouncing in multi-turn conversations by scanning a sliding
// window of recent messages for complexity evidence. Two signals:
//
//  1. Assistant response length — if the model produced a long response,
//     the conversation involves real work. Checking length is O(1).
//  2. User keyword matches — if a user message matches configured keyword
//     signals, it's a complex request. Keyword check is ~0.2ms.
//
// Both signals are essentially free. The conversation history is already
// in every Chat Completions request — CRM just reads what's there.
//
//   - Fast attack: a complex message immediately routes to the expensive model
//   - Slow release: takes `window` trivial turns to decay back to the cheap model
//
// Fully stateless. Zero server-side memory. Constant cost regardless of
// conversation length (bounded by window size, not history size).

// conversationMessage is a lightweight representation of a message in the
// conversation history, carrying only what CRM needs: role and text length.
type conversationMessage struct {
	Role    string
	Text    string
	TextLen int
}

// applyCRMOverride checks if recent conversation history shows evidence of
// complexity. If so, overrides the current routing decision to keep the
// conversation on the expensive model.
// Returns empty strings if no override is needed.
func (r *OpenAIRouter) applyCRMOverride(
	ctx *RequestContext,
	currentDecision string,
	currentModel string,
	cfg *config.RoutingMomentumConfig,
) (string, string) {
	history := ctx.ConversationHistory

	// Need at least 2 messages (a prior turn + current) for CRM to matter
	if len(history) < 2 {
		return "", ""
	}

	// If the current message already routed to the momentum-enabled decision,
	// no override needed — the classifier got it right.
	for i := range r.Config.Decisions {
		if r.Config.Decisions[i].Name == currentDecision {
			if m := r.Config.Decisions[i].GetMomentumConfig(); m != nil && m.Enabled {
				return "", ""
			}
		}
	}

	// Scan the window for complexity evidence, excluding the last message
	// (current user message — already classified by the full pipeline).
	windowTurns := cfg.GetWindow()
	responseThreshold := cfg.GetResponseThreshold()

	// Window is in turns (user+assistant pairs). Convert to message count:
	// each turn is typically 2 messages (user + assistant), so window of
	// N turns = last N*2 messages before the current one.
	windowMsgs := windowTurns * 2
	endIdx := len(history) - 1 // exclude current message
	startIdx := endIdx - windowMsgs
	if startIdx < 0 {
		startIdx = 0
	}

	for i := startIdx; i < endIdx; i++ {
		msg := history[i]

		// Signal 1: assistant response length — the model's own output is
		// ground truth for complexity. Long response = complex task.
		if msg.Role == "assistant" && msg.TextLen >= responseThreshold {
			logging.Infof("[CRM] Long response (%d chars) at position %d/%d (threshold=%d, window=%d turns), overriding",
				msg.TextLen, i+1, len(history), responseThreshold, windowTurns)
			return r.findMomentumDecisionOverride(currentDecision)
		}

		// Signal 2: user keyword match — cheap complexity check on user input
		if msg.Role == "user" && r.isComplexByKeywords(msg.Text) {
			logging.Infof("[CRM] Keyword match at position %d/%d (window=%d turns), overriding",
				i+1, len(history), windowTurns)
			return r.findMomentumDecisionOverride(currentDecision)
		}
	}

	logging.Debugf("[CRM] No complexity in window (window=%d turns, msgs=%d, response_threshold=%d), no override",
		windowTurns, len(history), responseThreshold)
	return "", ""
}

// isComplexByKeywords checks if a user message matches any configured keyword
// signals. Uses the keyword classifier (~0.2ms).
func (r *OpenAIRouter) isComplexByKeywords(text string) bool {
	if r.Classifier == nil || text == "" {
		return false
	}
	category, _, err := r.Classifier.ClassifyKeywordsOnly(text)
	return err == nil && category != ""
}

// extractConversationHistory extracts all user and assistant messages from
// the conversation in chronological order. Only carries role, text, and
// text length — the minimum CRM needs.
func extractConversationHistory(req *openai.ChatCompletionNewParams) []conversationMessage {
	var history []conversationMessage
	for _, msg := range req.Messages {
		switch {
		case msg.OfUser != nil:
			text := extractUserMessageText(msg.OfUser)
			if text != "" {
				history = append(history, conversationMessage{
					Role:    "user",
					Text:    text,
					TextLen: len(text),
				})
			}
		case msg.OfAssistant != nil:
			text := extractAssistantMessageText(msg.OfAssistant)
			if text != "" {
				history = append(history, conversationMessage{
					Role:    "assistant",
					Text:    text,
					TextLen: len(text),
				})
			}
		}
	}
	return history
}

// extractUserMessageText extracts the text content from a user message.
func extractUserMessageText(msg *openai.ChatCompletionUserMessageParam) string {
	if msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	var parts []string
	for _, part := range msg.Content.OfArrayOfContentParts {
		if part.OfText != nil {
			parts = append(parts, part.OfText.Text)
		}
	}
	return strings.Join(parts, " ")
}

// extractAssistantMessageText extracts the text content from an assistant message.
func extractAssistantMessageText(msg *openai.ChatCompletionAssistantMessageParam) string {
	if msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	var parts []string
	for _, part := range msg.Content.OfArrayOfContentParts {
		if part.OfText != nil {
			parts = append(parts, part.OfText.Text)
		}
	}
	return strings.Join(parts, " ")
}

// findEnabledMomentumConfig returns the first enabled momentum config from any decision.
// Returns nil if no decision has momentum enabled.
func findEnabledMomentumConfig(cfg *config.RouterConfig) *config.RoutingMomentumConfig {
	if cfg == nil {
		return nil
	}
	for i := range cfg.Decisions {
		if m := cfg.Decisions[i].GetMomentumConfig(); m != nil && m.Enabled {
			return m
		}
	}
	return nil
}

// findMomentumDecisionOverride finds the decision that has momentum enabled
// and returns it as an override target.
func (r *OpenAIRouter) findMomentumDecisionOverride(currentDecision string) (string, string) {
	for _, d := range r.Config.Decisions {
		if d.Name == currentDecision {
			continue
		}
		m := d.GetMomentumConfig()
		if m == nil || !m.Enabled || len(d.ModelRefs) == 0 {
			continue
		}
		logging.Infof("[CRM] Overriding decision %s→%s", currentDecision, d.Name)
		return d.Name, d.ModelRefs[0].Model
	}
	return "", ""
}
