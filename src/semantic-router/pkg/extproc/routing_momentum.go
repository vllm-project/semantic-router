package extproc

import (
	"strings"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Conversational Routing Momentum (CRM)
//
// An asymmetric low-pass filter for LLM routing signals, inspired by audio
// compressor attack/release dynamics. Prevents model bouncing in multi-turn
// conversations by applying different time constants for escalation vs
// de-escalation:
//
//   - Fast attack: quickly respond to complexity increases (don't give
//     hard questions to weak models)
//   - Slow release: gradually decay after complexity drops (don't bounce
//     a conversation to a different model mid-flow)
//
// The algorithm is stateless — momentum is computed from the conversation
// history present in each Chat Completions request.

// ComputeRoutingMomentum applies the asymmetric low-pass filter to a
// sequence of complexity signals and returns the current momentum value.
//
// Parameters:
//   - signals: per-turn complexity scores (0.0=trivial, 1.0=complex)
//   - attack: escalation coefficient (lower = faster, e.g. 0.3)
//   - release: de-escalation coefficient (higher = slower, e.g. 0.9)
//
// Returns momentum in [0.0, 1.0]. Compare against threshold to decide routing.
func ComputeRoutingMomentum(signals []float64, attack, release float64) float64 {
	if len(signals) == 0 {
		return 0.5 // neutral starting point
	}

	momentum := 0.5
	for _, signal := range signals {
		// Clamp signal to [0, 1]
		if signal < 0 {
			signal = 0
		} else if signal > 1 {
			signal = 1
		}

		if signal > momentum {
			// Escalation: fast attack — respond quickly to complexity increase
			momentum = attack*momentum + (1-attack)*signal
		} else {
			// De-escalation: slow release — resist dropping to cheaper model
			momentum = release*momentum + (1-release)*signal
		}
	}

	return momentum
}

// scoreMessageComplexity evaluates a single message's complexity using the
// classifier's signal evaluation. Returns a score in [0.0, 1.0].
// Uses the same classifier that evaluates the current message — no heuristics.
func (r *OpenAIRouter) scoreMessageComplexity(text string) float64 {
	if r.Classifier == nil || text == "" {
		return 0.5
	}
	signals := r.Classifier.EvaluateAllSignals(text)
	if signals == nil {
		return 0.5
	}

	// If complexity rules matched, use them directly
	score := complexityScoreFromRules(signals.MatchedComplexityRules)
	if len(signals.MatchedComplexityRules) > 0 {
		return score
	}

	// If the message matched keyword or embedding signals, it's complex
	if len(signals.MatchedKeywordRules) > 0 || len(signals.MatchedEmbeddingRules) > 0 {
		return 0.85
	}

	// No complexity, keyword, or embedding signals — message is trivial
	return 0.1
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

// extractAllUserMessages returns all user message texts from the conversation
// history in chronological order. Used by CRM to compute complexity signal
// history across conversation turns.
func extractAllUserMessages(req *openai.ChatCompletionNewParams) []string {
	var messages []string
	for _, msg := range req.Messages {
		if msg.OfUser == nil {
			continue
		}
		if text := extractUserMessageText(msg.OfUser); text != "" {
			messages = append(messages, text)
		}
	}
	return messages
}

// complexityScoreFromRules extracts a numerical score (0.0-1.0) from matched
// complexity rules like "rulename:hard", "rulename:easy", "rulename:medium".
func complexityScoreFromRules(rules []string) float64 {
	for _, rule := range rules {
		parts := strings.SplitN(rule, ":", 2)
		if len(parts) == 2 {
			switch parts[1] {
			case "hard":
				return 0.85
			case "medium":
				return 0.5
			case "easy":
				return 0.15
			}
		}
	}
	return 0.5 // neutral if no rules matched
}

// findEnabledMomentumConfig returns the first enabled momentum config from any decision.
// Returns nil if no decision has momentum enabled.
// Called on every request — O(n) over decisions, but decisions are typically < 10
// and the loop exits on first match.
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

// applyCRMOverride checks if conversational routing momentum disagrees with
// the current routing decision and returns an override if needed.
// Returns empty strings if no override is needed.
func (r *OpenAIRouter) applyCRMOverride(ctx *RequestContext, currentDecision string, currentModel string, cfg *config.RoutingMomentumConfig) (string, string) {
	allMsgs := ctx.AllUserMessages
	attack := cfg.GetAttack()
	release := cfg.GetRelease()
	threshold := cfg.GetThreshold()

	// Single message — no history to compute momentum
	if len(allMsgs) <= 1 {
		return "", ""
	}

	// Evaluate complexity for every message in the conversation.
	// Fully stateless — each request re-evaluates the entire history.
	signalHistory := make([]float64, len(allMsgs))
	for i, msg := range allMsgs {
		signalHistory[i] = r.scoreMessageComplexity(msg)
	}
	currentScore := signalHistory[len(signalHistory)-1]

	momentum := ComputeRoutingMomentum(signalHistory, attack, release)
	ctx.RoutingMomentum = momentum

	logging.Infof("[CRM] Routing momentum: %.3f (threshold=%.2f, attack=%.2f, release=%.2f, turns=%d, current_signal=%.3f)",
		momentum, threshold, attack, release, len(allMsgs), currentScore)

	// If momentum is high but the current message scored low, the conversation
	// is still in a "complex" flow — override to the momentum-enabled decision
	// to prevent mid-conversation model downgrade.
	if momentum > threshold && currentScore < threshold {
		return r.findMomentumDecisionOverride(currentDecision, momentum, threshold)
	}

	return "", ""
}

// findMomentumDecisionOverride finds the decision that has momentum enabled
// and returns it as an override target. This works regardless of what signal
// types the decision uses (domain, complexity, keyword, etc.).
func (r *OpenAIRouter) findMomentumDecisionOverride(currentDecision string, momentum, threshold float64) (string, string) {
	for _, d := range r.Config.Decisions {
		if d.Name == currentDecision {
			continue
		}
		m := d.GetMomentumConfig()
		if m == nil || !m.Enabled || len(d.ModelRefs) == 0 {
			continue
		}
		logging.Infof("[CRM] Overriding decision %s→%s (momentum=%.3f > threshold=%.2f)",
			currentDecision, d.Name, momentum, threshold)
		return d.Name, d.ModelRefs[0].Model
	}
	return "", ""
}
