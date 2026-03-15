package extproc

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"sync"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// complexityScoreCache stores real classifier scores for messages that were
// previously classified. When a message becomes historical in a later turn,
// the cached score is used instead of the cheap length-based proxy.
// Key: SHA-256 hash of message text (first 16 bytes). Value: float64 score.
var complexityScoreCache sync.Map

// cacheComplexityScore stores a message's real complexity score for reuse.
func cacheComplexityScore(message string, score float64) {
	hash := sha256.Sum256([]byte(message))
	key := fmt.Sprintf("%x", hash[:16])
	complexityScoreCache.Store(key, score)
}

// getCachedComplexityScore retrieves a previously cached score.
func getCachedComplexityScore(message string) (float64, bool) {
	hash := sha256.Sum256([]byte(message))
	key := fmt.Sprintf("%x", hash[:16])
	val, ok := complexityScoreCache.Load(key)
	if !ok {
		return 0, false
	}
	return val.(float64), true
}

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

// EstimateComplexityFromLength returns a cheap complexity proxy based on
// message character length. Used for historical messages in the conversation
// to avoid expensive embedding computation on every turn.
//
// The mapping is a simple piecewise linear function:
//
//	len < 30   → 0.1  (trivial: "yes", "ok", "thanks")
//	len 30-100 → 0.3  (short: "can you explain that?")
//	len 100-300 → 0.6 (medium: brief technical questions)
//	len > 300  → 0.8  (long: detailed technical prompts)
func EstimateComplexityFromLength(messageLen int) float64 {
	switch {
	case messageLen < 30:
		return 0.1
	case messageLen < 100:
		return 0.3
	case messageLen < 300:
		return 0.6
	default:
		return 0.8
	}
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

// applyCRMOverride checks if conversational routing momentum disagrees with
// the current routing decision and returns an override if needed.
// Returns empty strings if no override is needed.
func (r *OpenAIRouter) applyCRMOverride(ctx *RequestContext, currentDecision string, currentModel string) (string, string) {
	allMsgs := ctx.AllUserMessages
	cfg := r.Config.RoutingMomentum
	attack := cfg.GetAttack()
	release := cfg.GetRelease()
	threshold := cfg.GetThreshold()

	// Cache the current message's complexity score for future turns
	currentScore := complexityScoreFromRules(ctx.VSRMatchedComplexity)
	if len(ctx.VSRMatchedComplexity) == 0 {
		currentScore = EstimateComplexityFromLength(len(allMsgs[len(allMsgs)-1]))
	}
	cacheComplexityScore(allMsgs[len(allMsgs)-1], currentScore)

	// Single message — no history to compute momentum, just cache the score
	if len(allMsgs) <= 1 {
		return "", ""
	}

	// Build signal history: cached real score for historical, current score for latest
	signalHistory := make([]float64, len(allMsgs))
	for i, msg := range allMsgs {
		if i == len(allMsgs)-1 {
			signalHistory[i] = currentScore
		} else if cached, ok := getCachedComplexityScore(msg); ok {
			signalHistory[i] = cached
		} else {
			signalHistory[i] = EstimateComplexityFromLength(len(msg))
		}
	}

	momentum := ComputeRoutingMomentum(signalHistory, attack, release)
	ctx.RoutingMomentum = momentum

	logging.Infof("[CRM] Routing momentum: %.3f (threshold=%.2f, attack=%.2f, release=%.2f, turns=%d, current_signal=%.3f)",
		momentum, threshold, attack, release, len(allMsgs), currentScore)

	// If momentum says "complex" but current message classified as "simple",
	// override with the complex decision to prevent mid-conversation model switch.
	if momentum > threshold && currentScore < threshold {
		return r.findComplexDecisionOverride(currentDecision, momentum, threshold)
	}

	return "", ""
}

// findComplexDecisionOverride searches for a decision that handles complex
// queries and returns its name and model as an override.
func (r *OpenAIRouter) findComplexDecisionOverride(currentDecision string, momentum, threshold float64) (string, string) {
	for _, d := range r.Config.Decisions {
		if d.Name == currentDecision {
			continue
		}
		if name, model := matchComplexDecision(d, currentDecision, momentum, threshold); name != "" {
			return name, model
		}
	}
	return "", ""
}

// matchComplexDecision checks if a decision handles complexity:hard and returns its model.
func matchComplexDecision(d config.Decision, currentDecision string, momentum, threshold float64) (string, string) {
	for _, rule := range d.Rules.Conditions {
		if rule.Type != "complexity" || !strings.HasSuffix(rule.Name, ":hard") {
			continue
		}
		if len(d.ModelRefs) == 0 {
			continue
		}
		logging.Infof("[CRM] Overriding decision %s→%s (momentum=%.3f > threshold=%.2f)",
			currentDecision, d.Name, momentum, threshold)
		return d.Name, d.ModelRefs[0].Model
	}
	return "", ""
}
