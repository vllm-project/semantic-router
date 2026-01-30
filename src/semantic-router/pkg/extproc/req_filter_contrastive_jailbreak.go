package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// performContrastiveJailbreakCheck performs multi-turn jailbreak detection using contrastive embedding similarity
// This is a SEPARATE filter from the BERT-based jailbreak check and runs independently
// It analyzes the maximum contrastive score across all conversation turns to detect gradual escalation attacks
//
// Multi-turn Conversation Tracking:
// The OpenAI-compatible chat API is STATELESS - each request contains the full conversation history.
// The CLIENT APPLICATION is responsible for maintaining conversation state per user/session.
// Example request with multi-turn history:
//
//	{
//	  "model": "llama-3.1-8b",
//	  "messages": [
//	    {"role": "user", "content": "Turn 1..."},
//	    {"role": "assistant", "content": "Response 1..."},
//	    {"role": "user", "content": "Turn 2..."},
//	    {"role": "assistant", "content": "Response 2..."},
//	    {"role": "user", "content": "Turn 3 (current)..."}
//	  ]
//	}
//
// This function extracts all user messages from the messages array and computes the max contrastive
// score across them. It assumes the conversation history in a single request belongs to one session.
// If an attacker sends separate single-turn requests (without history), the multi-turn detection
// cannot correlate them - but this also defeats the attack's purpose since the model won't see
// the gradual "warming up" context that makes multi-turn attacks effective.
func (r *OpenAIRouter) performContrastiveJailbreakCheck(ctx *RequestContext, userContent string, nonUserMessages []string, categoryName string) (*ext_proc.ProcessingResponse, bool) {
	// Check if contrastive jailbreak detection is enabled
	if r.ContrastiveJailbreakClassifier == nil {
		return nil, false
	}

	// Check if enabled in config
	if r.Config == nil || r.Config.PromptGuard.ContrastiveJailbreak == nil || !r.Config.PromptGuard.ContrastiveJailbreak.Enabled {
		return nil, false
	}

	// Start contrastive jailbreak plugin span
	spanCtx, span := tracing.StartPluginSpan(ctx.TraceContext, "contrastive_jailbreak", categoryName)

	// Build list of all user messages for multi-turn analysis
	// Include current message + history (non-user messages may contain user content from prior turns)
	allUserMessages := make([]string, 0, len(nonUserMessages)+1)

	// Add historical messages (filter for user content if needed)
	// For simplicity, we include all non-user messages as they may contain relevant context
	// In practice, you might want to parse and extract only user role messages from history
	allUserMessages = append(allUserMessages, nonUserMessages...)

	// Add current user message
	if userContent != "" {
		allUserMessages = append(allUserMessages, userContent)
	}

	if len(allUserMessages) == 0 {
		tracing.EndPluginSpan(span, "skipped", 0, "no_messages")
		return nil, false
	}

	startTime := time.Now()
	result, err := r.ContrastiveJailbreakClassifier.AnalyzeConversation(allUserMessages)
	detectionTime := time.Since(startTime).Milliseconds()

	if err != nil {
		logging.Errorf("[Contrastive Jailbreak] Error performing analysis: %v", err)
		tracing.RecordError(span, err)
		tracing.EndPluginSpan(span, "error", detectionTime, "analysis_failed")
		// Continue processing despite analysis error
		metrics.RecordRequestError(ctx.RequestModel, "contrastive_jailbreak_failed")
		ctx.TraceContext = spanCtx
		return nil, false
	}

	if result.IsJailbreak {
		// Set tracing attributes
		tracing.SetSpanAttributes(span,
			attribute.Bool("contrastive_jailbreak.detected", true),
			attribute.Float64("contrastive_jailbreak.max_score", float64(result.MaxScore)),
			attribute.Int("contrastive_jailbreak.flagged_turn", result.FlaggedIndex),
			attribute.Int("contrastive_jailbreak.total_turns", len(result.ScoresPerTurn)),
			attribute.String(tracing.AttrSecurityAction, "blocked"))

		// End plugin span with blocked status
		tracing.EndPluginSpan(span, "blocked", detectionTime, "multi_turn_jailbreak_detected")

		logging.Warnf("[Contrastive Jailbreak] MULTI-TURN ATTACK BLOCKED: max_score=%.4f, threshold=%.4f, flagged_turn=%d/%d",
			result.MaxScore, r.ContrastiveJailbreakClassifier.GetThreshold(), result.FlaggedIndex, len(result.ScoresPerTurn))

		// Structured log for security block
		logging.LogEvent("security_block", map[string]interface{}{
			"reason_code":    "multi_turn_jailbreak_detected",
			"max_score":      result.MaxScore,
			"threshold":      r.ContrastiveJailbreakClassifier.GetThreshold(),
			"flagged_turn":   result.FlaggedIndex,
			"total_turns":    len(result.ScoresPerTurn),
			"request_id":     ctx.RequestID,
		})

		// Count this as a blocked request
		metrics.RecordRequestError(ctx.RequestModel, "contrastive_jailbreak_block")

		// Return immediate jailbreak violation response
		jailbreakResponse := http.CreateJailbreakViolationResponse(
			"multi_turn_escalation",
			result.MaxScore,
			ctx.ExpectStreamingResponse,
		)
		ctx.TraceContext = spanCtx
		return jailbreakResponse, true
	}

	// Not a jailbreak
	tracing.SetSpanAttributes(span,
		attribute.Bool("contrastive_jailbreak.detected", false),
		attribute.Float64("contrastive_jailbreak.max_score", float64(result.MaxScore)),
		attribute.Int("contrastive_jailbreak.total_turns", len(result.ScoresPerTurn)))

	tracing.EndPluginSpan(span, "success", detectionTime, "no_multi_turn_jailbreak")

	logging.Infof("[Contrastive Jailbreak] No multi-turn attack detected: max_score=%.4f, threshold=%.4f, turns=%d",
		result.MaxScore, r.ContrastiveJailbreakClassifier.GetThreshold(), len(result.ScoresPerTurn))

	ctx.TraceContext = spanCtx
	return nil, false
}
