package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// classifyTurnOutcome derives the outcome category from signals the response
// path already has. Provider failures stay non-attributable so infrastructure
// noise is never read as model regression; attribution itself is derived from
// the category by sessiontelemetry.
func classifyTurnOutcome(ctx *RequestContext, usage responseUsageMetrics) sessiontelemetry.TurnOutcomeCategory {
	if ctx == nil {
		return sessiontelemetry.TurnMissing
	}
	if status := ctx.UpstreamStatusCode; status == 429 || status >= 500 {
		return sessiontelemetry.TurnProviderError
	}
	// TODO(ayrnb): classify failed tool results as TurnToolError once the
	// signal layer exposes a tool success/failure flag.
	if usage.invalid {
		return sessiontelemetry.TurnMissing
	}
	// Reported-but-empty output means the turn completed without progress;
	// an unreported count means the outcome could not be observed at all.
	if usage.completionTokensReported && usage.completionTokens <= 0 {
		return sessiontelemetry.TurnNoProgress
	}
	if usage.completionTokens <= 0 {
		return sessiontelemetry.TurnMissing
	}
	return sessiontelemetry.TurnProgress
}

// recordSessionTurnOutcome captures one terminal outcome, independently of billing.
func recordSessionTurnOutcome(ctx *RequestContext, usage responseUsageMetrics, pricing ...sessiontelemetry.TurnPricing) {
	if ctx == nil || ctx.VSRProgressGateConfig == nil || !ctx.VSRProgressGateConfig.Enabled ||
		ctx.VSRProgressOutcomeRecorded || requestBypassesRouting(ctx) {
		return
	}
	key := progressEvidenceStateKey(ctx)
	if key == "" {
		return
	}
	ctx.VSRProgressOutcomeRecorded = true
	at := ctx.StartTime
	if at.IsZero() {
		at = time.Now()
	}
	outcome := sessiontelemetry.TurnOutcome{
		RequestID: ctx.RequestID, TurnIndex: ctx.TurnIndex, Model: ctx.RequestModel,
		Category: classifyTurnOutcome(ctx, usage), OutputTokens: int64(usage.completionTokens),
		LatencyMs: int64(ctx.TTFTSeconds * 1000), LatencyKnown: ctx.TTFTRecorded,
		Source: sessiontelemetry.TurnSourceRouterObserved,
	}
	if len(pricing) > 0 && (pricing[0].PromptPer1M > 0 || pricing[0].CompletionPer1M > 0) && !usage.invalid {
		outcome.Cost = sessionTurnCost(usage, pricing[0])
		outcome.CostKnown = true
	}
	sessiontelemetry.RecordTurnOutcome(key, outcome, at)
}

// verdictOutcomeCategory maps an ingested learning verdict onto the evidence
// vocabulary. overprovisioned is deliberately unmapped: it reports cost, not a
// quality regression, so it must not feed the escalation streak.
func verdictOutcomeCategory(verdict routerLearningOutcomeVerdict) (sessiontelemetry.TurnOutcomeCategory, bool) {
	switch verdict {
	case routerLearningOutcomeGoodFit:
		return sessiontelemetry.TurnProgress, true
	case routerLearningOutcomeUnderpowered, routerLearningOutcomeFailed:
		return sessiontelemetry.TurnRegression, true
	default:
		return "", false
	}
}

// recordIngestedTurnOutcome mirrors an accepted model verdict into the evidence
// window. The verdict grades an earlier turn, so it is written with that turn's
// event time and the window inserts it in order rather than at the tail.
//
// The replay record stores the bare session ID, so the routing namespace is
// rebuilt here: response-side capture keys the window by
// routingSessionStateKey, and both writers must land in the same window.
func recordIngestedTurnOutcome(
	record routerreplay.RoutingRecord,
	model string,
	verdict routerLearningOutcomeVerdict,
	score float64,
	scoreProvided bool,
) {
	category, ok := verdictOutcomeCategory(verdict)
	if !ok || record.SessionID == "" {
		return
	}
	eventTime := record.Timestamp
	if eventTime.IsZero() {
		return
	}
	if record.ResponseStatus == 429 || record.ResponseStatus >= 500 {
		category = sessiontelemetry.TurnProviderError
	}
	sessionKey := config.RoutingNamespaceKey(config.RecipeName(record.Recipe), record.SessionID)
	sessiontelemetry.RecordTurnOutcome(sessionKey, sessiontelemetry.TurnOutcome{
		RequestID:       record.RequestID,
		TurnIndex:       record.TurnIndex,
		Model:           model,
		Category:        category,
		Confidence:      score,
		ConfidenceKnown: scoreProvided,
		Source:          sessiontelemetry.TurnSourceOutcomeIngest,
	}, eventTime)
}
