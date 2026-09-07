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
//
// Tool-result failures are not classified here: the conversation facts carry
// tool-call shape but no tool error status, so a failed tool cannot yet be
// distinguished from a successful one. Those turns fall through to the usage
// checks rather than being guessed at.
// TODO: add more detailed classification logic

func classifyTurnOutcome(ctx *RequestContext, usage responseUsageMetrics) sessiontelemetry.TurnOutcomeCategory {
	if ctx == nil {
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

// recordSessionTurnOutcome captures one router-observed outcome into the
// bounded evidence window. It runs on the response path next to session usage
// accounting and never blocks it: an unresolvable session is simply skipped.
func recordSessionTurnOutcome(ctx *RequestContext, usage responseUsageMetrics) {
	if ctx == nil || ctx.SessionID == "" || requestBypassesRouting(ctx) {
		return
	}
	now := time.Now()
	sessiontelemetry.RecordTurnOutcome(routingSessionStateKey(ctx), sessiontelemetry.TurnOutcome{
		TurnIndex:    ctx.TurnIndex,
		Model:        ctx.RequestModel,
		Category:     classifyTurnOutcome(ctx, usage),
		OutputTokens: int64(usage.completionTokens),
		LatencyMs:    int64(ctx.TTFTSeconds * 1000),
		Source:       sessiontelemetry.TurnSourceRouterObserved,
	}, now)
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
) {
	category, ok := verdictOutcomeCategory(verdict)
	if !ok || record.SessionID == "" {
		return
	}
	eventTime := record.Timestamp
	if eventTime.IsZero() {
		eventTime = time.Now()
	}
	sessionKey := config.RoutingNamespaceKey(config.RecipeName(record.Recipe), record.SessionID)
	sessiontelemetry.RecordTurnOutcome(sessionKey, sessiontelemetry.TurnOutcome{
		TurnIndex:  record.TurnIndex,
		Model:      model,
		Category:   category,
		Confidence: score,
		Source:     sessiontelemetry.TurnSourceOutcomeIngest,
	}, eventTime)
}
