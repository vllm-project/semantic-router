package extproc

import (
	"context"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

const learningOutcomeIdempotencyTTL = 10 * time.Minute

func (rt *routerLearningRuntime) UpdateOutcome(
	_ context.Context,
	outcome *routerruntime.RouterOutcome,
) routerruntime.RouterOutcomeResult {
	if rt == nil || outcome == nil || strings.TrimSpace(outcome.ReplayID) == "" {
		return routerruntime.RouterOutcomeResult{
			Code:    routerruntime.RouterOutcomeCodeInvalid,
			Message: "replay_id is required",
		}
	}
	if result, duplicate := rt.idempotentOutcomeResult(outcome.IdempotencyKey); duplicate {
		return result
	}

	record, ok := rt.replayRecord(outcome.ReplayID)
	if !ok {
		return routerruntime.RouterOutcomeResult{
			Code:    routerruntime.RouterOutcomeCodeReplayNotFound,
			Message: "replay_id does not reference an owned routing event",
		}
	}

	if outcome.Target == routerruntime.RouterOutcomeTargetModel {
		return rt.updateOwnedModelOutcome(outcome, record)
	}
	return rt.recordOwnedNonModelOutcome(outcome)
}

func (rt *routerLearningRuntime) updateOwnedModelOutcome(
	outcome *routerruntime.RouterOutcome,
	record routerreplay.RoutingRecord,
) routerruntime.RouterOutcomeResult {
	verdict, ok := routerOutcomeVerdict(outcome.Verdict)
	if !ok {
		return routerruntime.RouterOutcomeResult{
			Code:    routerruntime.RouterOutcomeCodeInvalid,
			Message: "verdict must be one of good_fit, underpowered, overprovisioned, failed",
		}
	}
	model, ownershipErr := ownedReplayModel(outcome, record)
	if ownershipErr != "" {
		return routerruntime.RouterOutcomeResult{
			Code:    routerruntime.RouterOutcomeCodeOwnershipMismatch,
			Message: ownershipErr,
		}
	}
	outcome.TargetRef = model

	if !rt.appendReplayOutcome(outcome) {
		return routerruntime.RouterOutcomeResult{
			Code:    routerruntime.RouterOutcomeCodeReplayNotFound,
			Message: "failed to append outcome to owned routing event",
		}
	}
	rt.rememberIdempotencyKey(outcome.IdempotencyKey)

	decisionName, decisionTier := rt.resolveOutcomeDecisionContext(outcome)
	rt.recordModelExperience(decisionName, decisionTier, model, verdict, outcome.Score)
	return routerruntime.RouterOutcomeResult{Updated: 1, Recorded: true}
}

func (rt *routerLearningRuntime) recordOwnedNonModelOutcome(
	outcome *routerruntime.RouterOutcome,
) routerruntime.RouterOutcomeResult {
	if !rt.appendReplayOutcome(outcome) {
		return routerruntime.RouterOutcomeResult{
			Code:    routerruntime.RouterOutcomeCodeReplayNotFound,
			Message: "failed to append outcome to owned routing event",
		}
	}
	rt.rememberIdempotencyKey(outcome.IdempotencyKey)
	return routerruntime.RouterOutcomeResult{Recorded: true}
}

func ownedReplayModel(
	outcome *routerruntime.RouterOutcome,
	record routerreplay.RoutingRecord,
) (string, string) {
	replayModel := strings.TrimSpace(record.SelectedModel)
	if replayModel == "" && record.RouteDiagnostics != nil {
		replayModel = strings.TrimSpace(record.RouteDiagnostics.SelectedModel)
	}
	if replayModel == "" {
		return "", "owned routing event has no selected model binding"
	}

	claimed := strings.TrimSpace(outcome.TargetRef)
	if claimed == "" && outcome.Metadata != nil {
		claimed = strings.TrimSpace(outcome.Metadata["model"])
		if claimed == "" {
			claimed = strings.TrimSpace(outcome.Metadata["selected_model"])
		}
	}
	if claimed != "" && claimed != replayModel {
		return "", "outcome model binding does not match the owned routing event"
	}
	return replayModel, ""
}

func (rt *routerLearningRuntime) idempotentOutcomeResult(
	key string,
) (routerruntime.RouterOutcomeResult, bool) {
	key = strings.TrimSpace(key)
	if key == "" || rt == nil {
		return routerruntime.RouterOutcomeResult{}, false
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	rt.pruneIdempotencyLocked(time.Now())
	if _, exists := rt.idempotencyKeys[key]; exists {
		return routerruntime.RouterOutcomeResult{
			Code:    routerruntime.RouterOutcomeCodeDuplicate,
			Message: "idempotent learning outcome already applied",
		}, true
	}
	return routerruntime.RouterOutcomeResult{}, false
}

func (rt *routerLearningRuntime) rememberIdempotencyKey(key string) {
	key = strings.TrimSpace(key)
	if rt == nil || key == "" {
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.idempotencyKeys == nil {
		rt.idempotencyKeys = map[string]time.Time{}
	}
	now := time.Now()
	rt.pruneIdempotencyLocked(now)
	rt.idempotencyKeys[key] = now
}

func (rt *routerLearningRuntime) pruneIdempotencyLocked(now time.Time) {
	if rt == nil || len(rt.idempotencyKeys) == 0 {
		return
	}
	cutoff := now.Add(-learningOutcomeIdempotencyTTL)
	for key, seenAt := range rt.idempotencyKeys {
		if seenAt.Before(cutoff) {
			delete(rt.idempotencyKeys, key)
		}
	}
}
