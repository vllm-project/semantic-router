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
	ctx context.Context,
	outcome *routerruntime.RouterOutcome,
) (result routerruntime.RouterOutcomeResult) {
	if rt == nil || outcome == nil || strings.TrimSpace(outcome.ReplayID) == "" {
		return routerruntime.RouterOutcomeResult{
			Code:    routerruntime.RouterOutcomeCodeInvalid,
			Message: "replay_id is required",
		}
	}
	claim, duplicate, claimErr := rt.claimIdempotencyKey(ctx, outcome.IdempotencyKey)
	if claimErr != nil {
		return routerruntime.RouterOutcomeResult{
			Code:    routerruntime.RouterOutcomeCodeInvalid,
			Message: claimErr.Error(),
		}
	}
	if duplicate {
		return routerruntime.RouterOutcomeResult{
			Code:    routerruntime.RouterOutcomeCodeDuplicate,
			Message: "idempotent learning outcome already applied",
		}
	}
	if claim != nil {
		defer func() {
			rt.finishIdempotencyClaim(
				outcome.IdempotencyKey,
				claim,
				result.Code == routerruntime.RouterOutcomeCodeOK && result.Recorded,
			)
		}()
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

func (rt *routerLearningRuntime) claimIdempotencyKey(
	ctx context.Context,
	key string,
) (*outcomeIdempotencyClaim, bool, error) {
	key = strings.TrimSpace(key)
	if key == "" || rt == nil {
		return nil, false, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	for {
		rt.shared.mu.Lock()
		now := time.Now()
		rt.pruneIdempotencyLocked(now)
		if rt.shared.idempotencyKeys == nil {
			rt.shared.idempotencyKeys = map[string]*outcomeIdempotencyClaim{}
		}
		claim, exists := rt.shared.idempotencyKeys[key]
		if !exists {
			claim = &outcomeIdempotencyClaim{done: make(chan struct{})}
			rt.shared.idempotencyKeys[key] = claim
			rt.shared.mu.Unlock()
			return claim, false, nil
		}
		if !claim.committedAt.IsZero() {
			rt.shared.mu.Unlock()
			return nil, true, nil
		}
		done := claim.done
		rt.shared.mu.Unlock()

		select {
		case <-done:
			// The owner either committed (the next loop returns duplicate) or
			// aborted and removed the claim (the waiter becomes the new owner).
		case <-ctx.Done():
			return nil, false, ctx.Err()
		}
	}
}

func (rt *routerLearningRuntime) finishIdempotencyClaim(
	key string,
	claim *outcomeIdempotencyClaim,
	commit bool,
) {
	key = strings.TrimSpace(key)
	if rt == nil || key == "" || claim == nil {
		return
	}
	rt.shared.mu.Lock()
	defer rt.shared.mu.Unlock()
	if current := rt.shared.idempotencyKeys[key]; current != claim {
		return
	}
	if commit {
		claim.committedAt = time.Now()
	} else {
		delete(rt.shared.idempotencyKeys, key)
	}
	close(claim.done)
}

func (rt *routerLearningRuntime) pruneIdempotencyLocked(now time.Time) {
	if rt == nil || rt.shared == nil || len(rt.shared.idempotencyKeys) == 0 {
		return
	}
	cutoff := now.Add(-learningOutcomeIdempotencyTTL)
	for key, claim := range rt.shared.idempotencyKeys {
		if claim != nil && !claim.committedAt.IsZero() && claim.committedAt.Before(cutoff) {
			delete(rt.shared.idempotencyKeys, key)
		}
	}
}
