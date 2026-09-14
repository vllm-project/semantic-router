package sessiontelemetry

import (
	"encoding/json"
	"sort"
	"time"
)

// RouterSessionStateMerger is implemented by stores that can fold a locally
// observed snapshot into the shared one without clobbering a concurrent writer.
// The merge must be atomic with respect to other writers: a read-modify-write
// without compare-and-swap does not satisfy this. Stores that do not implement
// it keep the whole-snapshot Save contract.
type RouterSessionStateMerger interface {
	Merge(local RouterSessionSnapshot, ttl time.Duration) error
}

// mergeStoredSnapshot folds a stored payload into the local snapshot. An
// unreadable payload is an error rather than an empty base, because writing
// over it would discard the facts this merge exists to keep.
func mergeStoredSnapshot(stored []byte, local RouterSessionSnapshot) (RouterSessionSnapshot, error) {
	var remote RouterSessionSnapshot
	if err := json.Unmarshal(stored, &remote); err != nil {
		return RouterSessionSnapshot{}, err
	}
	return mergeRouterSessionSnapshots(remote, local), nil
}

// mergeRouterSessionSnapshots folds a remote snapshot into the locally observed
// one. Every rule is monotonic: a merge may add facts but never drop or
// regress one, so retries and out-of-order persistence are safe.
func mergeRouterSessionSnapshots(remote, local RouterSessionSnapshot) RouterSessionSnapshot {
	if local.SessionID == "" {
		return remote
	}
	if remote.SessionID != local.SessionID {
		return local
	}
	// A remote snapshot the local replica has already moved well past is idle
	// history, not a concurrent writer: reusing it would resurrect a session
	// that has since expired.
	if local.LastSeen.After(remote.LastSeen) && local.LastSeen.Sub(remote.LastSeen) > routerMemoryTTL {
		return local
	}

	now := time.Now()
	size, windowTTL := mergedWindowPolicy(remote, local)
	merged := newerRouterSessionSnapshot(remote, local)

	merged.RecentOutcomes = mergeTurnOutcomeWindows(remote.RecentOutcomes, local.RecentOutcomes, size, windowTTL, now)
	merged.SwitchTimestamps = mergeSwitchTimestamps(remote.SwitchTimestamps, local.SwitchTimestamps, windowTTL, now)
	merged.LastSwitchAt = laterTime(remote.LastSwitchAt, local.LastSwitchAt)

	// Counters are per-replica accumulations of a shared baseline. Taking the
	// maximum keeps whatever each replica observed without double counting a
	// replayed save; it cannot sum concurrent per-replica deltas.
	merged.TurnCount = max(remote.TurnCount, local.TurnCount)
	merged.SwitchCount = max(remote.SwitchCount, local.SwitchCount)
	merged.ModelTurns = mergeModelTurns(remote.ModelTurns, local.ModelTurns)
	merged.CumulativePromptTokens = max(remote.CumulativePromptTokens, local.CumulativePromptTokens)
	merged.CumulativeCachedTokens = max(remote.CumulativeCachedTokens, local.CumulativeCachedTokens)
	merged.CumulativeCacheWriteTokens = max(remote.CumulativeCacheWriteTokens, local.CumulativeCacheWriteTokens)
	merged.CumulativeEstimatedCachedTokens = max(remote.CumulativeEstimatedCachedTokens, local.CumulativeEstimatedCachedTokens)
	merged.CumulativeCompletionTokens = max(remote.CumulativeCompletionTokens, local.CumulativeCompletionTokens)
	merged.CumulativeCost = max(remote.CumulativeCost, local.CumulativeCost)
	merged.CumulativeEstimatedCacheSavings = max(remote.CumulativeEstimatedCacheSavings, local.CumulativeEstimatedCacheSavings)

	// The widest configured policy wins so neither writer's window is narrowed.
	merged.OutcomeWindowSize = max(remote.OutcomeWindowSize, local.OutcomeWindowSize)
	merged.OutcomeWindowTTLSeconds = max(remote.OutcomeWindowTTLSeconds, local.OutcomeWindowTTLSeconds)
	// IdleFor is recomputed by the loader for the reader's clock.
	merged.IdleFor = 0
	return merged
}

// newerRouterSessionSnapshot returns the view with the later LastSeen; the
// local snapshot wins ties because it is the one this replica just observed.
func newerRouterSessionSnapshot(remote, local RouterSessionSnapshot) RouterSessionSnapshot {
	if remote.LastSeen.After(local.LastSeen) {
		return remote
	}
	return local
}

// mergedWindowPolicy picks the widest window the two writers configured, so a
// merge never trims evidence a reader on either replica would still consider.
func mergedWindowPolicy(remote, local RouterSessionSnapshot) (int, time.Duration) {
	size := max(remote.OutcomeWindowSize, local.OutcomeWindowSize)
	ttlSeconds := max(remote.OutcomeWindowTTLSeconds, local.OutcomeWindowTTLSeconds)
	if ttlSeconds <= 0 {
		return normalizeWindowPolicy(size, 0)
	}
	return normalizeWindowPolicy(size, time.Duration(ttlSeconds)*time.Second)
}

// mergeTurnOutcomeWindows unions two writers' windows. appendTurnOutcome
// already folds two observations of the same turn and inserts by event time.
func mergeTurnOutcomeWindows(remote, local []TurnOutcome, size int, ttl time.Duration, now time.Time) []TurnOutcome {
	merged := pruneTurnOutcomes(remote, ttl, now)
	for _, outcome := range local {
		merged = appendTurnOutcome(merged, outcome, now, size, ttl)
	}
	if len(merged) == 0 {
		return nil
	}
	return merged
}

// mergeSwitchTimestamps unions and de-duplicates the model-change series so a
// replayed save cannot inflate the oscillation guard.
func mergeSwitchTimestamps(remote, local []int64, ttl time.Duration, now time.Time) []int64 {
	if len(remote) == 0 && len(local) == 0 {
		return nil
	}
	seen := make(map[int64]struct{}, len(remote)+len(local))
	merged := make([]int64, 0, len(remote)+len(local))
	for _, series := range [][]int64{remote, local} {
		for _, ts := range series {
			if _, duplicate := seen[ts]; duplicate {
				continue
			}
			seen[ts] = struct{}{}
			merged = append(merged, ts)
		}
	}
	sort.Slice(merged, func(i, j int) bool { return merged[i] < merged[j] })
	return pruneSwitchTimestamps(merged, ttl, now)
}

func mergeModelTurns(remote, local map[string]int) map[string]int {
	if len(remote) == 0 && len(local) == 0 {
		return nil
	}
	merged := make(map[string]int, len(remote)+len(local))
	for _, turns := range []map[string]int{remote, local} {
		for model, count := range turns {
			if count > merged[model] {
				merged[model] = count
			}
		}
	}
	return merged
}

func laterTime(a, b time.Time) time.Time {
	if a.After(b) {
		return a
	}
	return b
}
