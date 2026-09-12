package sessiontelemetry

import (
	"sort"
	"time"
)

// RouterSessionStateMerger is implemented by stores that can fold a locally
// observed snapshot into the shared one without clobbering a concurrent writer.
// Stores that do not implement it keep the whole-snapshot Save contract.
//
// The merge must be atomic with respect to other writers: two replicas that
// loaded the same session, appended different outcomes and then persisted must
// both survive. A read-modify-write without compare-and-swap does not satisfy
// this, because the second writer overwrites the first.
type RouterSessionStateMerger interface {
	Merge(local RouterSessionSnapshot, ttl time.Duration) error
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

	// Counters are per-replica accumulations. Taking the maximum keeps whatever
	// each replica observed without double counting a replayed save.
	merged.TurnCount = maxInt(remote.TurnCount, local.TurnCount)
	merged.SwitchCount = maxInt(remote.SwitchCount, local.SwitchCount)
	merged.ModelTurns = mergeModelTurns(remote.ModelTurns, local.ModelTurns)
	merged.CumulativePromptTokens = maxInt64(remote.CumulativePromptTokens, local.CumulativePromptTokens)
	merged.CumulativeCachedTokens = maxInt64(remote.CumulativeCachedTokens, local.CumulativeCachedTokens)
	merged.CumulativeCacheWriteTokens = maxInt64(remote.CumulativeCacheWriteTokens, local.CumulativeCacheWriteTokens)
	merged.CumulativeEstimatedCachedTokens = maxInt64(remote.CumulativeEstimatedCachedTokens, local.CumulativeEstimatedCachedTokens)
	merged.CumulativeCompletionTokens = maxInt64(remote.CumulativeCompletionTokens, local.CumulativeCompletionTokens)
	merged.CumulativeCost = maxFloat64(remote.CumulativeCost, local.CumulativeCost)
	merged.CumulativeEstimatedCacheSavings = maxFloat64(remote.CumulativeEstimatedCacheSavings, local.CumulativeEstimatedCacheSavings)

	// The widest configured policy wins so neither writer's window is narrowed.
	merged.OutcomeWindowSize = maxInt(remote.OutcomeWindowSize, local.OutcomeWindowSize)
	merged.OutcomeWindowTTLSeconds = maxInt(remote.OutcomeWindowTTLSeconds, local.OutcomeWindowTTLSeconds)
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
	size := maxInt(remote.OutcomeWindowSize, local.OutcomeWindowSize)
	ttlSeconds := maxInt(remote.OutcomeWindowTTLSeconds, local.OutcomeWindowTTLSeconds)
	if ttlSeconds <= 0 {
		return normalizeWindowPolicy(size, 0)
	}
	return normalizeWindowPolicy(size, time.Duration(ttlSeconds)*time.Second)
}

// mergeTurnOutcomeWindows unions two writers' windows. appendTurnOutcome
// already knows how to fold two observations of the same turn and insert by
// event time, so the remote window is the base and the local one is replayed
// into it.
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

// mergeSwitchTimestamps unions and de-duplicates the model-change series. Two
// replicas observing the same switch must not inflate the oscillation guard.
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

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func maxFloat64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
