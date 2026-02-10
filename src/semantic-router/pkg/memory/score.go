package memory

import (
	"math"
	"sort"
	"time"
)

const (
	// DefaultInitialStrengthDays is the fallback S0 when config omits initial_strength_days or sets it to 0.
	DefaultInitialStrengthDays = 30

	// DefaultPruneThreshold is the fallback delta when config omits prune_threshold or sets it to 0.
	DefaultPruneThreshold = 0.1

	// MinStrength prevents division by zero when S = S0 + AccessCount somehow reaches 0 or below.
	MinStrength = 1.0
)

// RetentionScore computes the MemoryBank-style retention score R = exp(-t_days/S).
// t_days = (now - lastAccessed) in fractional days; S = initialStrengthDays + accessCount.
// Higher R means the memory is "stronger" (keep); lower R means prune when R < threshold.
func RetentionScore(lastAccessed time.Time, accessCount int, initialStrengthDays int, now time.Time) float64 {
	if initialStrengthDays <= 0 {
		initialStrengthDays = DefaultInitialStrengthDays
	}
	tDays := now.Sub(lastAccessed).Hours() / 24
	if tDays < 0 {
		tDays = 0
	}
	S := float64(initialStrengthDays) + float64(accessCount)
	if S <= 0 {
		S = MinStrength
	}
	return math.Exp(-tDays / S)
}

// PruneCandidates returns IDs to delete: entries with R < delta, then lowest-R entries
// until at most maxPerUser remain. initialStrength and delta use the same semantics as
// RetentionScore and PruneThreshold; maxPerUser 0 means no cap.
func PruneCandidates(entries []MemoryPruneEntry, now time.Time, initialStrength int, delta float64, maxPerUser int) []string {
	if len(entries) == 0 {
		return nil
	}
	if initialStrength <= 0 {
		initialStrength = DefaultInitialStrengthDays
	}
	if delta <= 0 {
		delta = DefaultPruneThreshold
	}

	type scored struct {
		id string
		R  float64
	}
	scoredList := make([]scored, 0, len(entries))
	for _, e := range entries {
		R := RetentionScore(e.LastAccessed, e.AccessCount, initialStrength, now)
		scoredList = append(scoredList, scored{id: e.ID, R: R})
	}
	sort.Slice(scoredList, func(i, j int) bool { return scoredList[i].R < scoredList[j].R })

	var toDelete []string
	for _, s := range scoredList {
		if s.R < delta {
			toDelete = append(toDelete, s.id)
		}
	}
	keepCount := len(scoredList) - len(toDelete)
	if maxPerUser > 0 && keepCount > maxPerUser {
		needToDelete := keepCount - maxPerUser
		for i := 0; i < len(scoredList) && needToDelete > 0; i++ {
			if scoredList[i].R >= delta {
				toDelete = append(toDelete, scoredList[i].id)
				needToDelete--
			}
		}
	}
	return toDelete
}
