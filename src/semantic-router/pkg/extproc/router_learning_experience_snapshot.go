package extproc

import (
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// ExperienceSnapshots exports the current in-process experience state as
// versioned, typed snapshots. This is read-only and does not itself gate or
// change routing; it exists so an operator or offline materializer can
// inspect what Router Learning has observed so far (#2240).
func (rt *routerLearningRuntime) ExperienceSnapshots() []routerruntime.RouterExperienceSnapshot {
	if rt == nil || rt.shared == nil {
		return nil
	}
	rt.shared.mu.Lock()
	defer rt.shared.mu.Unlock()
	snapshots := make([]routerruntime.RouterExperienceSnapshot, 0, len(rt.shared.experience))
	for key, exp := range rt.shared.experience {
		decision := key.decision
		if decision == "_global" {
			decision = ""
		}
		snapshots = append(snapshots, routerruntime.RouterExperienceSnapshot{
			SchemaVersion:           routerruntime.RouterExperienceSnapshotSchemaVersion,
			Decision:                decision,
			Tier:                    key.tier,
			Model:                   key.model,
			QualitySeed:             exp.QualitySeed,
			SeedWeight:              exp.SeedWeight,
			GoodFitCount:            exp.GoodFitCount,
			UnderpoweredCount:       exp.UnderpoweredCount,
			OverprovisionedCount:    exp.OverprovisionedCount,
			FailedCount:             exp.FailedCount,
			LatencyEWMA:             exp.LatencyEWMA,
			CacheHitEWMA:            exp.CacheHitEWMA,
			CacheWriteEWMA:          exp.CacheWriteEWMA,
			InputCostMultiplierEWMA: exp.InputCostMultiplierEWMA,
			SampleCount:             exp.GoodFitCount + exp.UnderpoweredCount + exp.OverprovisionedCount + exp.FailedCount,
			Source:                  routerruntime.RouterExperienceSourceRuntime,
			LastUpdated:             exp.LastUpdated,
		})
	}
	// Map iteration order is randomized, so identical learned state must be
	// sorted back into a stable order here rather than relying on encoding
	// to do it — otherwise this "versioned" artifact hashes differently
	// across calls with no actual change in state.
	sort.Slice(snapshots, func(i, j int) bool {
		return experienceSnapshotLess(snapshots[i], snapshots[j])
	})
	return snapshots
}

// experienceSnapshotLess orders snapshots by their persisted key tuple
// (decision, tier, model) so exported output is deterministic.
func experienceSnapshotLess(a, b routerruntime.RouterExperienceSnapshot) bool {
	if a.Decision != b.Decision {
		return a.Decision < b.Decision
	}
	if a.Tier != b.Tier {
		return a.Tier < b.Tier
	}
	return a.Model < b.Model
}
