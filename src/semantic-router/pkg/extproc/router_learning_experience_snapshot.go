package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"

// ExperienceSnapshots exports the current in-process experience state as
// versioned, typed snapshots. This is read-only and does not itself gate or
// change routing; it exists so an operator or offline materializer can
// inspect what Router Learning has observed so far (#2240).
func (rt *routerLearningRuntime) ExperienceSnapshots() []routerruntime.RouterExperienceSnapshot {
	if rt == nil {
		return nil
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	snapshots := make([]routerruntime.RouterExperienceSnapshot, 0, len(rt.experience))
	for key, exp := range rt.experience {
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
	return snapshots
}
