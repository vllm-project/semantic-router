package routerruntime

import "time"

// RouterExperienceSnapshotSchemaVersion is the schema version of
// RouterExperienceSnapshot. Bump it when the exported shape changes.
const RouterExperienceSnapshotSchemaVersion = 1

// RouterExperienceSource identifies where a RouterExperienceSnapshot's
// evidence came from. Runtime is the only source today; offline seed-pack
// import is a separate, not-yet-built follow-up (#2240 scope item 4).
type RouterExperienceSource string

const RouterExperienceSourceRuntime RouterExperienceSource = "runtime"

// RouterExperienceSnapshot is a versioned, read-only export of one
// decision/tier/model context's Router Learning experience state. It is a
// separate, stable-contract type from the runtime's internal experience
// representation so the two can evolve independently.
type RouterExperienceSnapshot struct {
	SchemaVersion int

	Decision string
	Tier     int
	Model    string

	QualitySeed             float64
	SeedWeight              float64
	GoodFitCount            int
	UnderpoweredCount       int
	OverprovisionedCount    int
	FailedCount             int
	LatencyEWMA             float64
	CacheHitEWMA            float64
	CacheWriteEWMA          float64
	InputCostMultiplierEWMA float64

	SampleCount int
	Source      RouterExperienceSource
	LastUpdated time.Time
}

// ExperienceRuntime is the API-server seam for reading materialized Router
// Learning experience state. The implementation lives with the router
// runtime; callers only need typed snapshots, not extproc internals.
type ExperienceRuntime interface {
	ExperienceSnapshots() []RouterExperienceSnapshot
}
