package cache

import "context"

// PolarityVerifyFunc scores how strongly incomingQuery contradicts cachedQuery as
// a probability in [0, 1]. The NLI direction is fixed: premise = cached query,
// hypothesis = incoming query. It runs once per lookup on the single winning
// candidate, never per entry inside the scan.
type PolarityVerifyFunc func(ctx context.Context, cachedQuery, incomingQuery string) (contradiction float32, err error)

// PolarityGuardOptions configures the optional NLI polarity tier (#2751) of the
// in-memory semantic cache. The lexical tier (#2691) is the unconditional floor;
// this tier verifies the best above-threshold candidate outside the cache lock
// and rejects the hit when the contradiction probability exceeds
// ContradictionThreshold.
type PolarityGuardOptions struct {
	UseNLI                 bool
	ContradictionThreshold float32
}

// The verifier is injected once at startup by the classification lifecycle
// (classification.initializeSemanticCacheNLI), mirroring
// looper.SetGroundingBackends, so pkg/cache never imports pkg/classification and
// the guard stays unit-testable with a fake.
var polarityVerifier PolarityVerifyFunc

// SetPolarityVerifier wires the NLI backend used by the polarity guard. Safe to
// call again to replace it; nil leaves the tier unavailable, and lookups then
// fail open (the threshold-verified hit is served).
func SetPolarityVerifier(fn PolarityVerifyFunc) {
	polarityVerifier = fn
}

// polarityGuardTierNLI labels NLI-tier telemetry so it can be told apart from
// the lexical tier's events.
const polarityGuardTierNLI = "nli"
