package cache

import (
	"context"
	"sync/atomic"
)

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

// The verifier is injected by the classification lifecycle
// (classification.initializeSemanticCacheNLI), mirroring
// looper.SetGroundingBackends, so pkg/cache never imports pkg/classification and
// the guard stays unit-testable with a fake. It is held atomically because a
// config reload re-runs the classifier runtime tasks — and therefore this
// injection — while the previous router is still serving lookups.
var polarityVerifier atomic.Pointer[PolarityVerifyFunc]

// SetPolarityVerifier wires the NLI backend used by the polarity guard. Safe to
// call again, from any goroutine, to replace it; nil leaves the tier
// unavailable, and lookups then fail open (the threshold-verified hit is served).
func SetPolarityVerifier(fn PolarityVerifyFunc) {
	if fn == nil {
		polarityVerifier.Store(nil)
		return
	}
	polarityVerifier.Store(&fn)
}

// loadPolarityVerifier returns the currently injected verifier, or nil.
func loadPolarityVerifier() PolarityVerifyFunc {
	if p := polarityVerifier.Load(); p != nil {
		return *p
	}
	return nil
}

// polarityGuardTierNLI labels NLI-tier telemetry so it can be told apart from
// the lexical tier's events.
const polarityGuardTierNLI = "nli"
