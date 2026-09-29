package cache

import (
	"context"
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
	Verifier               PolarityVerifyFunc
	UseNLI                 bool
	ContradictionThreshold float32
}

// polarityGuardTierNLI labels NLI-tier telemetry so it can be told apart from
// the lexical tier's events.
const polarityGuardTierNLI = "nli"
