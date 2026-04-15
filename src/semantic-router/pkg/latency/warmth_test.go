package latency

import (
	"math"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// EstimateCacheProbability tests
// ---------------------------------------------------------------------------

// seedVariedTTFTs populates the TTFT cache for model with n observations spread
// linearly from lo to hi. Returns the time after seeding for use as Now.
func seedVariedTTFTs(model string, n int, lo, hi float64) time.Time {
	if n == 1 {
		UpdateTTFT(model, lo)
		return time.Now()
	}
	for i := 0; i < n; i++ {
		v := lo + (hi-lo)*float64(i)/float64(n-1)
		UpdateTTFT(model, v)
	}
	return time.Now()
}

func TestEstimateCacheProbability_WarmObservation(t *testing.T) {
	ResetTTFT()
	// 30 observations ranging 0.1s–1.0s. p20 ≈ 0.28, p80 ≈ 0.82.
	// Observed near p10 → should be warm (> 0.7).
	now := seedVariedTTFTs("m-warm", 30, 0.1, 1.0)
	prob := EstimateCacheProbability(CacheEstimationInput{
		Model:       "m-warm",
		TTFTSeconds: 0.15,
		Now:         now,
	})
	t.Logf("warm observation: prob=%.4f", prob)
	if prob <= 0.7 {
		t.Errorf("expected > 0.7 for warm observation, got %.4f", prob)
	}
}

func TestEstimateCacheProbability_ColdObservation(t *testing.T) {
	ResetTTFT()
	now := seedVariedTTFTs("m-cold", 30, 0.1, 1.0)
	prob := EstimateCacheProbability(CacheEstimationInput{
		Model:       "m-cold",
		TTFTSeconds: 0.95,
		Now:         now,
	})
	t.Logf("cold observation: prob=%.4f", prob)
	if prob >= 0.3 {
		t.Errorf("expected < 0.3 for cold observation, got %.4f", prob)
	}
}

func TestEstimateCacheProbability_MidRange(t *testing.T) {
	ResetTTFT()
	now := seedVariedTTFTs("m-mid", 30, 0.1, 1.0)
	prob := EstimateCacheProbability(CacheEstimationInput{
		Model:       "m-mid",
		TTFTSeconds: 0.55,
		Now:         now,
	})
	t.Logf("mid-range observation: prob=%.4f", prob)
	if prob < 0.35 || prob > 0.65 {
		t.Errorf("expected ≈ 0.5 for mid-range observation, got %.4f", prob)
	}
}

func TestEstimateCacheProbability_FewObservations(t *testing.T) {
	ResetTTFT()
	// 2 observations → countReliability = clamp((2-5)/45, 0, 1) = 0 → CacheWarmthPrior
	UpdateTTFT("m-few", 0.5)
	UpdateTTFT("m-few", 1.0)
	now := time.Now()
	prob := EstimateCacheProbability(CacheEstimationInput{
		Model:       "m-few",
		TTFTSeconds: 0.3,
		Now:         now,
	})
	t.Logf("few observations: prob=%.4f", prob)
	if prob != CacheWarmthPrior {
		t.Errorf("expected CacheWarmthPrior (%.1f) for 2 observations, got %.4f", CacheWarmthPrior, prob)
	}
}

func TestEstimateCacheProbability_GrowingConfidence(t *testing.T) {
	// As observation count grows, the result for the same warm observation
	// should move monotonically further from 0.5.
	// Start at 6 (not 5) since n=CountReliabilityStart gives countReliability=0,
	// which always returns CacheWarmthPrior regardless of the observation.
	prevDelta := 0.0
	for _, n := range []int{6, 10, 20, 50} {
		ResetTTFT()
		now := seedVariedTTFTs("m-grow", n, 0.1, 1.0)
		prob := EstimateCacheProbability(CacheEstimationInput{
			Model:       "m-grow",
			TTFTSeconds: 0.15,
			Now:         now,
		})
		delta := math.Abs(prob - CacheWarmthPrior)
		t.Logf("n=%d: prob=%.4f delta=%.4f", n, prob, delta)
		if delta < prevDelta-0.001 {
			t.Errorf("confidence should grow: n=%d delta=%.4f < prev=%.4f", n, delta, prevDelta)
		}
		prevDelta = delta
	}
}

func TestEstimateCacheProbability_TightDistribution(t *testing.T) {
	ResetTTFT()
	// 30 identical observations → p20 == p80 → cold <= warm → CacheWarmthPrior
	for i := 0; i < 30; i++ {
		UpdateTTFT("m-tight", 0.5)
	}
	now := time.Now()
	prob := EstimateCacheProbability(CacheEstimationInput{
		Model:       "m-tight",
		TTFTSeconds: 0.45,
		Now:         now,
	})
	t.Logf("tight distribution: prob=%.4f", prob)
	if prob != CacheWarmthPrior {
		t.Errorf("expected CacheWarmthPrior (%.1f) for identical observations, got %.4f", CacheWarmthPrior, prob)
	}
}

func TestEstimateCacheProbability_NoHistory(t *testing.T) {
	ResetTTFT()
	prob := EstimateCacheProbability(CacheEstimationInput{
		Model:       "no-such-model",
		TTFTSeconds: 0.5,
		Now:         time.Now(),
	})
	if prob != CacheWarmthPrior {
		t.Errorf("expected CacheWarmthPrior (%.1f) for unknown model, got %.4f", CacheWarmthPrior, prob)
	}
}

func TestEstimateCacheProbability_InvalidInputs(t *testing.T) {
	ResetTTFT()
	now := time.Now()

	tests := []struct {
		name  string
		input CacheEstimationInput
	}{
		{"empty model", CacheEstimationInput{Model: "", TTFTSeconds: 0.5, Now: now}},
		{"zero TTFT", CacheEstimationInput{Model: "m", TTFTSeconds: 0, Now: now}},
		{"negative TTFT", CacheEstimationInput{Model: "m", TTFTSeconds: -1.0, Now: now}},
		{"whitespace model", CacheEstimationInput{Model: "   ", TTFTSeconds: 0.5, Now: now}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			prob := EstimateCacheProbability(tc.input)
			if prob != CacheWarmthPrior {
				t.Errorf("expected CacheWarmthPrior (%.1f), got %.4f", CacheWarmthPrior, prob)
			}
		})
	}
}

func TestEstimateCacheProbability_FreshnessDecay(t *testing.T) {
	ResetTTFT()
	now := seedVariedTTFTs("m-fresh", 50, 0.1, 1.0)

	probFresh := EstimateCacheProbability(CacheEstimationInput{
		Model:       "m-fresh",
		TTFTSeconds: 0.15,
		Now:         now,
	})
	// 120s later the data is stale; result should be closer to CacheWarmthPrior.
	probStale := EstimateCacheProbability(CacheEstimationInput{
		Model:       "m-fresh",
		TTFTSeconds: 0.15,
		Now:         now.Add(120 * time.Second),
	})
	t.Logf("fresh=%.4f stale=%.4f", probFresh, probStale)

	freshDelta := math.Abs(probFresh - CacheWarmthPrior)
	staleDelta := math.Abs(probStale - CacheWarmthPrior)
	if staleDelta >= freshDelta {
		t.Errorf("stale should be closer to CacheWarmthPrior: freshDelta=%.4f staleDelta=%.4f", freshDelta, staleDelta)
	}
}

func TestEstimateCacheProbability_Continuity(t *testing.T) {
	// No large jump at the CountReliabilityStart boundary.
	var probBefore, probAfter float64
	for _, n := range []int{CountReliabilityStart - 1, CountReliabilityStart + 1} {
		ResetTTFT()
		now := seedVariedTTFTs("m-cont", n, 0.1, 1.0)
		p := EstimateCacheProbability(CacheEstimationInput{
			Model:       "m-cont",
			TTFTSeconds: 0.15,
			Now:         now,
		})
		if n < CountReliabilityStart {
			probBefore = p
		} else {
			probAfter = p
		}
	}
	t.Logf("before=%d: %.4f  after=%d: %.4f", CountReliabilityStart-1, probBefore, CountReliabilityStart+1, probAfter)
	if jump := math.Abs(probAfter - probBefore); jump > 0.15 {
		t.Errorf("expected smooth transition, got jump=%.4f", jump)
	}
}

func TestEstimateCacheProbability_Deterministic(t *testing.T) {
	ResetTTFT()
	now := seedVariedTTFTs("m-det", 30, 0.1, 1.0)
	input := CacheEstimationInput{
		Model:       "m-det",
		TTFTSeconds: 0.3,
		Now:         now,
	}
	if a, b := EstimateCacheProbability(input), EstimateCacheProbability(input); a != b {
		t.Errorf("not deterministic: %f != %f", a, b)
	}
}

func TestEstimateCacheProbability_ZeroNowUsesTimeNow(t *testing.T) {
	ResetTTFT()
	_ = seedVariedTTFTs("m-zero-now", 30, 0.1, 1.0)
	prob := EstimateCacheProbability(CacheEstimationInput{
		Model:       "m-zero-now",
		TTFTSeconds: 0.15,
	})
	t.Logf("zero-Now: prob=%.4f", prob)
	if prob < 0 || prob > 1 {
		t.Errorf("probability out of [0,1]: %.4f", prob)
	}
}

// ---------------------------------------------------------------------------
// Snapshot and helper tests
// ---------------------------------------------------------------------------

func TestGetTTFTLastUpdated(t *testing.T) {
	ResetTTFT()
	if _, ok := GetTTFTLastUpdated("unknown"); ok {
		t.Error("expected false for unknown model")
	}

	UpdateTTFT("m-lu", 0.5)
	ts, ok := GetTTFTLastUpdated("m-lu")
	if !ok {
		t.Fatal("expected true after UpdateTTFT")
	}
	if time.Since(ts) > 2*time.Second {
		t.Errorf("LastUpdated too old: %v", ts)
	}
}
