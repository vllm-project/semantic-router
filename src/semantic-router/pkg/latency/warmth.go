package latency

import (
	"math"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// CacheEstimationInput describes one cache-warmth estimate request.
type CacheEstimationInput struct {
	Model       string
	TTFTSeconds float64
	Now         time.Time
}

// CacheWarmthPrior is returned when the estimator lacks reliable evidence.
const CacheWarmthPrior = 0.5

// CountReliabilityStart is the observation count where count reliability starts increasing.
const CountReliabilityStart = 5

// CountReliabilityFull is the observation count where count reliability reaches 1.
const CountReliabilityFull = 50

// TargetRelativeSpread is the relative warm/cold spread that yields full spread reliability.
const TargetRelativeSpread = 0.15

// FreshnessHalfLifeSeconds is the TTFT-history freshness half-life.
const FreshnessHalfLifeSeconds = 60.0

// MinRelativeScaleFloor is the minimum effective scale relative to ref.
const MinRelativeScaleFloor = 0.10

// ttftSnapshot copies the TTFT stats needed for one estimate under a single lock.
type ttftSnapshot struct {
	averageTTFT      float64
	recentTTFTs      []float64
	observationCount int
	lastUpdated      time.Time
}

// getTTFTSnapshot returns a copy of model TTFT stats under one RLock.
func getTTFTSnapshot(model string) (ttftSnapshot, bool) {
	globalTTFTCache.mu.RLock()
	defer globalTTFTCache.mu.RUnlock()

	stats, exists := globalTTFTCache.cache[model]
	if !exists {
		return ttftSnapshot{}, false
	}

	recent := make([]float64, len(stats.RecentTTFTs))
	copy(recent, stats.RecentTTFTs)

	return ttftSnapshot{
		averageTTFT:      stats.AverageTTFT,
		recentTTFTs:      recent,
		observationCount: stats.ObservationCount,
		lastUpdated:      stats.LastUpdated,
	}, true
}

// percentileFromSorted returns the pct percentile from a sorted slice.
func percentileFromSorted(sorted []float64, pct float64) (float64, bool) {
	n := len(sorted)
	if n == 0 {
		return 0, false
	}
	index := pct * float64(n-1)
	lower := int(index)
	upper := lower + 1
	if upper >= n {
		return sorted[n-1], true
	}
	weight := index - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight, true
}

// EstimateCacheProbability estimates cache warmth from TTFT history.
// It returns CacheWarmthPrior when evidence is missing or unreliable. A zero input.Now uses time.Now().
func EstimateCacheProbability(input CacheEstimationInput) float64 {
	model := strings.TrimSpace(input.Model)
	if model == "" || input.TTFTSeconds <= 0 {
		return CacheWarmthPrior
	}

	now := input.Now
	if now.IsZero() {
		now = time.Now()
	}

	snap, ok := getTTFTSnapshot(model)
	if !ok {
		return CacheWarmthPrior
	}

	// Sort once; read p20, p50, p80 from the same sorted slice.
	sort.Float64s(snap.recentTTFTs)

	warm, warmOk := percentileFromSorted(snap.recentTTFTs, 0.20)
	cold, coldOk := percentileFromSorted(snap.recentTTFTs, 0.80)
	if !warmOk || !coldOk || cold <= warm {
		return CacheWarmthPrior
	}

	ref, refOk := percentileFromSorted(snap.recentTTFTs, 0.50)
	if !refOk || ref <= 0 {
		ref = snap.averageTTFT
	}
	if ref <= 0 {
		return CacheWarmthPrior
	}

	spread := cold - warm
	effectiveScale := math.Max(spread, ref*MinRelativeScaleFloor)
	raw := clamp((cold-input.TTFTSeconds)/effectiveScale, 0, 1)

	countReliability := clamp(
		float64(snap.observationCount-CountReliabilityStart)/
			float64(CountReliabilityFull-CountReliabilityStart),
		0, 1,
	)
	spreadReliability := clamp((spread/ref)/TargetRelativeSpread, 0, 1)

	ageSeconds := now.Sub(snap.lastUpdated).Seconds()
	freshnessReliability := 1.0
	if ageSeconds > 0 {
		freshnessReliability = math.Exp(-math.Ln2 * ageSeconds / FreshnessHalfLifeSeconds)
	}

	reliability := countReliability * spreadReliability * freshnessReliability
	prob := clamp(reliability*raw+(1.0-reliability)*CacheWarmthPrior, 0, 1)

	logging.Debugf("EstimateCacheProbability: model=%q observed=%.4fs ref=%.4fs raw=%.4f "+
		"reliability=%.4f (count=%.2f spread=%.2f fresh=%.2f) prob=%.4f",
		model, input.TTFTSeconds, ref, raw,
		reliability, countReliability, spreadReliability, freshnessReliability, prob)

	return prob
}

func clamp(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(v, hi))
}

// GetTTFTLastUpdated returns the most recent TTFT timestamp for model.
func GetTTFTLastUpdated(model string) (time.Time, bool) {
	model = strings.TrimSpace(model)
	if model == "" {
		return time.Time{}, false
	}
	globalTTFTCache.mu.RLock()
	defer globalTTFTCache.mu.RUnlock()
	stats, exists := globalTTFTCache.cache[model]
	if !exists {
		return time.Time{}, false
	}
	return stats.LastUpdated, true
}
