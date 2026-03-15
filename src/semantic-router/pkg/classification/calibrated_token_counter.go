package classification

import (
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

const (
	defaultBytesPerToken = 4.0
	defaultEMADecay      = 0.995
	minSamplesForUse     = 10
	defaultCategory      = "_default"
	maxCategories        = 256
	quantileAlpha        = 0.90
)

var (
	calibrationRatioGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "vsr_token_calibration_ratio",
		Help: "Learned bytes-per-token ratio per category",
	}, []string{"category"})

	calibrationErrorHist = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "vsr_token_estimation_error_pct",
		Help:    "Signed estimation error as percentage of actual tokens",
		Buckets: []float64{-50, -30, -20, -10, -5, 0, 5, 10, 20, 30, 50},
	}, []string{"category"})

	calibrationSamplesGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "vsr_token_calibration_samples",
		Help: "Number of calibration samples collected per category",
	}, []string{"category"})
)

// categoryState holds the EMA-calibrated bytes-per-token ratio for one
// content category (e.g. "code", "rag", "chat"). It uses both a mean
// EMA and a quantile EMA: the mean gives the best point estimate while
// the quantile (P90 by default) gives a conservative upper bound that
// minimises the chance of underestimating and misrouting to the short pool.
type categoryState struct {
	mu             sync.RWMutex
	meanRatio      float64
	quantileRatio  float64
	samples        int64
	lastUpdateTime time.Time
}

func newCategoryState() *categoryState {
	return &categoryState{
		meanRatio:     defaultBytesPerToken,
		quantileRatio: defaultBytesPerToken,
	}
}

// CalibratedTokenCounter estimates token counts using a per-category
// bytes-per-token ratio that is continuously calibrated from actual
// usage.prompt_tokens returned by the LLM.
//
// On each response the caller invokes Observe(category, byteLen, actualTokens)
// which updates the EMA for that category. On the next request the Estimate
// method uses the learned ratio instead of the fixed chars/4 heuristic.
//
// Thread-safe for concurrent request/response processing.
type CalibratedTokenCounter struct {
	decay         float64
	conservative  bool // when true, use quantile (P90) estimate instead of mean
	categories    sync.Map
	categoryCount atomic.Int64
}

// CalibratedTokenCounterOption configures a CalibratedTokenCounter.
type CalibratedTokenCounterOption func(*CalibratedTokenCounter)

// WithDecay sets the EMA decay factor (0,1). Higher values make the
// estimator more stable; lower values adapt faster. Default: 0.995.
func WithDecay(decay float64) CalibratedTokenCounterOption {
	return func(c *CalibratedTokenCounter) {
		if decay > 0 && decay < 1 {
			c.decay = decay
		}
	}
}

// WithConservativeEstimate makes CountTokens return the P90 quantile
// estimate instead of the mean. Recommended for routing decisions where
// underestimation causes preemption (asymmetric cost).
func WithConservativeEstimate() CalibratedTokenCounterOption {
	return func(c *CalibratedTokenCounter) {
		c.conservative = true
	}
}

// NewCalibratedTokenCounter creates a token counter that learns
// bytes-per-token ratios per content category from actual LLM usage.
func NewCalibratedTokenCounter(opts ...CalibratedTokenCounterOption) *CalibratedTokenCounter {
	c := &CalibratedTokenCounter{
		decay:        defaultEMADecay,
		conservative: false,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// CountTokens estimates the token count for text using the default category.
// Implements the TokenCounter interface.
func (c *CalibratedTokenCounter) CountTokens(text string) (int, error) {
	return c.Estimate(defaultCategory, len(text)), nil
}

// Estimate returns the estimated token count for a request with the
// given content category and byte length.
func (c *CalibratedTokenCounter) Estimate(category string, byteLen int) int {
	if byteLen == 0 {
		return 0
	}
	if category == "" {
		category = defaultCategory
	}

	state := c.getOrCreateCategory(category)
	state.mu.RLock()
	samples := state.samples
	var ratio float64
	if c.conservative {
		ratio = state.quantileRatio
	} else {
		ratio = state.meanRatio
	}
	state.mu.RUnlock()

	if samples < minSamplesForUse {
		ratio = defaultBytesPerToken
	}

	tokens := int(math.Ceil(float64(byteLen) / ratio))
	if tokens < 1 {
		tokens = 1
	}
	return tokens
}

// EstimateWithCategory returns the estimated token count and the
// current calibrated ratio for observability.
func (c *CalibratedTokenCounter) EstimateWithCategory(category string, byteLen int) (tokens int, ratio float64, calibrated bool) {
	if byteLen == 0 {
		return 0, defaultBytesPerToken, false
	}
	if category == "" {
		category = defaultCategory
	}

	state := c.getOrCreateCategory(category)
	state.mu.RLock()
	samples := state.samples
	if c.conservative {
		ratio = state.quantileRatio
	} else {
		ratio = state.meanRatio
	}
	state.mu.RUnlock()

	calibrated = samples >= minSamplesForUse
	if !calibrated {
		ratio = defaultBytesPerToken
	}

	tokens = int(math.Ceil(float64(byteLen) / ratio))
	if tokens < 1 {
		tokens = 1
	}
	return tokens, ratio, calibrated
}

// Observe records an actual (byteLen, actualTokens) observation for the
// given category, updating the EMA ratios. Called on the response path
// after the LLM returns usage.prompt_tokens.
func (c *CalibratedTokenCounter) Observe(category string, byteLen int, actualTokens int) {
	if byteLen <= 0 || actualTokens <= 0 {
		return
	}
	if category == "" {
		category = defaultCategory
	}

	observedRatio := float64(byteLen) / float64(actualTokens)

	state := c.getOrCreateCategory(category)
	state.mu.Lock()

	if state.samples == 0 {
		state.meanRatio = observedRatio
		state.quantileRatio = observedRatio
	} else {
		state.meanRatio = c.decay*state.meanRatio + (1-c.decay)*observedRatio

		// Asymmetric EMA for quantile estimation (P90):
		// Move faster toward observations ABOVE the current quantile
		// (underestimation is more costly → we want the upper bound).
		if observedRatio > state.quantileRatio {
			state.quantileRatio += (1 - quantileAlpha) * (observedRatio - state.quantileRatio)
		} else {
			state.quantileRatio += quantileAlpha * (observedRatio - state.quantileRatio)
		}
	}

	state.samples++
	state.lastUpdateTime = time.Now()
	meanRatio := state.meanRatio
	state.mu.Unlock()

	// Record estimation error for this observation
	estimatedTokens := float64(byteLen) / meanRatio
	errorPct := (estimatedTokens - float64(actualTokens)) / float64(actualTokens) * 100
	calibrationErrorHist.WithLabelValues(category).Observe(errorPct)
	calibrationRatioGauge.WithLabelValues(category).Set(meanRatio)
	calibrationSamplesGauge.WithLabelValues(category).Set(float64(state.samples))
}

// GetRatio returns the current calibrated bytes-per-token ratio for a
// category, and whether the category has enough samples to be considered
// calibrated. Useful for observability and debugging.
func (c *CalibratedTokenCounter) GetRatio(category string) (mean, quantile float64, samples int64, calibrated bool) {
	if category == "" {
		category = defaultCategory
	}
	val, ok := c.categories.Load(category)
	if !ok {
		return defaultBytesPerToken, defaultBytesPerToken, 0, false
	}
	state := val.(*categoryState)
	state.mu.RLock()
	defer state.mu.RUnlock()
	return state.meanRatio, state.quantileRatio, state.samples, state.samples >= minSamplesForUse
}

// Categories returns a snapshot of all tracked category names.
func (c *CalibratedTokenCounter) Categories() []string {
	var cats []string
	c.categories.Range(func(key, _ any) bool {
		cats = append(cats, key.(string))
		return true
	})
	return cats
}

func (c *CalibratedTokenCounter) getOrCreateCategory(category string) *categoryState {
	if val, ok := c.categories.Load(category); ok {
		return val.(*categoryState)
	}

	if c.categoryCount.Load() >= maxCategories {
		if val, ok := c.categories.Load(defaultCategory); ok {
			return val.(*categoryState)
		}
	}

	state := newCategoryState()
	actual, loaded := c.categories.LoadOrStore(category, state)
	if !loaded {
		c.categoryCount.Add(1)
	}
	return actual.(*categoryState)
}
