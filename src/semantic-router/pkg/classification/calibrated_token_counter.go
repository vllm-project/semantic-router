package classification

import (
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

const (
	defaultBytesPerToken        = float64(CharactersPerToken)
	defaultCalibrationEMADecay  = 0.995
	minCalibrationSamplesForUse = 10
	defaultCalibrationCategory  = "_default"
	maxCalibrationCategories    = 256
	conservativeRatioAlpha      = 0.90
)

var (
	tokenCalibrationRatioGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "vsr_token_calibration_ratio",
		Help: "Learned bytes-per-token ratio per token calibration category.",
	}, []string{"category"})

	tokenCalibrationErrorHist = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "vsr_token_estimation_error_pct",
		Help:    "Signed token estimation error percentage after calibration observations.",
		Buckets: []float64{-50, -30, -20, -10, -5, 0, 5, 10, 20, 30, 50},
	}, []string{"category"})

	tokenCalibrationSamplesGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "vsr_token_calibration_samples",
		Help: "Number of token calibration samples collected per category.",
	}, []string{"category"})
)

type tokenCalibrationState struct {
	mu                sync.RWMutex
	meanRatio         float64
	conservativeRatio float64
	samples           int64
	lastUpdateTime    time.Time
}

func newTokenCalibrationState() *tokenCalibrationState {
	return &tokenCalibrationState{
		meanRatio:         defaultBytesPerToken,
		conservativeRatio: defaultBytesPerToken,
	}
}

// CalibratedTokenCounter estimates token counts using observed provider usage.
//
// It starts with the same bytes-per-token heuristic as CharacterBasedTokenCounter
// and switches to a learned EMA ratio after enough successful response-usage
// samples have been observed. The counter is safe for concurrent request and
// response processing.
type CalibratedTokenCounter struct {
	decay        float64
	conservative bool
	categories   sync.Map
	categoryCnt  atomic.Int64
}

// CalibratedTokenCounterOption configures a CalibratedTokenCounter.
type CalibratedTokenCounterOption func(*CalibratedTokenCounter)

// WithDecay sets the EMA decay factor. Values outside (0,1) are ignored.
func WithDecay(decay float64) CalibratedTokenCounterOption {
	return func(c *CalibratedTokenCounter) {
		if decay > 0 && decay < 1 {
			c.decay = decay
		}
	}
}

// WithConservativeEstimate returns the lower-tail bytes-per-token ratio, which
// intentionally estimates more tokens when observations are noisy.
func WithConservativeEstimate() CalibratedTokenCounterOption {
	return func(c *CalibratedTokenCounter) {
		c.conservative = true
	}
}

// NewCalibratedTokenCounter creates a token counter backed by online
// bytes-per-token calibration.
func NewCalibratedTokenCounter(opts ...CalibratedTokenCounterOption) *CalibratedTokenCounter {
	c := &CalibratedTokenCounter{
		decay: defaultCalibrationEMADecay,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// CountTokens estimates token count for the default aggregate category.
func (c *CalibratedTokenCounter) CountTokens(text string) (int, error) {
	return c.Estimate(defaultCalibrationCategory, len(text)), nil
}

// Estimate returns the calibrated token estimate for byteLen in category.
func (c *CalibratedTokenCounter) Estimate(category string, byteLen int) int {
	tokens, _, _ := c.EstimateWithCategory(category, byteLen)
	return tokens
}

// EstimateWithCategory returns the estimate, ratio, and whether enough samples
// exist for the category to use learned calibration.
func (c *CalibratedTokenCounter) EstimateWithCategory(category string, byteLen int) (int, float64, bool) {
	if byteLen <= 0 {
		return 0, defaultBytesPerToken, false
	}

	state := c.getOrCreateCategory(normalizeCalibrationCategory(category))
	state.mu.RLock()
	ratio := state.meanRatio
	if c.conservative {
		ratio = state.conservativeRatio
	}
	samples := state.samples
	state.mu.RUnlock()

	calibrated := samples >= minCalibrationSamplesForUse
	if !calibrated {
		ratio = defaultBytesPerToken
	}
	tokens := int(math.Ceil(float64(byteLen) / ratio))
	if tokens < 1 {
		tokens = 1
	}
	return tokens, ratio, calibrated
}

// Observe records an actual provider prompt-token usage sample.
func (c *CalibratedTokenCounter) Observe(category string, byteLen int, actualTokens int) {
	if byteLen <= 0 || actualTokens <= 0 {
		return
	}

	category = normalizeCalibrationCategory(category)
	observedRatio := float64(byteLen) / float64(actualTokens)
	state := c.getOrCreateCategory(category)

	state.mu.Lock()
	if state.samples == 0 {
		state.meanRatio = observedRatio
		state.conservativeRatio = observedRatio
	} else {
		state.meanRatio = c.decay*state.meanRatio + (1-c.decay)*observedRatio
		if observedRatio < state.conservativeRatio {
			state.conservativeRatio += conservativeRatioAlpha * (observedRatio - state.conservativeRatio)
		} else {
			state.conservativeRatio += (1 - conservativeRatioAlpha) * (observedRatio - state.conservativeRatio)
		}
	}
	state.samples++
	state.lastUpdateTime = time.Now()
	meanRatio := state.meanRatio
	samples := state.samples
	state.mu.Unlock()

	estimatedTokens := float64(byteLen) / meanRatio
	errorPct := (estimatedTokens - float64(actualTokens)) / float64(actualTokens) * 100
	tokenCalibrationErrorHist.WithLabelValues(category).Observe(errorPct)
	tokenCalibrationRatioGauge.WithLabelValues(category).Set(meanRatio)
	tokenCalibrationSamplesGauge.WithLabelValues(category).Set(float64(samples))
}

// GetRatio returns the learned mean and conservative ratios for category.
func (c *CalibratedTokenCounter) GetRatio(category string) (mean, conservative float64, samples int64, calibrated bool) {
	category = normalizeCalibrationCategory(category)
	val, ok := c.categories.Load(category)
	if !ok {
		return defaultBytesPerToken, defaultBytesPerToken, 0, false
	}
	state := val.(*tokenCalibrationState)
	state.mu.RLock()
	defer state.mu.RUnlock()
	return state.meanRatio, state.conservativeRatio, state.samples, state.samples >= minCalibrationSamplesForUse
}

// Categories returns a deterministic snapshot of tracked category names.
func (c *CalibratedTokenCounter) Categories() []string {
	categories := []string{}
	c.categories.Range(func(key, _ any) bool {
		categories = append(categories, key.(string))
		return true
	})
	sort.Strings(categories)
	return categories
}

func (c *CalibratedTokenCounter) getOrCreateCategory(category string) *tokenCalibrationState {
	if val, ok := c.categories.Load(category); ok {
		return val.(*tokenCalibrationState)
	}
	if c.categoryCnt.Load() >= maxCalibrationCategories {
		category = defaultCalibrationCategory
		if val, ok := c.categories.Load(category); ok {
			return val.(*tokenCalibrationState)
		}
	}

	state := newTokenCalibrationState()
	actual, loaded := c.categories.LoadOrStore(category, state)
	if !loaded {
		c.categoryCnt.Add(1)
	}
	return actual.(*tokenCalibrationState)
}

func normalizeCalibrationCategory(category string) string {
	if category == "" {
		return defaultCalibrationCategory
	}
	return category
}
