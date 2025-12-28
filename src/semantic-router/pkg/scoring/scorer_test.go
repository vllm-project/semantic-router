package scoring

import (
	"math"
	"testing"
	"time"
)

// MockMetricsProvider implements MetricsProvider for testing
type MockMetricsProvider struct {
	metrics map[string]ModelMetrics
}

func NewMockMetricsProvider() *MockMetricsProvider {
	return &MockMetricsProvider{
		metrics: make(map[string]ModelMetrics),
	}
}

func (m *MockMetricsProvider) SetModelMetrics(model string, metrics ModelMetrics) {
	m.metrics[model] = metrics
}

func (m *MockMetricsProvider) GetModelMetrics(model string, window time.Duration) (ModelMetrics, error) {
	if metrics, ok := m.metrics[model]; ok {
		return metrics, nil
	}
	return ModelMetrics{}, nil
}

func (m *MockMetricsProvider) GetAllModelMetrics(window time.Duration) (map[string]ModelMetrics, error) {
	result := make(map[string]ModelMetrics, len(m.metrics))
	for k, v := range m.metrics {
		result[k] = v
	}
	return result, nil
}

func TestNewDynamicScorer(t *testing.T) {
	provider := NewMockMetricsProvider()

	tests := []struct {
		name    string
		config  DynamicScoringConfig
		wantErr bool
	}{
		{
			name: "default config",
			config: DynamicScoringConfig{
				Enabled: true,
			},
			wantErr: false,
		},
		{
			name: "custom weights",
			config: DynamicScoringConfig{
				Enabled: true,
				Weights: ScoreWeights{
					Accuracy: 0.6,
					Latency:  0.3,
					Cost:     0.1,
				},
			},
			wantErr: false,
		},
		{
			name: "unnormalized weights get normalized",
			config: DynamicScoringConfig{
				Enabled: true,
				Weights: ScoreWeights{
					Accuracy: 2.0,
					Latency:  1.0,
					Cost:     1.0,
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scorer, err := NewDynamicScorer(tt.config, provider)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewDynamicScorer() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if scorer == nil {
				t.Error("NewDynamicScorer() returned nil")
			}
		})
	}
}

func TestDynamicScorer_ComputeScore(t *testing.T) {
	provider := NewMockMetricsProvider()

	config := DynamicScoringConfig{
		Enabled:      true,
		MinSamples:   5,
		DefaultScore: 0.5,
		DecayFactor:  0.1,
		Weights: ScoreWeights{
			Accuracy: 0.5,
			Latency:  0.3,
			Cost:     0.2,
		},
		Normalization: NormalizationConfig{
			LatencyTarget: 1.0,
			LatencyMax:    10.0,
			CostTarget:    0.01,
			CostMax:       0.1,
		},
	}

	scorer, _ := NewDynamicScorer(config, provider)

	tests := []struct {
		name          string
		model         string
		metrics       ModelMetrics
		minScore      float64
		maxScore      float64
		expectDefault bool
	}{
		{
			name:  "perfect model - no errors, low latency, low cost",
			model: "perfect-model",
			metrics: ModelMetrics{
				RequestCount:      100,
				ErrorCount:        0,
				ErrorRate:         0.0,
				AvgLatencySeconds: 0.1,
				CostPerKTokens:    0.001,
			},
			minScore:      0.8,
			maxScore:      1.0,
			expectDefault: false,
		},
		{
			name:  "high error rate model",
			model: "error-prone-model",
			metrics: ModelMetrics{
				RequestCount:      100,
				ErrorCount:        50,
				ErrorRate:         0.5,
				AvgLatencySeconds: 0.5,
				CostPerKTokens:    0.005,
			},
			minScore:      0.2,
			maxScore:      0.6,
			expectDefault: false,
		},
		{
			name:  "slow model",
			model: "slow-model",
			metrics: ModelMetrics{
				RequestCount:      100,
				ErrorCount:        0,
				ErrorRate:         0.0,
				AvgLatencySeconds: 8.0,
				CostPerKTokens:    0.001,
			},
			minScore:      0.4,
			maxScore:      0.7,
			expectDefault: false,
		},
		{
			name:  "expensive model",
			model: "expensive-model",
			metrics: ModelMetrics{
				RequestCount:      100,
				ErrorCount:        0,
				ErrorRate:         0.0,
				AvgLatencySeconds: 0.5,
				CostPerKTokens:    0.09,
			},
			minScore:      0.5,
			maxScore:      0.85,
			expectDefault: false,
		},
		{
			name:  "insufficient samples - should use default",
			model: "new-model",
			metrics: ModelMetrics{
				RequestCount:      3, // Below MinSamples
				ErrorCount:        0,
				ErrorRate:         0.0,
				AvgLatencySeconds: 0.1,
				CostPerKTokens:    0.001,
			},
			minScore:      0.5,
			maxScore:      0.5,
			expectDefault: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := scorer.computeScore(tt.model, tt.metrics)

			if score.IsDefault != tt.expectDefault {
				t.Errorf("expected IsDefault=%v, got %v", tt.expectDefault, score.IsDefault)
			}

			if score.CompositeScore < tt.minScore || score.CompositeScore > tt.maxScore {
				t.Errorf("expected score in range [%.2f, %.2f], got %.4f",
					tt.minScore, tt.maxScore, score.CompositeScore)
			}
		})
	}
}

func TestDynamicScorer_NormalizeLatency(t *testing.T) {
	config := DynamicScoringConfig{
		Enabled: true,
		Normalization: NormalizationConfig{
			LatencyTarget: 1.0,
			LatencyMax:    10.0,
		},
	}

	scorer, _ := NewDynamicScorer(config, nil)

	tests := []struct {
		name     string
		latency  float64
		minScore float64
		maxScore float64
	}{
		{"zero latency", 0.0, 1.0, 1.0},
		{"very low latency", 0.1, 0.8, 1.0},
		{"target latency", 1.0, 0.3, 0.5},
		{"high latency", 5.0, 0.0, 0.2},
		{"max latency", 10.0, 0.0, 0.0},
		{"above max latency", 15.0, 0.0, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := scorer.normalizeLatency(tt.latency)
			if score < tt.minScore || score > tt.maxScore {
				t.Errorf("normalizeLatency(%.1f) = %.4f, expected in range [%.2f, %.2f]",
					tt.latency, score, tt.minScore, tt.maxScore)
			}
		})
	}
}

func TestDynamicScorer_NormalizeCost(t *testing.T) {
	config := DynamicScoringConfig{
		Enabled: true,
		Normalization: NormalizationConfig{
			CostTarget: 0.01,
			CostMax:    0.1,
		},
	}

	scorer, _ := NewDynamicScorer(config, nil)

	tests := []struct {
		name     string
		cost     float64
		minScore float64
		maxScore float64
	}{
		{"free", 0.0, 1.0, 1.0},
		{"very cheap", 0.001, 0.9, 1.0},
		{"target cost", 0.01, 0.45, 0.55},
		{"expensive", 0.05, 0.2, 0.35},
		{"max cost", 0.1, 0.0, 0.05},
		{"above max cost", 0.2, 0.0, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := scorer.normalizeCost(tt.cost)
			if score < tt.minScore || score > tt.maxScore {
				t.Errorf("normalizeCost(%.3f) = %.4f, expected in range [%.2f, %.2f]",
					tt.cost, score, tt.minScore, tt.maxScore)
			}
		})
	}
}

func TestDynamicScorer_GetBestModel(t *testing.T) {
	provider := NewMockMetricsProvider()

	// Set up test models with different scores
	provider.SetModelMetrics("model-a", ModelMetrics{
		RequestCount:      100,
		ErrorRate:         0.1,
		AvgLatencySeconds: 1.0,
		CostPerKTokens:    0.01,
	})
	provider.SetModelMetrics("model-b", ModelMetrics{
		RequestCount:      100,
		ErrorRate:         0.0,
		AvgLatencySeconds: 0.5,
		CostPerKTokens:    0.005,
	})
	provider.SetModelMetrics("model-c", ModelMetrics{
		RequestCount:      100,
		ErrorRate:         0.3,
		AvgLatencySeconds: 2.0,
		CostPerKTokens:    0.02,
	})

	config := DynamicScoringConfig{
		Enabled:      true,
		MinSamples:   10,
		DefaultScore: 0.5,
		DecayFactor:  1.0, // No smoothing for test
		Weights: ScoreWeights{
			Accuracy: 0.5,
			Latency:  0.3,
			Cost:     0.2,
		},
		Normalization: NormalizationConfig{
			LatencyTarget: 1.0,
			LatencyMax:    10.0,
			CostTarget:    0.01,
			CostMax:       0.1,
		},
	}

	scorer, _ := NewDynamicScorer(config, provider)
	scorer.updateScores() // Force score computation

	tests := []struct {
		name       string
		candidates []string
		wantModel  string
	}{
		{
			name:       "select best from all",
			candidates: []string{"model-a", "model-b", "model-c"},
			wantModel:  "model-b", // Best: no errors, low latency, low cost
		},
		{
			name:       "select from subset",
			candidates: []string{"model-a", "model-c"},
			wantModel:  "model-a", // Better than model-c
		},
		{
			name:       "single candidate",
			candidates: []string{"model-c"},
			wantModel:  "model-c",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bestModel, _ := scorer.GetBestModel(tt.candidates)
			if bestModel != tt.wantModel {
				t.Errorf("GetBestModel() = %s, want %s", bestModel, tt.wantModel)
			}
		})
	}
}

func TestDynamicScorer_WeightsNormalization(t *testing.T) {
	provider := NewMockMetricsProvider()

	// Weights that don't sum to 1.0
	config := DynamicScoringConfig{
		Enabled: true,
		Weights: ScoreWeights{
			Accuracy: 1.0,
			Latency:  1.0,
			Cost:     1.0,
		},
	}

	scorer, _ := NewDynamicScorer(config, provider)

	// Check weights were normalized
	totalWeight := scorer.config.Weights.Accuracy +
		scorer.config.Weights.Latency +
		scorer.config.Weights.Cost

	if math.Abs(totalWeight-1.0) > 0.01 {
		t.Errorf("weights should sum to 1.0, got %f", totalWeight)
	}

	// Each should be ~0.333
	expectedWeight := 1.0 / 3.0
	if math.Abs(scorer.config.Weights.Accuracy-expectedWeight) > 0.01 {
		t.Errorf("accuracy weight = %f, want ~%f", scorer.config.Weights.Accuracy, expectedWeight)
	}
}

func TestDynamicScorer_EMASmooothing(t *testing.T) {
	provider := NewMockMetricsProvider()

	config := DynamicScoringConfig{
		Enabled:      true,
		MinSamples:   5,
		DefaultScore: 0.5,
		DecayFactor:  0.3, // EMA decay
		Weights: ScoreWeights{
			Accuracy: 0.5,
			Latency:  0.3,
			Cost:     0.2,
		},
		Normalization: NormalizationConfig{
			LatencyTarget: 1.0,
			LatencyMax:    10.0,
			CostTarget:    0.01,
			CostMax:       0.1,
		},
	}

	scorer, _ := NewDynamicScorer(config, provider)

	// Set initial good metrics
	provider.SetModelMetrics("test-model", ModelMetrics{
		RequestCount:      100,
		ErrorRate:         0.0,
		AvgLatencySeconds: 0.1,
		CostPerKTokens:    0.001,
	})

	scorer.updateScores()
	initialScore, _ := scorer.GetScore("test-model")

	// Suddenly degrade metrics
	provider.SetModelMetrics("test-model", ModelMetrics{
		RequestCount:      100,
		ErrorRate:         0.5,
		AvgLatencySeconds: 5.0,
		CostPerKTokens:    0.05,
	})

	scorer.updateScores()
	afterDegradeScore, _ := scorer.GetScore("test-model")

	// With EMA, score should not drop immediately to the new "bad" value
	// It should be somewhere between initial and new raw score
	// Since decay is 0.3, the new score = 0.3 * newRaw + 0.7 * initial
	if afterDegradeScore >= initialScore {
		t.Errorf("score should decrease after degradation: initial=%.4f, after=%.4f",
			initialScore, afterDegradeScore)
	}
}

func TestWindowedMetricsProvider_RecordAndGet(t *testing.T) {
	provider := NewWindowedMetricsProvider(10)

	// Record some requests
	for i := 0; i < 20; i++ {
		isError := i%5 == 0 // 20% error rate
		provider.RecordRequest("test-model", 0.5, 100, 50, isError, 0.001)
	}

	// Get metrics
	metrics, err := provider.GetModelMetrics("test-model", 5*time.Minute)
	if err != nil {
		t.Fatalf("GetModelMetrics() error: %v", err)
	}

	if metrics.RequestCount != 20 {
		t.Errorf("RequestCount = %d, want 20", metrics.RequestCount)
	}

	if metrics.ErrorCount != 4 {
		t.Errorf("ErrorCount = %d, want 4", metrics.ErrorCount)
	}

	expectedErrorRate := 0.2
	if math.Abs(metrics.ErrorRate-expectedErrorRate) > 0.01 {
		t.Errorf("ErrorRate = %f, want %f", metrics.ErrorRate, expectedErrorRate)
	}
}

func TestWindowedMetricsProvider_TimeWindow(t *testing.T) {
	provider := NewWindowedMetricsProvider(10)

	// Record a request
	provider.RecordRequest("test-model", 0.5, 100, 50, false, 0.001)

	// Should be visible in 5-minute window
	metrics, _ := provider.GetModelMetrics("test-model", 5*time.Minute)
	if metrics.RequestCount != 1 {
		t.Errorf("RequestCount = %d, want 1", metrics.RequestCount)
	}

	// Should not be visible in zero-duration window (future only)
	metrics, _ = provider.GetModelMetrics("test-model", 0)
	if metrics.RequestCount != 0 {
		t.Errorf("RequestCount for zero window = %d, want 0", metrics.RequestCount)
	}
}

func TestComputePercentile(t *testing.T) {
	tests := []struct {
		name       string
		values     []float64
		percentile float64
		want       float64
		tolerance  float64
	}{
		{"empty slice", []float64{}, 0.5, 0.0, 0.0},
		{"single value", []float64{5.0}, 0.5, 5.0, 0.01},
		{"p50 of two values", []float64{1.0, 9.0}, 0.5, 5.0, 0.01},
		{"p95 of sorted list", []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, 0.95, 9.55, 0.1},
		{"p0", []float64{1.0, 2.0, 3.0}, 0.0, 1.0, 0.01},
		{"p100", []float64{1.0, 2.0, 3.0}, 1.0, 3.0, 0.01},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := computePercentile(tt.values, tt.percentile)
			if math.Abs(got-tt.want) > tt.tolerance {
				t.Errorf("computePercentile() = %v, want %v (tolerance %v)", got, tt.want, tt.tolerance)
			}
		})
	}
}
