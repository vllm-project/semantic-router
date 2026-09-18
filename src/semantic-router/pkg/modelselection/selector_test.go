/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package modelselection

import (
	"math"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var testModels = []config.ModelRef{
	{Model: "model-a"},
	{Model: "model-b"},
}

func TestNewSelector(t *testing.T) {
	tests := []struct {
		name        string
		cfg         *config.MLModelSelectionConfig
		expectName  string
		expectError bool
	}{
		{
			name:        "knn selector",
			cfg:         &config.MLModelSelectionConfig{Type: "knn", K: 5},
			expectName:  "knn",
			expectError: false,
		},
		{
			name:        "knn with default k",
			cfg:         &config.MLModelSelectionConfig{Type: "knn"},
			expectName:  "knn",
			expectError: false,
		},
		{
			name:        "kmeans selector",
			cfg:         &config.MLModelSelectionConfig{Type: "kmeans", NumClusters: 3},
			expectName:  "kmeans",
			expectError: false,
		},
		{
			name:        "svm selector",
			cfg:         &config.MLModelSelectionConfig{Type: "svm", Kernel: "linear"},
			expectName:  "svm",
			expectError: false,
		},
		{
			name:        "svm linear kernel",
			cfg:         &config.MLModelSelectionConfig{Type: "svm", Kernel: "linear"},
			expectName:  "svm",
			expectError: false,
		},
		{
			name:        "svm poly kernel",
			cfg:         &config.MLModelSelectionConfig{Type: "svm", Kernel: "poly"},
			expectName:  "svm",
			expectError: false,
		},
		{
			name:        "unknown selector type",
			cfg:         &config.MLModelSelectionConfig{Type: "unknown"},
			expectName:  "",
			expectError: true,
		},
		{
			name:        "empty selector type",
			cfg:         &config.MLModelSelectionConfig{Type: ""},
			expectName:  "",
			expectError: true,
		},
		{
			name:        "nil config",
			cfg:         nil,
			expectName:  "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			selector, err := NewSelector(tt.cfg)
			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}
			if selector.Name() != tt.expectName {
				t.Errorf("Expected name %s, got %s", tt.expectName, selector.Name())
			}
		})
	}
}

func TestStatsTracker_UpdateAndGet(t *testing.T) {
	tracker := NewStatsTracker()

	// Test non-existent model
	stats := tracker.GetStats("non-existent")
	if stats != nil {
		t.Error("Expected nil for non-existent model")
	}

	// Update stats for model-a
	tracker.UpdateStats("model-a", 100*time.Millisecond, 0.9, true)
	tracker.UpdateStats("model-a", 200*time.Millisecond, 0.8, true)
	tracker.UpdateStats("model-a", 150*time.Millisecond, 0.0, false)

	stats = tracker.GetStats("model-a")
	if stats == nil {
		t.Fatal("Expected stats, got nil")
	}

	if stats.RequestCount != 3 {
		t.Errorf("Expected 3 requests, got %d", stats.RequestCount)
	}

	// Check average latency: (100 + 200 + 150) / 3 = 150
	expectedLatency := 150.0
	if math.Abs(stats.AverageLatency-expectedLatency) > 1.0 {
		t.Errorf("Expected latency ~%.0f, got %.2f", expectedLatency, stats.AverageLatency)
	}

	// Check success rate: 2 success / 3 total ≈ 0.667
	expectedSuccessRate := 2.0 / 3.0
	if math.Abs(stats.SuccessRate-expectedSuccessRate) > 0.01 {
		t.Errorf("Expected success rate ~%.2f, got %.2f", expectedSuccessRate, stats.SuccessRate)
	}

	// Check quality score: (0.9 + 0.8 + 0.0) / 3 ≈ 0.567
	expectedQuality := (0.9 + 0.8 + 0.0) / 3.0
	if math.Abs(stats.QualityScore-expectedQuality) > 0.01 {
		t.Errorf("Expected quality ~%.2f, got %.2f", expectedQuality, stats.QualityScore)
	}
}

func TestStatsTracker_GetAllStats(t *testing.T) {
	tracker := NewStatsTracker()

	// Add stats for multiple models
	tracker.UpdateStats("model-a", 100*time.Millisecond, 0.9, true)
	tracker.UpdateStats("model-b", 200*time.Millisecond, 0.8, true)
	tracker.UpdateStats("model-c", 150*time.Millisecond, 0.7, true)

	allStats := tracker.GetAllStats()
	if len(allStats) != 3 {
		t.Errorf("Expected 3 models, got %d", len(allStats))
	}

	// Verify each model exists
	for _, name := range []string{"model-a", "model-b", "model-c"} {
		if _, exists := allStats[name]; !exists {
			t.Errorf("Missing stats for %s", name)
		}
	}

	// Verify returned map is a copy (modifying it shouldn't affect tracker)
	delete(allStats, "model-a")
	if tracker.GetStats("model-a") == nil {
		t.Error("Deleting from returned map affected tracker")
	}
}

func TestStatsTracker_Concurrent(t *testing.T) {
	tracker := NewStatsTracker()
	done := make(chan bool)

	// Concurrent writers
	for i := 0; i < 10; i++ {
		go func(id int) {
			for j := 0; j < 100; j++ {
				tracker.UpdateStats("model-a", time.Duration(id)*time.Millisecond, 0.9, true)
			}
			done <- true
		}(i)
	}

	// Concurrent readers
	for i := 0; i < 5; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				_ = tracker.GetStats("model-a")
				_ = tracker.GetAllStats()
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 15; i++ {
		<-done
	}

	stats := tracker.GetStats("model-a")
	if stats == nil {
		t.Fatal("Expected stats after concurrent access")
	}
	if stats.RequestCount != 1000 {
		t.Errorf("Expected 1000 requests, got %d", stats.RequestCount)
	}
}

func TestSelector_EmptyRefs(t *testing.T) {
	selectors := []Selector{
		NewKNNSelector(3),
		NewKMeansSelector(3),
		NewSVMSelector("rbf"),
	}

	for _, selector := range selectors {
		t.Run(selector.Name(), func(t *testing.T) {
			result, err := selector.Select(&SelectionContext{}, []config.ModelRef{})
			// Selectors should return an error when no refs are provided (no fallback)
			if err == nil {
				t.Error("Expected error for empty refs, got nil")
			}
			if result != nil {
				t.Error("Expected nil result for empty refs")
			}
		})
	}
}

func TestSelector_SingleModel(t *testing.T) {
	selectors := []Selector{
		NewKNNSelector(3),
		NewKMeansSelector(3),
		NewSVMSelector("rbf"),
	}

	singleModel := []config.ModelRef{{Model: "only-model"}}

	for _, selector := range selectors {
		t.Run(selector.Name(), func(t *testing.T) {
			result, err := selector.Select(&SelectionContext{}, singleModel)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if result.Model != "only-model" {
				t.Errorf("Expected only-model, got %s", result.Model)
			}
		})
	}
}

func TestSelector_NoEmbedding(t *testing.T) {
	selectors := []Selector{
		NewKNNSelector(3),
		NewKMeansSelector(3),
		NewSVMSelector("rbf"),
	}

	for _, selector := range selectors {
		t.Run(selector.Name(), func(t *testing.T) {
			ctx := &SelectionContext{
				QueryText: "test query without embedding",
				// No embedding
			}

			_, err := selector.Select(ctx, testModels)
			// Should return an error when no embedding is provided (not trained or no embedding)
			// This is expected behavior - ML selectors require embeddings to work
			if err == nil {
				t.Fatal("expected an error for a multi-model selection without an embedding")
			}
		})
	}
}

func TestProductionScenario_ColdStart(t *testing.T) {
	algorithms := []string{"knn", "kmeans", "svm"}

	for _, algoType := range algorithms {
		t.Run(algoType, func(t *testing.T) {
			selector, _ := NewSelector(&config.MLModelSelectionConfig{Type: algoType})

			// No trained artifact means a multi-model decision must fail.
			ctx := &SelectionContext{
				QueryEmbedding: []float64{1, 0},
				QueryText:      "What is 2+2?",
			}

			_, err := selector.Select(ctx, testModels)
			// Cold start (untrained) selectors should return an error (no fallback)
			// This is expected behavior - must load pretrained model first
			if err == nil {
				t.Fatal("expected an error for a multi-model selection without a trained artifact")
			}
		})
	}
}

func TestNormalizeVector(t *testing.T) {
	v := []float64{3, 4, 0}
	normalized := NormalizeVector(v)

	// Should have unit length
	var sumSquares float64
	for _, val := range normalized {
		sumSquares += val * val
	}
	norm := math.Sqrt(sumSquares)

	if math.Abs(norm-1.0) > 0.001 {
		t.Errorf("Normalized vector has norm %.4f, expected 1.0", norm)
	}
}

func TestSoftmax(t *testing.T) {
	input := []float64{1.0, 2.0, 3.0}
	output := Softmax(input)

	// Sum should be 1
	var sum float64
	for _, v := range output {
		sum += v
	}

	if math.Abs(sum-1.0) > 0.001 {
		t.Errorf("Softmax sum is %.4f, expected 1.0", sum)
	}

	// Values should be ordered (larger input → larger output)
	if output[0] >= output[1] || output[1] >= output[2] {
		t.Error("Softmax did not preserve ordering")
	}
}

func TestFloat32ToFloat64(t *testing.T) {
	input := []float32{1.5, 2.5, 3.5}
	output := Float32ToFloat64(input)

	if len(output) != len(input) {
		t.Errorf("Length mismatch: %d vs %d", len(output), len(input))
	}

	for i, v := range output {
		if math.Abs(v-float64(input[i])) > 0.0001 {
			t.Errorf("Conversion error at %d: %.4f vs %.4f", i, v, input[i])
		}
	}
}

func TestDecisionIntegration_ConfigValidation(t *testing.T) {
	t.Run("Invalid algorithm type returns error", func(t *testing.T) {
		cfg := &config.MLModelSelectionConfig{
			Type: "invalid-algorithm",
		}
		_, err := NewSelector(cfg)
		if err == nil {
			t.Error("Expected error for invalid algorithm type")
		}
	})

	t.Run("Empty algorithm type returns error", func(t *testing.T) {
		cfg := &config.MLModelSelectionConfig{
			Type: "",
		}
		_, err := NewSelector(cfg)
		if err == nil {
			t.Error("Expected error for empty algorithm type")
		}
	})

	t.Run("Nil config returns error", func(t *testing.T) {
		_, err := NewSelector(nil)
		if err == nil {
			t.Error("Expected error for nil config")
		}
	})

	t.Run("Valid configs with defaults work", func(t *testing.T) {
		// KNN with default K
		knn, err := NewSelector(&config.MLModelSelectionConfig{Type: "knn"})
		if err != nil {
			t.Errorf("KNN with defaults failed: %v", err)
		}
		if knn.Name() != "knn" {
			t.Errorf("Expected knn, got %s", knn.Name())
		}

		// SVM with default kernel
		svm, err := NewSelector(&config.MLModelSelectionConfig{Type: "svm"})
		if err != nil {
			t.Errorf("SVM with defaults failed: %v", err)
		}
		if svm.Name() != "svm" {
			t.Errorf("Expected svm, got %s", svm.Name())
		}
	})
}

func TestKMeans_EfficiencyWeight(t *testing.T) {
	t.Run("Default efficiency weight", func(t *testing.T) {
		selector := NewKMeansSelector(3)
		// Default should be 0.3 (70% performance, 30% efficiency)
		if selector.efficiencyWeight != 0.3 {
			t.Errorf("Expected default efficiency weight 0.3, got %f", selector.efficiencyWeight)
		}
	})

	t.Run("Custom efficiency weight", func(t *testing.T) {
		selector := NewKMeansSelectorWithEfficiency(3, 0.7)
		if selector.efficiencyWeight != 0.7 {
			t.Errorf("Expected efficiency weight 0.7, got %f", selector.efficiencyWeight)
		}
	})

	t.Run("Clamped efficiency weight", func(t *testing.T) {
		// Test values outside [0, 1] are clamped
		selector := NewKMeansSelectorWithEfficiency(3, 1.5)
		if selector.efficiencyWeight != 1.0 {
			t.Errorf("Expected clamped efficiency weight 1.0, got %f", selector.efficiencyWeight)
		}

		selector = NewKMeansSelectorWithEfficiency(3, -0.5)
		if selector.efficiencyWeight != 0.0 {
			t.Errorf("Expected clamped efficiency weight 0.0, got %f", selector.efficiencyWeight)
		}
	})
}

func TestKMeans_ConfigWithEfficiencyWeight(t *testing.T) {
	effWeight := 0.5
	cfg := &config.MLModelSelectionConfig{
		Type:             "kmeans",
		NumClusters:      4,
		EfficiencyWeight: &effWeight,
	}

	selector, err := NewSelector(cfg)
	if err != nil {
		t.Fatalf("Failed to create selector: %v", err)
	}

	if selector.Name() != "kmeans" {
		t.Errorf("Expected kmeans, got %s", selector.Name())
	}

	// Verify it's a KMeansSelector with correct efficiency weight
	kmeans, ok := selector.(*KMeansSelector)
	if !ok {
		t.Fatal("Expected KMeansSelector type")
	}
	if kmeans.efficiencyWeight != 0.5 {
		t.Errorf("Expected efficiency weight 0.5, got %f", kmeans.efficiencyWeight)
	}
}

func BenchmarkCosineSimilarity(b *testing.B) {
	a, other := make([]float64, 384), make([]float64, 384)
	a[0], other[1] = 1, 1
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CosineSimilarity(a, other)
	}
}

func BenchmarkEuclideanDistance(b *testing.B) {
	a, other := make([]float64, 384), make([]float64, 384)
	a[0], other[1] = 1, 1
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EuclideanDistance(a, other)
	}
}
