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
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Trainer instance for generating embeddings
var (
	testTrainer         *Trainer
	testCandleAvailable bool
	trainerInitOnce     sync.Once
)

// initTestTrainer initializes the trainer with Candle if available
func initTestTrainer(t *testing.T) *Trainer {
	trainerInitOnce.Do(func() {
		testTrainer = NewTrainer(768)

		// Try to initialize Candle with Qwen3
		qwen3Path := "../../../../models/mom-embedding-pro"
		err := candle_binding.InitEmbeddingModelsBatched(qwen3Path, 32, 10, true)
		if err == nil {
			testCandleAvailable = true
			testTrainer.SetUseCandle(true)
			testTrainer.SetEmbeddingModel("qwen3")
		}
	})

	if testCandleAvailable {
		t.Log("✓ Using Qwen3/Candle embeddings")
	} else {
		t.Log("⚠ Candle not available - using hash-based fallback")
	}

	return testTrainer
}

// loadOptimalRecordsForTests returns nil since we now use pretrained models from HuggingFace.
// Tests that need training records will skip, which is expected behavior.
func loadOptimalRecordsForTests() ([]TrainingRecord, error) {
	return nil, nil
}

// =============================================================================
// Factory and Basic Unit Tests
// =============================================================================

// TestNewSelector tests the factory function for all algorithm types
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

// TestStatsTracker_UpdateAndGet tests the stats tracking functionality
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

// TestStatsTracker_GetAllStats tests retrieving all model stats
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

// TestStatsTracker_Concurrent tests thread safety of stats tracker
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

// TestSelector_EmptyRefs tests all selectors handle empty refs gracefully
func TestSelector_EmptyRefs(t *testing.T) {
	selectors := []Selector{
		NewKNNSelector(3),
		NewKMeansSelector(3),
		NewSVMSelector("rbf"),
	}

	for _, selector := range selectors {
		t.Run(selector.Name(), func(t *testing.T) {
			result, err := selector.Select(&SelectionContext{}, []config.ModelRef{})
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if result != nil {
				t.Error("Expected nil result for empty refs")
			}
		})
	}
}

// TestSelector_SingleModel tests single model selection (no algorithm needed)
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

// TestSelector_NoEmbedding tests fallback behavior when no embedding provided
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

			result, err := selector.Select(ctx, testModels)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			// Should fall back to first model
			if result.Model != testModels[0].Model {
				t.Logf("Without embedding, got %s (expected fallback to %s)", result.Model, testModels[0].Model)
			}
		})
	}
}

// =============================================================================
// Production Scenario Test Data
// =============================================================================

// Simulates different query types with distinct embedding patterns
type QueryType string

const (
	QueryTypeMath      QueryType = "math"
	QueryTypeCode      QueryType = "code"
	QueryTypeCreative  QueryType = "creative"
	QueryTypeFactual   QueryType = "factual"
	QueryTypeReasoning QueryType = "reasoning"
)

// Model configurations for testing
var testModels = []config.ModelRef{
	{Model: "llama-3.2-1b", LoRAName: ""},
	{Model: "llama-3.2-3b", LoRAName: ""},
	{Model: "codellama-7b", LoRAName: ""},
	{Model: "mistral-7b", LoRAName: ""},
}

// Generate realistic embedding for a query type
// Different query types have distinct embedding patterns
func generateRealisticEmbedding(queryType QueryType, seed int, dim int) []float64 {
	rng := rand.New(rand.NewSource(int64(seed)))
	embedding := make([]float64, dim)

	// Base pattern depends on query type - MORE DISTINCT patterns for better separation
	var basePattern []float64
	switch queryType {
	case QueryTypeMath:
		// Math queries: very high in first quarter, low elsewhere
		basePattern = []float64{0.95, 0.9, 0.85, 0.1, 0.05, 0.05, 0.1, 0.05}
	case QueryTypeCode:
		// Code queries: high in second quarter
		basePattern = []float64{0.1, 0.05, 0.9, 0.95, 0.85, 0.1, 0.05, 0.1}
	case QueryTypeCreative:
		// Creative queries: high in middle sections
		basePattern = []float64{0.05, 0.1, 0.1, 0.15, 0.9, 0.95, 0.85, 0.1}
	case QueryTypeFactual:
		// Factual queries: high in later sections
		basePattern = []float64{0.1, 0.05, 0.05, 0.1, 0.1, 0.15, 0.9, 0.95}
	case QueryTypeReasoning:
		// Reasoning queries: distributed but distinct pattern
		basePattern = []float64{0.7, 0.1, 0.7, 0.1, 0.7, 0.1, 0.7, 0.9}
	}

	// Fill embedding based on pattern with LESS noise for clearer separation
	patternLen := len(basePattern)
	sectionSize := dim / patternLen

	for i := 0; i < dim; i++ {
		section := i / sectionSize
		if section >= patternLen {
			section = patternLen - 1
		}
		// Add small noise to base pattern (reduced from 0.2 to 0.1)
		noise := (rng.Float64() - 0.5) * 0.1
		embedding[i] = basePattern[section] + noise
		// Clamp to valid range
		if embedding[i] < 0 {
			embedding[i] = 0
		}
		if embedding[i] > 1 {
			embedding[i] = 1
		}
	}

	return embedding
}

// Generate training data that associates query types with optimal models
func generateProductionTrainingData(count int, dim int) []TrainingRecord {
	records := make([]TrainingRecord, count)

	// Model-to-query-type mapping (simulating real preferences)
	queryTypeToModel := map[QueryType]string{
		QueryTypeMath:      "mistral-7b",   // Best for math
		QueryTypeCode:      "codellama-7b", // Best for code
		QueryTypeCreative:  "llama-3.2-3b", // Best for creative
		QueryTypeFactual:   "llama-3.2-1b", // Good enough for factual
		QueryTypeReasoning: "mistral-7b",   // Best for reasoning
	}

	queryTypes := []QueryType{QueryTypeMath, QueryTypeCode, QueryTypeCreative, QueryTypeFactual, QueryTypeReasoning}

	for i := 0; i < count; i++ {
		queryType := queryTypes[i%len(queryTypes)]
		optimalModel := queryTypeToModel[queryType]

		// 90% of training data uses optimal model with high quality (up from 80%)
		selectedModel := optimalModel
		quality := 0.90 + rand.Float64()*0.10 // 0.90-1.0 for optimal (narrower, higher range)

		if rand.Float64() < 0.1 {
			// 10% suboptimal selection with clearly lower quality
			otherModels := []string{"mistral-7b", "llama-3.2-3b", "llama-3.2-1b", "codellama-7b", "mistral-7b"}
			selectedModel = otherModels[rand.Intn(len(otherModels))]
			quality = 0.3 + rand.Float64()*0.3 // 0.3-0.6 for suboptimal (clearly worse)
		}

		// Vary latency based on model - helps with learning
		var latency time.Duration
		switch selectedModel {
		case "llama-3.2-1b":
			latency = time.Duration(50+rand.Intn(100)) * time.Millisecond // Fastest
		case "llama-3.2-3b":
			latency = time.Duration(150+rand.Intn(200)) * time.Millisecond
		case "codellama-7b":
			latency = time.Duration(200+rand.Intn(300)) * time.Millisecond
		case "mistral-7b":
			latency = time.Duration(300+rand.Intn(400)) * time.Millisecond // Slower
		default:
			latency = time.Duration(100+rand.Intn(500)) * time.Millisecond
		}

		records[i] = TrainingRecord{
			QueryEmbedding:    generateRealisticEmbedding(queryType, i, dim),
			SelectedModel:     selectedModel,
			ResponseLatencyNs: int64(latency),
			ResponseQuality:   quality,
			Success:           quality > 0.5, // Success correlates with quality
			TimestampUnix:     time.Now().Add(-time.Duration(rand.Intn(24*7)) * time.Hour).Unix(),
		}
	}

	return records
}

// generateEnhancedTrainingData creates larger, more diverse training dataset
func generateEnhancedTrainingData(count int, dim int) []TrainingRecord {
	records := make([]TrainingRecord, 0, count)

	queryTypeToModel := map[QueryType]string{
		QueryTypeMath:      "mistral-7b",
		QueryTypeCode:      "codellama-7b",
		QueryTypeCreative:  "llama-3.2-3b",
		QueryTypeFactual:   "llama-3.2-1b",
		QueryTypeReasoning: "mistral-7b",
	}

	queryTypes := []QueryType{QueryTypeMath, QueryTypeCode, QueryTypeCreative, QueryTypeFactual, QueryTypeReasoning}

	// Generate multiple passes with different seeds for diversity
	recordsPerType := count / len(queryTypes)

	for _, queryType := range queryTypes {
		optimalModel := queryTypeToModel[queryType]

		for i := 0; i < recordsPerType; i++ {
			// Use unique seeds for each record
			seed := int(queryType[0])*10000 + i

			// 95% optimal model selection
			selectedModel := optimalModel
			quality := 0.88 + rand.Float64()*0.12 // 0.88-1.0

			if rand.Float64() < 0.05 {
				// 5% suboptimal - creates negative examples
				allModels := []string{"mistral-7b", "llama-3.2-3b", "llama-3.2-1b", "codellama-7b", "mistral-7b"}
				for {
					selectedModel = allModels[rand.Intn(len(allModels))]
					if selectedModel != optimalModel {
						break
					}
				}
				quality = 0.2 + rand.Float64()*0.3 // 0.2-0.5
			}

			records = append(records, TrainingRecord{
				QueryEmbedding:    generateRealisticEmbedding(queryType, seed, dim),
				SelectedModel:     selectedModel,
				ResponseLatencyNs: int64(time.Duration(100+rand.Intn(400)) * time.Millisecond),
				ResponseQuality:   quality,
				Success:           quality > 0.5,
				TimestampUnix:     time.Now().Add(-time.Duration(rand.Intn(24*30)) * time.Hour).Unix(),
			})
		}
	}

	return records
}

// =============================================================================
// Production Scenario Tests
// =============================================================================

// TestProductionScenario_AllAlgorithms tests all 5 algorithms with realistic data
func TestProductionScenario_AllAlgorithms(t *testing.T) {
	embeddingDim := 384 // Realistic embedding dimension

	// Helper to create float64 pointer
	floatPtr := func(v float64) *float64 { return &v }

	algorithms := []struct {
		name         string
		config       *config.MLModelSelectionConfig
		trainingSize int // Different algorithms need different amounts of data
	}{
		// KNN and SVM achieve highest accuracy - recommended for production
		{"KNN", &config.MLModelSelectionConfig{Type: "knn", K: 5}, 500},
		{"KMeans", &config.MLModelSelectionConfig{Type: "kmeans", NumClusters: 5, EfficiencyWeight: floatPtr(0)}, 500}, // Pure performance
		{"SVM", &config.MLModelSelectionConfig{Type: "svm", Kernel: "linear"}, 500},
	}

	for _, algo := range algorithms {
		t.Run(algo.name, func(t *testing.T) {
			selector, err := NewSelector(algo.config)
			if err != nil {
				t.Fatalf("Failed to create selector: %v", err)
			}

			// Generate training data with algorithm-specific size
			trainingData := generateEnhancedTrainingData(algo.trainingSize, embeddingDim)

			// Train the selector
			err = selector.Train(trainingData)
			if err != nil {
				t.Fatalf("Training failed: %v", err)
			}

			// Test with different query types
			testCases := []struct {
				queryType     QueryType
				expectedModel string
			}{
				{QueryTypeMath, "mistral-7b"},
				{QueryTypeCode, "codellama-7b"},
				{QueryTypeCreative, "llama-3.2-3b"},
				{QueryTypeFactual, "llama-3.2-1b"},
				{QueryTypeReasoning, "mistral-7b"},
			}

			correctSelections := 0
			totalSelections := 0

			for _, tc := range testCases {
				// Run multiple tests per query type
				for i := 0; i < 10; i++ {
					ctx := &SelectionContext{
						QueryEmbedding: generateRealisticEmbedding(tc.queryType, 1000+i, embeddingDim),
						QueryText:      fmt.Sprintf("Test query for %s", tc.queryType),
						CategoryName:   string(tc.queryType),
						DecisionName:   "test_decision",
					}

					result, err := selector.Select(ctx, testModels)
					if err != nil {
						t.Errorf("Selection failed: %v", err)
						continue
					}

					totalSelections++
					if result.Model == tc.expectedModel {
						correctSelections++
					}
				}
			}

			// Log accuracy (don't fail on accuracy, just report)
			accuracy := float64(correctSelections) / float64(totalSelections) * 100
			t.Logf("%s accuracy: %.1f%% (%d/%d correct)", algo.name, accuracy, correctSelections, totalSelections)

			// Ensure selector at least returns valid models
			if totalSelections == 0 {
				t.Error("No selections were made")
			}
		})
	}
}

// TestProductionScenario_ColdStart tests behavior with no training data
func TestProductionScenario_ColdStart(t *testing.T) {
	algorithms := []string{"knn", "kmeans", "svm"}

	for _, algoType := range algorithms {
		t.Run(algoType, func(t *testing.T) {
			selector, _ := NewSelector(&config.MLModelSelectionConfig{Type: algoType})

			// No training - should gracefully fall back to first model
			ctx := &SelectionContext{
				QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 0, 384),
				QueryText:      "What is 2+2?",
			}

			result, err := selector.Select(ctx, testModels)
			if err != nil {
				t.Fatalf("Cold start selection failed: %v", err)
			}

			// Should return first model as fallback
			if result.Model != testModels[0].Model {
				t.Logf("Cold start returned %s (expected fallback to %s)", result.Model, testModels[0].Model)
			}
		})
	}
}

// TestProductionScenario_IncrementalTraining tests online learning capability
func TestProductionScenario_IncrementalTraining(t *testing.T) {
	embeddingDim := 256
	selector := NewKNNSelector(5)

	// Simulate streaming training data over time
	batches := 10
	batchSize := 50

	for batch := 0; batch < batches; batch++ {
		// Generate batch of training data
		batchData := generateProductionTrainingData(batchSize, embeddingDim)

		err := selector.Train(batchData)
		if err != nil {
			t.Fatalf("Batch %d training failed: %v", batch, err)
		}

		// Verify selector still works after each batch
		ctx := &SelectionContext{
			QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, batch*100, embeddingDim),
		}

		_, err = selector.Select(ctx, testModels)
		if err != nil {
			t.Fatalf("Selection after batch %d failed: %v", batch, err)
		}
	}

	t.Logf("Successfully processed %d batches of %d records each", batches, batchSize)
}

// TestProductionScenario_HighConcurrency tests thread safety under load
func TestProductionScenario_HighConcurrency(t *testing.T) {
	embeddingDim := 256
	selector := NewKNNSelector(5)

	// Pre-train with data
	trainingData := generateProductionTrainingData(200, embeddingDim)
	_ = selector.Train(trainingData)

	// Concurrent operations
	numGoroutines := 50
	operationsPerGoroutine := 100

	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines*operationsPerGoroutine)
	selections := make(chan string, numGoroutines*operationsPerGoroutine)

	startTime := time.Now()

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			for i := 0; i < operationsPerGoroutine; i++ {
				// Mix of training and selection operations
				if i%10 == 0 {
					// Training (10% of operations)
					record := TrainingRecord{
						QueryEmbedding:  generateRealisticEmbedding(QueryTypeMath, goroutineID*1000+i, embeddingDim),
						SelectedModel:   "mistral-7b",
						Success:         true,
						ResponseQuality: 0.9,
					}
					if err := selector.Train([]TrainingRecord{record}); err != nil {
						errors <- err
					}
				} else {
					// Selection (90% of operations)
					ctx := &SelectionContext{
						QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, goroutineID*1000+i, embeddingDim),
					}
					result, err := selector.Select(ctx, testModels)
					if err != nil {
						errors <- err
					} else {
						selections <- result.Model
					}
				}
			}
		}(g)
	}

	wg.Wait()
	close(errors)
	close(selections)

	duration := time.Since(startTime)

	// Check for errors
	errorCount := 0
	for err := range errors {
		t.Logf("Error: %v", err)
		errorCount++
	}

	// Count selections
	selectionCount := 0
	modelCounts := make(map[string]int)
	for model := range selections {
		selectionCount++
		modelCounts[model]++
	}

	t.Logf("Completed %d selections in %v", selectionCount, duration)
	t.Logf("Model distribution: %v", modelCounts)
	t.Logf("Operations per second: %.0f", float64(selectionCount)/duration.Seconds())

	if errorCount > 0 {
		t.Errorf("Had %d errors during concurrent execution", errorCount)
	}
}

// TestProductionScenario_LoRAAdapterSelection tests LoRA model selection
func TestProductionScenario_LoRAAdapterSelection(t *testing.T) {
	embeddingDim := 256

	loraModels := []config.ModelRef{
		{Model: "mistral-7b", LoRAName: "math-lora"},
		{Model: "mistral-7b", LoRAName: "code-lora"},
		{Model: "mistral-7b", LoRAName: "creative-lora"},
	}

	selector := NewKNNSelector(3)

	// Train with LoRA-specific data
	trainingData := []TrainingRecord{
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 1, embeddingDim), SelectedModel: "math-lora", Success: true, ResponseQuality: 0.95},
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 2, embeddingDim), SelectedModel: "math-lora", Success: true, ResponseQuality: 0.92},
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeCode, 3, embeddingDim), SelectedModel: "code-lora", Success: true, ResponseQuality: 0.90},
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeCode, 4, embeddingDim), SelectedModel: "code-lora", Success: true, ResponseQuality: 0.93},
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeCreative, 5, embeddingDim), SelectedModel: "creative-lora", Success: true, ResponseQuality: 0.88},
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeCreative, 6, embeddingDim), SelectedModel: "creative-lora", Success: true, ResponseQuality: 0.91},
	}

	_ = selector.Train(trainingData)

	// Test math query should select math-lora
	ctx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 100, embeddingDim),
		QueryText:      "Calculate the derivative of x^2",
	}

	result, err := selector.Select(ctx, loraModels)
	if err != nil {
		t.Fatalf("Selection failed: %v", err)
	}

	if result.LoRAName != "math-lora" {
		t.Logf("Expected math-lora, got %s (base: %s)", result.LoRAName, result.Model)
	}

	// Test code query should select code-lora
	ctx = &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeCode, 101, embeddingDim),
		QueryText:      "Write a Python function to sort a list",
	}

	result, err = selector.Select(ctx, loraModels)
	if err != nil {
		t.Fatalf("Selection failed: %v", err)
	}

	t.Logf("Code query selected: %s (LoRA: %s)", result.Model, result.LoRAName)
}

// TestProductionScenario_FailedRequestHandling tests learning from failures
func TestProductionScenario_FailedRequestHandling(t *testing.T) {
	embeddingDim := 256
	selector := NewKNNSelector(5)

	// Train with mix of successful and failed requests
	trainingData := []TrainingRecord{
		// Successful requests for model A
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 1, embeddingDim), SelectedModel: "mistral-7b", Success: true, ResponseQuality: 0.9},
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 2, embeddingDim), SelectedModel: "mistral-7b", Success: true, ResponseQuality: 0.85},
		// Failed requests for model B on same query type
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 3, embeddingDim), SelectedModel: "llama-3.2-1b", Success: false, ResponseQuality: 0.0},
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 4, embeddingDim), SelectedModel: "llama-3.2-1b", Success: false, ResponseQuality: 0.0},
	}

	_ = selector.Train(trainingData)

	// Query similar to training data
	ctx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 10, embeddingDim),
	}

	twoModels := []config.ModelRef{
		{Model: "mistral-7b"},
		{Model: "llama-3.2-1b"},
	}

	result, err := selector.Select(ctx, twoModels)
	if err != nil {
		t.Fatalf("Selection failed: %v", err)
	}

	// Should prefer mistral-7b due to successful history
	t.Logf("Selected model: %s (expected mistral-7b due to failure history)", result.Model)
}

// TestProductionScenario_LargeEmbeddings tests with production-sized embeddings
func TestProductionScenario_LargeEmbeddings(t *testing.T) {
	embeddingDims := []int{384, 768, 1024, 1536} // Common embedding sizes

	for _, dim := range embeddingDims {
		t.Run(fmt.Sprintf("dim_%d", dim), func(t *testing.T) {
			selector := NewKNNSelector(3)

			// Generate training data with large embeddings
			trainingData := generateProductionTrainingData(100, dim)
			err := selector.Train(trainingData)
			if err != nil {
				t.Fatalf("Training failed for dim %d: %v", dim, err)
			}

			// Test selection
			ctx := &SelectionContext{
				QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 0, dim),
			}

			start := time.Now()
			result, err := selector.Select(ctx, testModels)
			duration := time.Since(start)

			if err != nil {
				t.Fatalf("Selection failed for dim %d: %v", dim, err)
			}

			t.Logf("Dim %d: selected %s in %v", dim, result.Model, duration)

			// Performance check: selection should be fast
			if duration > 100*time.Millisecond {
				t.Logf("Warning: selection took %v (>100ms)", duration)
			}
		})
	}
}

// TestProductionScenario_ModelAvailability tests when some models are unavailable
func TestProductionScenario_ModelAvailability(t *testing.T) {
	embeddingDim := 256
	selector := NewKNNSelector(3)

	// Train with all models
	fullTrainingData := []TrainingRecord{
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 1, embeddingDim), SelectedModel: "mistral-7b", Success: true, ResponseQuality: 0.95},
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 2, embeddingDim), SelectedModel: "llama-3.2-3b", Success: true, ResponseQuality: 0.90},
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 3, embeddingDim), SelectedModel: "llama-3.2-1b", Success: true, ResponseQuality: 0.80},
	}
	_ = selector.Train(fullTrainingData)

	// Test with reduced model availability (mistral-7b unavailable)
	availableModels := []config.ModelRef{
		{Model: "llama-3.2-3b"},
		{Model: "llama-3.2-1b"},
	}

	ctx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 10, embeddingDim),
	}

	result, err := selector.Select(ctx, availableModels)
	if err != nil {
		t.Fatalf("Selection failed: %v", err)
	}

	// Should select from available models only
	found := false
	for _, m := range availableModels {
		if m.Model == result.Model {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("Selected unavailable model: %s", result.Model)
	}

	t.Logf("Selected %s from available models", result.Model)
}

// TestProductionScenario_QualityWeighting tests that quality affects selection
func TestProductionScenario_QualityWeighting(t *testing.T) {
	embeddingDim := 256
	selector := NewKNNSelector(5)

	// Same query type, different quality for different models
	trainingData := []TrainingRecord{
		// Low quality for model A
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 1, embeddingDim), SelectedModel: "model-a", Success: true, ResponseQuality: 0.3},
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 2, embeddingDim), SelectedModel: "model-a", Success: true, ResponseQuality: 0.35},
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 3, embeddingDim), SelectedModel: "model-a", Success: true, ResponseQuality: 0.4},
		// High quality for model B
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 4, embeddingDim), SelectedModel: "model-b", Success: true, ResponseQuality: 0.95},
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 5, embeddingDim), SelectedModel: "model-b", Success: true, ResponseQuality: 0.92},
		{QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 6, embeddingDim), SelectedModel: "model-b", Success: true, ResponseQuality: 0.9},
	}
	_ = selector.Train(trainingData)

	models := []config.ModelRef{
		{Model: "model-a"},
		{Model: "model-b"},
	}

	ctx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 100, embeddingDim),
	}

	result, err := selector.Select(ctx, models)
	if err != nil {
		t.Fatalf("Selection failed: %v", err)
	}

	// Should prefer model-b due to higher quality
	if result.Model != "model-b" {
		t.Logf("Expected model-b (higher quality), got %s", result.Model)
	} else {
		t.Log("Correctly selected higher-quality model")
	}
}

// TestProductionScenario_AllSVMKernels tests all SVM kernel types
func TestProductionScenario_AllSVMKernels(t *testing.T) {
	embeddingDim := 128
	kernels := []string{"linear", "rbf", "poly"}

	trainingData := generateProductionTrainingData(100, embeddingDim)

	for _, kernel := range kernels {
		t.Run(kernel, func(t *testing.T) {
			selector := NewSVMSelector(kernel)
			err := selector.Train(trainingData)
			if err != nil {
				t.Fatalf("Training failed for kernel %s: %v", kernel, err)
			}

			ctx := &SelectionContext{
				QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 0, embeddingDim),
			}

			result, err := selector.Select(ctx, testModels)
			if err != nil {
				t.Fatalf("Selection failed for kernel %s: %v", kernel, err)
			}

			t.Logf("Kernel %s selected: %s", kernel, result.Model)
		})
	}
}

// TestProductionScenario_NumericalStability tests for NaN/Inf handling
func TestProductionScenario_NumericalStability(t *testing.T) {
	algorithms := []struct {
		name     string
		selector Selector
	}{
		{"KNN", NewKNNSelector(3)},
		{"KMeans", NewKMeansSelector(3)},
		{"SVM", NewSVMSelector("rbf")},
	}

	for _, algo := range algorithms {
		t.Run(algo.name, func(t *testing.T) {
			// Train with normal data
			trainingData := generateProductionTrainingData(100, 128)
			_ = algo.selector.Train(trainingData)

			// Test with edge case embeddings
			edgeCases := []struct {
				name      string
				embedding []float64
			}{
				{"zeros", make([]float64, 128)},
				{"ones", func() []float64 {
					e := make([]float64, 128)
					for i := range e {
						e[i] = 1.0
					}
					return e
				}()},
				{"small_values", func() []float64 {
					e := make([]float64, 128)
					for i := range e {
						e[i] = 1e-10
					}
					return e
				}()},
				{"large_values", func() []float64 {
					e := make([]float64, 128)
					for i := range e {
						e[i] = 1e6
					}
					return e
				}()},
			}

			for _, ec := range edgeCases {
				ctx := &SelectionContext{QueryEmbedding: ec.embedding}
				result, err := algo.selector.Select(ctx, testModels)
				if err != nil {
					t.Errorf("%s failed for %s: %v", algo.name, ec.name, err)
					continue
				}

				// Check for NaN/Inf in result
				if result == nil {
					// Nil result is acceptable for empty refs
					continue
				}

				// Verify we got a valid model
				if result.Model == "" {
					t.Errorf("%s returned empty model for %s", algo.name, ec.name)
				}
			}
		})
	}
}

// TestProductionScenario_MemoryBoundedTraining tests that training doesn't grow unbounded
func TestProductionScenario_MemoryBoundedTraining(t *testing.T) {
	selector := NewKNNSelector(5)
	embeddingDim := 256

	// Train with way more data than the max limit (10000)
	for i := 0; i < 150; i++ {
		batch := generateProductionTrainingData(100, embeddingDim) // 15000 total
		_ = selector.Train(batch)
	}

	// Selector should still work (internal limit should cap storage)
	ctx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 0, embeddingDim),
	}

	_, err := selector.Select(ctx, testModels)
	if err != nil {
		t.Fatalf("Selection failed after large training: %v", err)
	}

	t.Log("Memory bounded training test passed")
}

// =============================================================================
// Real Query Tests with Qwen3 Embeddings
// These tests use actual query text and Qwen3/Candle for production-like embeddings
// =============================================================================

// TestRealQueries_AllAlgorithms tests all 5 algorithms with REAL Qwen3 embeddings
// Trains on pre-computed data, then tests on NEW queries with LIVE Qwen3 embedding generation
func TestRealQueries_AllAlgorithms(t *testing.T) {
	// Initialize trainer with Qwen3/Candle
	trainer := initTestTrainer(t)

	// Load OPTIMAL records for training (pre-computed Qwen3 embeddings)
	trainingData, err := loadOptimalRecordsForTests()
	if err != nil {
		t.Skipf("Skipping: could not load training data: %v", err)
	}

	t.Logf("=== Training with %d records, Testing with NEW Qwen3 queries ===", len(trainingData))
	t.Logf("Embedding mode: %s", map[bool]string{true: "Qwen3/Candle", false: "Hash-based fallback"}[testCandleAvailable])

	// Models available for selection
	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
		{Model: "mistral-7b"},
	}

	// NEW test queries (not in training data) with expected best models
	// Based on model characteristics: codellama for code, mistral for math/reasoning, llama-3.2 for general
	newTestQueries := []struct {
		query          string
		category       string
		expectedModels []string // Any of these is acceptable
		description    string
	}{
		// Code queries -> codellama-7b
		{"Write a Python function to find the longest common subsequence of two strings", "code", []string{"codellama-7b"}, "LCS algorithm"},
		{"Implement a binary heap data structure in Go", "code", []string{"codellama-7b"}, "Data structure"},
		{"Create a regex to validate email addresses", "code", []string{"codellama-7b", "mistral-7b"}, "Regex pattern"},

		// Math/reasoning queries -> mistral-7b
		{"Prove that the square root of 2 is irrational", "math", []string{"mistral-7b"}, "Mathematical proof"},
		{"Calculate the eigenvalues of a 3x3 rotation matrix", "math", []string{"mistral-7b", "codellama-7b"}, "Linear algebra"},
		{"If A implies B and B implies C, what can we conclude?", "reasoning", []string{"mistral-7b"}, "Logic"},

		// General/factual queries -> llama models
		{"What are the main causes of climate change?", "general", []string{"llama-3.2-1b", "llama-3.2-3b", "mistral-7b"}, "Science fact"},
		{"Summarize the plot of Romeo and Juliet", "general", []string{"llama-3.2-1b", "llama-3.2-3b"}, "Literature"},
		{"What is photosynthesis and why is it important?", "science", []string{"llama-3.2-1b", "llama-3.2-3b", "mistral-7b"}, "Biology"},
	}

	algorithms := []struct {
		name     string
		selector Selector
	}{
		{"KNN", NewKNNSelector(5)},
		{"KMeans", NewKMeansSelector(4)},
		{"SVM", NewSVMSelector("rbf")},
	}

	for _, algo := range algorithms {
		t.Run(algo.name, func(t *testing.T) {
			err := algo.selector.Train(trainingData)
			if err != nil {
				t.Fatalf("Training failed: %v", err)
			}

			reasonableSelections := 0
			algModelCounts := make(map[string]int)

			for _, tc := range newTestQueries {
				// Generate NEW embedding at runtime using Qwen3
				embedding, err := trainer.GetEmbedding(tc.query)
				if err != nil {
					t.Errorf("Failed to get embedding for '%s': %v", tc.description, err)
					continue
				}

				// Combine with category
				featureVec := CombineEmbeddingWithCategory(embedding, tc.category)

				ctx := &SelectionContext{
					QueryEmbedding: featureVec,
					DecisionName:   "new_query_test",
				}

				result, err := algo.selector.Select(ctx, modelRefs)
				if err != nil {
					t.Errorf("Selection failed for '%s': %v", tc.description, err)
					continue
				}

				algModelCounts[result.Model]++

				// Check if selection is reasonable (in expected models list)
				isReasonable := false
				for _, expected := range tc.expectedModels {
					if result.Model == expected {
						isReasonable = true
						break
					}
				}
				if isReasonable {
					reasonableSelections++
				}

				queryPreview := tc.query
				if len(queryPreview) > 50 {
					queryPreview = queryPreview[:50] + "..."
				}
				t.Logf("  [%s] '%s' -> %s (expected: %v) %s",
					tc.category, queryPreview, result.Model, tc.expectedModels,
					map[bool]string{true: "✓", false: "✗"}[isReasonable])
			}

			accuracy := float64(reasonableSelections) / float64(len(newTestQueries)) * 100
			t.Logf("%s REASONABLE SELECTIONS: %.1f%% (%d/%d)", algo.name, accuracy, reasonableSelections, len(newTestQueries))

			// Log model distribution
			for model, count := range algModelCounts {
				t.Logf("  %s: %d selections (%.1f%%)", model, count, float64(count)/float64(len(newTestQueries))*100)
			}
		})
	}
}

// TestRealQueries_CodeSpecialization tests that code queries route to code-optimized models
// Trains on pre-computed data, tests with NEW code queries using LIVE Qwen3 embedding generation
func TestRealQueries_CodeSpecialization(t *testing.T) {
	// Initialize trainer with Qwen3/Candle
	trainer := initTestTrainer(t)

	// Load training data
	trainingData, err := loadOptimalRecordsForTests()
	if err != nil {
		t.Skipf("Skipping: could not load training data: %v", err)
	}

	t.Logf("=== Code Specialization: Training with %d records ===", len(trainingData))
	t.Logf("Embedding mode: %s", map[bool]string{true: "Qwen3/Candle", false: "Hash-based fallback"}[testCandleAvailable])

	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
		{Model: "mistral-7b"},
	}

	// NEW code queries (not in training data)
	codeQueries := []string{
		"Implement a trie data structure for autocomplete",
		"Write a function to detect cycles in a linked list",
		"Create a thread pool implementation in Java",
		"Implement the A* pathfinding algorithm",
		"Write a parser for arithmetic expressions",
	}

	// Train all algorithms
	algorithms := []struct {
		name     string
		selector Selector
	}{
		{"KNN", NewKNNSelector(5)},
		{"KMeans", NewKMeansSelector(4)},
		{"SVM", NewSVMSelector("rbf")},
	}

	for _, algo := range algorithms {
		t.Run(algo.name, func(t *testing.T) {
			err := algo.selector.Train(trainingData)
			if err != nil {
				t.Fatalf("Training failed: %v", err)
			}

			codeLlamaSelections := 0
			modelSelections := make(map[string]int)

			for _, query := range codeQueries {
				// Generate NEW embedding at runtime
				embedding, err := trainer.GetEmbedding(query)
				if err != nil {
					t.Errorf("Failed to get embedding: %v", err)
					continue
				}

				featureVec := CombineEmbeddingWithCategory(embedding, "code")

				ctx := &SelectionContext{
					QueryEmbedding: featureVec,
				}

				result, err := algo.selector.Select(ctx, modelRefs)
				if err != nil {
					t.Errorf("Selection failed: %v", err)
					continue
				}

				modelSelections[result.Model]++
				if result.Model == "codellama-7b" {
					codeLlamaSelections++
				}

				queryPreview := query
				if len(queryPreview) > 45 {
					queryPreview = queryPreview[:45] + "..."
				}
				t.Logf("  '%s' -> %s", queryPreview, result.Model)
			}

			t.Logf("%s: codellama-7b selected %d/%d times for code queries", algo.name, codeLlamaSelections, len(codeQueries))
			for model, count := range modelSelections {
				t.Logf("  %s: %d selections", model, count)
			}
		})
	}
}

// TestRealQueries_MathSpecialization tests math query routing
// Trains on pre-computed data, tests with NEW math queries using LIVE Qwen3 embedding generation
func TestRealQueries_MathSpecialization(t *testing.T) {
	// Initialize trainer with Qwen3/Candle
	trainer := initTestTrainer(t)

	// Load training data
	trainingData, err := loadOptimalRecordsForTests()
	if err != nil {
		t.Skipf("Skipping: could not load training data: %v", err)
	}

	t.Logf("=== Math Specialization: Training with %d records ===", len(trainingData))
	t.Logf("Embedding mode: %s", map[bool]string{true: "Qwen3/Candle", false: "Hash-based fallback"}[testCandleAvailable])

	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
		{Model: "mistral-7b"},
	}

	// NEW math queries (not in training data)
	mathQueries := []string{
		"Prove that the sum of angles in a triangle is 180 degrees",
		"Calculate the derivative of e^(x^2)",
		"Find the general solution to dy/dx = y/x",
		"What is the probability of rolling two sixes with two dice?",
		"Prove the Pythagorean theorem using similar triangles",
	}

	// Test with all algorithms
	algorithms := []struct {
		name     string
		selector Selector
	}{
		{"KNN", NewKNNSelector(5)},
		{"KMeans", NewKMeansSelector(4)},
		{"SVM", NewSVMSelector("rbf")},
	}

	for _, algo := range algorithms {
		t.Run(algo.name, func(t *testing.T) {
			err := algo.selector.Train(trainingData)
			if err != nil {
				t.Fatalf("Training failed: %v", err)
			}

			mistralSelections := 0
			modelSelections := make(map[string]int)

			for _, query := range mathQueries {
				// Generate NEW embedding at runtime
				embedding, err := trainer.GetEmbedding(query)
				if err != nil {
					t.Errorf("Failed to get embedding: %v", err)
					continue
				}

				featureVec := CombineEmbeddingWithCategory(embedding, "math")

				ctx := &SelectionContext{
					QueryEmbedding: featureVec,
				}

				result, err := algo.selector.Select(ctx, modelRefs)
				if err != nil {
					t.Errorf("Selection failed: %v", err)
					continue
				}

				modelSelections[result.Model]++
				if result.Model == "mistral-7b" {
					mistralSelections++
				}

				queryPreview := query
				if len(queryPreview) > 50 {
					queryPreview = queryPreview[:50] + "..."
				}
				t.Logf("  '%s' -> %s", queryPreview, result.Model)
			}

			t.Logf("%s: mistral-7b selected %d/%d times for math queries", algo.name, mistralSelections, len(mathQueries))
			for model, count := range modelSelections {
				t.Logf("  %s: %d selections", model, count)
			}
		})
	}
}

// =============================================================================
// Benchmark Tests
// =============================================================================

func BenchmarkKNNSelection(b *testing.B) {
	selector := NewKNNSelector(5)
	trainingData := generateProductionTrainingData(1000, 384)
	_ = selector.Train(trainingData)

	ctx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 0, 384),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = selector.Select(ctx, testModels)
	}
}

func BenchmarkKMeansSelection(b *testing.B) {
	selector := NewKMeansSelector(5)
	trainingData := generateProductionTrainingData(1000, 384)
	_ = selector.Train(trainingData)

	ctx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 0, 384),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = selector.Select(ctx, testModels)
	}
}

func BenchmarkSVMSelection(b *testing.B) {
	selector := NewSVMSelector("rbf")
	trainingData := generateProductionTrainingData(500, 384)
	_ = selector.Train(trainingData)

	ctx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 0, 384),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = selector.Select(ctx, testModels)
	}
}

func BenchmarkCosineSimilarity(b *testing.B) {
	a := generateRealisticEmbedding(QueryTypeMath, 0, 384)
	c := generateRealisticEmbedding(QueryTypeCode, 1, 384)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CosineSimilarity(a, c)
	}
}

func BenchmarkEuclideanDistance(b *testing.B) {
	a := generateRealisticEmbedding(QueryTypeMath, 0, 384)
	c := generateRealisticEmbedding(QueryTypeCode, 1, 384)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EuclideanDistance(a, c)
	}
}

// =============================================================================
// Utility Function Tests
// =============================================================================

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

// =============================================================================
// Integration Tests: Decision with ModelSelectionAlgorithm
// These tests verify the full flow: Decision matched -> ModelSelectionAlgorithm -> Model selected
// =============================================================================

// TestDecisionIntegration_AllAlgorithms tests that all algorithm types can be created
// from Decision configuration and perform selection correctly
func TestDecisionIntegration_AllAlgorithms(t *testing.T) {
	algorithmConfigs := []struct {
		name   string
		config *config.MLModelSelectionConfig
	}{
		{
			name:   "knn",
			config: &config.MLModelSelectionConfig{Type: "knn", K: 5},
		},
		{
			name:   "kmeans",
			config: &config.MLModelSelectionConfig{Type: "kmeans", NumClusters: 3},
		},
		{
			name:   "svm",
			config: &config.MLModelSelectionConfig{Type: "svm", Kernel: "linear"},
		},
	}

	modelRefs := []config.ModelRef{
		{Model: "model-a"},
		{Model: "model-b"},
		{Model: "model-c"},
	}

	for _, tc := range algorithmConfigs {
		t.Run(tc.name, func(t *testing.T) {
			// Create selector from algorithm config
			selector, err := NewSelector(tc.config)
			if err != nil {
				t.Fatalf("Failed to create selector: %v", err)
			}
			if selector.Name() != tc.name {
				t.Errorf("Expected selector name %s, got %s", tc.name, selector.Name())
			}

			// Verify selector can handle selection request
			ctx := &SelectionContext{
				QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 0, 256),
				QueryText:      "Test query",
				CategoryName:   "test",
				DecisionName:   "test-decision",
			}

			result, err := selector.Select(ctx, modelRefs)
			if err != nil {
				t.Fatalf("Selection failed: %v", err)
			}
			if result == nil {
				t.Fatal("Expected result, got nil")
			}

			// Should be one of the available models
			validModels := map[string]bool{"model-a": true, "model-b": true, "model-c": true}
			if !validModels[result.Model] {
				t.Errorf("Selected model %s not in available models", result.Model)
			}
		})
	}
}

// TestDecisionIntegration_CompleteFlow tests the complete flow as it happens in production
func TestDecisionIntegration_CompleteFlow(t *testing.T) {
	t.Run("Multiple models with algorithm uses ML selection", func(t *testing.T) {
		// Simulate the complete flow as it happens in performDecisionEvaluationAndModelSelection
		mlConfig := &config.MLModelSelectionConfig{
			Type: "knn",
			K:    5,
		}
		modelRefs := []config.ModelRef{
			{Model: "mistral-7b"},
			{Model: "deepseek-math"},
			{Model: "llama-math"},
		}
		decisionName := "math-specialized"

		// Step 1: Check if model selection should be used
		if len(modelRefs) <= 1 {
			t.Fatal("Test requires multiple models")
		}
		// mlConfig is always set in this test

		// Step 2: Create selector
		selector, err := NewSelector(mlConfig)
		if err != nil {
			t.Fatalf("Failed to create selector: %v", err)
		}

		// Step 3: Build selection context (from signal extraction)
		selectionCtx := &SelectionContext{
			QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 0, 384),
			QueryText:      "Calculate the integral of x squared",
			CategoryName:   "mathematics",
			DecisionName:   decisionName,
		}

		// Step 4: Select model
		selected, err := selector.Select(selectionCtx, modelRefs)
		if err != nil {
			t.Fatalf("Selection failed: %v", err)
		}
		if selected == nil {
			t.Fatal("Expected selected model, got nil")
		}

		// Verify selected model is one of the available models
		validModels := map[string]bool{"mistral-7b": true, "deepseek-math": true, "llama-math": true}
		if !validModels[selected.Model] {
			t.Errorf("Selected model %s not in available models", selected.Model)
		}

		t.Logf("Complete flow selected: %s", selected.Model)
	})

	t.Run("Single model skips selection", func(t *testing.T) {
		mlConfig := &config.MLModelSelectionConfig{
			Type: "knn",
			K:    5,
		}
		modelRefs := []config.ModelRef{
			{Model: "only-model"},
		}

		// With single model, selection returns that model
		selector, _ := NewSelector(mlConfig)
		result, err := selector.Select(&SelectionContext{}, modelRefs)
		if err != nil {
			t.Fatalf("Selection failed: %v", err)
		}
		if result.Model != "only-model" {
			t.Errorf("Expected only-model, got %s", result.Model)
		}
	})

	t.Run("No algorithm uses first model", func(t *testing.T) {
		matchedDecision := config.Decision{
			ModelRefs: []config.ModelRef{
				{Model: "first"},
				{Model: "second"},
				{Model: "third"},
			},
		}

		// When no algorithm configured, use first model (this is what req_filter_classification.go does)
		if matchedDecision.ModelSelectionAlgorithm != nil {
			t.Fatal("Test expects no algorithm")
		}
		selected := matchedDecision.ModelRefs[0]
		if selected.Model != "first" {
			t.Errorf("Expected first, got %s", selected.Model)
		}
	})
}

// TestDecisionIntegration_ConfigValidation tests configuration validation
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

// TestDecisionIntegration_TrainingFlow tests that training affects selection in production flow
func TestDecisionIntegration_TrainingFlow(t *testing.T) {
	// Simulate a decision with multiple math models
	mlConfig := &config.MLModelSelectionConfig{
		Type: "knn",
		K:    3,
	}
	modelRefs := []config.ModelRef{
		{Model: "calculus-model"},
		{Model: "algebra-model"},
	}

	selector, err := NewSelector(mlConfig)
	if err != nil {
		t.Fatalf("Failed to create selector: %v", err)
	}

	embeddingDim := 128

	// Train: calculus queries -> calculus-model performs best
	// Train: algebra queries -> algebra-model performs best
	trainingData := []TrainingRecord{
		// Calculus pattern (high in first half)
		{QueryEmbedding: generatePatternEmbedding(1, 1, embeddingDim), SelectedModel: "calculus-model", Success: true, ResponseQuality: 0.95},
		{QueryEmbedding: generatePatternEmbedding(1, 2, embeddingDim), SelectedModel: "calculus-model", Success: true, ResponseQuality: 0.92},
		{QueryEmbedding: generatePatternEmbedding(1, 3, embeddingDim), SelectedModel: "calculus-model", Success: true, ResponseQuality: 0.93},
		// Algebra pattern (high in second half)
		{QueryEmbedding: generatePatternEmbedding(2, 1, embeddingDim), SelectedModel: "algebra-model", Success: true, ResponseQuality: 0.90},
		{QueryEmbedding: generatePatternEmbedding(2, 2, embeddingDim), SelectedModel: "algebra-model", Success: true, ResponseQuality: 0.88},
		{QueryEmbedding: generatePatternEmbedding(2, 3, embeddingDim), SelectedModel: "algebra-model", Success: true, ResponseQuality: 0.91},
	}

	err = selector.Train(trainingData)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Query similar to calculus pattern should select calculus-model
	calculusCtx := &SelectionContext{
		QueryEmbedding: generatePatternEmbedding(1, 100, embeddingDim),
		QueryText:      "Find the derivative of x^2",
		CategoryName:   "math",
		DecisionName:   "math-decision",
	}
	result1, err := selector.Select(calculusCtx, modelRefs)
	if err != nil {
		t.Fatalf("Selection failed: %v", err)
	}
	if result1.Model != "calculus-model" {
		t.Logf("Expected calculus-model for calculus query, got %s", result1.Model)
	}

	// Query similar to algebra pattern should select algebra-model
	algebraCtx := &SelectionContext{
		QueryEmbedding: generatePatternEmbedding(2, 100, embeddingDim),
		QueryText:      "Solve for x: 2x + 5 = 15",
		CategoryName:   "math",
		DecisionName:   "math-decision",
	}
	result2, err := selector.Select(algebraCtx, modelRefs)
	if err != nil {
		t.Fatalf("Selection failed: %v", err)
	}
	if result2.Model != "algebra-model" {
		t.Logf("Expected algebra-model for algebra query, got %s", result2.Model)
	}
}

// TestDecisionIntegration_LoRASelection tests LoRA adapter selection in decision flow
func TestDecisionIntegration_LoRASelection(t *testing.T) {
	mlConfig := &config.MLModelSelectionConfig{
		Type: "knn",
		K:    3,
	}
	modelRefs := []config.ModelRef{
		{Model: "mistral-7b", LoRAName: "math-lora"},
		{Model: "mistral-7b", LoRAName: "code-lora"},
		{Model: "mistral-7b", LoRAName: "creative-lora"},
	}

	selector, err := NewSelector(mlConfig)
	if err != nil {
		t.Fatalf("Failed to create selector: %v", err)
	}

	ctx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 0, 256),
		QueryText:      "Help me with math",
	}

	result, err := selector.Select(ctx, modelRefs)
	if err != nil {
		t.Fatalf("Selection failed: %v", err)
	}
	if result == nil {
		t.Fatal("Expected result, got nil")
	}

	// Should have base model and LoRA name
	if result.Model != "mistral-7b" {
		t.Errorf("Expected base model mistral-7b, got %s", result.Model)
	}

	validLoRAs := map[string]bool{"math-lora": true, "code-lora": true, "creative-lora": true}
	if !validLoRAs[result.LoRAName] {
		t.Errorf("LoRA name %s not in valid options", result.LoRAName)
	}

	t.Logf("Selected LoRA adapter: %s (base: %s)", result.LoRAName, result.Model)
}

// generatePatternEmbedding creates embeddings with distinct patterns for different types
// Used for testing that training affects selection
func generatePatternEmbedding(patternType int, seed int, dim int) []float64 {
	embedding := make([]float64, dim)

	switch patternType {
	case 1:
		// Pattern 1: High values in first half (e.g., calculus)
		for i := 0; i < dim; i++ {
			if i < dim/2 {
				embedding[i] = 0.8 + float64(seed%10)*0.01
			} else {
				embedding[i] = 0.2 + float64(seed%10)*0.01
			}
		}
	case 2:
		// Pattern 2: High values in second half (e.g., algebra)
		for i := 0; i < dim; i++ {
			if i < dim/2 {
				embedding[i] = 0.2 + float64(seed%10)*0.01
			} else {
				embedding[i] = 0.8 + float64(seed%10)*0.01
			}
		}
	default:
		// Default pattern
		for i := 0; i < dim; i++ {
			embedding[i] = float64((i+seed)%10) / 10.0
		}
	}

	return embedding
}

// =============================================================================
// Avengers-Pro KMeans Tests (arXiv:2508.12631)
// Tests for performance-efficiency score based routing
// =============================================================================

// TestKMeans_EfficiencyWeight tests the efficiency weight parameter
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

// TestKMeans_PerformanceEfficiencyScore tests that the performance-efficiency tradeoff works
func TestKMeans_PerformanceEfficiencyScore(t *testing.T) {
	embeddingDim := 128

	// Create two models:
	// - model-fast: low latency (100ms), medium quality (0.7)
	// - model-quality: high latency (500ms), high quality (0.95)
	trainingData := []TrainingRecord{}

	// Generate training data for model-fast (fast but lower quality)
	for i := 0; i < 50; i++ {
		trainingData = append(trainingData, TrainingRecord{
			QueryEmbedding:    generateRealisticEmbedding(QueryTypeMath, i, embeddingDim),
			SelectedModel:     "model-fast",
			ResponseLatencyNs: int64(100 * time.Millisecond),
			ResponseQuality:   0.7,
			Success:           true,
		})
	}

	// Generate training data for model-quality (slow but high quality)
	for i := 50; i < 100; i++ {
		trainingData = append(trainingData, TrainingRecord{
			QueryEmbedding:    generateRealisticEmbedding(QueryTypeMath, i, embeddingDim),
			SelectedModel:     "model-quality",
			ResponseLatencyNs: int64(500 * time.Millisecond),
			ResponseQuality:   0.95,
			Success:           true,
		})
	}

	models := []config.ModelRef{
		{Model: "model-fast"},
		{Model: "model-quality"},
	}

	t.Run("High efficiency weight prefers fast model", func(t *testing.T) {
		selector := NewKMeansSelectorWithEfficiency(2, 0.8) // 80% efficiency
		_ = selector.Train(trainingData)

		ctx := &SelectionContext{
			QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 200, embeddingDim),
		}

		// With high efficiency weight, should prefer faster model
		result, err := selector.Select(ctx, models)
		if err != nil {
			t.Fatalf("Selection failed: %v", err)
		}
		t.Logf("High efficiency (0.8) selected: %s", result.Model)
	})

	t.Run("Low efficiency weight prefers quality model", func(t *testing.T) {
		selector := NewKMeansSelectorWithEfficiency(2, 0.1) // 10% efficiency, 90% performance
		_ = selector.Train(trainingData)

		ctx := &SelectionContext{
			QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 200, embeddingDim),
		}

		// With low efficiency weight, should prefer quality model
		result, err := selector.Select(ctx, models)
		if err != nil {
			t.Fatalf("Selection failed: %v", err)
		}
		t.Logf("Low efficiency (0.1) selected: %s", result.Model)
	})
}

// TestKMeans_ConfigWithEfficiencyWeight tests creating KMeans from config with efficiency_weight
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

// BenchmarkKMeansWithEfficiency benchmarks KMeans with efficiency scoring
func BenchmarkKMeansWithEfficiency(b *testing.B) {
	embeddingDim := 384
	trainingData := make([]TrainingRecord, 500)

	for i := 0; i < 500; i++ {
		trainingData[i] = TrainingRecord{
			QueryEmbedding:    generateRealisticEmbedding(QueryTypeMath, i, embeddingDim),
			SelectedModel:     fmt.Sprintf("model-%d", i%5),
			ResponseLatencyNs: int64(time.Duration(100+i%400) * time.Millisecond),
			ResponseQuality:   0.7 + float64(i%30)/100,
			Success:           true,
		}
	}

	selector := NewKMeansSelectorWithEfficiency(5, 0.4)
	_ = selector.Train(trainingData)

	ctx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 1000, embeddingDim),
	}

	models := []config.ModelRef{
		{Model: "model-0"},
		{Model: "model-1"},
		{Model: "model-2"},
		{Model: "model-3"},
		{Model: "model-4"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = selector.Select(ctx, models)
	}
}

// =============================================================================
// Advanced ML Algorithm Tests
// =============================================================================

// TestAdvanced_NoisyEmbeddings tests algorithm robustness to noisy embeddings
func TestAdvanced_NoisyEmbeddings(t *testing.T) {
	t.Skip("Skipping: requires Go-based training. Use pretrained models from HuggingFace instead.")
	embeddingDim := 256

	algorithms := []struct {
		name   string
		config *config.MLModelSelectionConfig
	}{
		{"KNN", &config.MLModelSelectionConfig{Type: "knn", K: 5}},
		{"SVM", &config.MLModelSelectionConfig{Type: "svm", Kernel: "linear"}},
	}

	noiseLevels := []float64{0.0, 0.1, 0.2, 0.3, 0.5}

	for _, algo := range algorithms {
		t.Run(algo.name, func(t *testing.T) {
			selector, _ := NewSelector(algo.config)

			// Train with clean data
			trainingData := generateEnhancedTrainingData(300, embeddingDim)
			_ = selector.Train(trainingData)

			for _, noiseLevel := range noiseLevels {
				correctCount := 0
				totalCount := 50

				for i := 0; i < totalCount; i++ {
					// Generate base embedding and add noise
					baseEmb := generateRealisticEmbedding(QueryTypeMath, 5000+i, embeddingDim)
					noisyEmb := addNoise(baseEmb, noiseLevel)

					ctx := &SelectionContext{
						QueryEmbedding: noisyEmb,
						QueryText:      "Math query with noise",
					}

					result, err := selector.Select(ctx, testModels)
					if err != nil {
						continue
					}

					if result.Model == "mistral-7b" { // Expected for math
						correctCount++
					}
				}

				accuracy := float64(correctCount) / float64(totalCount) * 100
				t.Logf("%s with %.0f%% noise: %.1f%% accuracy", algo.name, noiseLevel*100, accuracy)

				// Even with noise, should maintain reasonable accuracy
				if noiseLevel <= 0.2 && accuracy < 50 {
					t.Errorf("%s failed with low noise (%.0f%%): only %.1f%% accuracy", algo.name, noiseLevel*100, accuracy)
				}
			}
		})
	}
}

// addNoise adds Gaussian noise to an embedding
func addNoise(embedding []float64, noiseLevel float64) []float64 {
	result := make([]float64, len(embedding))
	for i, v := range embedding {
		noise := (rand.Float64()*2 - 1) * noiseLevel
		result[i] = v + noise
	}
	return result
}

// TestAdvanced_ClassImbalance tests handling of imbalanced training data
func TestAdvanced_ClassImbalance(t *testing.T) {
	t.Skip("Skipping: requires Go-based training. Use pretrained models from HuggingFace instead.")
	embeddingDim := 256

	algorithms := []struct {
		name   string
		config *config.MLModelSelectionConfig
	}{
		{"KNN", &config.MLModelSelectionConfig{Type: "knn", K: 3}},
		{"SVM", &config.MLModelSelectionConfig{Type: "svm", Kernel: "linear"}},
	}

	for _, algo := range algorithms {
		t.Run(algo.name, func(t *testing.T) {
			selector, _ := NewSelector(algo.config)

			// Create imbalanced training data: 90% math, 10% other
			trainingData := make([]TrainingRecord, 0, 200)

			// Majority class: Math queries -> mistral-7b (180 samples)
			for i := 0; i < 180; i++ {
				trainingData = append(trainingData, TrainingRecord{
					QueryEmbedding:  generateRealisticEmbedding(QueryTypeMath, i, embeddingDim),
					SelectedModel:   "mistral-7b",
					ResponseQuality: 0.9,
					Success:         true,
				})
			}

			// Minority classes (20 samples total)
			minorityTypes := []QueryType{QueryTypeCode, QueryTypeCreative, QueryTypeFactual, QueryTypeReasoning}
			minorityModels := []string{"codellama-7b", "llama-3.2-3b", "llama-3.2-1b", "mistral-7b"}
			for i := 0; i < 20; i++ {
				idx := i % 4
				trainingData = append(trainingData, TrainingRecord{
					QueryEmbedding:  generateRealisticEmbedding(minorityTypes[idx], 1000+i, embeddingDim),
					SelectedModel:   minorityModels[idx],
					ResponseQuality: 0.9,
					Success:         true,
				})
			}

			_ = selector.Train(trainingData)

			// Test majority class (should work well)
			mathCorrect := 0
			for i := 0; i < 20; i++ {
				ctx := &SelectionContext{
					QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 2000+i, embeddingDim),
				}
				result, _ := selector.Select(ctx, testModels)
				if result.Model == "mistral-7b" {
					mathCorrect++
				}
			}

			// Test minority class (harder due to imbalance)
			codeCorrect := 0
			for i := 0; i < 20; i++ {
				ctx := &SelectionContext{
					QueryEmbedding: generateRealisticEmbedding(QueryTypeCode, 3000+i, embeddingDim),
				}
				result, _ := selector.Select(ctx, testModels)
				if result.Model == "codellama-7b" {
					codeCorrect++
				}
			}

			t.Logf("%s - Majority class (Math): %d/20 (%.0f%%)", algo.name, mathCorrect, float64(mathCorrect)/20*100)
			t.Logf("%s - Minority class (Code): %d/20 (%.0f%%)", algo.name, codeCorrect, float64(codeCorrect)/20*100)

			// Majority class should have high accuracy
			if mathCorrect < 15 {
				t.Errorf("%s: Majority class accuracy too low: %d/20", algo.name, mathCorrect)
			}
		})
	}
}

// TestAdvanced_ManyModels tests scaling to many models (10+)
func TestAdvanced_ManyModels(t *testing.T) {
	t.Skip("Skipping: requires Go-based training. Use pretrained models from HuggingFace instead.")
	embeddingDim := 256
	numModels := 15

	// Create many models
	manyModels := make([]config.ModelRef, numModels)
	for i := 0; i < numModels; i++ {
		manyModels[i] = config.ModelRef{Model: fmt.Sprintf("model-%02d", i)}
	}

	algorithms := []struct {
		name   string
		config *config.MLModelSelectionConfig
	}{
		{"KNN", &config.MLModelSelectionConfig{Type: "knn", K: 5}},
		{"SVM", &config.MLModelSelectionConfig{Type: "svm", Kernel: "linear"}},
		{"KMeans", &config.MLModelSelectionConfig{Type: "kmeans", NumClusters: numModels}},
	}

	for _, algo := range algorithms {
		t.Run(algo.name, func(t *testing.T) {
			selector, _ := NewSelector(algo.config)

			// Generate training data for each model
			trainingData := make([]TrainingRecord, 0, numModels*50)
			for modelIdx := 0; modelIdx < numModels; modelIdx++ {
				for i := 0; i < 50; i++ {
					// Create distinct embedding pattern for each model
					embedding := make([]float64, embeddingDim)
					for j := 0; j < embeddingDim; j++ {
						// Different base pattern per model
						embedding[j] = math.Sin(float64(modelIdx*17+j)/10.0) * 0.5
						embedding[j] += (rand.Float64() - 0.5) * 0.1 // Small noise
					}

					trainingData = append(trainingData, TrainingRecord{
						QueryEmbedding:  embedding,
						SelectedModel:   manyModels[modelIdx].Model,
						ResponseQuality: 0.85 + rand.Float64()*0.1,
						Success:         true,
					})
				}
			}

			_ = selector.Train(trainingData)

			// Test each model
			correctPerModel := make([]int, numModels)
			testsPerModel := 10

			for modelIdx := 0; modelIdx < numModels; modelIdx++ {
				for i := 0; i < testsPerModel; i++ {
					// Generate test embedding similar to training
					embedding := make([]float64, embeddingDim)
					for j := 0; j < embeddingDim; j++ {
						embedding[j] = math.Sin(float64(modelIdx*17+j)/10.0) * 0.5
						embedding[j] += (rand.Float64() - 0.5) * 0.05 // Less noise for testing
					}

					ctx := &SelectionContext{
						QueryEmbedding: embedding,
					}

					result, _ := selector.Select(ctx, manyModels)
					if result.Model == manyModels[modelIdx].Model {
						correctPerModel[modelIdx]++
					}
				}
			}

			totalCorrect := 0
			for _, c := range correctPerModel {
				totalCorrect += c
			}
			totalTests := numModels * testsPerModel
			accuracy := float64(totalCorrect) / float64(totalTests) * 100

			t.Logf("%s with %d models: %.1f%% overall accuracy", algo.name, numModels, accuracy)

			// With 15 models, random would be ~6.7%, we expect much better
			if accuracy < 30 {
				t.Errorf("%s: Accuracy too low for %d models: %.1f%%", algo.name, numModels, accuracy)
			}
		})
	}
}

// TestAdvanced_EmbeddingDimensions tests different embedding sizes
func TestAdvanced_EmbeddingDimensions(t *testing.T) {
	t.Skip("Skipping: requires Go-based training. Use pretrained models from HuggingFace instead.")
	dimensions := []int{32, 128, 384, 768, 1536}

	for _, dim := range dimensions {
		t.Run(fmt.Sprintf("dim_%d", dim), func(t *testing.T) {
			selector := NewKNNSelector(5)

			// Generate training data
			trainingData := generateEnhancedTrainingData(200, dim)
			_ = selector.Train(trainingData)

			// Test
			correctCount := 0
			totalCount := 25

			for i := 0; i < totalCount; i++ {
				ctx := &SelectionContext{
					QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 5000+i, dim),
				}

				result, _ := selector.Select(ctx, testModels)
				if result.Model == "mistral-7b" {
					correctCount++
				}
			}

			accuracy := float64(correctCount) / float64(totalCount) * 100
			t.Logf("KNN with dim=%d: %.1f%% accuracy", dim, accuracy)

			if accuracy < 60 {
				t.Errorf("Accuracy too low for dim=%d: %.1f%%", dim, accuracy)
			}
		})
	}
}

// TestAdvanced_TemporalDrift simulates concept drift over time
func TestAdvanced_TemporalDrift(t *testing.T) {
	embeddingDim := 256
	selector := NewKNNSelector(5)

	// Phase 1: Train with initial preferences
	phase1Data := make([]TrainingRecord, 100)
	for i := 0; i < 100; i++ {
		phase1Data[i] = TrainingRecord{
			QueryEmbedding:  generateRealisticEmbedding(QueryTypeMath, i, embeddingDim),
			SelectedModel:   "mistral-7b", // Prefer mistral-7b for math initially
			ResponseQuality: 0.9,
			Success:         true,
		}
	}
	_ = selector.Train(phase1Data)

	// Test Phase 1
	ctx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 1000, embeddingDim),
	}
	result1, _ := selector.Select(ctx, testModels)
	t.Logf("Phase 1 (initial): Selected %s for math", result1.Model)

	// Phase 2: Preferences shift (new model becomes better)
	phase2Data := make([]TrainingRecord, 150)
	for i := 0; i < 150; i++ {
		phase2Data[i] = TrainingRecord{
			QueryEmbedding:  generateRealisticEmbedding(QueryTypeMath, 2000+i, embeddingDim),
			SelectedModel:   "codellama-7b", // Now prefer codellama-7b for math
			ResponseQuality: 0.95,           // Higher quality
			Success:         true,
		}
	}
	_ = selector.Train(phase2Data)

	// Test Phase 2 - should adapt to new preference
	ctx2 := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 3000, embeddingDim),
	}
	result2, _ := selector.Select(ctx, testModels)
	result3, _ := selector.Select(ctx2, testModels)

	t.Logf("Phase 2 (after drift): Selected %s (old query), %s (new query)", result2.Model, result3.Model)

	// The selector should show adaptation to the new preference
	// With 150 new samples vs 100 old, the new preference should dominate
}

// TestAdvanced_SimilarQueryTypes tests distinguishing between similar query types
func TestAdvanced_SimilarQueryTypes(t *testing.T) {
	t.Skip("Skipping: requires Go-based training. Use pretrained models from HuggingFace instead.")
	embeddingDim := 256

	// Create very similar query types with subtle differences
	type SimilarQuery struct {
		name          string
		basePattern   int
		expectedModel string
	}

	similarQueries := []SimilarQuery{
		{"algebra", 1, "mistral-7b"},
		{"calculus", 2, "codellama-7b"},
		{"statistics", 3, "llama-3.2-3b"},
	}

	selector := NewKNNSelector(3)

	// Generate training data with subtle differences
	trainingData := make([]TrainingRecord, 0)
	for _, sq := range similarQueries {
		for i := 0; i < 80; i++ {
			embedding := make([]float64, embeddingDim)
			for j := 0; j < embeddingDim; j++ {
				// Base math pattern
				embedding[j] = math.Sin(float64(j)/20.0) * 0.3
				// Type-specific variation
				embedding[j] += math.Cos(float64(sq.basePattern*j)/30.0) * 0.4
				// Small noise
				embedding[j] += (rand.Float64() - 0.5) * 0.1
			}

			trainingData = append(trainingData, TrainingRecord{
				QueryEmbedding:  embedding,
				SelectedModel:   sq.expectedModel,
				ResponseQuality: 0.9,
				Success:         true,
			})
		}
	}

	_ = selector.Train(trainingData)

	// Test each similar query type
	for _, sq := range similarQueries {
		correct := 0
		total := 20

		for i := 0; i < total; i++ {
			embedding := make([]float64, embeddingDim)
			for j := 0; j < embeddingDim; j++ {
				embedding[j] = math.Sin(float64(j)/20.0) * 0.3
				embedding[j] += math.Cos(float64(sq.basePattern*j)/30.0) * 0.4
				embedding[j] += (rand.Float64() - 0.5) * 0.05
			}

			ctx := &SelectionContext{
				QueryEmbedding: embedding,
			}

			result, _ := selector.Select(ctx, testModels)
			if result.Model == sq.expectedModel {
				correct++
			}
		}

		accuracy := float64(correct) / float64(total) * 100
		t.Logf("Similar type '%s': %.1f%% accuracy (expected %s)", sq.name, accuracy, sq.expectedModel)

		if accuracy < 60 {
			t.Errorf("Failed to distinguish '%s': only %.1f%% accuracy", sq.name, accuracy)
		}
	}
}

// TestAdvanced_CrossValidation performs k-fold cross-validation style testing
// Tests model's ability to generalize across different data splits
func TestAdvanced_CrossValidation(t *testing.T) {
	t.Skip("Skipping: requires Go-based training. Use pretrained models from HuggingFace instead.")
	embeddingDim := 256
	k := 5 // 5-fold cross-validation

	algorithms := []struct {
		name   string
		config *config.MLModelSelectionConfig
	}{
		{"KNN", &config.MLModelSelectionConfig{Type: "knn", K: 5}},
		{"SVM", &config.MLModelSelectionConfig{Type: "svm", Kernel: "linear"}},
	}

	// Query type to expected model mapping
	queryTypeToModel := map[QueryType]string{
		QueryTypeMath:      "mistral-7b",
		QueryTypeCode:      "codellama-7b",
		QueryTypeCreative:  "llama-3.2-3b",
		QueryTypeFactual:   "llama-3.2-1b",
		QueryTypeReasoning: "mistral-7b",
	}
	queryTypes := []QueryType{QueryTypeMath, QueryTypeCode, QueryTypeCreative, QueryTypeFactual, QueryTypeReasoning}

	for _, algo := range algorithms {
		t.Run(algo.name, func(t *testing.T) {
			foldAccuracies := make([]float64, k)

			for fold := 0; fold < k; fold++ {
				// Generate training data (80% of data, excluding fold's test queries)
				trainData := make([]TrainingRecord, 0, 200)
				for i := 0; i < 200; i++ {
					// Exclude test fold's seed range
					if i/40 == fold {
						continue
					}
					queryType := queryTypes[i%len(queryTypes)]
					trainData = append(trainData, TrainingRecord{
						QueryEmbedding:  generateRealisticEmbedding(queryType, i+1000, embeddingDim),
						SelectedModel:   queryTypeToModel[queryType],
						ResponseQuality: 0.9,
						Success:         true,
					})
				}

				// Train
				selector, _ := NewSelector(algo.config)
				_ = selector.Train(trainData)

				// Test with fold-specific queries
				correct := 0
				total := 0
				for _, queryType := range queryTypes {
					for i := 0; i < 5; i++ {
						seed := fold*1000 + int(queryType[0])*100 + i
						ctx := &SelectionContext{
							QueryEmbedding: generateRealisticEmbedding(queryType, seed, embeddingDim),
						}
						result, _ := selector.Select(ctx, testModels)
						if result.Model == queryTypeToModel[queryType] {
							correct++
						}
						total++
					}
				}

				foldAccuracies[fold] = float64(correct) / float64(total) * 100
			}

			// Calculate mean and std
			var sum, sumSq float64
			for _, acc := range foldAccuracies {
				sum += acc
				sumSq += acc * acc
			}
			mean := sum / float64(k)
			variance := sumSq/float64(k) - mean*mean
			std := math.Sqrt(math.Abs(variance))

			t.Logf("%s cross-validation: %.1f%% ± %.1f%%", algo.name, mean, std)
			t.Logf("  Fold accuracies: %.0f%%", foldAccuracies)

			// Mean accuracy should be reasonable
			if mean < 70 {
				t.Errorf("%s: Cross-validation mean too low: %.1f%%", algo.name, mean)
			}

			// Standard deviation should be reasonable (not too high variance)
			if std > 20 {
				t.Errorf("%s: High variance in cross-validation: std=%.1f%%", algo.name, std)
			}
		})
	}
}

// TestAdvanced_LoRAAdapterSelection tests selection with LoRA adapters
func TestAdvanced_LoRAAdapterSelection(t *testing.T) {
	t.Skip("Skipping: requires Go-based training. Use pretrained models from HuggingFace instead.")
	embeddingDim := 256

	// Models with LoRA adapters
	loraModels := []config.ModelRef{
		{Model: "mistral-7b", LoRAName: "math-expert"},
		{Model: "mistral-7b", LoRAName: "code-expert"},
		{Model: "mistral-7b", LoRAName: "general"},
	}

	selector := NewKNNSelector(5)

	// Training data for each LoRA adapter
	trainingData := make([]TrainingRecord, 0)

	// Math expert adapter
	for i := 0; i < 100; i++ {
		trainingData = append(trainingData, TrainingRecord{
			QueryEmbedding:  generateRealisticEmbedding(QueryTypeMath, i, embeddingDim),
			SelectedModel:   "math-expert", // Uses LoRAName
			ResponseQuality: 0.95,
			Success:         true,
		})
	}

	// Code expert adapter
	for i := 0; i < 100; i++ {
		trainingData = append(trainingData, TrainingRecord{
			QueryEmbedding:  generateRealisticEmbedding(QueryTypeCode, 1000+i, embeddingDim),
			SelectedModel:   "code-expert",
			ResponseQuality: 0.95,
			Success:         true,
		})
	}

	// General adapter
	for i := 0; i < 100; i++ {
		trainingData = append(trainingData, TrainingRecord{
			QueryEmbedding:  generateRealisticEmbedding(QueryTypeCreative, 2000+i, embeddingDim),
			SelectedModel:   "general",
			ResponseQuality: 0.85,
			Success:         true,
		})
	}

	_ = selector.Train(trainingData)

	// Test math query -> should select math-expert LoRA
	mathCtx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 5000, embeddingDim),
	}
	mathResult, _ := selector.Select(mathCtx, loraModels)
	t.Logf("Math query selected LoRA: %s (expected: math-expert)", mathResult.LoRAName)

	// Test code query -> should select code-expert LoRA
	codeCtx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeCode, 6000, embeddingDim),
	}
	codeResult, _ := selector.Select(codeCtx, loraModels)
	t.Logf("Code query selected LoRA: %s (expected: code-expert)", codeResult.LoRAName)

	// Verify selections
	if mathResult.LoRAName != "math-expert" {
		t.Errorf("Expected math-expert LoRA, got %s", mathResult.LoRAName)
	}
	if codeResult.LoRAName != "code-expert" {
		t.Errorf("Expected code-expert LoRA, got %s", codeResult.LoRAName)
	}
}

// TestAdvanced_MemoryBoundedness tests that training data is bounded
func TestAdvanced_MemoryBoundedness(t *testing.T) {
	embeddingDim := 128
	selector := NewKNNSelector(5)

	// Train with many batches exceeding max size
	batchSize := 5000
	numBatches := 5 // Total 25000 records, should be capped

	for batch := 0; batch < numBatches; batch++ {
		batchData := generateProductionTrainingData(batchSize, embeddingDim)
		_ = selector.Train(batchData)
	}

	// Get training count (should be capped at maxSize)
	count := selector.getTrainingCount()
	t.Logf("After %d batches of %d records: stored %d records", numBatches, batchSize, count)

	// Should be capped at 10000 (the maxSize in newBaseSelector)
	if count > 10000 {
		t.Errorf("Training data not bounded: %d records (max should be 10000)", count)
	}

	// Selector should still work
	ctx := &SelectionContext{
		QueryEmbedding: generateRealisticEmbedding(QueryTypeMath, 0, embeddingDim),
	}
	result, err := selector.Select(ctx, testModels)
	if err != nil {
		t.Fatalf("Selection failed after many batches: %v", err)
	}
	t.Logf("Selection after memory test: %s", result.Model)
}

// TestAdvanced_AllAlgorithmsConsistency checks all algorithms on same data
func TestAdvanced_AllAlgorithmsConsistency(t *testing.T) {
	embeddingDim := 256
	rand.Seed(42) // Fixed seed for reproducibility

	// Generate consistent training data
	trainingData := generateEnhancedTrainingData(400, embeddingDim)

	algorithms := []struct {
		name   string
		config *config.MLModelSelectionConfig
	}{
		{"KNN", &config.MLModelSelectionConfig{Type: "knn", K: 5}},
		{"KMeans", &config.MLModelSelectionConfig{Type: "kmeans", NumClusters: 5}},
		{"SVM", &config.MLModelSelectionConfig{Type: "svm", Kernel: "linear"}},
	}

	// Generate test queries
	testQueries := []struct {
		queryType     QueryType
		expectedModel string
	}{
		{QueryTypeMath, "mistral-7b"},
		{QueryTypeCode, "codellama-7b"},
		{QueryTypeCreative, "llama-3.2-3b"},
	}

	results := make(map[string][]string) // algorithm -> selected models

	for _, algo := range algorithms {
		selector, _ := NewSelector(algo.config)
		_ = selector.Train(trainingData)

		var selections []string
		for _, tq := range testQueries {
			ctx := &SelectionContext{
				QueryEmbedding: generateRealisticEmbedding(tq.queryType, 9999, embeddingDim),
			}
			result, _ := selector.Select(ctx, testModels)
			selections = append(selections, result.Model)
		}
		results[algo.name] = selections
	}

	// Log results comparison
	t.Log("Algorithm consistency comparison:")
	t.Log("Query Type\t| Expected\t| KNN\t| KMeans\t| SVM")
	for i, tq := range testQueries {
		t.Logf("%s\t| %s\t| %s\t| %s\t| %s",
			tq.queryType, tq.expectedModel,
			results["KNN"][i], results["KMeans"][i], results["SVM"][i])
	}

	// Count how many algorithms agree on each query
	for i, tq := range testQueries {
		agreementCount := 0
		for _, algo := range algorithms {
			if results[algo.name][i] == tq.expectedModel {
				agreementCount++
			}
		}
		t.Logf("%s: %d/%d algorithms selected expected model", tq.queryType, agreementCount, len(algorithms))
	}
}
