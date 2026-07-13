//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

package onnx_binding

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// Test constants
const (
	TestText1        = "I love machine learning"
	TestText2        = "I enjoy artificial intelligence"
	TestText3        = "The weather is nice today"
	TestEpsilon      = 1e-6
	TestModelPathEnv = "MMBERT_MODEL_PATH"
)

// getModelPath returns the model path from environment variable or skips the test
func getModelPath(t *testing.T) string {
	path := os.Getenv(TestModelPathEnv)
	if path == "" {
		t.Skipf("Skipping test: %s environment variable not set", TestModelPathEnv)
	}
	return path
}

// TestInitMmBertEmbeddingModel tests the model initialization
func TestInitMmBertEmbeddingModel(t *testing.T) {
	modelPath := getModelPath(t)

	t.Run("InitWithValidPath", func(t *testing.T) {
		err := InitMmBertEmbeddingModel(modelPath, true)
		if err != nil {
			t.Fatalf("Failed to initialize model: %v", err)
		}

		rustState, goState := IsModelInitialized()
		if !rustState || !goState {
			t.Fatal("Model should be initialized")
		}
	})

	t.Run("InitWithEmptyPath", func(t *testing.T) {
		err := InitMmBertEmbeddingModel("", true)
		if err == nil {
			t.Fatal("Expected error for empty path")
		}
	})

	t.Run("SubsequentInitShouldBeIgnored", func(t *testing.T) {
		// First init should succeed
		err := InitMmBertEmbeddingModel(modelPath, true)
		if err != nil {
			t.Fatalf("First init failed: %v", err)
		}

		// Second init should be a no-op
		err = InitMmBertEmbeddingModel(modelPath, true)
		if err != nil {
			t.Fatalf("Second init should not fail: %v", err)
		}

		rustState, goState := IsModelInitialized()
		if !rustState || !goState {
			t.Fatal("Model should still be initialized")
		}
	})
}

// TestGetEmbedding2DMatryoshka tests the 2D Matryoshka embedding generation
func TestGetEmbedding2DMatryoshka(t *testing.T) {
	modelPath := getModelPath(t)
	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	t.Run("FullModelFullDimension", testFullModelFullDimension)
	t.Run("DimensionTruncation", testDimensionTruncation)
	t.Run("LayerEarlyExit", testLayerEarlyExit)
	t.Run("UnsupportedLayerFailsClosed", func(t *testing.T) {
		if _, err := GetEmbedding2DMatryoshka(TestText1, "mmbert", 1, 0); err == nil {
			t.Fatal("unsupported early-exit layer must not fall back to the full model")
		}
	})
	t.Run("2DMatryoshkaCombinations", testMatryoshkaCombinations)
	t.Run("EmbeddingConsistency", testEmbeddingConsistency)
	t.Run("EmbeddingNormalization", testEmbeddingNormalization)
}

func testFullModelFullDimension(t *testing.T) {
	output, err := GetEmbedding2DMatryoshka(TestText1, "mmbert", 0, 0)
	if err != nil {
		t.Fatalf("Failed to generate embedding: %v", err)
	}
	if len(output.Embedding) != 768 {
		t.Errorf("Expected 768 dimensions, got %d", len(output.Embedding))
	}
	if output.ModelType != "mmbert" {
		t.Errorf("Expected model type 'mmbert', got '%s'", output.ModelType)
	}
	if output.ProcessingTimeMs <= 0 {
		t.Errorf("Processing time should be positive, got %f", output.ProcessingTimeMs)
	}
	for i, value := range output.Embedding {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("Invalid embedding value at index %d: %f", i, value)
		}
	}
	t.Logf("Generated 768-dim embedding in %.2fms", output.ProcessingTimeMs)
}

func testDimensionTruncation(t *testing.T) {
	for _, dimension := range []int{512, 256, 128, 64} {
		output, err := GetEmbedding2DMatryoshka(TestText1, "mmbert", 0, dimension)
		if err != nil {
			t.Fatalf("Failed to generate %d-dim embedding: %v", dimension, err)
		}
		if len(output.Embedding) != dimension {
			t.Errorf("Expected %d dimensions, got %d", dimension, len(output.Embedding))
		}
		t.Logf("Generated %d-dim embedding in %.2fms", dimension, output.ProcessingTimeMs)
	}
}

func testLayerEarlyExit(t *testing.T) {
	for _, layer := range availableMatryoshkaLayers(t) {
		output, err := GetEmbedding2DMatryoshka(TestText1, "mmbert", layer, 0)
		if err != nil {
			t.Fatalf("Failed to generate embedding with layer %d: %v", layer, err)
		}
		if len(output.Embedding) == 0 {
			t.Errorf("Embedding should not be empty for layer %d", layer)
		}
		t.Logf("Generated embedding with layer %d in %.2fms", layer, output.ProcessingTimeMs)
	}
}

func testMatryoshkaCombinations(t *testing.T) {
	for _, layer := range availableMatryoshkaLayers(t) {
		for _, dimension := range []int{256, 64} {
			output, err := GetEmbedding2DMatryoshka(TestText1, "mmbert", layer, dimension)
			if err != nil {
				t.Fatalf("Failed with layer=%d, dim=%d: %v", layer, dimension, err)
			}
			if len(output.Embedding) != dimension {
				t.Errorf("Expected %d dimensions for layer=%d, got %d", dimension, layer, len(output.Embedding))
			}
			t.Logf("L%d/D%d: %.2fms", layer, dimension, output.ProcessingTimeMs)
		}
	}
}

func availableMatryoshkaLayers(t *testing.T) []int {
	t.Helper()
	config, err := GetMatryoshkaConfig()
	if err != nil {
		t.Fatalf("get loaded Matryoshka capabilities: %v", err)
	}
	var layers []int
	for _, value := range strings.Split(config.Layers, ",") {
		layer, err := strconv.Atoi(value)
		if err != nil {
			t.Fatalf("invalid Matryoshka layer %q: %v", value, err)
		}
		layers = append(layers, layer)
	}
	if len(layers) == 0 {
		t.Fatal("loaded Matryoshka capability has no layers")
	}
	return layers
}

func testEmbeddingConsistency(t *testing.T) {
	first, err := GetEmbedding2DMatryoshka(TestText1, "mmbert", 0, 0)
	if err != nil {
		t.Fatalf("First embedding failed: %v", err)
	}
	second, err := GetEmbedding2DMatryoshka(TestText1, "mmbert", 0, 0)
	if err != nil {
		t.Fatalf("Second embedding failed: %v", err)
	}
	if len(first.Embedding) != len(second.Embedding) {
		t.Fatalf("Embedding lengths differ: %d vs %d", len(first.Embedding), len(second.Embedding))
	}
	for i := range first.Embedding {
		if math.Abs(float64(first.Embedding[i]-second.Embedding[i])) > TestEpsilon {
			t.Fatalf("Embedding values differ at index %d: %f vs %f", i, first.Embedding[i], second.Embedding[i])
		}
	}
}

func testEmbeddingNormalization(t *testing.T) {
	output, err := GetEmbedding2DMatryoshka(TestText1, "mmbert", 0, 0)
	if err != nil {
		t.Fatalf("Failed to generate embedding: %v", err)
	}
	var sumSquared float64
	for _, value := range output.Embedding {
		sumSquared += float64(value) * float64(value)
	}
	norm := math.Sqrt(sumSquared)
	if math.Abs(norm-1.0) > 0.01 {
		t.Errorf("Embedding should be L2 normalized (norm=1.0), got norm=%.4f", norm)
	}
	t.Logf("Embedding L2 norm: %.6f", norm)
}

// TestGetEmbedding tests the convenience function
func TestGetEmbedding(t *testing.T) {
	modelPath := getModelPath(t)
	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	output, err := GetEmbedding(TestText1, 0)
	if err != nil {
		t.Fatalf("Failed to generate embedding: %v", err)
	}

	if len(output) != 768 {
		t.Errorf("Expected 768 dimensions, got %d", len(output))
	}
}

// TestGetEmbeddingWithDim tests the dimension-only convenience function
func TestGetEmbeddingWithDim(t *testing.T) {
	modelPath := getModelPath(t)
	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	output, err := GetEmbeddingWithDim(TestText1, 0, 0, 256)
	if err != nil {
		t.Fatalf("Failed to generate embedding: %v", err)
	}

	if len(output) != 256 {
		t.Errorf("Expected 256 dimensions, got %d", len(output))
	}
}

// TestGetEmbeddingsBatch tests batch embedding generation
func TestGetEmbeddingsBatch(t *testing.T) {
	modelPath := getModelPath(t)
	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	texts := []string{TestText1, TestText2, TestText3}
	t.Run("BatchEmbedding", func(t *testing.T) { testBatchEmbedding(t, texts) })
	t.Run("BatchWith2DMatryoshka", func(t *testing.T) { testBatchMatryoshka(t, texts) })
	t.Run("EmptyBatch", testEmptyEmbeddingBatch)
}

func testBatchEmbedding(t *testing.T, texts []string) {
	outputs, err := GetEmbeddingsBatch(texts, 0, 0)
	if err != nil {
		t.Fatalf("Failed to generate batch embeddings: %v", err)
	}
	if len(outputs) != len(texts) {
		t.Fatalf("Expected %d outputs, got %d", len(texts), len(outputs))
	}
	var totalTime float32
	for i, output := range outputs {
		if len(output.Embedding) != 768 {
			t.Errorf("Text %d: expected 768 dimensions, got %d", i, len(output.Embedding))
		}
		totalTime += output.ProcessingTimeMs
	}
	t.Logf("Batch of %d texts: total %.2fms", len(texts), totalTime)
}

func testBatchMatryoshka(t *testing.T, texts []string) {
	outputs, err := GetEmbeddingsBatch(texts, 6, 256)
	if err != nil {
		t.Fatalf("Failed to generate batch embeddings: %v", err)
	}
	for i, output := range outputs {
		if len(output.Embedding) != 256 {
			t.Errorf("Text %d: expected 256 dimensions, got %d", i, len(output.Embedding))
		}
	}
}

func testEmptyEmbeddingBatch(t *testing.T) {
	if _, err := GetEmbeddingsBatch(nil, 0, 0); err == nil {
		t.Fatal("Expected error for empty batch")
	}
}

// TestCalculateEmbeddingSimilarity tests similarity calculation
func TestCalculateEmbeddingSimilarity(t *testing.T) {
	modelPath := getModelPath(t)
	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	t.Run("SimilarTexts", func(t *testing.T) {
		result, err := CalculateEmbeddingSimilarity(TestText1, TestText2, "mmbert", 0)
		if err != nil {
			t.Fatalf("Failed to calculate similarity: %v", err)
		}

		if result.Similarity < 0.5 {
			t.Errorf("Similar texts should have high similarity, got %.4f", result.Similarity)
		}

		t.Logf("Similarity between '%s' and '%s': %.4f (%.2fms)",
			TestText1, TestText2, result.Similarity, result.ProcessingTimeMs)
	})

	t.Run("IdenticalTexts", func(t *testing.T) {
		result, err := CalculateEmbeddingSimilarity(TestText1, TestText1, "mmbert", 0)
		if err != nil {
			t.Fatalf("Failed to calculate similarity: %v", err)
		}

		if result.Similarity < 0.99 {
			t.Errorf("Identical texts should have similarity ~1.0, got %.4f", result.Similarity)
		}

		t.Logf("Similarity for identical texts: %.4f", result.Similarity)
	})

	t.Run("DifferentTexts", func(t *testing.T) {
		result, err := CalculateEmbeddingSimilarity(TestText1, TestText3, "mmbert", 0)
		if err != nil {
			t.Fatalf("Failed to calculate similarity: %v", err)
		}

		// Different texts should have lower similarity
		t.Logf("Similarity between '%s' and '%s': %.4f",
			TestText1, TestText3, result.Similarity)
	})

	t.Run("SimilarityWith2DMatryoshka", func(t *testing.T) {
		// Compare similarity calculations at different layer/dim settings
		result1, err := CalculateEmbeddingSimilarity(TestText1, TestText2, "mmbert", 0)
		if err != nil {
			t.Fatalf("Full model failed: %v", err)
		}

		result2, err := CalculateEmbeddingSimilarity(TestText1, TestText2, "mmbert", 256)
		if err != nil {
			t.Fatalf("L6/D256 failed: %v", err)
		}

		t.Logf("Full model similarity: %.4f (%.2fms)", result1.Similarity, result1.ProcessingTimeMs)
		t.Logf("L6/D256 similarity: %.4f (%.2fms)", result2.Similarity, result2.ProcessingTimeMs)

		// Similarities should be correlated (both should agree on similar/different)
		if (result1.Similarity > 0.5) != (result2.Similarity > 0.5) {
			t.Logf("Warning: similarity agreement differs between settings")
		}
	})
}

// TestCalculateSimilarityBatch tests batch similarity calculation
func TestCalculateSimilarityBatch(t *testing.T) {
	modelPath := getModelPath(t)
	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	query := "I enjoy machine learning"
	candidates := []string{
		"Machine learning is fascinating",
		"The weather is sunny today",
		"I love artificial intelligence",
		"Programming is fun",
		"Deep learning is a subset of ML",
	}

	t.Run("TopKMatches", func(t *testing.T) { testTopKSimilarity(t, query, candidates) })
	t.Run("AllMatches", func(t *testing.T) { testAllSimilarities(t, query, candidates) })
	t.Run("BatchWithDimensionTruncation", func(t *testing.T) { testDimensionSimilarity(t, query, candidates) })
	t.Run("EmptyCandidates", func(t *testing.T) { testEmptySimilarityCandidates(t, query) })
}

func testTopKSimilarity(t *testing.T, query string, candidates []string) {
	result, err := CalculateSimilarityBatch(query, candidates, 3, "mmbert", 0)
	if err != nil {
		t.Fatalf("Failed to calculate batch similarity: %v", err)
	}
	if len(result.Matches) != 3 {
		t.Errorf("Expected exactly 3 matches, got %d", len(result.Matches))
	}
	for i := 1; i < len(result.Matches); i++ {
		if result.Matches[i].Similarity > result.Matches[i-1].Similarity {
			t.Fatal("Matches not sorted by similarity")
		}
	}
	logSimilarityMatches(t, result, candidates)
}

func testAllSimilarities(t *testing.T, query string, candidates []string) {
	result, err := CalculateSimilarityBatch(query, candidates, 0, "mmbert", 0)
	if err != nil {
		t.Fatalf("Failed to calculate batch similarity: %v", err)
	}
	if len(result.Matches) != len(candidates) {
		t.Errorf("Expected %d matches, got %d", len(candidates), len(result.Matches))
	}
}

func testDimensionSimilarity(t *testing.T, query string, candidates []string) {
	result, err := CalculateSimilarityBatch(query, candidates, 3, "mmbert", 256)
	if err != nil {
		t.Fatalf("Failed with dimension truncation: %v", err)
	}
	if len(result.Matches) != 3 {
		t.Fatalf("Expected exactly 3 dimension-truncated matches, got %d", len(result.Matches))
	}
	logSimilarityMatches(t, result, candidates)
}

func testEmptySimilarityCandidates(t *testing.T, query string) {
	if _, err := CalculateSimilarityBatch(query, nil, 3, "mmbert", 0); err == nil {
		t.Fatal("Expected error for empty candidates")
	}
}

func logSimilarityMatches(t *testing.T, result *BatchSimilarityOutput, candidates []string) {
	t.Helper()
	t.Logf("Top %d matches (%.2fms):", len(result.Matches), result.ProcessingTimeMs)
	for _, match := range result.Matches {
		t.Logf("  [%d] %.4f: %s", match.Index, match.Similarity, candidates[match.Index])
	}
}

// TestGetEmbeddingModelsInfo tests the model info retrieval
func TestGetEmbeddingModelsInfo(t *testing.T) {
	modelPath := getModelPath(t)
	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	info, err := GetEmbeddingModelsInfo()
	if err != nil {
		t.Fatalf("Failed to get model info: %v", err)
	}

	if len(info.Models) == 0 {
		t.Fatal("Expected at least one model")
	}

	model := info.Models[0]
	if model.ModelName != "mmbert" {
		t.Errorf("Expected model name 'mmbert', got '%s'", model.ModelName)
	}

	if !model.IsLoaded {
		t.Error("Model should be loaded")
	}

	t.Logf("Model: %s", model.ModelName)
	t.Logf("  Loaded: %v", model.IsLoaded)
	t.Logf("  Max Sequence Length: %d", model.MaxSequenceLength)
	t.Logf("  Default Dimension: %d", model.DefaultDimension)
	t.Logf("  Supports Layer Exit: %v", model.SupportsLayerExit)
	t.Logf("  Available Layers: %s", model.AvailableLayers)
	t.Logf("  Path: %s", model.ModelPath)
}

// TestGetMatryoshkaConfig tests the Matryoshka config retrieval
func TestGetMatryoshkaConfig(t *testing.T) {
	modelPath := getModelPath(t)
	if err := InitMmBertEmbeddingModel(modelPath, true); err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}
	config, err := GetMatryoshkaConfig()
	if err != nil {
		t.Fatalf("Failed to get Matryoshka config: %v", err)
	}

	if config.Dimensions == "" {
		t.Error("Dimensions should not be empty")
	}

	if config.Layers == "" {
		t.Error("Layers should not be empty")
	}

	models, err := GetEmbeddingModelsInfo()
	if err != nil || len(models.Models) == 0 {
		t.Fatalf("Failed to get loaded model capabilities: %v", err)
	}
	loaded := models.Models[0]
	if config.Layers != loaded.AvailableLayers {
		t.Errorf("Matryoshka layers %q do not match loaded capability %q", config.Layers, loaded.AvailableLayers)
	}
	if config.Supports2D != loaded.SupportsLayerExit {
		t.Errorf("Matryoshka support %v does not match loaded capability %v", config.Supports2D, loaded.SupportsLayerExit)
	}

	t.Logf("Matryoshka Config:")
	t.Logf("  Dimensions: %s", config.Dimensions)
	t.Logf("  Layers: %s", config.Layers)
	t.Logf("  Supports 2D: %v", config.Supports2D)
}

// TestConcurrentEmbeddings tests thread safety
func TestConcurrentEmbeddings(t *testing.T) {
	modelPath := getModelPath(t)
	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	texts := []string{
		"First test sentence",
		"Second test sentence",
		"Third test sentence",
		"Fourth test sentence",
		"Fifth test sentence",
	}

	const numGoroutines = 5
	const iterationsPerGoroutine = 3

	var wg sync.WaitGroup
	errCh := make(chan error, numGoroutines*iterationsPerGoroutine)

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < iterationsPerGoroutine; i++ {
				text := texts[(id+i)%len(texts)]
				output, err := GetEmbedding2DMatryoshka(text, "mmbert", 0, 0)
				if err != nil {
					errCh <- fmt.Errorf("goroutine %d, iter %d: %v", id, i, err)
					return
				}
				if len(output.Embedding) != 768 {
					errCh <- fmt.Errorf("goroutine %d, iter %d: wrong dim %d", id, i, len(output.Embedding))
					return
				}
			}
		}(g)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("Concurrent error: %v", err)
	}

	t.Logf("Completed %d concurrent embedding generations", numGoroutines*iterationsPerGoroutine)
}

// Benchmark tests
func BenchmarkGetEmbedding2DMatryoshka(b *testing.B) {
	modelPath := os.Getenv(TestModelPathEnv)
	if modelPath == "" {
		b.Skipf("Skipping benchmark: %s not set", TestModelPathEnv)
	}

	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		b.Fatalf("Failed to initialize model: %v", err)
	}

	b.Run("FullModel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = GetEmbedding2DMatryoshka(TestText1, "mmbert", 0, 0)
		}
	})

	b.Run("Layer6_Dim256", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = GetEmbedding2DMatryoshka(TestText1, "mmbert", 6, 256)
		}
	})

	b.Run("Layer11_Dim64", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = GetEmbedding2DMatryoshka(TestText1, "mmbert", 11, 64)
		}
	})
}

func BenchmarkBatchEmbeddings(b *testing.B) {
	modelPath := os.Getenv(TestModelPathEnv)
	if modelPath == "" {
		b.Skipf("Skipping benchmark: %s not set", TestModelPathEnv)
	}

	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		b.Fatalf("Failed to initialize model: %v", err)
	}

	texts := []string{
		"First test sentence",
		"Second test sentence",
		"Third test sentence",
		"Fourth test sentence",
		"Fifth test sentence",
	}

	b.Run("Batch5", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = GetEmbeddingsBatch(texts, 0, 0)
		}
	})
}

// ============================================================================
// Tests for candle_binding compatible API
// ============================================================================

// TestInitEmbeddingModels tests the candle_binding compatible initialization
func TestInitEmbeddingModels(t *testing.T) {
	modelPath := getModelPath(t)

	t.Run("InitWithMmBertOnly", func(t *testing.T) {
		// Only mmbert path provided (qwen3 and gemma empty)
		err := InitEmbeddingModels("", "", modelPath, true)
		if err != nil {
			t.Fatalf("Failed to initialize with mmbert only: %v", err)
		}

		rustState, goState := IsModelInitialized()
		if !rustState || !goState {
			t.Errorf("Model should be initialized: rust=%v, go=%v", rustState, goState)
		}
	})

	t.Run("InitWithMissingMmBert", func(t *testing.T) {
		err := InitEmbeddingModels("", "", "", true)
		if err == nil {
			t.Fatal("Expected error when mmbert path is empty")
		}
	})
}

func TestInitEmbeddingModelsRejectsUnsupportedPartialUnion(t *testing.T) {
	if err := InitEmbeddingModels("models/qwen3", "", "models/mmbert", true); err == nil || !strings.Contains(err.Error(), "supports only mmBERT") {
		t.Fatalf("qwen3 + mmBERT partial union error = %v", err)
	}
	if err := InitEmbeddingModels("", "models/gemma", "models/mmbert", true); err == nil || !strings.Contains(err.Error(), "supports only mmBERT") {
		t.Fatalf("gemma + mmBERT partial union error = %v", err)
	}
}

func TestInitEmbeddingModelsBatchedRejectsUnsupportedQwen3(t *testing.T) {
	if err := InitEmbeddingModelsBatched("models/qwen3", 64, 10, true); err == nil || !strings.Contains(err.Error(), "does not support") {
		t.Fatalf("batched qwen3 error = %v", err)
	}
}

// TestGetEmbeddingBatched tests the batched embedding API
func TestGetEmbeddingBatched(t *testing.T) {
	modelPath := getModelPath(t)
	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	t.Run("BatchedEmbedding", func(t *testing.T) {
		output, err := GetEmbeddingBatched(TestText1, "mmbert", 0)
		if err != nil {
			t.Fatalf("GetEmbeddingBatched failed: %v", err)
		}

		if len(output.Embedding) != 768 {
			t.Errorf("Expected 768 dims, got %d", len(output.Embedding))
		}

		if output.ModelType != "mmbert" {
			t.Errorf("Expected model type 'mmbert', got '%s'", output.ModelType)
		}
	})
}

// TestGetEmbeddingWithModelType tests model type selection
func TestGetEmbeddingWithModelType(t *testing.T) {
	modelPath := getModelPath(t)
	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	output, err := GetEmbeddingWithModelType(TestText1, "mmbert", 0)
	if err != nil {
		t.Fatalf("GetEmbeddingWithModelType(mmbert) failed: %v", err)
	}
	if len(output.Embedding) != 768 {
		t.Errorf("Expected 768 dims, got %d", len(output.Embedding))
	}

	for _, modelType := range []string{"qwen3", "gemma", "unknown"} {
		if _, err := GetEmbeddingWithModelType(TestText1, modelType, 0); err == nil {
			t.Errorf("unsupported model type %q must fail closed", modelType)
		}
	}
}

// TestFindMostSimilar tests the legacy similarity API
func TestFindMostSimilar(t *testing.T) {
	modelPath := getModelPath(t)
	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	query := "machine learning algorithms"
	candidates := []string{
		"artificial intelligence methods",
		"cooking recipes",
		"deep neural networks",
		"gardening tips",
	}

	t.Run("FindMostSimilarIndex", func(t *testing.T) {
		result := FindMostSimilar(query, candidates, 0)
		if result.Index < 0 {
			t.Fatal("Expected valid index")
		}
		if result.Score <= 0 {
			t.Errorf("Expected positive score, got %f", result.Score)
		}
		t.Logf("Most similar: [%d] '%s' with score %.4f", result.Index, candidates[result.Index], result.Score)
	})

	t.Run("FindMostSimilarDefault", func(t *testing.T) {
		result := FindMostSimilarDefault(query, candidates)
		if result.Index < 0 {
			t.Fatal("Expected valid index")
		}
		t.Logf("Most similar (default): [%d] score %.4f", result.Index, result.Score)
	})
}

// TestCalculateSimilarity tests the legacy similarity functions
func TestCalculateSimilarity(t *testing.T) {
	modelPath := getModelPath(t)
	err := InitMmBertEmbeddingModel(modelPath, true)
	if err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	t.Run("CalculateSimilarity", func(t *testing.T) {
		sim := CalculateSimilarity(TestText1, TestText2, 0)
		if sim <= 0 || sim > 1.0 {
			t.Errorf("Expected similarity in (0, 1], got %f", sim)
		}
		t.Logf("Similarity: %.4f", sim)
	})

	t.Run("CalculateSimilarityDefault", func(t *testing.T) {
		sim := CalculateSimilarityDefault(TestText1, TestText2)
		if sim <= 0 || sim > 1.0 {
			t.Errorf("Expected similarity in (0, 1], got %f", sim)
		}
	})
}

// ============================================================================
// Tests for Classification API (stubs return errors, but should not panic)
// ============================================================================

// TestClassificationStubs tests that classification stubs work without panic
func TestClassificationStubs(t *testing.T) {
	// These are stub tests - the classifiers are not initialized,
	// so they should return errors, but not panic

	t.Run("ClassifyMmBert32KIntent", func(t *testing.T) {
		result, err := ClassifyMmBert32KIntent("test text")
		// Expected to fail since classifier not initialized
		if err == nil {
			t.Logf("Unexpected success: class=%d, confidence=%.2f", result.Class, result.Confidence)
		}
	})

	t.Run("ClassifyMmBert32KJailbreak", func(t *testing.T) {
		result, err := ClassifyMmBert32KJailbreak("test text")
		if err == nil {
			t.Logf("Unexpected success: class=%d, confidence=%.2f", result.Class, result.Confidence)
		}
	})

	t.Run("ClassifyMmBert32KFeedback", func(t *testing.T) {
		result, err := ClassifyMmBert32KFeedback("test text")
		if err == nil {
			t.Logf("Unexpected success: class=%d, confidence=%.2f", result.Class, result.Confidence)
		}
	})

	t.Run("ClassifyMmBert32KFactcheck", func(t *testing.T) {
		result, err := ClassifyMmBert32KFactcheck("test text")
		if err == nil {
			t.Logf("Unexpected success: class=%d, confidence=%.2f", result.Class, result.Confidence)
		}
	})

	t.Run("ClassifyMmBert32KPII", func(t *testing.T) {
		entities, err := ClassifyMmBert32KPII("My SSN is 123-45-6789")
		if err == nil {
			t.Logf("Unexpected success: %d entities", len(entities))
		}
	})
}

// TestIsClassifierLoaded tests classifier status check
func TestIsClassifierLoaded(t *testing.T) {
	classifiers := []string{"intent", "jailbreak", "feedback", "factcheck", "pii"}

	for _, name := range classifiers {
		t.Run(name, func(t *testing.T) {
			// Should return false since classifiers are not initialized
			loaded := IsClassifierLoaded(name)
			if loaded {
				t.Errorf("Classifier %s should not be loaded", name)
			}
		})
	}
}

// ============================================================================
// Tests for NLI and Hallucination stubs
// ============================================================================

// TestNLIStubs tests NLI-related stub functions
func TestNLIStubs(t *testing.T) {
	t.Run("NLIConstants", func(t *testing.T) {
		// Verify NLI constants are defined correctly
		if NLIEntailment != 0 {
			t.Errorf("NLIEntailment should be 0, got %d", NLIEntailment)
		}
		if NLINeutral != 1 {
			t.Errorf("NLINeutral should be 1, got %d", NLINeutral)
		}
		if NLIContradiction != 2 {
			t.Errorf("NLIContradiction should be 2, got %d", NLIContradiction)
		}
		if NLIError != -1 {
			t.Errorf("NLIError should be -1, got %d", NLIError)
		}
	})

	t.Run("InitHallucinationModel", func(t *testing.T) {
		err := InitHallucinationModel("/fake/path", true)
		if err == nil {
			t.Fatal("Expected error for unimplemented function")
		}
	})

	t.Run("InitNLIModel", func(t *testing.T) {
		err := InitNLIModel("/fake/path", true)
		if err == nil {
			t.Fatal("Expected error for unimplemented function")
		}
	})

	t.Run("DetectHallucinations", func(t *testing.T) {
		result, err := DetectHallucinations("context", "question", "answer", 0.5)
		if err == nil {
			t.Fatalf("Expected error, got result: %v", result)
		}
	})

	t.Run("DetectHallucinationsWithNLI", func(t *testing.T) {
		result, err := DetectHallucinationsWithNLI("context", "question", "answer", 0.5)
		if err == nil {
			t.Fatalf("Expected error, got result: %v", result)
		}
	})

	t.Run("ClassifyNLI", func(t *testing.T) {
		result, err := ClassifyNLI("premise", "hypothesis")
		if err == nil {
			t.Fatalf("Expected error, got result: %v", result)
		}
	})
}

// ============================================================================
// Tests for legacy BERT classifier stubs
// ============================================================================

// TestLegacyClassifierStubs tests candle_binding compatible classifier stubs
func TestLegacyClassifierStubs(t *testing.T) {
	t.Run("InitCandleBertClassifier", func(t *testing.T) {
		// Should return false since model doesn't exist
		success := InitCandleBertClassifier("/fake/path", 2, true)
		if success {
			t.Error("Expected failure for non-existent model")
		}
	})

	t.Run("InitCandleBertTokenClassifier", func(t *testing.T) {
		success := InitCandleBertTokenClassifier("/fake/path", 10, true)
		if success {
			t.Error("Expected failure for non-existent model")
		}
	})

	t.Run("InitFactCheckClassifier", func(t *testing.T) {
		err := InitFactCheckClassifier("/fake/path", true)
		// Expected to fail since path doesn't exist
		if err == nil {
			t.Log("Unexpected success for non-existent model")
		}
	})

	t.Run("InitFeedbackDetector", func(t *testing.T) {
		err := InitFeedbackDetector("/fake/path", true)
		if err == nil {
			t.Log("Unexpected success for non-existent model")
		}
	})
}

// ============================================================================
// Tests for type definitions
// ============================================================================

// TestTypeDefinitions tests that all types are properly defined
func TestTypeDefinitions(t *testing.T) {
	t.Run("ClassResult", func(t *testing.T) {
		result := ClassResult{
			Class:      1,
			Confidence: 0.95,
			Categories: []string{"cat1", "cat2"},
		}
		if result.Class != 1 {
			t.Errorf("Expected Class=1, got %d", result.Class)
		}
		if result.Confidence != 0.95 {
			t.Errorf("Expected Confidence=0.95, got %f", result.Confidence)
		}
		if len(result.Categories) != 2 {
			t.Errorf("Expected 2 categories, got %d", len(result.Categories))
		}
	})

	t.Run("ClassResultWithProbs", testClassResultWithProbabilitiesType)
	t.Run("BatchSimilarityMatchCompatibility", testBatchSimilarityMatchType)
	t.Run("NLILabelStringCompatibility", testNLILabelStringType)

	t.Run("TokenEntity", func(t *testing.T) {
		entity := TokenEntity{
			Text:       "John",
			EntityType: "PERSON",
			Start:      0,
			End:        4,
			Confidence: 0.99,
		}
		if entity.Text != "John" {
			t.Errorf("Expected Text='John', got '%s'", entity.Text)
		}
	})

	t.Run("TokenClassificationResult", func(t *testing.T) {
		result := TokenClassificationResult{
			Entities: []TokenEntity{
				{Text: "test", EntityType: "ORG", Start: 0, End: 4, Confidence: 0.9},
			},
		}
		if len(result.Entities) != 1 {
			t.Errorf("Expected 1 entity, got %d", len(result.Entities))
		}
	})

	t.Run("EmbeddingOutput", func(t *testing.T) {
		output := EmbeddingOutput{
			Embedding:        make([]float32, 768),
			ModelType:        "mmbert",
			SequenceLength:   128,
			ProcessingTimeMs: 10.5,
		}
		if output.ModelType != "mmbert" {
			t.Errorf("Expected ModelType='mmbert', got '%s'", output.ModelType)
		}
	})

	t.Run("SimilarityOutput", func(t *testing.T) {
		output := SimilarityOutput{
			Similarity:       0.85,
			ModelType:        "mmbert",
			ProcessingTimeMs: 5.0,
		}
		if output.Similarity != 0.85 {
			t.Errorf("Expected Similarity=0.85, got %f", output.Similarity)
		}
	})

	t.Run("BatchSimilarityOutput", func(t *testing.T) {
		output := BatchSimilarityOutput{
			Matches: []SimilarityMatchResult{
				{Index: 0, Similarity: 0.9},
				{Index: 2, Similarity: 0.8},
			},
			ModelType:        "mmbert",
			ProcessingTimeMs: 15.0,
		}
		if len(output.Matches) != 2 {
			t.Errorf("Expected 2 matches, got %d", len(output.Matches))
		}
	})

	t.Run("NLIResult", testNLIResultType)
}

func testClassResultWithProbabilitiesType(t *testing.T) {
	result := ClassResultWithProbs{
		Class:         0,
		Confidence:    0.8,
		Probabilities: []float32{0.8, 0.15, 0.05},
		NumClasses:    3,
	}
	if len(result.Probabilities) != result.NumClasses {
		t.Errorf("Expected %d probabilities, got %d", result.NumClasses, len(result.Probabilities))
	}
}

func testBatchSimilarityMatchType(t *testing.T) {
	result := BatchSimilarityOutput{
		Matches: []BatchSimilarityMatch{{Index: 2, Similarity: 0.75}},
	}
	if result.Matches[0].Index != 2 {
		t.Fatalf("Expected Candle-compatible batch match, got %+v", result.Matches[0])
	}
}

func testNLILabelStringType(t *testing.T) {
	labels := map[NLILabel]string{
		NLIEntailment:    "ENTAILMENT",
		NLINeutral:       "NEUTRAL",
		NLIContradiction: "CONTRADICTION",
		NLIError:         "ERROR",
		NLILabel(99):     "ERROR",
	}
	for label, expected := range labels {
		if actual := label.String(); actual != expected {
			t.Errorf("NLILabel(%d).String() = %q, want %q", label, actual, expected)
		}
	}
}

func TestClassResultWithProbabilitiesOwnsNativeCopy(t *testing.T) {
	native := []float32{0.1, 0.7, 0.2}
	result := ownedClassResultWithProbabilities(1, 0.7, native)
	native[1] = 0

	if result.NumClasses != 3 || len(result.Probabilities) != 3 {
		t.Fatalf("probability count mismatch: %+v", result)
	}
	if result.Probabilities[1] != 0.7 {
		t.Fatalf("probabilities retained native ownership: %+v", result.Probabilities)
	}
}

func testNLIResultType(t *testing.T) {
	result := NLIResult{
		Label:             NLIEntailment,
		LabelStr:          "entailment",
		Confidence:        0.95,
		EntailmentProb:    0.95,
		NeutralProb:       0.03,
		ContradictionProb: 0.02,
		ContradictProb:    0.02,
	}
	if result.Label != NLIEntailment {
		t.Errorf("Expected Label=NLIEntailment, got %d", result.Label)
	}
}

// TestMultiModalEmbeddingOutput tests the MultiModalEmbeddingOutput struct.
func TestMultiModalEmbeddingOutput(t *testing.T) {
	output := MultiModalEmbeddingOutput{
		Embedding:        []float32{0.1, 0.2, 0.3},
		Modality:         "text",
		ProcessingTimeMs: 5.0,
	}
	if output.Modality != "text" {
		t.Errorf("Expected Modality='text', got '%s'", output.Modality)
	}
	if len(output.Embedding) != 3 {
		t.Errorf("Expected 3 dims, got %d", len(output.Embedding))
	}
}

// TestMultiModalEncodeTextNotInitialized verifies error when model is not loaded.
func TestMultiModalEncodeTextNotInitialized(t *testing.T) {
	_, err := MultiModalEncodeText("hello", 0)
	if err == nil {
		t.Error("Expected error when model not initialized, got nil")
	}
}

// TestMultiModalEncodeImageEmptyInput tests empty pixel data.
func TestMultiModalEncodeImageEmptyInput(t *testing.T) {
	_, err := MultiModalEncodeImage(nil, 0, 0, 0)
	if err == nil {
		t.Error("Expected error for empty pixelData, got nil")
	}
}

// TestMultiModalEncodeImageWrongSize tests mismatched pixel dimensions.
func TestMultiModalEncodeImageWrongSize(t *testing.T) {
	_, err := MultiModalEncodeImage([]float32{1.0, 2.0}, 512, 512, 0)
	if err == nil {
		t.Error("Expected error for wrong pixelData size, got nil")
	}
}

// TestMultiModalEncodeAudioEmptyInput tests empty mel data.
func TestMultiModalEncodeAudioEmptyInput(t *testing.T) {
	_, err := MultiModalEncodeAudio(nil, 0, 0, 0)
	if err == nil {
		t.Error("Expected error for empty melData, got nil")
	}
}

// TestMultiModalEncodeImageFromURLEmpty tests empty URL.
func TestMultiModalEncodeImageFromURLEmpty(t *testing.T) {
	_, err := MultiModalEncodeImageFromURL("", 0)
	if err == nil {
		t.Error("Expected error for empty URL, got nil")
	}
}

// TestMultiModalEncodeImageFromURLHttpBlocked tests that http:// is rejected.
func TestMultiModalEncodeImageFromURLHttpBlocked(t *testing.T) {
	_, err := MultiModalEncodeImageFromURL("http://example.com/image.png", 0)
	if err == nil {
		t.Error("Expected error for http URL, got nil")
	}
}

// TestMultiModalEncodeImageFromBase64Empty tests empty base64 input.
func TestMultiModalEncodeImageFromBase64Empty(t *testing.T) {
	_, err := MultiModalEncodeImageFromBase64("", 0)
	if err == nil {
		t.Error("Expected error for empty base64, got nil")
	}
}

// TestMultiModalEncodeImageFromBase64Invalid tests invalid base64.
func TestMultiModalEncodeImageFromBase64Invalid(t *testing.T) {
	_, err := MultiModalEncodeImageFromBase64("not-valid-base64!!!", 0)
	if err == nil {
		t.Error("Expected error for invalid base64, got nil")
	}
}

// TestMultiModalEncodeImageFromBytesEmpty tests empty byte input.
func TestMultiModalEncodeImageFromBytesEmpty(t *testing.T) {
	_, err := MultiModalEncodeImageFromBytes(nil, 0)
	if err == nil {
		t.Error("Expected error for empty bytes, got nil")
	}
}

// TestModalityToString tests the modalityToString helper.
func TestModalityToString(t *testing.T) {
	tests := []struct {
		input    int
		expected string
	}{
		{0, "text"},
		{1, "image"},
		{2, "audio"},
		{99, "unknown"},
	}
	for _, tt := range tests {
		got := modalityToString(tt.input)
		if got != tt.expected {
			t.Errorf("modalityToString(%d) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}
