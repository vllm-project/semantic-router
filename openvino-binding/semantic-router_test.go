//go:build !windows && cgo
// +build !windows,cgo

package openvino_binding

import (
	"math"
	"sync"
	"testing"
	"time"
)

// Test constants
const (
	DefaultEmbeddingModelPath   = "test_models/all-MiniLM-L6-v2/openvino_model.xml"
	CategoryClassifierModelPath = "test_models/category_classifier_modernbert/openvino_model.xml"
	TestMaxLength               = 512
	TestText1                   = "I love machine learning"
	TestText2                   = "I enjoy artificial intelligence"
	TestText3                   = "The weather is nice today"
	TestEpsilon                 = 1e-6
)

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

func TestInitEmbeddingModel(t *testing.T) {
	t.Run("InitWithValidPath", func(t *testing.T) {
		err := InitEmbeddingModel(DefaultEmbeddingModelPath, "CPU")
		if err != nil {
			t.Skipf("Skipping: model not available: %v", err)
		}

		if !IsEmbeddingModelInitialized() {
			t.Error("Model should be initialized")
		}
	})

	t.Run("InitWithEmptyPath", func(t *testing.T) {
		err := InitEmbeddingModel("", "CPU")
		if err == nil {
			t.Log("Empty path accepted (model may already be initialized)")
		} else {
			t.Logf("Got expected error: %v", err)
		}
	})

	t.Run("InitWithInvalidPath", func(t *testing.T) {
		err := InitEmbeddingModel("/nonexistent/model.xml", "CPU")
		if err == nil {
			t.Log("Invalid path accepted (model may already be initialized)")
		} else {
			t.Logf("Got expected error: %v", err)
		}
	})
}

func TestInitClassifier(t *testing.T) {
	t.Run("InitWithValidPath", func(t *testing.T) {
		err := InitModernBertClassifier(CategoryClassifierModelPath, 14, "CPU")
		if err != nil {
			t.Skipf("Skipping: classifier model not available: %v", err)
		}

		if !IsModernBertClassifierInitialized() {
			t.Error("Classifier should be initialized")
		}
	})

	t.Run("InitWithEmptyPath", func(t *testing.T) {
		err := InitModernBertClassifier("", 14, "CPU")
		if err == nil {
			t.Log("Empty path accepted (classifier may already be initialized)")
		} else {
			t.Logf("Got expected error: %v", err)
		}
	})

	t.Run("InitWithInvalidNumClasses", func(t *testing.T) {
		err := InitClassifier(CategoryClassifierModelPath, 1, "CPU")
		if err == nil {
			t.Error("Expected error for numClasses < 2")
		}
	})
}

func TestGetVersion(t *testing.T) {
	version := GetVersion()
	if version == "" {
		t.Error("Expected non-empty version string")
	}
	t.Logf("OpenVINO version: %s", version)
}

func TestGetAvailableDevices(t *testing.T) {
	devices := GetAvailableDevices()
	if len(devices) == 0 {
		t.Skip("No devices available")
	}
	t.Logf("Available devices: %v", devices)
}

// ============================================================================
// EMBEDDING TESTS
// ============================================================================

func TestEmbeddings(t *testing.T) {
	err := InitEmbeddingModel(DefaultEmbeddingModelPath, "CPU")
	if err != nil {
		t.Skipf("Skipping embedding tests: %v", err)
	}

	t.Run("GetEmbedding", func(t *testing.T) {
		embedding, err := GetEmbedding(TestText1, TestMaxLength)
		if err != nil {
			t.Fatalf("Failed to get embedding: %v", err)
		}

		if len(embedding) == 0 {
			t.Fatal("Embedding should not be empty")
		}

		// Check for valid values
		for i, val := range embedding {
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				t.Fatalf("Invalid embedding value at index %d: %f", i, val)
			}
		}

		t.Logf("Generated embedding of length %d", len(embedding))
	})

	t.Run("GetEmbeddingDefault", func(t *testing.T) {
		embedding, err := GetEmbeddingDefault(TestText1)
		if err != nil {
			t.Fatalf("Failed to get embedding with default: %v", err)
		}

		if len(embedding) == 0 {
			t.Fatal("Embedding should not be empty")
		}
	})

	t.Run("EmbeddingConsistency", func(t *testing.T) {
		embedding1, err := GetEmbedding(TestText1, TestMaxLength)
		if err != nil {
			t.Fatalf("Failed to get first embedding: %v", err)
		}

		embedding2, err := GetEmbedding(TestText1, TestMaxLength)
		if err != nil {
			t.Fatalf("Failed to get second embedding: %v", err)
		}

		if len(embedding1) != len(embedding2) {
			t.Fatalf("Embedding lengths differ: %d vs %d", len(embedding1), len(embedding2))
		}

		// Check identical values (deterministic)
		maxDiff := float32(0)
		for i := range embedding1 {
			diff := float32(math.Abs(float64(embedding1[i] - embedding2[i])))
			if diff > maxDiff {
				maxDiff = diff
			}
		}

		if maxDiff > 1e-6 {
			t.Errorf("Embeddings differ (max: %.9f) - should be deterministic", maxDiff)
		}

		t.Logf("✓ Embeddings identical (diff: %.9f)", maxDiff)
	})

	t.Run("EmbeddingDimensionsConsistent", func(t *testing.T) {
		texts := []string{TestText1, TestText2, TestText3, "short", "a very long text with many words"}

		var firstLen int
		for i, text := range texts {
			embedding, err := GetEmbedding(text, TestMaxLength)
			if err != nil {
				t.Fatalf("Failed to get embedding for text %d: %v", i, err)
			}

			if i == 0 {
				firstLen = len(embedding)
			} else if len(embedding) != firstLen {
				t.Errorf("Inconsistent dimensions: text %d has %d, expected %d", i, len(embedding), firstLen)
			}
		}

		t.Logf("✓ All embeddings have consistent dimension: %d", firstLen)
	})

	t.Run("EmptyStringEmbedding", func(t *testing.T) {
		embedding, err := GetEmbedding("", TestMaxLength)
		if err != nil {
			t.Errorf("Empty string embedding should not fail: %v", err)
		}
		if len(embedding) == 0 {
			t.Error("Empty string should still produce embedding")
		}
	})
}

// ============================================================================
// SIMILARITY TESTS
// ============================================================================

func TestSimilarity(t *testing.T) {
	err := InitEmbeddingModel(DefaultEmbeddingModelPath, "CPU")
	if err != nil {
		t.Skipf("Skipping similarity tests: %v", err)
	}

	t.Run("CalculateSimilarity", func(t *testing.T) {
		score := CalculateSimilarity(TestText1, TestText2, TestMaxLength)
		if score < 0 {
			t.Fatalf("Similarity calculation failed, got negative score: %f", score)
		}

		if score > 1.0 {
			t.Errorf("Similarity score should be <= 1.0, got %f", score)
		}

		t.Logf("Similarity between '%s' and '%s': %f", TestText1, TestText2, score)
	})

	t.Run("CalculateSimilarityDefault", func(t *testing.T) {
		score := CalculateSimilarityDefault(TestText1, TestText2)
		if score < 0 {
			t.Fatalf("Similarity calculation failed: %f", score)
		}
	})

	t.Run("IdenticalTextSimilarity", func(t *testing.T) {
		score := CalculateSimilarity(TestText1, TestText1, TestMaxLength)
		if score < 0.99 {
			t.Errorf("Identical text should have similarity ~1.0, got %f", score)
		}
		t.Logf("✓ Identical text similarity: %f", score)
	})

	t.Run("DifferentTextSimilarity", func(t *testing.T) {
		score := CalculateSimilarity(TestText1, TestText3, TestMaxLength)
		if score < 0 {
			t.Fatalf("Similarity calculation failed: %f", score)
		}

		// Different texts should have lower similarity
		identicalScore := CalculateSimilarity(TestText1, TestText1, TestMaxLength)
		if score >= identicalScore {
			t.Errorf("Different texts should have lower similarity than identical: %f vs %f",
				score, identicalScore)
		}

		t.Logf("✓ Different text similarity: %f (< identical %f)", score, identicalScore)
	})

	t.Run("SimilarTextsShouldHaveHighSimilarity", func(t *testing.T) {
		score := CalculateSimilarity(TestText1, TestText2, TestMaxLength)
		if score < 0.5 {
			t.Errorf("Semantically similar texts should have similarity > 0.5, got %f", score)
		}
		t.Logf("✓ Similar texts similarity: %f", score)
	})

	t.Run("EmptyStringSimilarity", func(t *testing.T) {
		score := CalculateSimilarity("", "", TestMaxLength)
		if score < 0 {
			t.Error("Empty string similarity should not fail")
		}
	})
}

// ============================================================================
// FIND MOST SIMILAR TESTS
// ============================================================================

func TestFindMostSimilar(t *testing.T) {
	err := InitEmbeddingModel(DefaultEmbeddingModelPath, "CPU")
	if err != nil {
		t.Skipf("Skipping FindMostSimilar tests: %v", err)
	}

	candidates := []string{
		"Machine learning is fascinating",
		"The weather is sunny today",
		"I love artificial intelligence",
		"Programming is fun",
	}

	t.Run("FindMostSimilar", func(t *testing.T) {
		query := "I enjoy machine learning"
		result := FindMostSimilar(query, candidates, TestMaxLength)

		if result.Index < 0 {
			t.Fatalf("Find most similar failed, got negative index: %d", result.Index)
		}

		if result.Index >= len(candidates) {
			t.Fatalf("Index out of bounds: %d >= %d", result.Index, len(candidates))
		}

		if result.Score < 0 {
			t.Fatalf("Invalid similarity score: %f", result.Score)
		}

		// Should pick index 0 or 2 (ML/AI related)
		if result.Index != 0 && result.Index != 2 {
			t.Errorf("Expected index 0 or 2 (ML/AI related), got %d", result.Index)
		}

		t.Logf("✓ Most similar to '%s' is candidate %d: '%s' (score: %f)",
			query, result.Index, candidates[result.Index], result.Score)
	})

	t.Run("FindMostSimilarDefault", func(t *testing.T) {
		query := "I enjoy machine learning"
		result := FindMostSimilarDefault(query, candidates)

		if result.Index < 0 {
			t.Fatalf("Find most similar failed: %d", result.Index)
		}
	})

	t.Run("FindMostSimilarEmptyCandidates", func(t *testing.T) {
		query := "test query"
		result := FindMostSimilar(query, []string{}, TestMaxLength)

		if result.Index != -1 || result.Score != -1.0 {
			t.Errorf("Expected index=-1 and score=-1.0 for empty candidates, got index=%d, score=%f",
				result.Index, result.Score)
		}
	})

	t.Run("FindMostSimilarSingleCandidate", func(t *testing.T) {
		query := "test query"
		singleCandidate := []string{"only one option"}
		result := FindMostSimilar(query, singleCandidate, TestMaxLength)

		if result.Index != 0 {
			t.Errorf("Expected index=0 for single candidate, got %d", result.Index)
		}
	})
}

// ============================================================================
// BATCH SIMILARITY TESTS
// ============================================================================

func TestBatchSimilarity(t *testing.T) {
	err := InitEmbeddingModel(DefaultEmbeddingModelPath, "CPU")
	if err != nil {
		t.Skipf("Skipping batch similarity tests: %v", err)
	}

	query := "machine learning algorithms"
	candidates := []string{
		"artificial intelligence systems",
		"weather forecast sunny",
		"deep neural networks",
		"cooking recipes pasta",
		"natural language processing",
	}

	t.Run("ManualBatchSimilarityTopK", func(t *testing.T) {
		// Manually calculate top K by iterating
		k := 3
		type result struct {
			Index int
			Score float32
		}

		results := make([]result, 0, len(candidates))
		for i, candidate := range candidates {
			score := CalculateSimilarity(query, candidate, TestMaxLength)
			results = append(results, result{Index: i, Score: score})
		}

		// Sort descending by score
		for i := 0; i < len(results); i++ {
			for j := i + 1; j < len(results); j++ {
				if results[j].Score > results[i].Score {
					results[i], results[j] = results[j], results[i]
				}
			}
		}

		// Take top K
		if len(results) > k {
			results = results[:k]
		}

		if len(results) != k {
			t.Errorf("Expected %d results, got %d", k, len(results))
		}

		// Check sorted descending
		for i := 1; i < len(results); i++ {
			if results[i].Score > results[i-1].Score {
				t.Errorf("Results not sorted: results[%d].Score (%.4f) > results[%d].Score (%.4f)",
					i, results[i].Score, i-1, results[i-1].Score)
			}
		}

		// Check indices are valid
		for i, result := range results {
			if result.Index < 0 || result.Index >= len(candidates) {
				t.Errorf("Invalid index at position %d: %d", i, result.Index)
			}
		}

		t.Logf("✓ Batch similarity top %d:", k)
		for i, result := range results {
			t.Logf("  %d. '%s' (score: %.4f)", i+1, candidates[result.Index], result.Score)
		}
	})
}

// ============================================================================
// CLASSIFICATION TESTS
// ============================================================================

func TestClassification(t *testing.T) {
	err := InitModernBertClassifier(CategoryClassifierModelPath, 14, "CPU")
	if err != nil {
		t.Skipf("Skipping classification tests: %v", err)
	}

	t.Run("BasicClassification", func(t *testing.T) {
		text := "What is the weather today?"
		result, err := ClassifyModernBert(text)
		if err != nil {
			t.Fatalf("Failed to classify: %v", err)
		}

		if result.Class < 0 || result.Class >= 14 {
			t.Errorf("Invalid class: %d", result.Class)
		}

		if result.Confidence < 0.0 || result.Confidence > 1.0 {
			t.Errorf("Confidence out of range: %f", result.Confidence)
		}

		t.Logf("✓ Classification: class=%d, confidence=%.4f", result.Class, result.Confidence)
	})

	t.Run("ClassificationConsistency", func(t *testing.T) {
		text := "How do I reset my password?"

		result1, err1 := ClassifyModernBert(text)
		result2, err2 := ClassifyModernBert(text)

		if err1 != nil || err2 != nil {
			t.Fatalf("Failed to classify: %v, %v", err1, err2)
		}

		if result1.Class != result2.Class {
			t.Errorf("Inconsistent classification: %d vs %d", result1.Class, result2.Class)
		}

		// Confidence should also be identical (deterministic)
		diffConf := math.Abs(float64(result1.Confidence - result2.Confidence))
		if diffConf > 1e-6 {
			t.Errorf("Inconsistent confidence: %.6f vs %.6f (diff: %.9f)",
				result1.Confidence, result2.Confidence, diffConf)
		}

		t.Logf("✓ Classification consistent: class=%d, confidence=%.4f", result1.Class, result1.Confidence)
	})

	t.Run("ClassificationWithProbabilities", func(t *testing.T) {
		// Skip this test - ClassifyWithProbabilities requires ModernBERT WithProbs function
		t.Skip("Probability distribution test skipped (requires ModernBERT WithProbs function)")
	})

	t.Run("ClassificationMultipleTexts", func(t *testing.T) {
		texts := []string{
			"What is the weather today?",
			"How do I reset my password?",
			"Tell me about machine learning",
			"I want to book a flight",
			"What are your business hours?",
		}

		for i, text := range texts {
			result, err := ClassifyModernBert(text)
			if err != nil {
				t.Errorf("Failed to classify text %d: %v", i, err)
				continue
			}

			if result.Confidence < 0.3 {
				t.Errorf("Low confidence for text %d: %.4f", i, result.Confidence)
			}

			t.Logf("  Text %d: class=%d, confidence=%.4f", i, result.Class, result.Confidence)
		}
	})

	t.Run("EmptyStringClassification", func(t *testing.T) {
		result, err := ClassifyModernBert("")
		if err != nil {
			t.Logf("Empty string classification returned error (acceptable): %v", err)
		} else {
			t.Logf("Empty string classified as class=%d, confidence=%.4f", result.Class, result.Confidence)
		}
	})
}

// ============================================================================
// CONCURRENCY TESTS
// ============================================================================

func TestConcurrency(t *testing.T) {
	t.Run("ConcurrentEmbedding", func(t *testing.T) {
		err := InitEmbeddingModel(DefaultEmbeddingModelPath, "CPU")
		if err != nil {
			t.Skipf("Skipping: %v", err)
		}

		const numGoroutines = 10
		const numIterations = 5

		var wg sync.WaitGroup
		errors := make(chan error, numGoroutines*numIterations)

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				for j := 0; j < numIterations; j++ {
					_, err := GetEmbedding(TestText1, TestMaxLength)
					if err != nil {
						errors <- err
					}
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		errorCount := 0
		for err := range errors {
			t.Errorf("Concurrent embedding error: %v", err)
			errorCount++
		}

		if errorCount == 0 {
			t.Logf("✓ %d concurrent embedding requests completed successfully", numGoroutines*numIterations)
		}
	})

	t.Run("ConcurrentSimilarity", func(t *testing.T) {
		err := InitEmbeddingModel(DefaultEmbeddingModelPath, "CPU")
		if err != nil {
			t.Skipf("Skipping: %v", err)
		}

		const numGoroutines = 10
		const numIterations = 5

		var wg sync.WaitGroup
		errors := make(chan error, numGoroutines*numIterations)

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				for j := 0; j < numIterations; j++ {
					score := CalculateSimilarity(TestText1, TestText2, TestMaxLength)
					if score < 0 {
						errors <- nil // Track failures
					}
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		errorCount := 0
		for range errors {
			errorCount++
		}

		if errorCount > 0 {
			t.Errorf("Concurrent similarity calculation had %d failures", errorCount)
		} else {
			t.Logf("✓ %d concurrent similarity requests completed successfully", numGoroutines*numIterations)
		}
	})

	t.Run("ConcurrentClassification", func(t *testing.T) {
		err := InitModernBertClassifier(CategoryClassifierModelPath, 14, "CPU")
		if err != nil {
			t.Skipf("Skipping: %v", err)
		}

		const numGoroutines = 20
		const numRequests = 100

		text := "What is the weather today?"
		var wg sync.WaitGroup
		var mu sync.Mutex
		var errorCount int32
		classResults := make(map[int]int)

		startTime := time.Now()

		for i := 0; i < numRequests; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				result, err := ClassifyModernBert(text)
				if err != nil {
					t.Errorf("Error in goroutine %d: %v", id, err)
					errorCount++
					return
				}

				mu.Lock()
				classResults[result.Class]++
				mu.Unlock()
			}(i)
		}

		wg.Wait()
		duration := time.Since(startTime)
		throughput := float64(numRequests) / duration.Seconds()

		if errorCount > 0 {
			t.Errorf("Had %d errors during concurrent classification", errorCount)
		}

		// Check consistency - all requests should return same class
		if len(classResults) != 1 {
			t.Errorf("Inconsistent classification: got %d different classes: %v", len(classResults), classResults)
		}

		t.Logf("✓ Concurrent inference: %d requests, %.2fs, %.1f req/s, %d unique classes",
			numRequests, duration.Seconds(), throughput, len(classResults))
	})
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

func TestErrorHandling(t *testing.T) {
	t.Run("UninitializedModelError", func(t *testing.T) {
		// Model is already initialized from previous tests, so this test is not applicable
		t.Skip("Model already initialized from previous tests")
	})

	t.Run("EmptyStringHandling", func(t *testing.T) {
		err := InitEmbeddingModel(DefaultEmbeddingModelPath, "CPU")
		if err != nil {
			t.Skipf("Skipping: %v", err)
		}

		// Test empty strings don't crash
		embedding, err := GetEmbedding("", TestMaxLength)
		if err != nil {
			t.Logf("Empty string returned error: %v", err)
		}
		if len(embedding) > 0 {
			t.Logf("Empty string produced embedding of length %d", len(embedding))
		}

		score := CalculateSimilarity("", "", TestMaxLength)
		t.Logf("Empty string similarity: %f", score)

		result := FindMostSimilar("", []string{"test"}, TestMaxLength)
		t.Logf("Empty query FindMostSimilar: index=%d, score=%f", result.Index, result.Score)
	})

	t.Run("InvalidMaxLength", func(t *testing.T) {
		err := InitEmbeddingModel(DefaultEmbeddingModelPath, "CPU")
		if err != nil {
			t.Skipf("Skipping: %v", err)
		}

		// Test with invalid max lengths
		_, err = GetEmbedding(TestText1, 0)
		if err != nil {
			t.Logf("max_length=0 returned error: %v", err)
		}

		_, err = GetEmbedding(TestText1, -1)
		if err != nil {
			t.Logf("max_length=-1 returned error: %v", err)
		}
	})

	t.Run("VeryLongText", func(t *testing.T) {
		err := InitEmbeddingModel(DefaultEmbeddingModelPath, "CPU")
		if err != nil {
			t.Skipf("Skipping: %v", err)
		}

		// Create very long text (> max_length tokens)
		longText := ""
		for i := 0; i < 1000; i++ {
			longText += "word "
		}

		embedding, err := GetEmbedding(longText, 128)
		if err != nil {
			t.Errorf("Failed to handle long text: %v", err)
		} else {
			t.Logf("Long text produced embedding of length %d", len(embedding))
		}
	})
}

// ============================================================================
// UTILITY FUNCTION TESTS
// ============================================================================

func TestUtilityFunctions(t *testing.T) {
	t.Run("IsModelInitialized", func(t *testing.T) {
		// Before initialization
		if IsEmbeddingModelInitialized() {
			t.Log("Embedding model already initialized (from previous tests)")
		}

		err := InitEmbeddingModel(DefaultEmbeddingModelPath, "CPU")
		if err != nil {
			t.Skipf("Skipping: %v", err)
		}

		// After initialization
		if !IsEmbeddingModelInitialized() {
			t.Error("Model should be initialized")
		}
	})

	t.Run("ClassifierInitializedCheck", func(t *testing.T) {
		err := InitModernBertClassifier(CategoryClassifierModelPath, 14, "CPU")
		if err != nil {
			t.Skipf("Skipping: %v", err)
		}

		// If init succeeded, classifier is ready to use
		_, err = ClassifyModernBert("test")
		if err != nil {
			t.Errorf("Classifier should be usable after initialization: %v", err)
		}
	})
}

// ============================================================================
// BENCHMARKS
// ============================================================================

func BenchmarkEmbedding(b *testing.B) {
	err := InitEmbeddingModel(DefaultEmbeddingModelPath, "CPU")
	if err != nil {
		b.Skipf("Skipping: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = GetEmbedding(TestText1, TestMaxLength)
	}
}

func BenchmarkSimilarity(b *testing.B) {
	err := InitEmbeddingModel(DefaultEmbeddingModelPath, "CPU")
	if err != nil {
		b.Skipf("Skipping: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CalculateSimilarity(TestText1, TestText2, TestMaxLength)
	}
}

func BenchmarkClassification(b *testing.B) {
	err := InitModernBertClassifier(CategoryClassifierModelPath, 14, "CPU")
	if err != nil {
		b.Skipf("Skipping: %v", err)
	}

	text := "What is the weather today?"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = ClassifyModernBert(text)
	}
}

func BenchmarkConcurrentClassification(b *testing.B) {
	err := InitModernBertClassifier(CategoryClassifierModelPath, 14, "CPU")
	if err != nil {
		b.Skipf("Skipping: %v", err)
	}

	text := "What is the weather today?"
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = ClassifyModernBert(text)
		}
	})
}
