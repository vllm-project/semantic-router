// Comprehensive example demonstrating Qwen3 Embedding Model
//
// This example shows:
// 1. Basic embedding generation
// 2. Similarity calculation between texts
// 3. Batch similarity search
// 4. Model information and diagnostics
//
// Usage:
//   cd ../../candle-binding
//   LD_LIBRARY_PATH=$(pwd)/target/release go run ../examples/candle-binding/qwen3_embedding_example.go

package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"time"
)

/*
#cgo LDFLAGS: -L${SRCDIR}/../../candle-binding/target/release -lcandle_semantic_router -ldl -lm
#include <stdlib.h>
#include <stdbool.h>

// Embedding result structure
typedef struct {
    float* data;
    int length;
    bool error;
    int model_type;
    int sequence_length;
    float processing_time_ms;
} EmbeddingResult;

// Embedding similarity result structure
typedef struct {
    float similarity;
    int model_type;
    float processing_time_ms;
    bool error;
} EmbeddingSimilarityResult;

// Batch similarity match structure
typedef struct {
    int index;
    float similarity;
} BatchSimilarityMatch;

// Batch similarity result structure
typedef struct {
    BatchSimilarityMatch* matches;
    int num_matches;
    float processing_time_ms;
    bool error;
    char* error_message;
} BatchSimilarityResult;

extern bool init_embedding_models(const char* qwen3_model_path, const char* gemma_model_path, bool use_cpu);
extern int get_embedding_with_model_type(const char* text, const char* model_type, int target_dim, EmbeddingResult* result);
extern int calculate_embedding_similarity(const char* text1, const char* text2, const char* model_type, int target_dim, EmbeddingSimilarityResult* result);
extern int calculate_similarity_batch(const char* query, const char** candidates, int num_candidates, int top_k, const char* model_type, int target_dim, BatchSimilarityResult* result);
extern void free_batch_similarity_result(BatchSimilarityResult* result);
*/
import "C"
import "unsafe"

func printHeader(title string) {
	fmt.Println()
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("  " + title)
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println()
}

func printSubHeader(title string) {
	fmt.Println()
	fmt.Println(strings.Repeat("-", 80))
	fmt.Println(title)
	fmt.Println(strings.Repeat("-", 80))
}

// GetEmbedding generates an embedding for the given text
func GetEmbedding(text string) ([]float32, time.Duration, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cModelType := C.CString("qwen3")
	defer C.free(unsafe.Pointer(cModelType))

	var result C.EmbeddingResult
	status := C.get_embedding_with_model_type(cText, cModelType, -1, &result)

	if status != 0 || result.error {
		return nil, 0, fmt.Errorf("failed to get embedding (status: %d)", status)
	}

	// Convert C array to Go slice
	length := int(result.length)
	embedding := make([]float32, length)
	cArray := (*[1 << 30]C.float)(unsafe.Pointer(result.data))[:length:length]
	for i := 0; i < length; i++ {
		embedding[i] = float32(cArray[i])
	}

	processingTime := time.Duration(float64(result.processing_time_ms) * float64(time.Millisecond))

	// Free the C memory
	C.free(unsafe.Pointer(result.data))

	return embedding, processingTime, nil
}

// CalculateSimilarity computes cosine similarity between two texts
func CalculateSimilarity(text1, text2 string) (float32, time.Duration, error) {
	cText1 := C.CString(text1)
	defer C.free(unsafe.Pointer(cText1))

	cText2 := C.CString(text2)
	defer C.free(unsafe.Pointer(cText2))

	cModelType := C.CString("qwen3")
	defer C.free(unsafe.Pointer(cModelType))

	var result C.EmbeddingSimilarityResult
	status := C.calculate_embedding_similarity(cText1, cText2, cModelType, -1, &result)

	if status != 0 || result.error {
		return 0, 0, fmt.Errorf("failed to calculate similarity (status: %d)", status)
	}

	similarity := float32(result.similarity)
	processingTime := time.Duration(float64(result.processing_time_ms) * float64(time.Millisecond))

	return similarity, processingTime, nil
}

// CalculateBatchSimilarity finds top-k most similar texts to a query
func CalculateBatchSimilarity(query string, candidates []string, topK int) ([]struct {
	Index      int
	Similarity float32
}, time.Duration, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	// Convert candidates to C array
	cCandidates := make([]*C.char, len(candidates))
	for i, candidate := range candidates {
		cCandidates[i] = C.CString(candidate)
		defer C.free(unsafe.Pointer(cCandidates[i]))
	}

	cModelType := C.CString("qwen3")
	defer C.free(unsafe.Pointer(cModelType))

	var result C.BatchSimilarityResult
	status := C.calculate_similarity_batch(
		cQuery,
		(**C.char)(unsafe.Pointer(&cCandidates[0])),
		C.int(len(candidates)),
		C.int(topK),
		cModelType,
		-1,
		&result,
	)

	if status != 0 || result.error {
		errMsg := ""
		if result.error_message != nil {
			errMsg = C.GoString(result.error_message)
		}
		return nil, 0, fmt.Errorf("failed to calculate batch similarity: %s (status: %d)", errMsg, status)
	}

	// Convert results
	numMatches := int(result.num_matches)
	matches := make([]struct {
		Index      int
		Similarity float32
	}, numMatches)

	cMatches := (*[1 << 30]C.BatchSimilarityMatch)(unsafe.Pointer(result.matches))[:numMatches:numMatches]
	for i := 0; i < numMatches; i++ {
		matches[i].Index = int(cMatches[i].index)
		matches[i].Similarity = float32(cMatches[i].similarity)
	}

	processingTime := time.Duration(float64(result.processing_time_ms) * float64(time.Millisecond))

	// Free C memory
	C.free_batch_similarity_result(&result)

	return matches, processingTime, nil
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return float32(dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)))
}

func main() {
	printHeader("Qwen3 Embedding Model Example")

	// Initialize model
	fmt.Println("🔧 Initializing Qwen3 Embedding Model...")
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "../models/Qwen3-Embedding-0.6B"
	}

	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	success := C.init_embedding_models(cModelPath, nil, false)
	if !success {
		log.Fatalf("❌ Failed to initialize embedding model from: %s", modelPath)
	}

	fmt.Printf("✅ Model loaded successfully from: %s\n", modelPath)

	// Example 1: Basic Embedding Generation
	printHeader("Example 1: Basic Embedding Generation")

	testTexts := []string{
		"The quick brown fox jumps over the lazy dog",
		"Machine learning is transforming the world",
		"What is the capital of France?",
	}

	for i, text := range testTexts {
		fmt.Printf("Text %d: %s\n", i+1, text)

		embedding, duration, err := GetEmbedding(text)
		if err != nil {
			log.Printf("   ❌ Error: %v\n", err)
			continue
		}

		fmt.Printf("   ✅ Embedding dimension: %d\n", len(embedding))
		fmt.Printf("   ⏱️  Processing time: %.2f ms\n", duration.Seconds()*1000)
		fmt.Printf("   📊 First 5 values: %.4f, %.4f, %.4f, %.4f, %.4f...\n",
			embedding[0], embedding[1], embedding[2], embedding[3], embedding[4])
		fmt.Println()
	}

	// Example 2: Similarity Calculation
	printHeader("Example 2: Text Similarity")

	pairs := []struct {
		text1 string
		text2 string
		desc  string
	}{
		{
			"I love programming in Python",
			"Python is my favorite programming language",
			"Similar sentences about Python programming",
		},
		{
			"The weather is sunny today",
			"Machine learning models are very powerful",
			"Completely different topics",
		},
		{
			"How do I reset my password?",
			"I forgot my password, can you help?",
			"Similar questions with different wording",
		},
	}

	for i, pair := range pairs {
		fmt.Printf("Pair %d: %s\n", i+1, pair.desc)
		fmt.Printf("  Text 1: %s\n", pair.text1)
		fmt.Printf("  Text 2: %s\n", pair.text2)

		similarity, duration, err := CalculateSimilarity(pair.text1, pair.text2)
		if err != nil {
			log.Printf("  ❌ Error: %v\n", err)
			continue
		}

		fmt.Printf("  📊 Similarity: %.4f\n", similarity)
		fmt.Printf("  ⏱️  Processing time: %.2f ms\n", duration.Seconds()*1000)
		fmt.Println()
	}

	// Example 3: Batch Similarity Search
	printHeader("Example 3: Batch Similarity Search (Semantic Search)")

	query := "How can I improve my machine learning model performance?"

	documents := []string{
		"Tips for optimizing neural network training",
		"The best chocolate cake recipe",
		"Techniques to prevent model overfitting",
		"How to change a flat tire",
		"Hyperparameter tuning strategies for deep learning",
		"Travel guide to Paris, France",
		"Cross-validation methods for model evaluation",
		"Best practices for data preprocessing",
	}

	fmt.Printf("Query: %s\n\n", query)
	fmt.Println("Documents:")
	for i, doc := range documents {
		fmt.Printf("  [%d] %s\n", i, doc)
	}
	fmt.Println()

	topK := 3
	fmt.Printf("Finding top-%d most similar documents...\n\n", topK)

	matches, duration, err := CalculateBatchSimilarity(query, documents, topK)
	if err != nil {
		log.Fatalf("❌ Error: %v", err)
	}

	fmt.Printf("✅ Found top-%d matches in %.2f ms:\n\n", topK, duration.Seconds()*1000)
	for rank, match := range matches {
		fmt.Printf("  Rank %d: [%d] %.4f - %s\n",
			rank+1, match.Index, match.Similarity, documents[match.Index])
	}

	// Example 4: Manual Similarity Verification
	printHeader("Example 4: Manual Similarity Verification")

	text1 := "Machine learning is amazing"
	text2 := "AI and ML are transforming technology"

	fmt.Printf("Computing embeddings for:\n")
	fmt.Printf("  Text 1: %s\n", text1)
	fmt.Printf("  Text 2: %s\n", text2)
	fmt.Println()

	emb1, dur1, err := GetEmbedding(text1)
	if err != nil {
		log.Fatalf("❌ Error getting embedding 1: %v", err)
	}

	emb2, dur2, err := GetEmbedding(text2)
	if err != nil {
		log.Fatalf("❌ Error getting embedding 2: %v", err)
	}

	manualSim := cosineSimilarity(emb1, emb2)
	fmt.Printf("✅ Manual cosine similarity: %.4f\n", manualSim)
	fmt.Printf("⏱️  Embedding 1 time: %.2f ms\n", dur1.Seconds()*1000)
	fmt.Printf("⏱️  Embedding 2 time: %.2f ms\n", dur2.Seconds()*1000)
	fmt.Println()

	// Verify with built-in similarity
	builtinSim, durSim, err := CalculateSimilarity(text1, text2)
	if err != nil {
		log.Fatalf("❌ Error calculating similarity: %v", err)
	}

	fmt.Printf("✅ Built-in similarity: %.4f\n", builtinSim)
	fmt.Printf("⏱️  Processing time: %.2f ms\n", durSim.Seconds()*1000)
	fmt.Printf("📊 Difference: %.6f (should be very small)\n", math.Abs(float64(manualSim-builtinSim)))

	printHeader("✅ All Examples Complete!")
	fmt.Println("The Qwen3 embedding model is working correctly!")
	fmt.Println()
}
