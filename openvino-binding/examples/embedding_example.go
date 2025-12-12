package main

import (
	"fmt"
	"log"
	"os"

	openvino "github.com/vllm-project/semantic-router/openvino-binding"
)

func main() {
	// Check command line arguments
	if len(os.Args) < 2 {
		fmt.Println("Usage: embedding_example <model_path.xml> [device]")
		fmt.Println("Example: embedding_example ./models/bert-base-uncased.xml CPU")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	device := "CPU"
	if len(os.Args) > 2 {
		device = os.Args[2]
	}

	// Initialize embedding model
	fmt.Printf("Initializing embedding model from: %s on %s\n", modelPath, device)
	err := openvino.InitEmbeddingModel(modelPath, device)
	if err != nil {
		log.Fatalf("Failed to initialize embedding model: %v", err)
	}
	fmt.Println("✓ Embedding model initialized successfully")
	fmt.Println()

	// Example 1: Generate embedding
	fmt.Println("=== Example 1: Generate Embedding ===")
	text := "Hello, world! This is a semantic embedding example."

	embedding, err := openvino.GetEmbeddingDefault(text)
	if err != nil {
		log.Fatalf("Failed to generate embedding: %v", err)
	}

	fmt.Printf("Text: %s\n", text)
	fmt.Printf("Embedding dimension: %d\n", len(embedding))
	fmt.Printf("First 10 values: %v\n", embedding[:10])
	fmt.Println()

	// Example 2: Embedding with metadata
	fmt.Println("=== Example 2: Embedding with Metadata ===")
	output, err := openvino.GetEmbeddingWithMetadata(text, 512)
	if err != nil {
		log.Fatalf("Failed to generate embedding with metadata: %v", err)
	}

	fmt.Printf("Text: %s\n", text)
	fmt.Printf("Embedding dimension: %d\n", len(output.Embedding))
	fmt.Printf("Processing time: %.2f ms\n", output.ProcessingTimeMs)
	fmt.Println()

	// Example 3: Batch similarity search
	fmt.Println("=== Example 3: Batch Similarity Search ===")
	query := "natural language processing"
	candidates := []string{
		"machine learning algorithms",
		"computer vision techniques",
		"text processing and analysis",
		"image recognition systems",
		"speech synthesis methods",
		"language understanding models",
	}

	batchResult, err := openvino.CalculateSimilarityBatch(query, candidates, 3, 512)
	if err != nil {
		log.Fatalf("Failed to calculate batch similarity: %v", err)
	}

	fmt.Printf("Query: %s\n", query)
	fmt.Printf("Top %d matches:\n", len(batchResult.Matches))
	for i, match := range batchResult.Matches {
		fmt.Printf("  %d. %s (similarity: %.4f)\n",
			i+1, candidates[match.Index], match.Similarity)
	}
	fmt.Printf("Processing time: %.2f ms\n", batchResult.ProcessingTimeMs)
	fmt.Println()

	// Example 4: Compare embeddings directly
	fmt.Println("=== Example 4: Embedding Similarity ===")
	text1 := "The quick brown fox jumps over the lazy dog"
	text2 := "A fast brown fox leaps over a sleepy dog"
	text3 := "Python programming language is great"

	simOutput12, err := openvino.CalculateEmbeddingSimilarity(text1, text2, 512)
	if err != nil {
		log.Fatalf("Failed to calculate similarity: %v", err)
	}

	simOutput13, err := openvino.CalculateEmbeddingSimilarity(text1, text3, 512)
	if err != nil {
		log.Fatalf("Failed to calculate similarity: %v", err)
	}

	fmt.Printf("Text 1: %s\n", text1)
	fmt.Printf("Text 2: %s\n", text2)
	fmt.Printf("Similarity: %.4f (%.2f ms)\n",
		simOutput12.Similarity, simOutput12.ProcessingTimeMs)
	fmt.Println()

	fmt.Printf("Text 1: %s\n", text1)
	fmt.Printf("Text 3: %s\n", text3)
	fmt.Printf("Similarity: %.4f (%.2f ms)\n",
		simOutput13.Similarity, simOutput13.ProcessingTimeMs)
	fmt.Println()

	fmt.Println("=== All examples completed successfully! ===")
}
