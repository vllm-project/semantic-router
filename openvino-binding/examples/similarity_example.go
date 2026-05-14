package main

import (
	"fmt"
	"log"
	"os"

	openvino "github.com/vllm-project/semantic-router/openvino-binding"
)

func runSimilarityExample() {
	text1 := "The cat sits on the mat"
	text2 := "A cat is sitting on a rug"
	text3 := "The weather is sunny today"

	sim12 := openvino.CalculateSimilarityDefault(text1, text2)
	sim13 := openvino.CalculateSimilarityDefault(text1, text3)

	fmt.Printf("Text 1: %s\nText 2: %s\nSimilarity: %.4f\n\n", text1, text2, sim12)
	fmt.Printf("Text 1: %s\nText 3: %s\nSimilarity: %.4f\n\n", text1, text3, sim13)
}

func runFindMostSimilarExample() {
	query := "machine learning and artificial intelligence"
	candidates := []string{
		"deep neural networks",
		"cooking recipes",
		"artificial intelligence research",
		"weather forecast",
		"natural language processing",
	}

	result := openvino.FindMostSimilarDefault(query, candidates)
	if result.Index >= 0 {
		fmt.Printf("Query: %s\nMost similar: %s (score: %.4f)\n",
			query, candidates[result.Index], result.Score)
	} else {
		fmt.Println("Failed to find most similar")
	}
}

func runTokenizationExample() {
	sampleText := "Hello world, this is a test"
	tokResult, err := openvino.TokenizeTextDefault(sampleText)
	if err != nil {
		log.Printf("Tokenization error: %v", err)
		return
	}

	fmt.Printf("Text: %s\n", sampleText)
	fmt.Printf("Token count: %d\n", len(tokResult.TokenIDs))

	switch {
	case len(tokResult.TokenIDs) > 10:
		fmt.Printf("Token IDs (first 10): %v\n", tokResult.TokenIDs[:10])
	case len(tokResult.TokenIDs) > 0:
		fmt.Printf("Token IDs: %v\n", tokResult.TokenIDs)
	default:
		fmt.Println("Note: Tokenization returned empty (tokenizer may not expose IDs directly)")
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: similarity_example <model_path.xml> [device]")
		fmt.Println("Example: similarity_example ./models/bert-base-uncased.xml CPU")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	device := "CPU"
	if len(os.Args) > 2 {
		device = os.Args[2]
	}

	fmt.Printf("OpenVINO version: %s\n", openvino.GetVersion())
	fmt.Printf("Available devices: %v\n\n", openvino.GetAvailableDevices())

	fmt.Printf("Initializing model from: %s on %s\n", modelPath, device)
	if err := openvino.InitModel(modelPath, device); err != nil {
		log.Fatalf("Failed to initialize model: %v", err)
	}
	fmt.Println("Model initialized successfully")

	fmt.Println("\n=== Example 1: Simple Similarity ===")
	runSimilarityExample()

	fmt.Println("=== Example 2: Find Most Similar ===")
	runFindMostSimilarExample()

	fmt.Println("\n=== Example 3: Tokenization ===")
	runTokenizationExample()

	fmt.Println("\n=== All examples completed successfully! ===")
}
