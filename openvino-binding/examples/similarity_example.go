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
		fmt.Println("Usage: similarity_example <model_path.xml> [device]")
		fmt.Println("Example: similarity_example ./models/bert-base-uncased.xml CPU")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	device := "CPU"
	if len(os.Args) > 2 {
		device = os.Args[2]
	}

	// Print OpenVINO version
	version := openvino.GetVersion()
	fmt.Printf("OpenVINO version: %s\n", version)

	// Check available devices
	devices := openvino.GetAvailableDevices()
	fmt.Printf("Available devices: %v\n", devices)
	fmt.Println()

	// Initialize model
	fmt.Printf("Initializing model from: %s on %s\n", modelPath, device)
	err := openvino.InitModel(modelPath, device)
	if err != nil {
		log.Fatalf("Failed to initialize model: %v", err)
	}
	fmt.Println("✓ Model initialized successfully")
	fmt.Println()

	// Example 1: Simple similarity
	fmt.Println("=== Example 1: Simple Similarity ===")
	text1 := "The cat sits on the mat"
	text2 := "A cat is sitting on a rug"
	text3 := "The weather is sunny today"

	sim12 := openvino.CalculateSimilarityDefault(text1, text2)
	sim13 := openvino.CalculateSimilarityDefault(text1, text3)

	fmt.Printf("Text 1: %s\n", text1)
	fmt.Printf("Text 2: %s\n", text2)
	fmt.Printf("Similarity: %.4f\n", sim12)
	fmt.Println()

	fmt.Printf("Text 1: %s\n", text1)
	fmt.Printf("Text 3: %s\n", text3)
	fmt.Printf("Similarity: %.4f\n", sim13)
	fmt.Println()

	// Example 2: Find most similar
	fmt.Println("=== Example 2: Find Most Similar ===")
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
		fmt.Printf("Query: %s\n", query)
		fmt.Printf("Most similar: %s (score: %.4f)\n",
			candidates[result.Index], result.Score)
	} else {
		fmt.Println("Failed to find most similar")
	}
	fmt.Println()

	// Example 3: Tokenization
	fmt.Println("=== Example 3: Tokenization ===")
	sampleText := "Hello world, this is a test"
	tokResult, err := openvino.TokenizeTextDefault(sampleText)
	if err != nil {
		log.Printf("Tokenization error: %v", err)
	} else {
		fmt.Printf("Text: %s\n", sampleText)
		fmt.Printf("Token count: %d\n", len(tokResult.TokenIDs))
		fmt.Printf("Token IDs: %v\n", tokResult.TokenIDs[:10])
	}
	fmt.Println()

	fmt.Println("=== All examples completed successfully! ===")
}
