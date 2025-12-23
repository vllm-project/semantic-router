package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	candle "github.com/vllm-project/semantic-router/candle-binding"
)

// CategoryMapping holds the mapping between indices and domain categories
type CategoryMapping struct {
	CategoryToIdx map[string]int    `json:"category_to_idx"`
	IdxToCategory map[string]string `json:"idx_to_category"`
}

// Global variable for category mappings
var categoryLabels map[int]string

// Configuration for model type
type ModelConfig struct {
	UseModernBERT bool
	ModelPath     string
	UseCPU        bool
}

// loadCategoryMapping loads the category mapping from a JSON file
func loadCategoryMapping(modelPath string) error {
	mappingPath := fmt.Sprintf("%s/category_mapping.json", modelPath)

	data, err := os.ReadFile(mappingPath)
	if err != nil {
		return fmt.Errorf("failed to read mapping file %s: %v", mappingPath, err)
	}

	var mapping CategoryMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return fmt.Errorf("failed to parse mapping JSON: %v", err)
	}

	// Convert string keys to int keys for easier lookup
	categoryLabels = make(map[int]string)
	for idxStr, label := range mapping.IdxToCategory {
		var idx int
		if _, err := fmt.Sscanf(idxStr, "%d", &idx); err != nil {
			return fmt.Errorf("failed to parse category index %s: %v", idxStr, err)
		}
		categoryLabels[idx] = label
	}

	fmt.Printf("Loaded %d category labels from %s\n", len(categoryLabels), mappingPath)
	return nil
}

// initializeClassifier initializes the classifier model based on config
func initializeClassifier(config ModelConfig) error {
	numClasses := len(categoryLabels)
	fmt.Printf("\nInitializing classifier (%s): %s\n",
		map[bool]string{true: "ModernBERT", false: "Linear"}[config.UseModernBERT],
		config.ModelPath)

	var err error
	if config.UseModernBERT {
		// Initialize ModernBERT classifier
		err = candle.InitModernBertClassifier(config.ModelPath, config.UseCPU)
	} else {
		// Initialize linear classifier
		err = candle.InitClassifier(config.ModelPath, numClasses, config.UseCPU)
	}

	if err != nil {
		return fmt.Errorf("failed to initialize classifier: %v", err)
	}

	fmt.Printf("Classifier initialized successfully!\n")
	if config.UseModernBERT {
		fmt.Println("Note: Number of classes auto-detected from model weights")
	} else {
		fmt.Printf("Note: Using %d classes from mapping file\n", numClasses)
	}

	return nil
}

// classifyText performs text classification using the appropriate model type
func classifyText(text string, config ModelConfig) (candle.ClassResult, error) {
	if config.UseModernBERT {
		return candle.ClassifyModernBertText(text)
	}
	return candle.ClassifyText(text)
}

func main() {
	// Parse command line flags
	var (
		useModernBERT = flag.Bool("modernbert", false, "Use ModernBERT models instead of linear classifier")
		modelPath     = flag.String("model", "../../../models/mom-domain-classifier", "Path to classifier model")
		useCPU        = flag.Bool("cpu", false, "Use CPU instead of GPU")
	)
	flag.Parse()

	config := ModelConfig{
		UseModernBERT: *useModernBERT,
		ModelPath:     *modelPath,
		UseCPU:        *useCPU,
	}

	fmt.Println("Domain Classifier Test")
	fmt.Println("======================")

	// Load category mapping
	err := loadCategoryMapping(config.ModelPath)
	if err != nil {
		log.Fatalf("Failed to load category mapping: %v", err)
	}

	// Initialize classifier
	err = initializeClassifier(config)
	if err != nil {
		log.Fatalf("Failed to initialize classifier: %v", err)
	}

	fmt.Println("===================================")

	// Test data with descriptions
	testQueries := []struct {
		text        string
		description string
	}{
		{"What is the derivative of e^x?", "Mathematics"},
		{"Explain the concept of supply and demand in economics.", "Economics"},
		{"How does DNA replication work in eukaryotic cells?", "Biology"},
		{"What is the difference between a civil law and common law system?", "Law"},
		{"Explain how transistors work in computer processors.", "Computer Science"},
		{"Why do stars twinkle?", "Physics"},
		{"How do I create a balanced portfolio for retirement?", "Economics"},
		{"What causes mental illnesses?", "Psychology"},
		{"How do computer algorithms work?", "Computer Science"},
		{"Explain the historical significance of the Roman Empire.", "History"},
		{"What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?", "Mathematics"},
		{"Describe the process of photosynthesis in plants.", "Biology"},
		{"What are the principles of macroeconomic policy?", "Economics"},
		{"How does machine learning classification work?", "Computer Science"},
		{"What is the capital of France?", "Other"},
	}

	// Process each query
	fmt.Println("\nClassifying queries:")
	fmt.Println("==================")

	for i, test := range testQueries {
		fmt.Printf("\nTest %d: %s\n", i+1, test.description)
		fmt.Printf("   Query: \"%s\"\n", test.text)

		// Classify the text
		result, err := classifyText(test.text, config)
		if err != nil {
			fmt.Printf("   Classification failed: %v\n", err)
			continue
		}

		// Get the category name
		categoryName := categoryLabels[result.Class]
		if categoryName == "" {
			categoryName = fmt.Sprintf("Class_%d", result.Class)
		}

		// Print the result
		fmt.Printf("   Classified as: %s (Class ID: %d, Confidence: %.4f)\n",
			categoryName, result.Class, result.Confidence)

	}
}
