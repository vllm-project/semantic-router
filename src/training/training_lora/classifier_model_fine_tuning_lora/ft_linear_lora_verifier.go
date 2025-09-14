package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	candle "github.com/vllm-project/semantic-router/candle-binding"
)

// ModelConfig represents the structure of config.json
type ModelConfig struct {
	Architectures []string `json:"architectures"`
}

// CategoryMapping holds the mapping between indices and domain categories
type CategoryMapping struct {
	CategoryToIdx map[string]int    `json:"category_to_idx"`
	IdxToCategory map[string]string `json:"idx_to_category"`
}

// Global variable for category mappings
var categoryLabels map[int]string

// Configuration for LoRA Intent model
type IntentLoRAConfig struct {
	UseModernBERT     bool
	ModelPath         string
	UseCPU            bool
	ModelArchitecture string // Added to track model architecture
}

// detectModelArchitecture reads config.json and determines the model architecture
func detectModelArchitecture(modelPath string) (string, error) {
	configPath := filepath.Join(modelPath, "config.json")

	configData, err := ioutil.ReadFile(configPath)
	if err != nil {
		return "", fmt.Errorf("failed to read config.json: %v", err)
	}

	var config ModelConfig
	err = json.Unmarshal(configData, &config)
	if err != nil {
		return "", fmt.Errorf("failed to parse config.json: %v", err)
	}

	if len(config.Architectures) == 0 {
		return "", fmt.Errorf("no architectures found in config.json")
	}

	architecture := config.Architectures[0]
	fmt.Printf("Detected model architecture: %s\n", architecture)

	return architecture, nil
}

// countLabelsFromConfig counts the number of labels in config.json
func countLabelsFromConfig(modelPath string) (int, error) {
	configPath := filepath.Join(modelPath, "config.json")

	configData, err := ioutil.ReadFile(configPath)
	if err != nil {
		return 0, fmt.Errorf("failed to read config.json: %v", err)
	}

	var configMap map[string]interface{}
	err = json.Unmarshal(configData, &configMap)
	if err != nil {
		return 0, fmt.Errorf("failed to parse config.json: %v", err)
	}

	if id2label, exists := configMap["id2label"].(map[string]interface{}); exists {
		return len(id2label), nil
	}

	return 0, fmt.Errorf("id2label not found in config.json")
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

	fmt.Printf("Loaded %d category mappings\n", len(categoryLabels))
	return nil
}

// initializeIntentClassifier initializes the intent classifier based on architecture
func initializeIntentClassifier(config IntentLoRAConfig) error {
	fmt.Printf("Initializing LoRA Intent classifier (%s): %s\n", config.ModelArchitecture, config.ModelPath)

	var err error

	// Choose initialization function based on model architecture
	switch {
	case strings.Contains(config.ModelArchitecture, "ModernBert"):
		err = candle.InitModernBertClassifier(config.ModelPath, config.UseCPU)
	case strings.Contains(config.ModelArchitecture, "Bert") || strings.Contains(config.ModelArchitecture, "Roberta"):
		// For BERT and RoBERTa, use new official Candle implementation
		numClasses, countErr := countLabelsFromConfig(config.ModelPath)
		if countErr != nil {
			return fmt.Errorf("failed to count labels: %v", countErr)
		}
		success := candle.InitCandleBertClassifier(config.ModelPath, numClasses, config.UseCPU)
		if !success {
			err = fmt.Errorf("failed to initialize Candle BERT classifier")
		}
	default:
		return fmt.Errorf("unsupported model architecture: %s", config.ModelArchitecture)
	}

	if err != nil {
		return fmt.Errorf("failed to initialize LoRA intent classifier: %v", err)
	}

	fmt.Printf("LoRA Intent Classifier initialized successfully!\n")
	return nil
}

// classifyIntentText performs intent classification using the appropriate classifier
func classifyIntentText(text string, config IntentLoRAConfig) (candle.ClassResult, error) {
	// Choose classification function based on model architecture
	switch {
	case strings.Contains(config.ModelArchitecture, "ModernBert"):
		return candle.ClassifyModernBertText(text)
	case strings.Contains(config.ModelArchitecture, "Bert") || strings.Contains(config.ModelArchitecture, "Roberta"):
		return candle.ClassifyCandleBertText(text)
	default:
		return candle.ClassResult{}, fmt.Errorf("unsupported model architecture: %s", config.ModelArchitecture)
	}
}

func main() {
	// Parse command line flags
	var (
		useModernBERT = flag.Bool("modernbert", true, "Use ModernBERT models (default for LoRA)")
		modelPath     = flag.String("model", "lora_intent_classifier_modernbert-base_r8", "Path to LoRA classifier model")
		useCPU        = flag.Bool("cpu", false, "Use CPU instead of GPU")
	)
	flag.Parse()

	config := IntentLoRAConfig{
		UseModernBERT: *useModernBERT,
		ModelPath:     *modelPath,
		UseCPU:        *useCPU,
	}

	// Detect model architecture
	modelArchitecture, err := detectModelArchitecture(*modelPath)
	if err != nil {
		log.Fatalf("Failed to detect model architecture: %v", err)
	}
	config.ModelArchitecture = modelArchitecture

	fmt.Println("LoRA Intent Classifier Test")
	fmt.Println("============================")

	// Load category mapping
	err = loadCategoryMapping(config.ModelPath)
	if err != nil {
		log.Fatalf("Failed to load category mapping: %v", err)
	}

	// Initialize classifier
	err = initializeIntentClassifier(config)
	if err != nil {
		log.Fatalf("Failed to initialize LoRA classifier: %v", err)
	}

	// Test samples for intent classification (matching Python demo_inference)
	testSamples := []string{
		"What is the best strategy for corporate mergers and acquisitions?",
		"How do antitrust laws affect business competition?",
		"What are the psychological factors that influence consumer behavior?",
		"Explain the legal requirements for contract formation",
		"What is the difference between civil and criminal law?",
		"How does cognitive bias affect decision making?",
	}

	fmt.Println("\nTesting LoRA Intent Classification:")
	fmt.Println("===================================")

	for i, sample := range testSamples {
		fmt.Printf("\nTest %d: %s\n", i+1, sample)

		result, err := classifyIntentText(sample, config)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		if label, exists := categoryLabels[result.Class]; exists {
			fmt.Printf("Classification: %s (Class ID: %d, Confidence: %.4f)\n", label, result.Class, result.Confidence)
		} else {
			fmt.Printf("Unknown category index: %d (Confidence: %.4f)\n", result.Class, result.Confidence)
		}
	}

	fmt.Println("\nLoRA Intent Classification test completed!")
}
