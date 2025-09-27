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

// initializeIntentClassifier initializes the LoRA intent classifier
func initializeIntentClassifier(config IntentLoRAConfig) error {
	fmt.Printf("Initializing LoRA Intent classifier: %s\n", config.ModelPath)

	// Use different initialization methods based on architecture (following PII LoRA pattern)
	switch config.ModelArchitecture {
	case "BertForSequenceClassification", "RobertaForSequenceClassification":
		fmt.Printf("Using Candle BERT Classifier for %s architecture\n", config.ModelArchitecture)

		// Count the number of labels from config.json
		numClasses, err := countLabelsFromConfig(config.ModelPath)
		if err != nil {
			return fmt.Errorf("failed to count labels: %v", err)
		}

		fmt.Printf("Detected %d classes from config.json\n", numClasses)

		// Use Candle BERT classifier which supports LoRA models
		success := candle.InitCandleBertClassifier(config.ModelPath, numClasses, config.UseCPU)
		if !success {
			return fmt.Errorf("failed to initialize LoRA BERT/RoBERTa classifier")
		}

	case "ModernBertForSequenceClassification":
		fmt.Printf("Using ModernBERT Classifier for ModernBERT architecture\n")
		// Use dedicated ModernBERT classifier for ModernBERT models
		err := candle.InitModernBertClassifier(config.ModelPath, config.UseCPU)
		if err != nil {
			return fmt.Errorf("failed to initialize ModernBERT classifier: %v", err)
		}

	default:
		return fmt.Errorf("unsupported model architecture: %s", config.ModelArchitecture)
	}

	fmt.Printf("LoRA Intent Classifier initialized successfully!\n")
	return nil
}

// classifyIntentText performs intent classification using the appropriate classifier
func classifyIntentText(text string, config IntentLoRAConfig) (candle.ClassResult, error) {
	switch config.ModelArchitecture {
	case "BertForSequenceClassification", "RobertaForSequenceClassification":
		// Use Candle BERT classifier for BERT and RoBERTa LoRA models
		result, err := candle.ClassifyCandleBertText(text)
		if err != nil {
			return candle.ClassResult{}, err
		}
		return result, nil

	case "ModernBertForSequenceClassification":
		// Use dedicated ModernBERT classifier
		result, err := candle.ClassifyModernBertText(text)
		if err != nil {
			return candle.ClassResult{}, err
		}
		return result, nil

	default:
		return candle.ClassResult{}, fmt.Errorf("unsupported architecture: %s", config.ModelArchitecture)
	}
}

func main() {
	// Parse command line flags
	var (
		useModernBERT = flag.Bool("modernbert", true, "Use ModernBERT models (default for LoRA)")
		modelPath     = flag.String("model", "../../../../models/lora_intent_classifier_bert-base-uncased_model", "Path to LoRA classifier model")
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

	// Test samples with expected intent categories for validation
	testSamples := []struct {
		text        string
		description string
		expected    string
	}{
		{
			"What is the best strategy for corporate mergers and acquisitions?",
			"Business strategy question",
			"business",
		},
		{
			"How do antitrust laws affect business competition?",
			"Business law question",
			"business",
		},
		{
			"What are the psychological factors that influence consumer behavior?",
			"Psychology and behavior question",
			"psychology",
		},
		{
			"Explain the legal requirements for contract formation",
			"Legal concepts question",
			"jurisprudence",
		},
		{
			"What is the difference between civil and criminal law?",
			"Legal system question",
			"jurisprudence",
		},
		{
			"How does cognitive bias affect decision making?",
			"Psychology and cognition question",
			"psychology",
		},
		{
			"What is the derivative of e^x?",
			"Mathematical calculus question",
			"mathematics",
		},
		{
			"Explain the concept of supply and demand in economics.",
			"Economic principles question",
			"economics",
		},
		{
			"How does DNA replication work in eukaryotic cells?",
			"Biology and genetics question",
			"biology",
		},
		{
			"What is the difference between a civil law and common law system?",
			"Legal systems comparison",
			"jurisprudence",
		},
		{
			"Explain how transistors work in computer processors.",
			"Computer engineering question",
			"computer_science",
		},
		{
			"Why do stars twinkle?",
			"Astronomical physics question",
			"physics",
		},
		{
			"How do I create a balanced portfolio for retirement?",
			"Financial planning question",
			"economics",
		},
		{
			"What causes mental illnesses?",
			"Mental health and psychology question",
			"psychology",
		},
		{
			"How do computer algorithms work?",
			"Computer science fundamentals",
			"computer_science",
		},
		{
			"Explain the historical significance of the Roman Empire.",
			"Historical analysis question",
			"history",
		},
		{
			"What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?",
			"Calculus problem",
			"mathematics",
		},
		{
			"Describe the process of photosynthesis in plants.",
			"Biological processes question",
			"biology",
		},
		{
			"What are the principles of macroeconomic policy?",
			"Economic policy question",
			"economics",
		},
		{
			"How does machine learning classification work?",
			"Machine learning concepts",
			"computer_science",
		},
		{
			"What is the capital of France?",
			"General knowledge question",
			"other",
		},
	}

	fmt.Println("\nTesting LoRA Intent Classification:")
	fmt.Println("===================================")

	// Statistics tracking
	var (
		totalTests     = len(testSamples)
		correctTests   = 0
		highConfidence = 0
		lowConfidence  = 0
	)

	for i, test := range testSamples {
		fmt.Printf("\nTest %d: %s\n", i+1, test.description)
		fmt.Printf("   Text: \"%s\"\n", test.text)

		result, err := classifyIntentText(test.text, config)
		if err != nil {
			fmt.Printf("   Classification failed: %v\n", err)
			continue
		}

		// Get the predicted label name
		labelName := "unknown"
		if label, exists := categoryLabels[result.Class]; exists {
			labelName = label
		}

		// Print the result
		fmt.Printf("   Classified as: %s (Class ID: %d, Confidence: %.4f)\n",
			labelName, result.Class, result.Confidence)

		// Check correctness
		isCorrect := labelName == test.expected
		if isCorrect {
			fmt.Printf("   ✓ CORRECT")
			correctTests++
		} else {
			fmt.Printf("   ✗ INCORRECT (Expected: %s)", test.expected)
		}

		// Add confidence assessment
		if result.Confidence > 0.7 {
			fmt.Printf(" - HIGH CONFIDENCE\n")
			highConfidence++
		} else if result.Confidence > 0.5 {
			fmt.Printf(" - MEDIUM CONFIDENCE\n")
		} else {
			fmt.Printf(" - LOW CONFIDENCE\n")
			lowConfidence++
		}
	}

	// Print comprehensive summary
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("INTENT CLASSIFICATION TEST SUMMARY")
	fmt.Println(strings.Repeat("=", 50))
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correct Predictions: %d/%d (%.1f%%)\n", correctTests, totalTests, float64(correctTests)/float64(totalTests)*100)
	fmt.Printf("High Confidence (>0.7): %d/%d (%.1f%%)\n", highConfidence, totalTests, float64(highConfidence)/float64(totalTests)*100)
	fmt.Printf("Low Confidence (<0.5): %d/%d (%.1f%%)\n", lowConfidence, totalTests, float64(lowConfidence)/float64(totalTests)*100)

	// Overall assessment
	accuracy := float64(correctTests) / float64(totalTests) * 100
	fmt.Printf("\nOVERALL ASSESSMENT: ")
	if accuracy >= 85.0 {
		fmt.Printf("EXCELLENT (%.1f%% accuracy)\n", accuracy)
	} else if accuracy >= 70.0 {
		fmt.Printf("GOOD (%.1f%% accuracy)\n", accuracy)
	} else if accuracy >= 50.0 {
		fmt.Printf("FAIR (%.1f%% accuracy) - Consider retraining\n", accuracy)
	} else {
		fmt.Printf("POOR (%.1f%% accuracy) - Requires retraining\n", accuracy)
	}

	fmt.Println("\nLoRA Intent Classification verification completed!")
}
