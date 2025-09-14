package main

import (
	"encoding/json"
	"flag"
	"fmt"
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

// Configuration for LoRA PII model type
type LoRAModelConfig struct {
	PIITokenModelPath         string
	UseCPU                    bool
	EnableTokenClassification bool
	ModelArchitecture         string // Added to track model architecture
}

// detectModelArchitecture reads config.json and determines the model architecture
func detectModelArchitecture(modelPath string) (string, error) {
	configPath := filepath.Join(modelPath, "config.json")

	configData, err := os.ReadFile(configPath)
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

// initializeModels initializes the LoRA PII token classifier based on architecture
func initializeModels(config LoRAModelConfig) error {
	// Initialize LoRA PII token classifier
	if config.EnableTokenClassification {
		fmt.Printf("\nInitializing LoRA PII token classifier (%s): %s\n", config.ModelArchitecture, config.PIITokenModelPath)

		var err error

		// Choose initialization function based on model architecture
		switch {
		case strings.Contains(config.ModelArchitecture, "ModernBert"):
			err = candle.InitModernBertPIITokenClassifier(config.PIITokenModelPath, config.UseCPU)
		case strings.Contains(config.ModelArchitecture, "Bert") || strings.Contains(config.ModelArchitecture, "Roberta"):
			// For BERT and RoBERTa, use new official Candle token classifier
			numClasses, countErr := countLabelsFromConfig(config.PIITokenModelPath)
			if countErr != nil {
				return fmt.Errorf("failed to count labels: %v", countErr)
			}
			success := candle.InitCandleBertTokenClassifier(config.PIITokenModelPath, numClasses, config.UseCPU)
			if !success {
				err = fmt.Errorf("failed to initialize Candle BERT token classifier")
			}
		default:
			return fmt.Errorf("unsupported model architecture: %s", config.ModelArchitecture)
		}

		if err != nil {
			return fmt.Errorf("failed to initialize LoRA PII token classifier: %v", err)
		}
		fmt.Printf("LoRA PII token classifier initialized successfully!\n")
		fmt.Println("   Note: Token-level entity detection enabled with LoRA fine-tuning")
	}

	return nil
}

// countLabelsFromConfig counts the number of labels in config.json
func countLabelsFromConfig(modelPath string) (int, error) {
	configPath := filepath.Join(modelPath, "config.json")

	configData, err := os.ReadFile(configPath)
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

// classifyPIITokens performs PII token classification using the appropriate classifier
func classifyPIITokens(text string, config LoRAModelConfig) (candle.TokenClassificationResult, error) {
	// Choose classification function based on model architecture
	switch {
	case strings.Contains(config.ModelArchitecture, "ModernBert"):
		configPath := fmt.Sprintf("%s/config.json", config.PIITokenModelPath)
		return candle.ClassifyModernBertPIITokens(text, configPath)
	case strings.Contains(config.ModelArchitecture, "Bert") || strings.Contains(config.ModelArchitecture, "Roberta"):
		// For BERT and RoBERTa, use new official Candle token classifier with proper label mapping
		labelMappingPath := fmt.Sprintf("%s/label_mapping.json", config.PIITokenModelPath)
		labelMappingData, err := os.ReadFile(labelMappingPath)
		if err != nil {
			fmt.Printf("Warning: Could not read label mapping from %s, using generic labels: %v\n", labelMappingPath, err)
			return candle.ClassifyCandleBertTokens(text)
		}

		// Parse label mapping to get id2label
		var labelMapping map[string]interface{}
		err = json.Unmarshal(labelMappingData, &labelMapping)
		if err != nil {
			fmt.Printf("Warning: Could not parse label mapping, using generic labels: %v\n", err)
			return candle.ClassifyCandleBertTokens(text)
		}

		// Extract id2label mapping
		id2labelInterface, exists := labelMapping["id_to_label"]
		if !exists {
			fmt.Printf("Warning: No id_to_label found in mapping, using generic labels\n")
			return candle.ClassifyCandleBertTokens(text)
		}

		id2labelJSON, err := json.Marshal(id2labelInterface)
		if err != nil {
			fmt.Printf("Warning: Could not serialize id2label mapping, using generic labels: %v\n", err)
			return candle.ClassifyCandleBertTokens(text)
		}

		return candle.ClassifyCandleBertTokensWithLabels(text, string(id2labelJSON))
	default:
		return candle.TokenClassificationResult{}, fmt.Errorf("unsupported model architecture: %s", config.ModelArchitecture)
	}
}

func main() {
	// Parse command line flags
	var (
		piiTokenPath              = flag.String("pii-token-model", "lora_pii_detector_modernbert-base_r8_token_model", "Path to LoRA PII token classifier model")
		enableTokenClassification = flag.Bool("token-classification", true, "Enable token-level PII classification")
		useCPU                    = flag.Bool("cpu", false, "Use CPU instead of GPU")
	)
	flag.Parse()

	config := LoRAModelConfig{
		PIITokenModelPath:         *piiTokenPath,
		EnableTokenClassification: *enableTokenClassification,
		UseCPU:                    *useCPU,
	}

	// Detect model architecture
	modelArchitecture, err := detectModelArchitecture(*piiTokenPath)
	if err != nil {
		log.Fatalf("Failed to detect model architecture: %v", err)
	}
	config.ModelArchitecture = modelArchitecture

	fmt.Println("LoRA PII Token Classifier Verifier")
	fmt.Println("===================================")

	// Initialize models
	err = initializeModels(config)
	if err != nil {
		log.Fatalf("Failed to initialize models: %v", err)
	}

	if config.EnableTokenClassification {
		fmt.Println("\nTesting LoRA PII Token Classification:")
		fmt.Println("======================================")

		// Test samples with various PII entities
		testSamples := []string{
			"My name is John Smith and my email is john.smith@example.com",
			"Please call me at 555-123-4567 or visit my address at 123 Main Street, New York, NY 10001",
			"The patient's social security number is 123-45-6789 and credit card is 4111-1111-1111-1111",
			"Contact Dr. Sarah Johnson at sarah.johnson@hospital.org for medical records",
			"My personal information: Phone: +1-800-555-0199, Address: 456 Oak Avenue, Los Angeles, CA 90210",
		}

		for i, sample := range testSamples {
			fmt.Printf("\nTest %d: %s\n", i+1, sample)

			result, err := classifyPIITokens(sample, config)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				continue
			}

			if len(result.Entities) == 0 {
				fmt.Printf("PII Entities: No entities detected\n")
			} else {
				fmt.Printf("PII Entities: %d entities detected:\n", len(result.Entities))

				for j, entity := range result.Entities {
					fmt.Printf("   %d. %s: \"%s\" [%d-%d] (confidence: %.3f)\n",
						j+1, entity.EntityType, entity.Text, entity.Start, entity.End, entity.Confidence)

					// Verify span extraction
					if entity.Start >= 0 && entity.End <= len(sample) && entity.Start < entity.End {
						extractedText := sample[entity.Start:entity.End]
						if extractedText != entity.Text {
							fmt.Printf("      WARNING: Span mismatch: expected '%s', extracted '%s'\n",
								entity.Text, extractedText)
						}
					} else {
						fmt.Printf("      WARNING: Invalid span: %d-%d for text length %d\n",
							entity.Start, entity.End, len(sample))
					}
				}
			}
		}
	}

	fmt.Println("\nLoRA PII classification test completed!")
}
