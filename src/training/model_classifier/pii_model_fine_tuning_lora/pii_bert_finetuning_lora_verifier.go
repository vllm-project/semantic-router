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

// ExpectedEntity represents an expected PII entity for testing
type ExpectedEntity struct {
	EntityType string
	Text       string
	Start      int
	End        int
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

// normalizeBIOLabel converts BIO format labels to simple format for comparison
func normalizeBIOLabel(label string) string {
	// Remove BIO prefixes (B-, I-, O-)
	if strings.HasPrefix(label, "B-") || strings.HasPrefix(label, "I-") {
		return label[2:]
	}
	if label == "O" {
		return ""
	}
	return label
}

// normalizeEntityType maps various entity type formats to standard format
func normalizeEntityType(entityType string) string {
	// First normalize BIO format
	normalized := normalizeBIOLabel(entityType)

	// Map common variations to expected format
	switch strings.ToUpper(normalized) {
	case "EMAIL_ADDRESS", "EMAIL":
		return "EMAIL"
	case "PHONE_NUMBER", "PHONE":
		return "PHONE_NUMBER"
	case "STREET_ADDRESS", "ADDRESS", "LOCATION", "GPE":
		return "LOCATION"
	case "US_SSN", "SSN":
		return "SSN"
	case "CREDIT_CARD", "CREDITCARD":
		return "CREDIT_CARD"
	case "PERSON", "PER":
		return "PERSON"
	case "ORGANIZATION", "ORG":
		return "ORGANIZATION"
	case "DOMAIN_NAME", "DOMAIN":
		return "DOMAIN_NAME"
	case "TITLE":
		return "TITLE"
	default:
		return strings.ToUpper(normalized)
	}
}

// combineBIOEntities combines individual BIO-tagged tokens into complete entities
func combineBIOEntities(rawEntities []candle.TokenEntity, originalText string) []candle.TokenEntity {
	if len(rawEntities) == 0 {
		return rawEntities
	}

	var combinedEntities []candle.TokenEntity
	var currentEntity *candle.TokenEntity

	for _, entity := range rawEntities {
		entityType := entity.EntityType

		if strings.HasPrefix(entityType, "B-") {
			// Beginning of new entity - save previous if exists
			if currentEntity != nil {
				combinedEntities = append(combinedEntities, *currentEntity)
			}

			// Start new entity
			baseType := entityType[2:] // Remove "B-" prefix
			currentEntity = &candle.TokenEntity{
				EntityType: baseType,
				Start:      entity.Start,
				End:        entity.End,
				Text:       entity.Text,
				Confidence: entity.Confidence,
			}
		} else if strings.HasPrefix(entityType, "I-") {
			// Inside current entity - extend if same type
			baseType := entityType[2:] // Remove "I-" prefix
			if currentEntity != nil && currentEntity.EntityType == baseType {
				// Extend current entity
				currentEntity.End = entity.End
				// Recalculate text from original text using character positions
				if currentEntity.Start >= 0 && currentEntity.End <= len(originalText) && currentEntity.Start < currentEntity.End {
					currentEntity.Text = originalText[currentEntity.Start:currentEntity.End]
				}
				// Update confidence (use minimum to be conservative)
				if entity.Confidence < currentEntity.Confidence {
					currentEntity.Confidence = entity.Confidence
				}
			} else {
				// Different entity type or no current entity - treat as standalone
				if currentEntity != nil {
					combinedEntities = append(combinedEntities, *currentEntity)
				}
				currentEntity = nil
			}
		} else {
			// "O" tag or other - finish current entity if exists
			if currentEntity != nil {
				combinedEntities = append(combinedEntities, *currentEntity)
				currentEntity = nil
			}

			// If it's not an "O" tag, treat as standalone entity
			if entityType != "O" && entityType != "" {
				combinedEntities = append(combinedEntities, entity)
			}
		}
	}

	// Don't forget the last entity
	if currentEntity != nil {
		combinedEntities = append(combinedEntities, *currentEntity)
	}

	return combinedEntities
}

func main() {
	// Parse command line flags
	var (
		piiModelPath = flag.String("pii-token-model", "../../../../models/mom-pii-classifier", "Path to LoRA PII classifier model")
		architecture = flag.String("architecture", "bert", "Model architecture (bert, roberta, modernbert)")
		useCPU       = flag.Bool("cpu", false, "Use CPU instead of GPU")
	)
	flag.Parse()

	if *piiModelPath == "" {
		log.Fatal("PII model path is required")
	}

	fmt.Println("LoRA PII Token Classifier Verifier")
	fmt.Printf("PII Model: %s\n", *piiModelPath)
	fmt.Printf("Architecture: %s\n", *architecture)

	// Detect model architecture from config.json
	modelArchitecture, err := detectModelArchitecture(*piiModelPath)
	if err != nil {
		log.Fatalf("Failed to detect model architecture: %v", err)
	}

	// Initialize PII token classifier based on architecture
	fmt.Printf("Detected model architecture: %s\n", modelArchitecture)

	var initSuccess bool
	switch {
	case strings.Contains(modelArchitecture, "ModernBert"):
		err = candle.InitModernBertPIITokenClassifier(*piiModelPath, *useCPU)
		initSuccess = (err == nil)
	case strings.Contains(modelArchitecture, "Bert") || strings.Contains(modelArchitecture, "Roberta"):
		numClasses, countErr := countLabelsFromConfig(*piiModelPath)
		if countErr != nil {
			log.Fatalf("Failed to count labels: %v", countErr)
		}
		initSuccess = candle.InitCandleBertTokenClassifier(*piiModelPath, numClasses, *useCPU)
	default:
		log.Fatalf("Unsupported model architecture: %s", modelArchitecture)
	}

	if !initSuccess {
		log.Fatalf("Failed to initialize PII token classifier")
	}

	fmt.Println("PII token classifier initialized successfully!")

	// Test cases for PII detection
	testCases := []struct {
		text          string
		description   string
		expectedPII   bool
		expectedTypes []string
	}{
		{
			text:          "My name is John Smith and my email is john.smith@example.com",
			description:   "Name and email detection",
			expectedPII:   true,
			expectedTypes: []string{"PERSON", "EMAIL"},
		},
		{
			text:          "Please call me at 555-123-4567 or visit my address at 123 Main Street, New York, NY 10001",
			description:   "Phone number and address detection",
			expectedPII:   true,
			expectedTypes: []string{"PHONE_NUMBER", "LOCATION"},
		},
		{
			text:          "The patient's social security number is 123-45-6789 and credit card is 4111-1111-1111-1111",
			description:   "SSN and credit card detection",
			expectedPII:   true,
			expectedTypes: []string{"SSN", "CREDIT_CARD"},
		},
		{
			text:          "Contact Dr. Sarah Johnson at sarah.johnson@hospital.org for medical records",
			description:   "Person name and email in medical context",
			expectedPII:   true,
			expectedTypes: []string{"PERSON", "EMAIL"},
		},
		{
			text:          "This is a normal sentence without any personal information.",
			description:   "No PII content",
			expectedPII:   false,
			expectedTypes: []string{},
		},
	}

	// Run tests using unified LoRA classifier
	fmt.Println("\nTesting PII Detection with Unified LoRA Classifier:")
	fmt.Println(strings.Repeat("=", 60))

	var (
		totalTests         = len(testCases)
		correctPredictions = 0
		totalTypesFound    = 0
		totalExpectedTypes = 0
	)

	for i, testCase := range testCases {
		fmt.Printf("\nTest %d: %s\n", i+1, testCase.description)
		fmt.Printf("Text: \"%s\"\n", testCase.text)

		// Use direct PII token classification
		var tokenResult candle.TokenClassificationResult
		var err error

		switch {
		case strings.Contains(modelArchitecture, "ModernBert"):
			configPath := filepath.Join(*piiModelPath, "config.json")
			tokenResult, err = candle.ClassifyModernBertPIITokens(testCase.text, configPath)
		case strings.Contains(modelArchitecture, "Bert") || strings.Contains(modelArchitecture, "Roberta"):
			configPath := filepath.Join(*piiModelPath, "config.json")
			configData, readErr := os.ReadFile(configPath)
			if readErr != nil {
				fmt.Printf("Failed to read config.json: %v\n", readErr)
				continue
			}

			var configMap map[string]interface{}
			if json.Unmarshal(configData, &configMap) != nil {
				fmt.Printf("Failed to parse config.json\n")
				continue
			}

			id2label, exists := configMap["id2label"]
			if !exists {
				fmt.Printf("id2label not found in config.json\n")
				continue
			}

			id2labelJSON, _ := json.Marshal(id2label)
			tokenResult, err = candle.ClassifyCandleBertTokensWithLabels(testCase.text, string(id2labelJSON))
		}

		if err != nil {
			fmt.Printf("Classification failed: %v\n", err)
			continue
		}

		// Combine BIO-tagged tokens into complete entities
		tokenResult.Entities = combineBIOEntities(tokenResult.Entities, testCase.text)

		// Extract unique PII types from detected entities
		piiTypes := make(map[string]bool)
		hasPII := false

		for _, entity := range tokenResult.Entities {
			if entity.Confidence >= 0.5 { // Use threshold
				normalizedType := normalizeEntityType(entity.EntityType)
				piiTypes[normalizedType] = true
				hasPII = true
			}
		}

		// Convert to slice
		var detectedTypes []string
		for piiType := range piiTypes {
			detectedTypes = append(detectedTypes, piiType)
		}

		fmt.Printf("Has PII: %v\n", hasPII)
		if len(detectedTypes) > 0 {
			fmt.Printf("Detected PII Types: %v\n", detectedTypes)
		}

		// Check if prediction matches expectation
		predictionCorrect := hasPII == testCase.expectedPII
		if predictionCorrect {
			fmt.Printf("✓ CORRECT: PII detection matches expectation\n")
			correctPredictions++
		} else {
			fmt.Printf("✗ INCORRECT: Expected HasPII=%v, got HasPII=%v\n",
				testCase.expectedPII, hasPII)
		}

		// Check detected types if PII was found
		if hasPII && len(testCase.expectedTypes) > 0 {
			fmt.Printf("Expected types: %v\n", testCase.expectedTypes)
			totalExpectedTypes += len(testCase.expectedTypes)

			// Check type matching with flexible comparison
			typesFound := 0
			for _, expectedType := range testCase.expectedTypes {
				for _, detectedType := range detectedTypes {
					expectedNorm := normalizeEntityType(expectedType)
					detectedNorm := normalizeEntityType(detectedType)

					if strings.EqualFold(expectedNorm, detectedNorm) {
						typesFound++
						break
					}
				}
			}

			totalTypesFound += typesFound
			if typesFound > 0 {
				fmt.Printf("✓ Found %d/%d expected PII types\n", typesFound, len(testCase.expectedTypes))
			} else {
				fmt.Printf("✗ No expected PII types found\n")
			}
		}
	}

	// Print comprehensive summary
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("UNIFIED LORA PII DETECTION TEST SUMMARY")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correct PII Predictions: %d/%d (%.1f%%)\n",
		correctPredictions, totalTests, float64(correctPredictions)/float64(totalTests)*100)

	if totalExpectedTypes > 0 {
		fmt.Printf("Expected PII Types Found: %d/%d (%.1f%%)\n",
			totalTypesFound, totalExpectedTypes, float64(totalTypesFound)/float64(totalExpectedTypes)*100)
	}

	// Overall assessment
	fmt.Printf("\nOVERALL ASSESSMENT: ")
	accuracy := float64(correctPredictions) / float64(totalTests) * 100
	if accuracy >= 90.0 {
		fmt.Printf("EXCELLENT (%.1f%% accuracy)\n", accuracy)
	} else if accuracy >= 80.0 {
		fmt.Printf("GOOD (%.1f%% accuracy)\n", accuracy)
	} else if accuracy >= 60.0 {
		fmt.Printf("FAIR (%.1f%% accuracy) - Consider retraining\n", accuracy)
	} else {
		fmt.Printf("POOR (%.1f%% accuracy) - Requires retraining\n", accuracy)
	}

}
