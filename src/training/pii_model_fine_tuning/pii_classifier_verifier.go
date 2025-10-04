// Copyright 2025 The vLLM Semantic Router Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"flag"
	"fmt"
	"log"

	candle "github.com/vllm-project/semantic-router/candle-binding"
)

// Configuration for model type
type ModelConfig struct {
	PIITokenModelPath         string
	UseCPU                    bool
	EnableTokenClassification bool
}

// initializeModels initializes the PII token classifier
func initializeModels(config ModelConfig) error {
	// Initialize PII token classifier
	if config.EnableTokenClassification {
		fmt.Printf("\nInitializing PII token classifier (ModernBERT): %s\n", config.PIITokenModelPath)

		err := candle.InitModernBertPIITokenClassifier(config.PIITokenModelPath, config.UseCPU)
		if err != nil {
			return fmt.Errorf("failed to initialize PII token classifier: %v", err)
		}
		fmt.Printf("PII token classifier initialized successfully!\n")
		fmt.Println("   Note: Token-level entity detection enabled")
	}

	return nil
}

// classifyPIITokens performs PII token classification using ModernBERT
func classifyPIITokens(text string, config ModelConfig) (candle.TokenClassificationResult, error) {
	// Construct config path from model path
	configPath := fmt.Sprintf("%s/config.json", config.PIITokenModelPath)
	return candle.ClassifyModernBertPIITokens(text, configPath)
}

func main() {
	// Parse command line flags
	var (
		piiTokenPath              = flag.String("pii-token-model", "../../../models/pii_classifier_modernbert-base_presidio_token_model", "Path to PII token classifier model")
		enableTokenClassification = flag.Bool("token-classification", true, "Enable token-level PII classification")
		useCPU                    = flag.Bool("cpu", false, "Use CPU instead of GPU")
	)
	flag.Parse()

	config := ModelConfig{
		PIITokenModelPath:         *piiTokenPath,
		EnableTokenClassification: *enableTokenClassification,
		UseCPU:                    *useCPU,
	}
	fmt.Println("PII Classifier Verifier")
	fmt.Println("========================")

	var err error
	// Initialize models
	err = initializeModels(config)
	if err != nil {
		log.Fatalf("Failed to initialize models: %v", err)
	}

	fmt.Println("===================================")

	testTexts := []struct {
		text        string
		description string
	}{
		{"What is the derivative of x^2 with respect to x?", "Math Question"},
		{"My email address is john.smith@example.com", "Email PII"},
		{"Explain the concept of supply and demand in economics", "Economics Question"},
		{"Please call me at (555) 123-4567 for more information", "Phone PII"},
		{"How does DNA replication work in eukaryotic cells?", "Biology Question"},
		{"My social security number is 123-45-6789", "SSN PII"},
		{"What are the fundamental principles of computer algorithms?", "CS Question"},
		{"This is just a normal sentence without any personal information", "Clean Text"},
		{"What is the difference between civil law and common law?", "Law Question"},
		{"I live at 123 Main Street, New York, NY 10001", "Address Info"},
		{"My credit card number is 4532-1234-5678-9012", "Credit Card PII"},
		{"Visit our website at https://example.com for details", "URL Reference"},
	}

	for i, test := range testTexts {
		fmt.Printf("\nTest %d: %s\n", i+1, test.description)
		fmt.Printf("   Text: \"%s\"\n", test.text)

		// PII token classification
		if config.EnableTokenClassification {
			tokenResult, err := classifyPIITokens(test.text, config)
			if err != nil {
				fmt.Printf("PII Tokens: Error - %v\n", err)
			} else {
				if len(tokenResult.Entities) == 0 {
					fmt.Printf("PII Tokens: No entities detected\n")
				} else {
					fmt.Printf("PII Tokens: %d entities detected:\n", len(tokenResult.Entities))

					// Group entities by type for summary
					entityTypes := make(map[string]int)

					for i, entity := range tokenResult.Entities {
						fmt.Printf("   %d. %s: \"%s\" [%d-%d] (confidence: %.3f)\n",
							i+1, entity.EntityType, entity.Text, entity.Start, entity.End, entity.Confidence)

						// Verify span extraction
						if entity.Start >= 0 && entity.End <= len(test.text) && entity.Start < entity.End {
							extractedText := test.text[entity.Start:entity.End]
							if extractedText != entity.Text {
								fmt.Printf("      WARNING: Span mismatch: expected '%s', extracted '%s'\n",
									entity.Text, extractedText)
							}
						} else {
							fmt.Printf("      WARNING: Invalid span: %d-%d for text length %d\n",
								entity.Start, entity.End, len(test.text))
						}

						entityTypes[entity.EntityType]++
					}

					// Display entity type summary
					if len(entityTypes) > 0 {
						fmt.Printf("   Entity types: ")
						first := true
						for entityType, count := range entityTypes {
							if !first {
								fmt.Printf(", ")
							}
							fmt.Printf("%s(%d)", entityType, count)
							first = false
						}
						fmt.Println()
					}
				}
			}
		}

	}
}
