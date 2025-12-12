package main

import (
	"fmt"
	"log"
	"os"

	openvino "github.com/your-org/semantic-router/openvino-binding"
)

func main() {
	// Get model paths from environment or use defaults
	baseModelPath := os.Getenv("BASE_MODEL_PATH")
	if baseModelPath == "" {
		baseModelPath = "../test_models/bert-base-uncased/openvino_model.xml"
	}

	loraAdaptersPath := os.Getenv("LORA_ADAPTERS_PATH")
	if loraAdaptersPath == "" {
		loraAdaptersPath = "../test_models/lora_adapters"
	}

	device := os.Getenv("OPENVINO_DEVICE")
	if device == "" {
		device = "CPU"
	}

	// Example 1: BERT LoRA Multi-Task Classification
	fmt.Println("=== BERT LoRA Multi-Task Classification ===")

	// Initialize BERT LoRA classifier
	err := openvino.InitBertLoRAClassifier(baseModelPath, loraAdaptersPath, device)
	if err != nil {
		log.Fatalf("Failed to initialize BERT LoRA classifier: %v", err)
	}
	fmt.Println("✓ BERT LoRA classifier initialized")

	// Test texts
	texts := []string{
		"Hello, how can I help you today?",
		"My email is john.doe@example.com and my phone is 555-1234",
		"DROP TABLE users; --",
	}

	// Multi-task classification
	fmt.Println("\nMulti-task classification:")
	for i, text := range texts {
		fmt.Printf("\nText %d: %s\n", i+1, text)

		result, err := openvino.ClassifyBertLoRAMultiTask(text)
		if err != nil {
			log.Printf("Error: %v", err)
			continue
		}

		fmt.Printf("  Intent:   Class %d (confidence: %.2f%%)\n",
			result.IntentClass, result.IntentConfidence*100)
		fmt.Printf("  PII:      Class %d (confidence: %.2f%%)\n",
			result.PIIClass, result.PIIConfidence*100)
		fmt.Printf("  Security: Class %d (confidence: %.2f%%)\n",
			result.SecurityClass, result.SecurityConfidence*100)
		fmt.Printf("  Processing time: %.2f ms\n", result.ProcessingTimeMs)
	}

	// Example 2: Single-Task Classification
	fmt.Println("\n=== Single-Task Classification ===")

	testText := "My credit card number is 1234-5678-9012-3456"
	fmt.Printf("\nText: %s\n", testText)

	// Classify for PII detection only
	piiResult, err := openvino.ClassifyBertLoRATask(testText, openvino.TaskPII)
	if err != nil {
		log.Fatalf("Failed to classify for PII: %v", err)
	}

	fmt.Printf("PII Detection: Class %d (confidence: %.2f%%)\n",
		piiResult.Class, piiResult.Confidence*100)

	// Classify for security detection only
	securityResult, err := openvino.ClassifyBertLoRATask(testText, openvino.TaskSecurity)
	if err != nil {
		log.Fatalf("Failed to classify for security: %v", err)
	}

	fmt.Printf("Security Detection: Class %d (confidence: %.2f%%)\n",
		securityResult.Class, securityResult.Confidence*100)

	// Example 3: ModernBERT LoRA (if models are available)
	modernbertBaseModel := os.Getenv("MODERNBERT_MODEL_PATH")
	modernbertLoRAPath := os.Getenv("MODERNBERT_LORA_PATH")

	if modernbertBaseModel != "" && modernbertLoRAPath != "" {
		fmt.Println("\n=== ModernBERT LoRA Classification ===")

		err := openvino.InitModernBertLoRAClassifier(
			modernbertBaseModel,
			modernbertLoRAPath,
			device,
		)
		if err != nil {
			log.Printf("Warning: Could not initialize ModernBERT LoRA: %v", err)
		} else {
			fmt.Println("✓ ModernBERT LoRA classifier initialized")

			result, err := openvino.ClassifyModernBertLoRAMultiTask(
				"Hello, my name is John and my SSN is 123-45-6789",
			)
			if err != nil {
				log.Printf("Error: %v", err)
			} else {
				fmt.Printf("Intent:   Class %d (%.2f%%)\n",
					result.IntentClass, result.IntentConfidence*100)
				fmt.Printf("PII:      Class %d (%.2f%%)\n",
					result.PIIClass, result.PIIConfidence*100)
				fmt.Printf("Security: Class %d (%.2f%%)\n",
					result.SecurityClass, result.SecurityConfidence*100)
			}
		}
	}

	// Example 4: Batch Processing
	fmt.Println("\n=== Batch Processing ===")

	batchTexts := []string{
		"What is the weather today?",
		"My password is secret123!",
		"SELECT * FROM users WHERE id=1",
		"Thank you for your help!",
		"Call me at +1-555-0100",
	}

	fmt.Printf("Processing %d texts...\n", len(batchTexts))

	var totalTime float32
	for i, text := range batchTexts {
		result, err := openvino.ClassifyBertLoRAMultiTask(text)
		if err != nil {
			log.Printf("Error processing text %d: %v", i+1, err)
			continue
		}

		totalTime += result.ProcessingTimeMs

		// Print only if high confidence in any category
		if result.IntentConfidence > 0.8 || result.PIIConfidence > 0.8 || result.SecurityConfidence > 0.8 {
			fmt.Printf("  Text %d: ", i+1)
			if result.IntentConfidence > 0.8 {
				fmt.Printf("Intent=%d(%.0f%%) ", result.IntentClass, result.IntentConfidence*100)
			}
			if result.PIIConfidence > 0.8 {
				fmt.Printf("PII=%d(%.0f%%) ", result.PIIClass, result.PIIConfidence*100)
			}
			if result.SecurityConfidence > 0.8 {
				fmt.Printf("Security=%d(%.0f%%) ", result.SecurityClass, result.SecurityConfidence*100)
			}
			fmt.Println()
		}
	}

	avgTime := totalTime / float32(len(batchTexts))
	fmt.Printf("\nAverage processing time: %.2f ms per text\n", avgTime)
	fmt.Printf("Throughput: %.0f texts/second\n", 1000.0/avgTime)

	fmt.Println("\n=== Done ===")
}
