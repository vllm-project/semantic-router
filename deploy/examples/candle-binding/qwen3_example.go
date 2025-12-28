// Comprehensive example demonstrating Qwen3 Multi-LoRA Classification
//
// This example shows:
// 1. Multi-LoRA adapter loading and switching
// 2. Zero-shot classification (no adapter required)
// 3. Benchmark dataset evaluation
// 4. Error handling
//
// Usage:
//   cd ../../candle-binding
//   LD_LIBRARY_PATH=$(pwd)/target/release go run ../examples/candle-binding/qwen3_example.go

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"
	"time"
	"unsafe"
)

/*
#cgo LDFLAGS: -L${SRCDIR}/../../candle-binding/target/release -lcandle_semantic_router -ldl -lm
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    int class_id;
    float confidence;
    char* category_name;
    float* probabilities;
    int num_categories;
    bool error;
    char* error_message;
} GenerativeClassificationResult;

extern int init_qwen3_multi_lora_classifier(const char* base_model_path);
extern int load_qwen3_lora_adapter(const char* adapter_name, const char* adapter_path);
extern int classify_with_qwen3_adapter(const char* text, const char* adapter_name, GenerativeClassificationResult* result);
extern int classify_zero_shot_qwen3(const char* text, const char** categories, int num_categories, GenerativeClassificationResult* result);
extern int get_qwen3_loaded_adapters(char*** adapters_out, int* num_adapters);
extern void free_generative_classification_result(GenerativeClassificationResult* result);
extern void free_categories(char** categories, int num_categories);
*/
import "C"

// ==================== Data Structures ====================

type TestSample struct {
	Text        string `json:"text"`
	TrueLabel   string `json:"true_label"`
	TrueLabelID int    `json:"true_label_id"`
}

type ClassificationResult struct {
	ClassID       int
	Confidence    float32
	CategoryName  string
	Probabilities []float32
	NumCategories int
}

// ==================== FFI Wrappers ====================

func InitQwen3MultiLoRAClassifier(baseModelPath string) error {
	cBaseModelPath := C.CString(baseModelPath)
	defer C.free(unsafe.Pointer(cBaseModelPath))

	result := C.init_qwen3_multi_lora_classifier(cBaseModelPath)
	if result != 0 {
		return fmt.Errorf("failed to initialize (error code: %d)", result)
	}
	log.Printf("✅ Initialized Qwen3 Multi-LoRA classifier: %s", baseModelPath)
	return nil
}

func LoadQwen3LoRAAdapter(adapterName, adapterPath string) error {
	cAdapterName := C.CString(adapterName)
	defer C.free(unsafe.Pointer(cAdapterName))

	cAdapterPath := C.CString(adapterPath)
	defer C.free(unsafe.Pointer(cAdapterPath))

	result := C.load_qwen3_lora_adapter(cAdapterName, cAdapterPath)
	if result != 0 {
		return fmt.Errorf("failed to load adapter '%s' (error code: %d)", adapterName, result)
	}
	log.Printf("✅ Loaded adapter '%s': %s", adapterName, adapterPath)
	return nil
}

func ClassifyWithAdapter(text, adapterName string) (*ClassificationResult, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cAdapterName := C.CString(adapterName)
	defer C.free(unsafe.Pointer(cAdapterName))

	var result C.GenerativeClassificationResult
	ret := C.classify_with_qwen3_adapter(cText, cAdapterName, &result)
	defer C.free_generative_classification_result(&result)

	if ret != 0 || result.error {
		errMsg := "classification failed"
		if result.error_message != nil {
			errMsg = C.GoString(result.error_message)
		}
		return nil, fmt.Errorf("%s", errMsg)
	}

	return convertResult(&result), nil
}

func ClassifyZeroShot(text string, categories []string) (*ClassificationResult, error) {
	if len(categories) == 0 {
		return nil, fmt.Errorf("categories list cannot be empty")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cCategories := make([]*C.char, len(categories))
	for i, cat := range categories {
		cCategories[i] = C.CString(cat)
		defer C.free(unsafe.Pointer(cCategories[i]))
	}

	var result C.GenerativeClassificationResult
	ret := C.classify_zero_shot_qwen3(cText, &cCategories[0], C.int(len(categories)), &result)
	defer C.free_generative_classification_result(&result)

	if ret != 0 || result.error {
		errMsg := "zero-shot classification failed"
		if result.error_message != nil {
			errMsg = C.GoString(result.error_message)
		}
		return nil, fmt.Errorf("%s", errMsg)
	}

	return convertResult(&result), nil
}

func GetLoadedAdapters() ([]string, error) {
	var adaptersPtr **C.char
	var numAdapters C.int

	result := C.get_qwen3_loaded_adapters(&adaptersPtr, &numAdapters)
	if result != 0 {
		return nil, fmt.Errorf("failed to get loaded adapters (error code: %d)", result)
	}

	if numAdapters == 0 {
		return []string{}, nil
	}

	adaptersSlice := (*[1000]*C.char)(unsafe.Pointer(adaptersPtr))[:numAdapters:numAdapters]
	adapters := make([]string, numAdapters)
	for i := 0; i < int(numAdapters); i++ {
		adapters[i] = C.GoString(adaptersSlice[i])
	}

	C.free_categories(adaptersPtr, numAdapters)
	return adapters, nil
}

func convertResult(cResult *C.GenerativeClassificationResult) *ClassificationResult {
	numCats := int(cResult.num_categories)
	probs := make([]float32, numCats)
	if cResult.probabilities != nil && numCats > 0 {
		probsSlice := (*[1000]C.float)(unsafe.Pointer(cResult.probabilities))[:numCats:numCats]
		for i := 0; i < numCats; i++ {
			probs[i] = float32(probsSlice[i])
		}
	}

	return &ClassificationResult{
		ClassID:       int(cResult.class_id),
		Confidence:    float32(cResult.confidence),
		CategoryName:  C.GoString(cResult.category_name),
		Probabilities: probs,
		NumCategories: numCats,
	}
}

// ==================== Example Scenarios ====================

func demonstrateZeroShot() {
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  ZERO-SHOT CLASSIFICATION (No Adapter Required)")
	fmt.Println(strings.Repeat("=", 70))

	testCases := []struct {
		name       string
		text       string
		categories []string
	}{
		{
			name:       "Sentiment Analysis",
			text:       "This movie was absolutely fantastic! I loved every minute of it.",
			categories: []string{"positive", "negative", "neutral"},
		},
		{
			name:       "Topic Classification",
			text:       "The stock market rallied today as investors reacted to positive economic data.",
			categories: []string{"science", "politics", "sports", "business"},
		},
		{
			name:       "Intent Detection",
			text:       "What time does the store open?",
			categories: []string{"question", "command", "statement"},
		},
	}

	correct := 0
	for i, tc := range testCases {
		fmt.Printf("\n[%d/%d] %s\n", i+1, len(testCases), tc.name)
		fmt.Printf("  Text: %s\n", tc.text)
		fmt.Printf("  Categories: %v\n", tc.categories)

		result, err := ClassifyZeroShot(tc.text, tc.categories)
		if err != nil {
			log.Printf("  ❌ Error: %v\n", err)
			continue
		}

		fmt.Printf("  ✅ Result: %s (%.2f%% confidence)\n",
			result.CategoryName, result.Confidence*100)

		// Simple accuracy check
		if strings.Contains(tc.text, "fantastic") && result.CategoryName == "positive" {
			correct++
		} else if strings.Contains(tc.text, "stock market") && result.CategoryName == "business" {
			correct++
		} else if strings.Contains(tc.text, "What time") && result.CategoryName == "question" {
			correct++
		}
	}

	fmt.Printf("\n  Accuracy: %d/%d (%.1f%%)\n", correct, len(testCases),
		float64(correct)/float64(len(testCases))*100)
}

func demonstrateMultiAdapter() {
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  MULTI-ADAPTER CLASSIFICATION")
	fmt.Println(strings.Repeat("=", 70))

	// Load adapter
	adapterPath := "../../models/qwen3_generative_classifier_r16"
	if err := LoadQwen3LoRAAdapter("category", adapterPath); err != nil {
		log.Fatalf("❌ Failed to load adapter: %v", err)
	}

	// Show loaded adapters
	adapters, err := GetLoadedAdapters()
	if err != nil {
		log.Printf("❌ Failed to get loaded adapters: %v", err)
	} else {
		fmt.Printf("\n  Loaded adapters: %v\n", adapters)
	}

	// Test classification
	testTexts := []string{
		"What is the weather like today?",
		"I want to book a flight to Paris",
		"Tell me a joke about programming",
	}

	fmt.Println("\n  Testing adapter classification:")
	for i, text := range testTexts {
		fmt.Printf("\n  [%d] Text: %s\n", i+1, text)
		result, err := ClassifyWithAdapter(text, "category")
		if err != nil {
			log.Printf("    ❌ Error: %v\n", err)
			continue
		}
		fmt.Printf("    ✅ Category: %s (%.2f%% confidence)\n",
			result.CategoryName, result.Confidence*100)
	}
}

func runBenchmarkEvaluation() {
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  BENCHMARK DATASET EVALUATION")
	fmt.Println(strings.Repeat("=", 70))

	// Load test data
	dataPath := "../../bench/test_data.json"
	data, err := ioutil.ReadFile(dataPath)
	if err != nil {
		log.Printf("⚠️  Could not load benchmark data: %v (skipping)", err)
		return
	}

	var samples []TestSample
	if err := json.Unmarshal(data, &samples); err != nil {
		log.Printf("❌ Failed to parse test data: %v", err)
		return
	}

	fmt.Printf("\n  Loaded %d test samples\n", len(samples))

	// Run evaluation
	correct := 0
	startTime := time.Now()

	for i, sample := range samples {
		result, err := ClassifyWithAdapter(sample.Text, "category")
		if err != nil {
			log.Printf("  [%d] Error: %v\n", i+1, err)
			continue
		}

		if result.CategoryName == sample.TrueLabel {
			correct++
		}

		if (i+1)%10 == 0 {
			fmt.Printf("  Progress: %d/%d samples processed\n", i+1, len(samples))
		}
	}

	duration := time.Since(startTime)

	// Results
	accuracy := float64(correct) / float64(len(samples)) * 100
	avgLatency := duration.Milliseconds() / int64(len(samples))

	fmt.Printf("\n  📊 Results:\n")
	fmt.Printf("    • Accuracy: %d/%d (%.2f%%)\n", correct, len(samples), accuracy)
	fmt.Printf("    • Total time: %v\n", duration)
	fmt.Printf("    • Avg latency: %dms per sample\n", avgLatency)
}

// ==================== Main ====================

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════════╗")
	fmt.Println("║  Qwen3 Multi-LoRA Classification - Comprehensive Example  ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════╝")

	// Check for model path override
	baseModelPath := os.Getenv("BASE_MODEL_PATH")
	if baseModelPath == "" {
		baseModelPath = "../../models/Qwen3-0.6B"
	}

	// Initialize base model
	fmt.Printf("\n🔧 Initializing base model: %s\n", baseModelPath)
	if err := InitQwen3MultiLoRAClassifier(baseModelPath); err != nil {
		log.Fatalf("❌ Initialization failed: %v\n", err)
	}

	// Run demonstration scenarios
	demonstrateZeroShot()
	demonstrateMultiAdapter()
	runBenchmarkEvaluation()

	// Summary
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  ✅ All examples completed successfully!")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("\nFor more examples, see:")
	fmt.Println("  • ../../candle-binding/semantic-router_test.go (unit tests)")
	fmt.Println("  • ../../candle-binding/ZERO_SHOT_CLASSIFICATION.md (documentation)")
	fmt.Println("  • ../../candle-binding/MULTI_ADAPTER_IMPLEMENTATION.md (architecture)")
	fmt.Println()
}
