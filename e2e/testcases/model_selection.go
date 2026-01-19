package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("model-selection", pkgtestcases.TestCase{
		Description: "Test ML-based model selection from multiple models within a decision",
		Tags:        []string{"model-selection", "signal-decision", "routing", "ml"},
		Fn:          testModelSelection,
	})
}

// ModelSelectionCase represents a test case for ML-based model selection
type ModelSelectionCase struct {
	Query          string   `json:"query"`
	Decision       string   `json:"decision"`        // Expected decision to match
	ExpectedModels []string `json:"expected_models"` // List of valid models for this query
	Description    string   `json:"description"`
	// Algorithm specifies which model selection algorithm is expected to be used
	// Supported: "knn", "kmeans", "mlp", "svm", "matrix_factorization"
	Algorithm string `json:"algorithm,omitempty"`
	// ExpectEfficient indicates if the test expects an efficiency-optimized selection (KMeans)
	// When true, expects faster/cheaper model; when false, expects higher quality model
	ExpectEfficient bool `json:"expect_efficient,omitempty"`
}

// ModelSelectionResult tracks the result of a single model selection test
type ModelSelectionResult struct {
	Query           string
	Decision        string
	ActualDecision  string
	SelectedModel   string
	ExpectedModels  []string
	Algorithm       string
	ExpectEfficient bool
	Correct         bool
	Error           string
}

func testModelSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing ML-based model selection from multiple models")
		fmt.Println("[Test] This test verifies that when a decision has multiple models,")
		fmt.Println("[Test] the ModelSelectionAlgorithm correctly selects the appropriate model.")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Load test cases from JSON file or use default
	testCases, err := loadModelSelectionCases("e2e/testcases/testdata/model_selection_cases.json")
	if err != nil {
		if opts.Verbose {
			fmt.Printf("[Test] Could not load test cases from file: %v, using default cases\n", err)
		}
		testCases = getDefaultModelSelectionCases()
	}

	// Run model selection tests
	var results []ModelSelectionResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSingleModelSelection(ctx, testCase, localPort, opts.Verbose)
		results = append(results, result)
		if result.Correct {
			correctTests++
		}
	}

	// Calculate accuracy
	accuracy := float64(correctTests) / float64(totalTests) * 100

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":   totalTests,
			"correct_tests": correctTests,
			"accuracy_rate": fmt.Sprintf("%.2f%%", accuracy),
			"failed_tests":  totalTests - correctTests,
		})
	}

	// Print results
	printModelSelectionResults(results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Model selection test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy)
	}

	// Return error if accuracy is 0%
	if correctTests == 0 {
		return fmt.Errorf("model selection test failed: 0%% accuracy (0/%d correct)", totalTests)
	}

	return nil
}

func testSingleModelSelection(ctx context.Context, testCase ModelSelectionCase, localPort string, verbose bool) ModelSelectionResult {
	result := ModelSelectionResult{
		Query:           testCase.Query,
		Decision:        testCase.Decision,
		ExpectedModels:  testCase.ExpectedModels,
		Algorithm:       testCase.Algorithm,
		ExpectEfficient: testCase.ExpectEfficient,
	}

	// Create chat completion request with MoM (Mixture of Models) to trigger decision engine
	requestBody := map[string]interface{}{
		"model": "MoM", // Use MoM to trigger auto model selection
		"messages": []map[string]string{
			{"role": "user", "content": testCase.Query},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		result.Error = fmt.Sprintf("failed to marshal request: %v", err)
		return result
	}

	// Send request
	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		result.Error = fmt.Sprintf("failed to create request: %v", err)
		return result
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 60 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		result.Error = fmt.Sprintf("failed to send request: %v", err)
		return result
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		result.Error = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(bodyBytes))

		if verbose {
			fmt.Printf("[Test] ✗ HTTP %d Error for query: %s\n", resp.StatusCode, truncateString(testCase.Query, 50))
			fmt.Printf("  Expected decision: %s\n", testCase.Decision)
			fmt.Printf("  Response: %s\n", truncateString(string(bodyBytes), 200))
		}

		return result
	}

	// Extract VSR headers to verify model selection
	result.ActualDecision = resp.Header.Get("x-vsr-selected-decision")
	result.SelectedModel = resp.Header.Get("x-vsr-selected-model")

	// Verify the decision matches
	decisionMatches := result.ActualDecision == testCase.Decision

	// Verify the selected model is one of the expected models
	modelValid := false
	for _, expectedModel := range testCase.ExpectedModels {
		if result.SelectedModel == expectedModel {
			modelValid = true
			break
		}
	}

	// For the test to pass, both decision and model must be correct
	// If no expected models are specified, just check the decision
	if len(testCase.ExpectedModels) == 0 {
		result.Correct = decisionMatches
	} else {
		result.Correct = decisionMatches && modelValid
	}

	if verbose {
		algoInfo := ""
		if testCase.Algorithm != "" {
			algoInfo = fmt.Sprintf(" [%s]", testCase.Algorithm)
			if testCase.ExpectEfficient {
				algoInfo += " (efficiency-optimized)"
			}
		}

		if result.Correct {
			fmt.Printf("[Test] ✓ Model selection correct%s\n", algoInfo)
			fmt.Printf("  Query: %s\n", truncateString(testCase.Query, 60))
			fmt.Printf("  Decision: %s\n", result.ActualDecision)
			fmt.Printf("  Selected Model: %s\n", result.SelectedModel)
		} else {
			fmt.Printf("[Test] ✗ Model selection incorrect%s\n", algoInfo)
			fmt.Printf("  Query: %s\n", truncateString(testCase.Query, 60))
			fmt.Printf("  Expected Decision: %s, Got: %s\n", testCase.Decision, result.ActualDecision)
			fmt.Printf("  Selected Model: %s\n", result.SelectedModel)
			fmt.Printf("  Expected Models: %v\n", testCase.ExpectedModels)
		}
	}

	return result
}

func loadModelSelectionCases(filepath string) ([]ModelSelectionCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, err
	}

	var cases []ModelSelectionCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, err
	}

	return cases, nil
}

func getDefaultModelSelectionCases() []ModelSelectionCase {
	return []ModelSelectionCase{
		// Math decision with multiple specialized models (KNN algorithm)
		{
			Query:          "Calculate the derivative of sin(x) * cos(x)",
			Decision:       "math_decision",
			ExpectedModels: []string{"gpt-4-turbo", "deepseek-math", "llama-math"},
			Description:    "Calculus query should match math decision and select appropriate math model",
			Algorithm:      "knn",
		},
		{
			Query:          "Solve the quadratic equation: x^2 + 5x + 6 = 0",
			Decision:       "math_decision",
			ExpectedModels: []string{"gpt-4-turbo", "deepseek-math", "llama-math"},
			Description:    "Algebra query should match math decision",
			Algorithm:      "knn",
		},
		{
			Query:          "What is the integral of e^x from 0 to infinity?",
			Decision:       "math_decision",
			ExpectedModels: []string{"gpt-4-turbo", "deepseek-math", "llama-math"},
			Description:    "Advanced calculus query",
			Algorithm:      "mlp",
		},
		// Code decision with multiple models (SVM algorithm)
		{
			Query:          "Write a Python function to sort a list using quicksort",
			Decision:       "code_decision",
			ExpectedModels: []string{"claude-3-opus", "gpt-4-turbo", "codellama"},
			Description:    "Python coding query should match code decision",
			Algorithm:      "svm",
		},
		{
			Query:          "Debug this JavaScript: const x = undefined; console.log(x.length)",
			Decision:       "code_decision",
			ExpectedModels: []string{"claude-3-opus", "gpt-4-turbo", "codellama"},
			Description:    "Debug query should match code decision",
			Algorithm:      "svm",
		},
		// Reasoning decision with models (KNN algorithm)
		{
			Query:          "Think step by step: If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
			Decision:       "thinking_decision",
			ExpectedModels: []string{"deepseek-v3", "gpt-4-turbo", "llama-70b"},
			Description:    "Logical reasoning query requiring step-by-step thinking",
			Algorithm:      "knn",
		},
		// Creative decision (Matrix Factorization - RouteLLM style)
		{
			Query:          "Write a short poem about the beauty of autumn leaves",
			Decision:       "creative_decision",
			ExpectedModels: []string{"gpt-4", "claude-3-opus"},
			Description:    "Creative writing query",
			Algorithm:      "matrix_factorization",
		},
		// Factual decision with efficiency optimization (KMeans - Avengers-Pro style)
		{
			Query:           "What is the capital of France?",
			Decision:        "factual_decision",
			ExpectedModels:  []string{"gpt-3.5-turbo", "gpt-4"},
			Description:     "Simple factual query - fast model preferred for efficiency",
			Algorithm:       "kmeans",
			ExpectEfficient: true,
		},
	}
}

func printModelSelectionResults(results []ModelSelectionResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("ML-BASED MODEL SELECTION TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correct Selections: %d\n", correctTests)
	fmt.Printf("Accuracy Rate: %.2f%%\n", accuracy)
	fmt.Println(separator)

	// Print model selection summary
	modelCounts := make(map[string]int)
	for _, result := range results {
		if result.SelectedModel != "" {
			modelCounts[result.SelectedModel]++
		}
	}

	if len(modelCounts) > 0 {
		fmt.Println("\nModel Selection Distribution:")
		for model, count := range modelCounts {
			fmt.Printf("  %s: %d selections\n", model, count)
		}
	}

	// Print failed cases
	failedCount := 0
	for _, result := range results {
		if !result.Correct && result.Error == "" {
			failedCount++
		}
	}

	if failedCount > 0 {
		fmt.Println("\nFailed Selections:")
		for _, result := range results {
			if !result.Correct && result.Error == "" {
				fmt.Printf("  - Query: %s\n", truncateString(result.Query, 60))
				fmt.Printf("    Expected Decision: %s, Got: %s\n", result.Decision, result.ActualDecision)
				fmt.Printf("    Selected Model: %s\n", result.SelectedModel)
				fmt.Printf("    Expected Models: %v\n", result.ExpectedModels)
			}
		}
	}

	// Print errors
	errorCount := 0
	for _, result := range results {
		if result.Error != "" {
			errorCount++
		}
	}

	if errorCount > 0 {
		fmt.Println("\nErrors:")
		for _, result := range results {
			if result.Error != "" {
				fmt.Printf("  - Query: %s\n", truncateString(result.Query, 60))
				fmt.Printf("    Error: %s\n", result.Error)
			}
		}
	}

	fmt.Println(separator + "\n")
}
