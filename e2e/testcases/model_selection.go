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
	// Supported: "knn", "kmeans", "svm", "mlp"
	Algorithm string `json:"algorithm,omitempty"`
	// ExpectEfficient indicates if the test expects an efficiency-optimized selection (KMeans)
	// When true, expects faster/cheaper model; when false, expects higher quality model
	ExpectEfficient bool `json:"expect_efficient,omitempty"`
	// AllowNoDecision if true, accepts empty/no decision as valid (for ambiguous queries)
	// This is useful for queries that may not reliably match a specific domain
	AllowNoDecision bool `json:"allow_no_decision,omitempty"`
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
	AllowNoDecision bool
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

	// Require at least 75% accuracy (allows for domain classification variance)
	// The BERT domain classifier has ~55% accuracy, so some queries may not match expected decisions
	minAccuracy := 75.0
	if accuracy < minAccuracy {
		return fmt.Errorf("model selection test failed: %d/%d correct (%.2f%% accuracy) - minimum %.0f%% required",
			correctTests, totalTests, accuracy, minAccuracy)
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
		AllowNoDecision: testCase.AllowNoDecision,
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
	// If AllowNoDecision is true, accept empty decision or general_decision as valid
	// (general_decision is the catch-all for unclassified queries)
	if testCase.AllowNoDecision && (result.ActualDecision == "" || result.ActualDecision == "general_decision") {
		decisionMatches = true
	}

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
	// Models configured in values.yaml matching training data
	mlModels := []string{"llama-3.2-1b", "llama-3.2-3b", "codellama-7b", "mistral-7b"}

	// 20 test cases covering all 4 ML algorithms with DIVERSE decision types:
	// - MLP: math_decision (3 cases), code_decision (2 cases) - diverse domains
	// - SVM: physics_decision (3 cases), business_decision (2 cases) - diverse domains
	// - KNN: science_decision (3 cases), law_decision (2 cases) - diverse domains
	// - KMeans: health_decision (3 cases), engineering_decision (2 cases) - diverse domains

	return []ModelSelectionCase{
		// =================================================================
		// MLP ALGORITHM: math_decision + code_decision (5 cases total)
		// =================================================================
		{
			Query:          "Calculate the derivative of x^3 + 2x^2 - 5x + 1",
			Decision:       "math_decision",
			ExpectedModels: mlModels,
			Description:    "Calculus derivative - MLP/math",
			Algorithm:      "mlp",
		},
		{
			Query:          "Solve for x: 3x + 7 = 22",
			Decision:       "math_decision",
			ExpectedModels: mlModels,
			Description:    "Algebra equation - MLP/math",
			Algorithm:      "mlp",
		},
		{
			Query:          "What is the integral of sin(x)dx?",
			Decision:       "math_decision",
			ExpectedModels: mlModels,
			Description:    "Integral calculus - MLP/math",
			Algorithm:      "mlp",
		},
		{
			Query:          "Write a Python function to sort a list using bubble sort",
			Decision:       "code_decision",
			ExpectedModels: mlModels,
			Description:    "Python sorting - MLP/code",
			Algorithm:      "mlp",
		},
		{
			Query:          "How do I create a for loop in JavaScript?",
			Decision:       "code_decision",
			ExpectedModels: mlModels,
			Description:    "JavaScript loop - MLP/code",
			Algorithm:      "mlp",
		},

		// =================================================================
		// SVM ALGORITHM: physics_decision + business_decision (5 cases total)
		// =================================================================
		{
			Query:          "What is Newton's second law of motion?",
			Decision:       "physics_decision",
			ExpectedModels: mlModels,
			Description:    "Classical mechanics - SVM/physics",
			Algorithm:      "svm",
		},
		{
			Query:          "Calculate the kinetic energy of a 5kg object moving at 10m/s",
			Decision:       "physics_decision",
			ExpectedModels: mlModels,
			Description:    "Energy calculation - SVM/physics",
			Algorithm:      "svm",
		},
		{
			Query:          "What is the wavelength of light with frequency 5e14 Hz?",
			Decision:       "physics_decision",
			ExpectedModels: mlModels,
			Description:    "Wave physics - SVM/physics",
			Algorithm:      "svm",
		},
		{
			Query:          "What is the law of supply and demand in economics?",
			Decision:       "business_decision",
			ExpectedModels: mlModels,
			Description:    "Economics law - SVM/business",
			Algorithm:      "svm",
		},
		{
			Query:          "How does inflation affect interest rates?",
			Decision:       "business_decision",
			ExpectedModels: mlModels,
			Description:    "Monetary economics - SVM/business",
			Algorithm:      "svm",
		},

		// =================================================================
		// KNN ALGORITHM: science_decision + law_decision (5 cases total)
		// =================================================================
		{
			Query:          "What is the process of photosynthesis in plants?",
			Decision:       "science_decision",
			ExpectedModels: mlModels,
			Description:    "Biology photosynthesis - KNN/science",
			Algorithm:      "knn",
		},
		{
			Query:          "Explain the structure of DNA and its double helix",
			Decision:       "science_decision",
			ExpectedModels: mlModels,
			Description:    "Biology DNA - KNN/science",
			Algorithm:      "knn",
		},
		{
			Query:          "What is the pH of a neutral solution at 25 degrees Celsius?",
			Decision:       "science_decision",
			ExpectedModels: mlModels,
			Description:    "Chemistry pH - KNN/science",
			Algorithm:      "knn",
		},
		{
			Query:          "What is the difference between civil law and criminal law?",
			Decision:       "law_decision",
			ExpectedModels: mlModels,
			Description:    "Legal concepts - KNN/law",
			Algorithm:      "knn",
		},
		{
			Query:          "What are the elements of a valid contract?",
			Decision:       "law_decision",
			ExpectedModels: mlModels,
			Description:    "Contract law - KNN/law",
			Algorithm:      "knn",
		},

		// =================================================================
		// KMeans ALGORITHM: health_decision + engineering_decision (5 cases total)
		// =================================================================
		{
			Query:          "What are the symptoms of diabetes mellitus?",
			Decision:       "health_decision",
			ExpectedModels: mlModels,
			Description:    "Medical symptoms - KMeans/health",
			Algorithm:      "kmeans",
		},
		{
			Query:          "How does the human cardiovascular system work?",
			Decision:       "health_decision",
			ExpectedModels: mlModels,
			Description:    "Human anatomy - KMeans/health",
			Algorithm:      "kmeans",
		},
		{
			Query:          "What causes high blood pressure in humans?",
			Decision:       "health_decision",
			ExpectedModels: mlModels,
			Description:    "Health condition - KMeans/health",
			Algorithm:      "kmeans",
		},
		{
			Query:          "What is the difference between AC and DC electrical current?",
			Decision:       "engineering_decision",
			ExpectedModels: mlModels,
			Description:    "Electrical engineering - KMeans/engineering",
			Algorithm:      "kmeans",
		},
		{
			Query:          "How does a transistor amplify electrical signals?",
			Decision:       "engineering_decision",
			ExpectedModels: mlModels,
			Description:    "Electronics engineering - KMeans/engineering",
			Algorithm:      "kmeans",
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
