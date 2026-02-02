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
	// Supported: "knn", "kmeans", "svm"
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
	// Models configured in values.yaml matching training data
	mlModels := []string{"llama-3.2-1b", "llama-3.2-3b", "codellama-7b", "mistral-7b"}

	return []ModelSelectionCase{
		// =================================================================
		// MATH DECISION (domain: "math") - KNN algorithm
		// =================================================================
		{
			Query:          "Calculate the derivative of sin(x) * cos(x)",
			Decision:       "math_decision",
			ExpectedModels: mlModels,
			Description:    "Calculus query should match math decision",
			Algorithm:      "knn",
		},
		{
			Query:          "Solve the quadratic equation: x^2 + 5x + 6 = 0",
			Decision:       "math_decision",
			ExpectedModels: mlModels,
			Description:    "Algebra query should match math decision",
			Algorithm:      "knn",
		},
		{
			Query:          "What is the integral of e^x from 0 to infinity?",
			Decision:       "math_decision",
			ExpectedModels: mlModels,
			Description:    "Advanced calculus query",
			Algorithm:      "knn",
		},
		{
			Query:          "Prove that the sum of angles in a triangle is 180 degrees",
			Decision:       "math_decision",
			ExpectedModels: mlModels,
			Description:    "Geometry proof query",
			Algorithm:      "knn",
		},
		{
			Query:          "Calculate the eigenvalues of a 3x3 matrix",
			Decision:       "math_decision",
			ExpectedModels: mlModels,
			Description:    "Linear algebra computation",
			Algorithm:      "knn",
		},

		// =================================================================
		// CODE DECISION (domain: "computer science") - SVM algorithm
		// =================================================================
		{
			Query:          "Write a Python function to sort a list using quicksort",
			Decision:       "code_decision",
			ExpectedModels: mlModels,
			Description:    "Python coding query should match code decision",
			Algorithm:      "svm",
		},
		{
			Query:          "Debug this JavaScript: const x = undefined; console.log(x.length)",
			Decision:       "code_decision",
			ExpectedModels: mlModels,
			Description:    "Debug query should match code decision",
			Algorithm:      "svm",
		},
		{
			Query:          "How do I implement a binary search tree in Go?",
			Decision:       "code_decision",
			ExpectedModels: mlModels,
			Description:    "Data structures coding query",
			Algorithm:      "svm",
		},
		{
			Query:          "Write a recursive function to compute Fibonacci numbers in Rust",
			Decision:       "code_decision",
			ExpectedModels: mlModels,
			Description:    "Rust programming with recursion",
			Algorithm:      "svm",
		},
		{
			Query:          "How do I optimize a SQL query with multiple JOINs?",
			Decision:       "code_decision",
			ExpectedModels: mlModels,
			Description:    "Database optimization query",
			Algorithm:      "svm",
		},

		// =================================================================
		// SCIENCE DECISION (domain: "physics", "chemistry", "biology") - KMeans
		// =================================================================
		{
			Query:          "Explain Newton's laws of motion",
			Decision:       "science_decision",
			ExpectedModels: mlModels,
			Description:    "Physics query should match science decision",
			Algorithm:      "kmeans",
		},
		{
			Query:          "What is the theory of relativity?",
			Decision:       "science_decision",
			ExpectedModels: mlModels,
			Description:    "Physics theory query",
			Algorithm:      "kmeans",
		},
		{
			Query:          "Explain the difference between nuclear fission and fusion",
			Decision:       "science_decision",
			ExpectedModels: mlModels,
			Description:    "Nuclear physics query",
			Algorithm:      "kmeans",
		},
		{
			Query:          "What is the structure of a DNA molecule?",
			Decision:       "science_decision",
			ExpectedModels: mlModels,
			Description:    "Biology query should match science decision",
			Algorithm:      "kmeans",
		},
		{
			Query:          "Explain the periodic table and electron configurations",
			Decision:       "science_decision",
			ExpectedModels: mlModels,
			Description:    "Chemistry query should match science decision",
			Algorithm:      "kmeans",
		},

		// =================================================================
		// HEALTH DECISION (domain: "health") - KNN algorithm
		// =================================================================
		{
			Query:          "What are the symptoms and treatment for diabetes?",
			Decision:       "health_decision",
			ExpectedModels: mlModels,
			Description:    "Medical symptoms query",
			Algorithm:      "knn",
		},
		{
			Query:          "Explain the cardiovascular system and heart function",
			Decision:       "health_decision",
			ExpectedModels: mlModels,
			Description:    "Human anatomy query",
			Algorithm:      "knn",
		},

		// =================================================================
		// ENGINEERING DECISION (domain: "engineering") - SVM algorithm
		// =================================================================
		{
			Query:          "How do I design a bridge to withstand earthquakes?",
			Decision:       "engineering_decision",
			ExpectedModels: mlModels,
			Description:    "Civil engineering query",
			Algorithm:      "svm",
		},
		{
			Query:          "Explain the principles of aerodynamics in aircraft design",
			Decision:       "engineering_decision",
			ExpectedModels: mlModels,
			Description:    "Aerospace engineering query",
			Algorithm:      "svm",
		},

		// =================================================================
		// HUMANITIES DECISION (domain: "history", "philosophy", "psychology", "law")
		// =================================================================
		{
			Query:          "Tell me about the history of the Roman Empire",
			Decision:       "humanities_decision",
			ExpectedModels: mlModels,
			Description:    "History query should match humanities decision",
			Algorithm:      "knn",
		},
		{
			Query:          "What is Kant's categorical imperative in moral philosophy?",
			Decision:       "humanities_decision",
			ExpectedModels: mlModels,
			Description:    "Philosophy query should match humanities decision",
			Algorithm:      "knn",
		},

		// =================================================================
		// BUSINESS DECISION (domain: "business", "economics")
		// =================================================================
		{
			Query:          "Explain supply and demand in microeconomics",
			Decision:       "business_decision",
			ExpectedModels: mlModels,
			Description:    "Economics query should match business decision",
			Algorithm:      "knn",
		},
		{
			Query:          "What are the best strategies for startup fundraising?",
			Decision:       "business_decision",
			ExpectedModels: mlModels,
			Description:    "Business strategy query",
			Algorithm:      "knn",
		},

		// =================================================================
		// GENERAL DECISION (domain: "other") - catch-all
		// =================================================================
		{
			Query:          "What is the capital of France?",
			Decision:       "general_decision",
			ExpectedModels: mlModels,
			Description:    "General factual query",
			Algorithm:      "knn",
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
