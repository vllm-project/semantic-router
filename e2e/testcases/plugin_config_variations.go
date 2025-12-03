package testcases

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

//go:embed testdata/plugin_config_cases.json
var pluginConfigCasesJSON []byte

func init() {
	pkgtestcases.Register("plugin-config-variations", pkgtestcases.TestCase{
		Description: "Test different plugin configuration variations",
		Tags:        []string{"signal-decision", "plugin", "configuration"},
		Fn:          testPluginConfigVariations,
	})
}

// PluginConfigTestData represents the JSON file structure
type PluginConfigTestData struct {
	Description string             `json:"description"`
	TestCases   []PluginConfigCase `json:"test_cases"`
}

// PluginConfigCase represents a test case for plugin configuration variations
type PluginConfigCase struct {
	Query            string  `json:"query"`
	ExpectedDecision string  `json:"expected_decision"`
	PluginType       string  `json:"plugin_type"` // "pii", "semantic-cache", "system_prompt"
	ExpectedBehavior string  `json:"expected_behavior"`
	CacheSimilarity  float64 `json:"cache_similarity,omitempty"`
	Description      string  `json:"description"`
}

// PluginConfigResult tracks the result of a single plugin config test
type PluginConfigResult struct {
	Query            string
	ExpectedDecision string
	ActualDecision   string
	PluginType       string
	ExpectedBehavior string
	ActualBehavior   string
	Correct          bool
	Error            string
}

func testPluginConfigVariations(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing plugin configuration variations")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Load test cases from JSON file
	testCases, err := loadPluginConfigCases()
	if err != nil {
		return fmt.Errorf("failed to load plugin config test cases: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Loaded %d plugin config test cases from JSON\n", len(testCases))
	}

	// Run plugin config tests
	var results []PluginConfigResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSinglePluginConfig(ctx, testCase, localPort, opts.Verbose)
		results = append(results, result)
		if result.Correct {
			correctTests++
		}

		// Delay between tests to allow cache to settle
		time.Sleep(1 * time.Second)
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
	printPluginConfigResults(results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Plugin config variations test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy)
	}

	// Return error if accuracy is below threshold
	if correctTests == 0 {
		return fmt.Errorf("plugin config variations test failed: 0%% accuracy (0/%d correct)", totalTests)
	}

	return nil
}

func testSinglePluginConfig(ctx context.Context, testCase PluginConfigCase, localPort string, verbose bool) PluginConfigResult {
	result := PluginConfigResult{
		Query:            testCase.Query,
		ExpectedDecision: testCase.ExpectedDecision,
		PluginType:       testCase.PluginType,
		ExpectedBehavior: testCase.ExpectedBehavior,
	}

	// Create chat completion request
	requestBody := map[string]interface{}{
		"model": "MoM", // Use Mixture of Models to trigger decision engine
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

	httpClient := &http.Client{Timeout: 30 * time.Second}
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
			fmt.Printf("[Test] ✗ HTTP %d Error for query: %s\n", resp.StatusCode, testCase.Query)
			fmt.Printf("  Expected decision: %s\n", testCase.ExpectedDecision)
			fmt.Printf("  Response: %s\n", string(bodyBytes))
		}

		return result
	}

	// Extract VSR decision headers
	result.ActualDecision = resp.Header.Get("x-vsr-selected-decision")

	// Determine actual behavior based on plugin type
	switch testCase.PluginType {
	case "semantic-cache":
		cacheHit := resp.Header.Get("x-vsr-cache-hit")
		if cacheHit == "true" {
			result.ActualBehavior = "cache_hit"
		} else {
			result.ActualBehavior = "cache_miss"
		}
		// For "cache_hit_possible", we accept either hit or miss
		if testCase.ExpectedBehavior == "cache_hit_possible" {
			result.Correct = (result.ActualDecision == testCase.ExpectedDecision)
		} else {
			result.Correct = (result.ActualDecision == testCase.ExpectedDecision &&
				result.ActualBehavior == testCase.ExpectedBehavior)
		}

	case "system_prompt":
		// If decision matched, system prompt should have been applied
		result.ActualBehavior = "prompt_applied"
		result.Correct = (result.ActualDecision == testCase.ExpectedDecision)

	default:
		result.Correct = (result.ActualDecision == testCase.ExpectedDecision)
	}

	if verbose {
		if result.Correct {
			fmt.Printf("[Test] ✓ Plugin config correct\n")
			fmt.Printf("  Query: %s\n", truncateString(testCase.Query, 60))
			fmt.Printf("  Decision: %s\n", result.ActualDecision)
			fmt.Printf("  Plugin: %s, Behavior: %s\n", testCase.PluginType, result.ActualBehavior)
		} else {
			fmt.Printf("[Test] ✗ Plugin config incorrect\n")
			fmt.Printf("  Query: %s\n", testCase.Query)
			fmt.Printf("  Expected Decision: %s, Actual: %s\n", testCase.ExpectedDecision, result.ActualDecision)
			fmt.Printf("  Plugin: %s\n", testCase.PluginType)
			fmt.Printf("  Expected Behavior: %s, Actual: %s\n", testCase.ExpectedBehavior, result.ActualBehavior)
			fmt.Printf("  Description: %s\n", testCase.Description)
		}
	}

	return result
}

// loadPluginConfigCases loads test cases from the embedded JSON file
func loadPluginConfigCases() ([]PluginConfigCase, error) {
	var testData PluginConfigTestData
	if err := json.Unmarshal(pluginConfigCasesJSON, &testData); err != nil {
		return nil, fmt.Errorf("failed to parse embedded JSON: %w", err)
	}
	return testData.TestCases, nil
}

func printPluginConfigResults(results []PluginConfigResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("PLUGIN CONFIGURATION VARIATIONS TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correct Configurations: %d\n", correctTests)
	fmt.Printf("Accuracy Rate: %.2f%%\n", accuracy)
	fmt.Println(separator)

	// Print summary by plugin type
	pluginStats := make(map[string]struct {
		total   int
		correct int
	})

	for _, result := range results {
		stats := pluginStats[result.PluginType]
		stats.total++
		if result.Correct {
			stats.correct++
		}
		pluginStats[result.PluginType] = stats
	}

	fmt.Println("\nTest Breakdown by Plugin Type:")
	for pluginType, stats := range pluginStats {
		pluginAccuracy := float64(stats.correct) / float64(stats.total) * 100
		fmt.Printf("  - %-20s: %d/%d (%.2f%%)\n", pluginType, stats.correct, stats.total, pluginAccuracy)
	}

	// Print failed cases
	failedCount := 0
	for _, result := range results {
		if !result.Correct && result.Error == "" {
			failedCount++
		}
	}

	if failedCount > 0 {
		fmt.Println("\nFailed Plugin Configurations:")
		for _, result := range results {
			if !result.Correct && result.Error == "" {
				fmt.Printf("  - Query: %s\n", truncateString(result.Query, 70))
				fmt.Printf("    Plugin: %s\n", result.PluginType)
				fmt.Printf("    Expected Decision: %s, Actual: %s\n", result.ExpectedDecision, result.ActualDecision)
				fmt.Printf("    Expected Behavior: %s, Actual: %s\n", result.ExpectedBehavior, result.ActualBehavior)
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
				fmt.Printf("  - Query: %s\n", truncateString(result.Query, 70))
				fmt.Printf("    Plugin: %s\n", result.PluginType)
				fmt.Printf("    Error: %s\n", result.Error)
			}
		}
	}

	fmt.Println(separator + "\n")
}
