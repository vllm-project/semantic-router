package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("jailbreak-detection", pkgtestcases.TestCase{
		Description: "Test jailbreak detection and blocking functionality",
		Tags:        []string{"kubernetes", "security", "jailbreak"},
		Fn:          testJailbreakDetection,
	})
}

// JailbreakTestCase represents a test case for jailbreak detection
type JailbreakTestCase struct {
	Description     string `json:"description"`
	Question        string `json:"question"`
	ExpectedBlocked bool   `json:"expected_blocked"`
}

// JailbreakResult tracks the result of a jailbreak detection test
type JailbreakResult struct {
	Description     string
	Question        string
	ExpectedBlocked bool
	ActuallyBlocked bool
	DetectedType    string
	Confidence      string
	Correct         bool
	Error           string
}

func testJailbreakDetection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing jailbreak detection functionality")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward() // Ensure port forwarding is stopped when test completes

	// Load test cases from JSON file
	testCases, err := loadJailbreakCases("e2e/testcases/testdata/jailbreak_detection_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	// Run jailbreak detection tests
	var results []JailbreakResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSingleJailbreakDetection(ctx, testCase, localPort, opts.Verbose)
		results = append(results, result)
		if result.Correct {
			correctTests++
		}
	}

	// Calculate detection rate and count blocked requests
	detectionRate := float64(correctTests) / float64(totalTests) * 100
	blockedCount := 0
	for _, result := range results {
		if result.ActuallyBlocked {
			blockedCount++
		}
	}

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":    totalTests,
			"correct_tests":  correctTests,
			"detection_rate": fmt.Sprintf("%.2f%%", detectionRate),
			"blocked_count":  blockedCount,
			"failed_tests":   totalTests - correctTests,
		})
	}

	// Print results
	printJailbreakResults(results, totalTests, correctTests, detectionRate)

	if opts.Verbose {
		fmt.Printf("[Test] Jailbreak detection test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, detectionRate)
	}

	// Return error if detection rate is 0%
	if correctTests == 0 {
		return fmt.Errorf("jailbreak detection test failed: 0%% accuracy (0/%d correct)", totalTests)
	}

	return nil
}

func loadJailbreakCases(filepath string) ([]JailbreakTestCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []JailbreakTestCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

func testSingleJailbreakDetection(ctx context.Context, testCase JailbreakTestCase, localPort string, verbose bool) JailbreakResult {
	result := JailbreakResult{
		Description:     testCase.Description,
		Question:        testCase.Question,
		ExpectedBlocked: testCase.ExpectedBlocked,
	}

	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", testCase.Question, 30*time.Second)
	if err != nil {
		result.Error = err.Error()
		return result
	}

	if response.StatusCode != http.StatusOK {
		result.Error = formatUnexpectedChatCompletionStatus(response)
		logUnexpectedChatCompletionStatus(verbose, response, testCase.Description,
			"Question: "+testCase.Question)
		return result
	}

	fastResponse := response.Headers.Get("x-vsr-fast-response")
	selectedDecision := response.Headers.Get("x-vsr-selected-decision")
	matchedJailbreak := response.Headers.Get("x-vsr-matched-jailbreak")
	jailbreakBlockedLegacy := response.Headers.Get("x-vsr-jailbreak-blocked")
	result.ActuallyBlocked = fastResponse == "true" || jailbreakBlockedLegacy == "true"
	result.DetectedType = selectedDecision
	if result.DetectedType == "" {
		result.DetectedType = matchedJailbreak
	}
	if result.DetectedType == "" {
		result.DetectedType = response.Headers.Get("x-vsr-jailbreak-type")
	}
	result.Confidence = response.Headers.Get("x-vsr-jailbreak-confidence")

	if result.ActuallyBlocked {
		bodyStr := string(response.Body)
		hasExpectedText := strings.Contains(bodyStr, "jailbreak attempt") ||
			strings.Contains(bodyStr, "jailbreak") ||
			strings.Contains(bodyStr, "security") ||
			fastResponse == "true"
		if !hasExpectedText {
			result.Error = "Jailbreak blocked but response message doesn't contain expected text"
		}
	}

	result.Correct = result.ActuallyBlocked == result.ExpectedBlocked

	if verbose {
		if result.Correct {
			if result.ActuallyBlocked {
				fmt.Printf("[Test] ✓ Correct: %s (blocked, type=%s, confidence=%s)\n",
					testCase.Description, result.DetectedType, result.Confidence)
			} else {
				fmt.Printf("[Test] ✓ Correct: %s (not blocked)\n", testCase.Description)
			}
		} else {
			fmt.Printf("[Test] ✗ Incorrect: %s (expected blocked=%v, actual=%v)\n",
				testCase.Description, result.ExpectedBlocked, result.ActuallyBlocked)
		}
	}

	return result
}

func printJailbreakResults(results []JailbreakResult, totalTests, correctTests int, blockRate float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("JAILBREAK DETECTION TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correctly Detected: %d\n", correctTests)
	fmt.Printf("Detection Accuracy: %.2f%%\n", blockRate)
	fmt.Println(separator)

	// Count blocked vs not blocked
	blockedCount := 0
	for _, result := range results {
		if result.ActuallyBlocked {
			blockedCount++
		}
	}
	fmt.Printf("\nBlocked Requests: %d/%d\n", blockedCount, totalTests)

	// Print blocked attacks with details
	blockedAttacks := 0
	for _, result := range results {
		if result.ActuallyBlocked {
			blockedAttacks++
		}
	}

	if blockedAttacks > 0 {
		fmt.Println("\nBlocked Attacks (with details):")
		for _, result := range results {
			if result.ActuallyBlocked {
				fmt.Printf("  - %s\n", result.Description)
				fmt.Printf("    Type: %s, Confidence: %s\n", result.DetectedType, result.Confidence)
			}
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
		fmt.Println("\nFailed Detections:")
		for _, result := range results {
			if !result.Correct && result.Error == "" {
				fmt.Printf("  - %s\n", result.Description)
				fmt.Printf("    Question: %s\n", result.Question)
				fmt.Printf("    Expected blocked: %v, Actually blocked: %v\n",
					result.ExpectedBlocked, result.ActuallyBlocked)
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
				fmt.Printf("  - %s\n", result.Description)
				fmt.Printf("    Error: %s\n", result.Error)
			}
		}
	}

	fmt.Println(separator + "\n")
}
