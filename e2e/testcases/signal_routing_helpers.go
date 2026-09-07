package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

// SignalRoutingCase is a single request/expectation pair for a signal that is
// evaluated from one standalone request (e.g. event, language): the signal
// either matches a configured rule on that request or it doesn't.
type SignalRoutingCase struct {
	Name                  string `json:"name"`
	Description           string `json:"description"`
	Query                 string `json:"query"`
	ExpectedDecision      string `json:"expected_decision"`
	ExpectedMatchedSignal string `json:"expected_matched_signal"`
	ShouldMatch           bool   `json:"should_match"`
}

// SignalRoutingResult tracks the result of a single SignalRoutingCase.
type SignalRoutingResult struct {
	Name                  string
	Query                 string
	ExpectedDecision      string
	ActualDecision        string
	ExpectedMatchedSignal string
	ActualMatchedSignal   string
	ShouldMatch           bool
	DecisionCorrect       bool
	MatchCorrect          bool
	Error                 string
}

// signalRoutingConfig parameterizes runSignalRoutingTest for one signal.
type signalRoutingConfig struct {
	// TestDataPath is the JSON file of SignalRoutingCase entries.
	TestDataPath string
	// MatchedHeader is the response header carrying the matched rule name
	// (e.g. "x-vsr-matched-event", "x-vsr-matched-language").
	MatchedHeader string
	// TargetDecision is the decision this signal's rule routes to; used to
	// check negative cases without pinning the exact fallback decision.
	TargetDecision string
	// ResultsTitle is printed as the results-table header.
	ResultsTitle string
	// LogLabel names the signal in progress/summary log lines (e.g. "Event").
	LogLabel string
}

func runSignalRoutingTest(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions, cfg signalRoutingConfig) error {
	if opts.Verbose {
		fmt.Printf("[Test] Testing %s signal routing\n", strings.ToLower(cfg.LogLabel))
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	testCases, err := loadSignalRoutingCases(cfg.TestDataPath)
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	var results []SignalRoutingResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSingleSignalRouting(ctx, testCase, localPort, opts.Verbose, cfg)
		results = append(results, result)
		if result.DecisionCorrect && result.MatchCorrect {
			correctTests++
		}
	}

	accuracy := float64(correctTests) / float64(totalTests) * 100

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":   totalTests,
			"correct_tests": correctTests,
			"accuracy_rate": fmt.Sprintf("%.2f%%", accuracy),
			"failed_tests":  totalTests - correctTests,
		})
	}

	printSignalRoutingResults(cfg.ResultsTitle, results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] %s routing test completed: %d/%d correct (%.2f%% accuracy)\n",
			cfg.LogLabel, correctTests, totalTests, accuracy)
	}

	if correctTests != totalTests {
		return fmt.Errorf("%s routing test failed: %d/%d correct", strings.ToLower(cfg.LogLabel), correctTests, totalTests)
	}

	return nil
}

func loadSignalRoutingCases(filepath string) ([]SignalRoutingCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []SignalRoutingCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

func testSingleSignalRouting(ctx context.Context, testCase SignalRoutingCase, localPort string, verbose bool, cfg signalRoutingConfig) SignalRoutingResult {
	result := SignalRoutingResult{
		Name:                  testCase.Name,
		Query:                 testCase.Query,
		ExpectedDecision:      testCase.ExpectedDecision,
		ExpectedMatchedSignal: testCase.ExpectedMatchedSignal,
		ShouldMatch:           testCase.ShouldMatch,
	}

	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", testCase.Query, 30*time.Second)
	if err != nil {
		result.Error = err.Error()
		return result
	}

	if response.StatusCode != http.StatusOK {
		result.Error = formatUnexpectedChatCompletionStatus(response)
		logUnexpectedChatCompletionStatus(verbose, response, "test case: "+testCase.Name,
			"Query: "+testCase.Query,
			"Should match: "+fmt.Sprintf("%v", testCase.ShouldMatch))
		return result
	}

	decision := response.Headers.Get("x-vsr-selected-decision")
	result.ActualDecision = strings.TrimSuffix(decision, "_decision")
	result.ActualMatchedSignal = response.Headers.Get(cfg.MatchedHeader)

	if testCase.ShouldMatch {
		result.DecisionCorrect = result.ActualDecision == testCase.ExpectedDecision
		result.MatchCorrect = result.ActualMatchedSignal == testCase.ExpectedMatchedSignal
	} else {
		result.DecisionCorrect = result.ActualDecision != cfg.TargetDecision
		result.MatchCorrect = result.ActualMatchedSignal == ""
	}

	if verbose && (!result.DecisionCorrect || !result.MatchCorrect) {
		fmt.Printf("[Test] Test case failed: %s\n", testCase.Name)
		if !result.DecisionCorrect {
			fmt.Printf("  Decision mismatch: query='%s', expected=%s, actual=%s\n",
				testCase.Query, testCase.ExpectedDecision, result.ActualDecision)
		}
		if !result.MatchCorrect {
			fmt.Printf("  Matched-signal mismatch: expected=%q, actual=%q\n",
				testCase.ExpectedMatchedSignal, result.ActualMatchedSignal)
		}
	}

	return result
}

func printSignalRoutingResults(title string, results []SignalRoutingResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println(title)
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correct: %d (%.2f%%)\n", correctTests, accuracy)
	fmt.Println(separator)

	for _, result := range results {
		if result.Error != "" {
			fmt.Printf("  - Test: %s\n    Query: %s\n    Error: %s\n", result.Name, result.Query, result.Error)
			continue
		}
		if !result.DecisionCorrect || !result.MatchCorrect {
			fmt.Printf("  - Test: %s\n    Query: %s\n    Expected decision: %s, actual: %s\n    Expected matched signal: %q, actual: %q\n",
				result.Name, result.Query, result.ExpectedDecision, result.ActualDecision,
				result.ExpectedMatchedSignal, result.ActualMatchedSignal)
		}
	}

	fmt.Println(separator + "\n")
}
