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

// targetReaskDecision is the decision configured to route on the
// repeated_billing_question reask rule (see e2e/profiles/ai-gateway/values.yaml).
const targetReaskDecision = "reask_escalation"

func init() {
	pkgtestcases.Register("reask-routing", pkgtestcases.TestCase{
		Description: "Test reask signal rule matching and routing",
		Tags:        []string{"kubernetes", "routing", "reask"},
		Fn:          testReaskRouting,
	})
}

// ReaskRoutingCase represents a test case for reask signal routing. Unlike
// the single-turn signals (event, language), reask needs prior-turn context,
// so a case carries a full conversation rather than one query string.
type ReaskRoutingCase struct {
	Name                  string              `json:"name"`
	Description           string              `json:"description"`
	Messages              []map[string]string `json:"messages"`
	ExpectedDecision      string              `json:"expected_decision"`
	ExpectedMatchedSignal string              `json:"expected_matched_signal"`
	ShouldMatch           bool                `json:"should_match"`
}

// ReaskRoutingResult tracks the result of a single ReaskRoutingCase.
type ReaskRoutingResult struct {
	Name                  string
	ExpectedDecision      string
	ActualDecision        string
	ExpectedMatchedSignal string
	ActualMatchedSignal   string
	ShouldMatch           bool
	DecisionCorrect       bool
	MatchCorrect          bool
	Error                 string
}

func testReaskRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing reask signal routing")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	testCases, err := loadReaskRoutingCases("e2e/testcases/testdata/reask_routing_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	var results []ReaskRoutingResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSingleReaskRouting(ctx, testCase, localPort, opts.Verbose)
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

	printReaskRoutingResults(results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Reask routing test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy)
	}

	if correctTests != totalTests {
		return fmt.Errorf("reask routing test failed: %d/%d correct", correctTests, totalTests)
	}

	return nil
}

func loadReaskRoutingCases(filepath string) ([]ReaskRoutingCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []ReaskRoutingCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

func testSingleReaskRouting(ctx context.Context, testCase ReaskRoutingCase, localPort string, verbose bool) ReaskRoutingResult {
	result := ReaskRoutingResult{
		Name:                  testCase.Name,
		ExpectedDecision:      testCase.ExpectedDecision,
		ExpectedMatchedSignal: testCase.ExpectedMatchedSignal,
		ShouldMatch:           testCase.ShouldMatch,
	}

	// testCase.Messages is a full multi-turn conversation (the reask signal
	// needs prior-turn context, unlike the single-query event/language cases).
	response, err := sendLocalChatConversation(ctx, localPort, "MoM", testCase.Messages, 30*time.Second)
	if err != nil {
		result.Error = err.Error()
		return result
	}

	if response.StatusCode != http.StatusOK {
		result.Error = formatUnexpectedChatCompletionStatus(response)
		logUnexpectedChatCompletionStatus(verbose, response, "test case: "+testCase.Name,
			"Should match: "+fmt.Sprintf("%v", testCase.ShouldMatch))
		return result
	}

	decision := response.Headers.Get("x-vsr-selected-decision")
	result.ActualDecision = strings.TrimSuffix(decision, "_decision")
	result.ActualMatchedSignal = response.Headers.Get("x-vsr-matched-reask")

	if testCase.ShouldMatch {
		result.DecisionCorrect = result.ActualDecision == testCase.ExpectedDecision
		result.MatchCorrect = result.ActualMatchedSignal == testCase.ExpectedMatchedSignal
	} else {
		result.DecisionCorrect = result.ActualDecision != targetReaskDecision
		result.MatchCorrect = result.ActualMatchedSignal == ""
	}

	if verbose && (!result.DecisionCorrect || !result.MatchCorrect) {
		fmt.Printf("[Test] Test case failed: %s\n", testCase.Name)
		if !result.DecisionCorrect {
			fmt.Printf("  Decision mismatch: expected=%s, actual=%s\n",
				testCase.ExpectedDecision, result.ActualDecision)
		}
		if !result.MatchCorrect {
			fmt.Printf("  Matched-signal mismatch: expected=%q, actual=%q\n",
				testCase.ExpectedMatchedSignal, result.ActualMatchedSignal)
		}
	}

	return result
}

func printReaskRoutingResults(results []ReaskRoutingResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("REASK ROUTING TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correct: %d (%.2f%%)\n", correctTests, accuracy)
	fmt.Println(separator)

	for _, result := range results {
		if result.Error != "" {
			fmt.Printf("  - Test: %s\n    Error: %s\n", result.Name, result.Error)
			continue
		}
		if !result.DecisionCorrect || !result.MatchCorrect {
			fmt.Printf("  - Test: %s\n    Expected decision: %s, actual: %s\n    Expected matched signal: %q, actual: %q\n",
				result.Name, result.ExpectedDecision, result.ActualDecision,
				result.ExpectedMatchedSignal, result.ActualMatchedSignal)
		}
	}

	fmt.Println(separator + "\n")
}
