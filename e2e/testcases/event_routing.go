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

// targetEventDecision is the decision configured to route on the
// critical_payment_event rule (see e2e/profiles/ai-gateway/values.yaml).
const targetEventDecision = "critical_event"

func init() {
	pkgtestcases.Register("event-routing", pkgtestcases.TestCase{
		Description: "Test event signal rule matching and routing",
		Tags:        []string{"kubernetes", "routing", "event"},
		Fn:          testEventRouting,
	})
}

// EventRoutingCase represents a test case for event signal routing.
type EventRoutingCase struct {
	Name                 string `json:"name"`
	Description          string `json:"description"`
	Query                string `json:"query"`
	ExpectedDecision     string `json:"expected_decision"`
	ExpectedMatchedEvent string `json:"expected_matched_event"`
	ShouldMatch          bool   `json:"should_match"`
}

// EventRoutingResult tracks the result of a single event routing test.
type EventRoutingResult struct {
	Name                 string
	Query                string
	ExpectedDecision     string
	ActualDecision       string
	ExpectedMatchedEvent string
	ActualMatchedEvent   string
	ShouldMatch          bool
	DecisionCorrect      bool
	MatchCorrect         bool
	Error                string
}

func testEventRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing event signal routing")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	testCases, err := loadEventRoutingCases("e2e/testcases/testdata/event_routing_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	var results []EventRoutingResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSingleEventRouting(ctx, testCase, localPort, opts.Verbose)
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

	printEventRoutingResults(results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Event routing test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy)
	}

	if correctTests != totalTests {
		return fmt.Errorf("event routing test failed: %d/%d correct", correctTests, totalTests)
	}

	return nil
}

func loadEventRoutingCases(filepath string) ([]EventRoutingCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []EventRoutingCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

func testSingleEventRouting(ctx context.Context, testCase EventRoutingCase, localPort string, verbose bool) EventRoutingResult {
	result := EventRoutingResult{
		Name:                 testCase.Name,
		Query:                testCase.Query,
		ExpectedDecision:     testCase.ExpectedDecision,
		ExpectedMatchedEvent: testCase.ExpectedMatchedEvent,
		ShouldMatch:          testCase.ShouldMatch,
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
	result.ActualMatchedEvent = response.Headers.Get("x-vsr-matched-event")

	if testCase.ShouldMatch {
		result.DecisionCorrect = result.ActualDecision == testCase.ExpectedDecision
		result.MatchCorrect = result.ActualMatchedEvent == testCase.ExpectedMatchedEvent
	} else {
		result.DecisionCorrect = result.ActualDecision != targetEventDecision
		result.MatchCorrect = result.ActualMatchedEvent == ""
	}

	if verbose && (!result.DecisionCorrect || !result.MatchCorrect) {
		fmt.Printf("[Test] Test case failed: %s\n", testCase.Name)
		if !result.DecisionCorrect {
			fmt.Printf("  Decision mismatch: query='%s', expected=%s, actual=%s\n",
				testCase.Query, testCase.ExpectedDecision, result.ActualDecision)
		}
		if !result.MatchCorrect {
			fmt.Printf("  Matched-event mismatch: expected=%q, actual=%q\n",
				testCase.ExpectedMatchedEvent, result.ActualMatchedEvent)
		}
	}

	return result
}

func printEventRoutingResults(results []EventRoutingResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("EVENT ROUTING TEST RESULTS")
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
			fmt.Printf("  - Test: %s\n    Query: %s\n    Expected decision: %s, actual: %s\n    Expected matched event: %q, actual: %q\n",
				result.Name, result.Query, result.ExpectedDecision, result.ActualDecision,
				result.ExpectedMatchedEvent, result.ActualMatchedEvent)
		}
	}

	fmt.Println(separator + "\n")
}
