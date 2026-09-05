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

// targetLanguageDecision is the decision configured to route on the
// language signal matching Spanish (see e2e/profiles/ai-gateway/values.yaml).
const targetLanguageDecision = "spanish_language"

func init() {
	pkgtestcases.Register("language-routing", pkgtestcases.TestCase{
		Description: "Test language signal rule matching and routing",
		Tags:        []string{"kubernetes", "routing", "language"},
		Fn:          testLanguageRouting,
	})
}

// LanguageRoutingCase represents a test case for language signal routing.
type LanguageRoutingCase struct {
	Name                    string `json:"name"`
	Description             string `json:"description"`
	Query                   string `json:"query"`
	ExpectedDecision        string `json:"expected_decision"`
	ExpectedMatchedLanguage string `json:"expected_matched_language"`
	ShouldMatch             bool   `json:"should_match"`
}

// LanguageRoutingResult tracks the result of a single language routing test.
type LanguageRoutingResult struct {
	Name                    string
	Query                   string
	ExpectedDecision        string
	ActualDecision          string
	ExpectedMatchedLanguage string
	ActualMatchedLanguage   string
	ShouldMatch             bool
	DecisionCorrect         bool
	MatchCorrect            bool
	Error                   string
}

func testLanguageRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing language signal routing")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	testCases, err := loadLanguageRoutingCases("e2e/testcases/testdata/language_routing_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	var results []LanguageRoutingResult
	totalTests := 0
	correctTests := 0

	for _, testCase := range testCases {
		totalTests++
		result := testSingleLanguageRouting(ctx, testCase, localPort, opts.Verbose)
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

	printLanguageRoutingResults(results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Language routing test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy)
	}

	if correctTests != totalTests {
		return fmt.Errorf("language routing test failed: %d/%d correct", correctTests, totalTests)
	}

	return nil
}

func loadLanguageRoutingCases(filepath string) ([]LanguageRoutingCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []LanguageRoutingCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

func testSingleLanguageRouting(ctx context.Context, testCase LanguageRoutingCase, localPort string, verbose bool) LanguageRoutingResult {
	result := LanguageRoutingResult{
		Name:                    testCase.Name,
		Query:                   testCase.Query,
		ExpectedDecision:        testCase.ExpectedDecision,
		ExpectedMatchedLanguage: testCase.ExpectedMatchedLanguage,
		ShouldMatch:             testCase.ShouldMatch,
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
	result.ActualMatchedLanguage = response.Headers.Get("x-vsr-matched-language")

	if testCase.ShouldMatch {
		result.DecisionCorrect = result.ActualDecision == testCase.ExpectedDecision
		result.MatchCorrect = result.ActualMatchedLanguage == testCase.ExpectedMatchedLanguage
	} else {
		result.DecisionCorrect = result.ActualDecision != targetLanguageDecision
		result.MatchCorrect = result.ActualMatchedLanguage == ""
	}

	if verbose && (!result.DecisionCorrect || !result.MatchCorrect) {
		fmt.Printf("[Test] Test case failed: %s\n", testCase.Name)
		if !result.DecisionCorrect {
			fmt.Printf("  Decision mismatch: query='%s', expected=%s, actual=%s\n",
				testCase.Query, testCase.ExpectedDecision, result.ActualDecision)
		}
		if !result.MatchCorrect {
			fmt.Printf("  Matched-language mismatch: expected=%q, actual=%q\n",
				testCase.ExpectedMatchedLanguage, result.ActualMatchedLanguage)
		}
	}

	return result
}

func printLanguageRoutingResults(results []LanguageRoutingResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("LANGUAGE ROUTING TEST RESULTS")
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
			fmt.Printf("  - Test: %s\n    Query: %s\n    Expected decision: %s, actual: %s\n    Expected matched language: %q, actual: %q\n",
				result.Name, result.Query, result.ExpectedDecision, result.ActualDecision,
				result.ExpectedMatchedLanguage, result.ActualMatchedLanguage)
		}
	}

	fmt.Println(separator + "\n")
}
