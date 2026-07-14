package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// responseWarningsHeader is the consolidated response-quality warnings header
// (v0.4). It carries the "hallucination" code when the detector flags the
// response. Kept as a literal to match the e2e convention of not importing the
// router module for header names.
const responseWarningsHeader = "x-vsr-response-warnings"

const hallucinationWarningCode = "hallucination"

func init() {
	pkgtestcases.Register("hallucination-detection", pkgtestcases.TestCase{
		Description: "Test the pluggable hallucination detector endpoint backend end-to-end",
		Tags:        []string{"kubernetes", "hallucination"},
		Fn:          testHallucinationDetection,
	})
}

// HallucinationTestCase is a factual query paired with the tool context that
// grounds it. The tool context makes the router run hallucination detection on
// the response, and the mock endpoint detector flags the answer.
type HallucinationTestCase struct {
	Description string `json:"description"`
	Question    string `json:"question"`
	Context     string `json:"context"`
}

// HallucinationResult tracks the outcome of a single detection test.
type HallucinationResult struct {
	Description         string
	Question            string
	SelectedDecision    string
	WarningsHeader      string
	HallucinationWarned bool
	Error               string
}

func testHallucinationDetection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing hallucination detector endpoint backend")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	testCases, err := loadHallucinationCases("e2e/testcases/testdata/hallucination_detection_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	var results []HallucinationResult
	warnedCount := 0
	for _, testCase := range testCases {
		result := testSingleHallucinationDetection(ctx, testCase, localPort, opts.Verbose)
		results = append(results, result)
		if result.HallucinationWarned {
			warnedCount++
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":  len(testCases),
			"warned_tests": warnedCount,
			"failed_tests": len(testCases) - warnedCount,
			"warning_rate": fmt.Sprintf("%.2f%%", detectionRate(warnedCount, len(testCases))),
		})
	}

	printHallucinationResults(results, warnedCount)

	// Fail only if the endpoint detection path never surfaced a warning: that
	// means the pluggable endpoint backend did not run end-to-end at all. Like
	// the other classifier e2e tests, per-case model behavior is tolerated.
	if warnedCount == 0 {
		return fmt.Errorf("hallucination detection test failed: no response surfaced the %q warning across %d cases",
			hallucinationWarningCode, len(testCases))
	}

	return nil
}

func loadHallucinationCases(path string) ([]HallucinationTestCase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []HallucinationTestCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

func testSingleHallucinationDetection(ctx context.Context, testCase HallucinationTestCase, localPort string, verbose bool) HallucinationResult {
	result := HallucinationResult{
		Description: testCase.Description,
		Question:    testCase.Question,
	}

	response, err := sendHallucinationChatCompletion(ctx, localPort, testCase, 60*time.Second)
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

	result.SelectedDecision = response.Headers.Get("x-vsr-selected-decision")
	result.WarningsHeader = response.Headers.Get(responseWarningsHeader)
	result.HallucinationWarned = warningsContain(result.WarningsHeader, hallucinationWarningCode)

	if verbose {
		if result.HallucinationWarned {
			fmt.Printf("[Test] ✓ Warned: %s (decision=%s, warnings=%q)\n",
				testCase.Description, result.SelectedDecision, result.WarningsHeader)
		} else {
			fmt.Printf("[Test] – No warning: %s (decision=%s, warnings=%q)\n",
				testCase.Description, result.SelectedDecision, result.WarningsHeader)
		}
	}

	return result
}

// sendHallucinationChatCompletion sends a factual question together with a tool
// result carrying the grounding context. The tool message is what makes the
// router run hallucination detection (HasToolsForFactCheck + tool context), and
// the answer returned by the mock backend is then checked by the endpoint
// detector.
func sendHallucinationChatCompletion(
	ctx context.Context,
	localPort string,
	testCase HallucinationTestCase,
	timeout time.Duration,
) (*localChatCompletionResponse, error) {
	requestBody := map[string]interface{}{
		"model": "openai/gpt-oss-20b",
		"messages": []map[string]interface{}{
			{"role": "user", "content": testCase.Question},
			{"role": "tool", "content": testCase.Context},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s%s", localPort, localChatCompletionsPath)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-vsr-debug", "true")

	resp, err := (&http.Client{Timeout: timeout}).Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	return &localChatCompletionResponse{
		StatusCode: resp.StatusCode,
		Headers:    resp.Header,
		Body:       bodyBytes,
	}, nil
}

func warningsContain(headerValue, code string) bool {
	for _, part := range strings.Split(headerValue, ",") {
		if strings.TrimSpace(part) == code {
			return true
		}
	}
	return false
}

func detectionRate(matched, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(matched) / float64(total) * 100
}

func printHallucinationResults(results []HallucinationResult, warnedCount int) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("HALLUCINATION DETECTION TEST RESULTS (endpoint backend)")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", len(results))
	fmt.Printf("Responses Warned: %d\n", warnedCount)
	fmt.Printf("Warning Rate: %.2f%%\n", detectionRate(warnedCount, len(results)))
	fmt.Println(separator)

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
				fmt.Printf("  - %s\n    Error: %s\n", result.Description, result.Error)
			}
		}
	}

	fmt.Println(separator + "\n")
}
