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

// Mirrors embedding_signal_routing.go for image-modality routing. The
// difference vs the text-only test is the request shape: this test sends
// chat completion content as an array with `image_url` parts, exercising the
// request-path image extractor + the embedding signal evaluator's
// image-modality dispatch (PR-2 of the multimodal-image-query-signals
// contribution series).

func init() {
	pkgtestcases.Register("embedding-signal-image-routing", pkgtestcases.TestCase{
		Description: "Test IntelligentRoute with image-modality EmbeddingSignal for content-aware multimodal routing",
		Tags:        []string{"signal-decision", "embedding", "routing", "semantic", "multimodal", "image"},
		Fn:          testEmbeddingSignalImageRouting,
	})
}

// EmbeddingSignalImageTestCase represents a test case for image-modality
// embedding signal routing. The optional ImageURL field carries a base64
// data URI that will be embedded into the chat completion content array as
// an `image_url` part; when empty, the request sends only text and verifies
// the image rule does NOT fire.
type EmbeddingSignalImageTestCase struct {
	Description      string `json:"description"`
	Query            string `json:"query"`
	ImageURL         string `json:"image_url,omitempty"`
	SignalName       string `json:"signal_name"`
	ExpectedMatch    bool   `json:"expected_match"`
	ExpectedDecision string `json:"expected_decision"`
	Category         string `json:"category"`
}

// EmbeddingSignalImageResult tracks the result of a single image-modality
// embedding signal test.
type EmbeddingSignalImageResult struct {
	Description      string
	Query            string
	HasImage         bool
	SignalName       string
	ExpectedMatch    bool
	ExpectedDecision string
	ActualDecision   string
	SignalTriggered  bool
	Correct          bool
	Error            string
	Category         string
}

// testEmbeddingSignalImageRouting tests IntelligentRoute with image-modality
// EmbeddingSignal configuration. Unlike the text-only embedding signal test,
// this constructs OpenAI-shape chat completion content arrays containing
// `image_url` parts so the router's request-path extractor pulls the image
// out and feeds it to the embedding signal evaluator's image-modality
// dispatch.
func testEmbeddingSignalImageRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing IntelligentRoute with image-modality EmbeddingSignal routing")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	testCases, err := loadEmbeddingSignalImageCases("e2e/testcases/testdata/embedding_signal_image_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	results := runEmbeddingSignalImageTests(ctx, testCases, localPort, opts.Verbose)

	totalTests := len(results)
	correctTests := countCorrectImageTests(results)
	accuracy := float64(correctTests) / float64(totalTests) * 100

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":   totalTests,
			"correct_tests": correctTests,
			"accuracy_rate": fmt.Sprintf("%.2f%%", accuracy),
			"failed_tests":  totalTests - correctTests,
		})
	}

	printEmbeddingSignalImageResults(results, totalTests, correctTests, accuracy)

	if opts.Verbose {
		fmt.Printf("[Test] Image-modality embedding signal routing test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy)
	}

	if correctTests == 0 {
		return fmt.Errorf("image-modality embedding signal routing test failed: 0%% accuracy (0/%d correct)", totalTests)
	}

	return nil
}

func loadEmbeddingSignalImageCases(filepath string) ([]EmbeddingSignalImageTestCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []EmbeddingSignalImageTestCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

func runEmbeddingSignalImageTests(ctx context.Context, testCases []EmbeddingSignalImageTestCase, localPort string, verbose bool) []EmbeddingSignalImageResult {
	results := make([]EmbeddingSignalImageResult, 0, len(testCases))

	for _, testCase := range testCases {
		result := testSingleEmbeddingSignalImage(ctx, testCase, localPort, verbose)
		results = append(results, result)
	}

	return results
}

// buildChatCompletionContent constructs the OpenAI chat completion content
// for a single user turn. When imageURL is empty, returns a plain string
// content (the existing pre-multimodal request shape). When imageURL is
// present, returns a content array with both text and image_url parts so
// the router's request-path extractor pulls the image out for routing.
func buildChatCompletionContent(query, imageURL string) interface{} {
	if imageURL == "" {
		return query
	}
	return []map[string]interface{}{
		{"type": "text", "text": query},
		{
			"type": "image_url",
			"image_url": map[string]string{
				"url": imageURL,
			},
		},
	}
}

func testSingleEmbeddingSignalImage(ctx context.Context, testCase EmbeddingSignalImageTestCase, localPort string, verbose bool) EmbeddingSignalImageResult {
	result := EmbeddingSignalImageResult{
		Description:      testCase.Description,
		Query:            testCase.Query,
		HasImage:         testCase.ImageURL != "",
		SignalName:       testCase.SignalName,
		ExpectedMatch:    testCase.ExpectedMatch,
		ExpectedDecision: testCase.ExpectedDecision,
		Category:         testCase.Category,
	}

	requestBody := map[string]interface{}{
		"model": "auto",
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": buildChatCompletionContent(testCase.Query, testCase.ImageURL),
			},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		result.Error = fmt.Sprintf("failed to marshal request: %v", err)
		return result
	}

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
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

	actualDecision := resp.Header.Get("x-vsr-selected-decision")
	result.ActualDecision = actualDecision

	if resp.StatusCode != http.StatusOK {
		if actualDecision == "" {
			bodyBytes, _ := io.ReadAll(resp.Body)
			result.Error = fmt.Sprintf("unexpected status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
			return result
		}
		if verbose {
			fmt.Printf("[Test] Request blocked with status %d, decision=%s\n", resp.StatusCode, actualDecision)
		}
	}

	result.SignalTriggered = (actualDecision == testCase.ExpectedDecision)
	result.Correct = (actualDecision == testCase.ExpectedDecision)

	if verbose {
		printImageTestResult(result, testCase)
	}

	return result
}

func countCorrectImageTests(results []EmbeddingSignalImageResult) int {
	correct := 0
	for _, result := range results {
		if result.Correct {
			correct++
		}
	}
	return correct
}

func printImageTestResult(result EmbeddingSignalImageResult, testCase EmbeddingSignalImageTestCase) {
	imageMarker := "no-image"
	if result.HasImage {
		imageMarker = "image"
	}
	if result.Correct {
		fmt.Printf("[Test] ✓ Correct (%s): %s\n", imageMarker, result.Description)
		fmt.Printf("       Signal: %s, Decision: %s\n", result.SignalName, result.ActualDecision)
	} else {
		fmt.Printf("[Test] ✗ Incorrect (%s): %s\n", imageMarker, result.Description)
		fmt.Printf("       Expected: signal_match=%v, decision=%s\n",
			testCase.ExpectedMatch, testCase.ExpectedDecision)
		fmt.Printf("       Actual:   decision=%s\n", result.ActualDecision)
		if result.Error != "" {
			fmt.Printf("       Error: %s\n", result.Error)
		}
	}
}

func printEmbeddingSignalImageResults(results []EmbeddingSignalImageResult, totalTests, correctTests int, accuracy float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("IMAGE-MODALITY EMBEDDING SIGNAL ROUTING TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correctly Routed: %d\n", correctTests)
	fmt.Printf("Routing Accuracy: %.2f%%\n", accuracy)
	fmt.Println(separator + "\n")

	for _, result := range results {
		if !result.Correct && result.Error == "" {
			imageMarker := "no-image"
			if result.HasImage {
				imageMarker = "image"
			}
			fmt.Printf("  - (%s) %s\n", imageMarker, result.Description)
			fmt.Printf("    Expected: decision=%s\n", result.ExpectedDecision)
			fmt.Printf("    Actual:   decision=%s\n", result.ActualDecision)
		}
		if result.Error != "" {
			fmt.Printf("  - ERROR: %s — %s\n", result.Description, result.Error)
		}
	}
}
