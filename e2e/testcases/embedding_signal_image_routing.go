package testcases

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

// Mirrors embedding_signal_routing.go for image-modality routing. The
// difference vs the text-only test is the request shape: this test sends
// chat completion content as an array with image_url parts, exercising the
// request-path image extractor + the embedding signal evaluator's
// image-modality dispatch wired through PRs #1867 and #1868.

// minAccuracyThreshold is the floor for the overall image-routing accuracy.
// Set to 75% per the Copilot review on the original draft: 0% pass-bar fails
// to detect a fully-broken signal pipeline. The medical-imagery anchors are
// short, distinctive phrases against synthetic 32x32 PNG fixtures, so a
// healthy classifier should clear this comfortably.
const minAccuracyThreshold = 0.75

func init() {
	pkgtestcases.Register("embedding-signal-image-routing", pkgtestcases.TestCase{
		Description: "Test IntelligentRoute with image-modality EmbeddingSignal for content-aware multimodal routing",
		Tags:        []string{"signal-decision", "embedding", "routing", "semantic", "multimodal", "image"},
		Fn:          testEmbeddingSignalImageRouting,
	})
}

// EmbeddingSignalImageTestCase represents a test case for image-modality
// embedding signal routing. The optional ImageFile field carries a path to
// an image file (relative to repo root) that will be read from disk,
// base64-encoded, and embedded into the chat completion content array as
// an image_url data URI part; when empty, the request sends only text and
// verifies the image rule does NOT fire. The SignalName field doubles as
// the expected embedding rule name that should appear in the
// x-vsr-matched-embeddings response header when ExpectedMatch is true.
type EmbeddingSignalImageTestCase struct {
	Description      string `json:"description"`
	Query            string `json:"query"`
	ImageFile        string `json:"image_file,omitempty"`
	SignalName       string `json:"signal_name"`
	ExpectedMatch    bool   `json:"expected_match"`
	ExpectedDecision string `json:"expected_decision"`
	Category         string `json:"category"`
}

// EmbeddingSignalImageResult tracks the result of a single image-modality
// embedding signal test.
type EmbeddingSignalImageResult struct {
	Description           string
	Query                 string
	HasImage              bool
	SignalName            string
	ExpectedMatch         bool
	ExpectedDecision      string
	ActualDecision        string
	MatchedEmbeddingRules []string
	SignalTriggered       bool
	Correct               bool
	Error                 string
	Category              string
}

// testEmbeddingSignalImageRouting tests IntelligentRoute with image-modality
// EmbeddingSignal configuration. Unlike the text-only embedding signal test,
// this constructs OpenAI-shape chat completion content arrays containing
// image_url parts so the router's request-path extractor pulls the image
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
	accuracy := float64(correctTests) / float64(totalTests)

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":            totalTests,
			"correct_tests":          correctTests,
			"accuracy_rate":          fmt.Sprintf("%.2f%%", accuracy*100),
			"failed_tests":           totalTests - correctTests,
			"min_accuracy_threshold": fmt.Sprintf("%.0f%%", minAccuracyThreshold*100),
		})
	}

	printEmbeddingSignalImageResults(results, totalTests, correctTests, accuracy*100)

	if opts.Verbose {
		fmt.Printf("[Test] Image-modality embedding signal routing test completed: %d/%d correct (%.2f%% accuracy)\n",
			correctTests, totalTests, accuracy*100)
	}

	if accuracy < minAccuracyThreshold {
		return fmt.Errorf("image-modality embedding signal routing accuracy %.2f%% below required %.0f%% threshold (%d/%d correct)",
			accuracy*100, minAccuracyThreshold*100, correctTests, totalTests)
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
// for a single user turn. When imageFile is empty, returns a plain string
// content (the existing pre-multimodal request shape). When imageFile is
// present, reads the file from disk relative to the repo root,
// base64-encodes the bytes, and returns a content array with both text and
// image_url parts so the router's request-path extractor pulls the image
// out for routing. The content type is inferred from the file extension
// (defaults to image/jpeg; .png produces image/png).
func buildChatCompletionContent(query, imageFile string) (interface{}, error) {
	if imageFile == "" {
		return query, nil
	}
	data, err := os.ReadFile(imageFile)
	if err != nil {
		return nil, fmt.Errorf("read image file %q: %w", imageFile, err)
	}
	contentType := "image/jpeg"
	if strings.HasSuffix(strings.ToLower(imageFile), ".png") {
		contentType = "image/png"
	}
	dataURI := fmt.Sprintf("data:%s;base64,%s", contentType, base64.StdEncoding.EncodeToString(data))
	return []map[string]interface{}{
		{"type": "text", "text": query},
		{
			"type": "image_url",
			"image_url": map[string]string{
				"url": dataURI,
			},
		},
	}, nil
}

// parseMatchedEmbeddingRules extracts the comma-separated rule list from
// the x-vsr-matched-embeddings response header. Empty header -> empty slice.
func parseMatchedEmbeddingRules(header string) []string {
	if header == "" {
		return nil
	}
	parts := strings.Split(header, ",")
	rules := make([]string, 0, len(parts))
	for _, p := range parts {
		if trimmed := strings.TrimSpace(p); trimmed != "" {
			rules = append(rules, trimmed)
		}
	}
	return rules
}

func testSingleEmbeddingSignalImage(ctx context.Context, testCase EmbeddingSignalImageTestCase, localPort string, verbose bool) EmbeddingSignalImageResult {
	result := EmbeddingSignalImageResult{
		Description:      testCase.Description,
		Query:            testCase.Query,
		HasImage:         testCase.ImageFile != "",
		SignalName:       testCase.SignalName,
		ExpectedMatch:    testCase.ExpectedMatch,
		ExpectedDecision: testCase.ExpectedDecision,
		Category:         testCase.Category,
	}

	content, err := buildChatCompletionContent(testCase.Query, testCase.ImageFile)
	if err != nil {
		result.Error = fmt.Sprintf("failed to build request content: %v", err)
		return result
	}

	requestBody := map[string]interface{}{
		"model": "auto",
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": content,
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

	result.ActualDecision = resp.Header.Get("x-vsr-selected-decision")
	result.MatchedEmbeddingRules = parseMatchedEmbeddingRules(resp.Header.Get("x-vsr-matched-embeddings"))

	if resp.StatusCode != http.StatusOK {
		if result.ActualDecision == "" {
			bodyBytes, _ := io.ReadAll(resp.Body)
			result.Error = fmt.Sprintf("unexpected status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
			return result
		}
		if verbose {
			fmt.Printf("[Test] Request blocked with status %d, decision=%s\n", resp.StatusCode, result.ActualDecision)
		}
	}

	// Derive SignalTriggered from the matched-embedding-rules header, NOT
	// from decision identity. A request can hit the correct fallback
	// decision without the embedding rule having fired (e.g., default route
	// matched because no rule scored above threshold). The header tells us
	// definitively whether the rule matched.
	result.SignalTriggered = slices.Contains(result.MatchedEmbeddingRules, testCase.SignalName)

	// Correctness combines decision-match AND signal-state-match. The
	// embedding rule's match state must agree with the test case's
	// expected_match (positive cases require the rule to fire; negative
	// cases require it not to fire), AND the routed decision must match.
	result.Correct = result.ActualDecision == testCase.ExpectedDecision && result.SignalTriggered == testCase.ExpectedMatch

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
		fmt.Printf("[Test] PASS (%s): %s\n", imageMarker, result.Description)
		fmt.Printf("       Signal=%s, Triggered=%v, Decision=%s\n", result.SignalName, result.SignalTriggered, result.ActualDecision)
	} else {
		fmt.Printf("[Test] FAIL (%s): %s\n", imageMarker, result.Description)
		fmt.Printf("       Expected: signal_match=%v, decision=%s\n",
			testCase.ExpectedMatch, testCase.ExpectedDecision)
		fmt.Printf("       Actual:   signal_match=%v (matched_rules=%v), decision=%s\n",
			result.SignalTriggered, result.MatchedEmbeddingRules, result.ActualDecision)
		if result.Error != "" {
			fmt.Printf("       Error: %s\n", result.Error)
		}
	}
}

func printEmbeddingSignalImageResults(results []EmbeddingSignalImageResult, totalTests, correctTests int, accuracyPct float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("IMAGE-MODALITY EMBEDDING SIGNAL ROUTING TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Tests: %d\n", totalTests)
	fmt.Printf("Correctly Routed: %d\n", correctTests)
	fmt.Printf("Routing Accuracy: %.2f%% (threshold: %.0f%%)\n", accuracyPct, minAccuracyThreshold*100)
	fmt.Println(separator + "\n")

	for _, result := range results {
		if !result.Correct && result.Error == "" {
			imageMarker := "no-image"
			if result.HasImage {
				imageMarker = "image"
			}
			fmt.Printf("  - (%s) %s\n", imageMarker, result.Description)
			fmt.Printf("    Expected: signal_match=%v, decision=%s\n",
				result.ExpectedMatch, result.ExpectedDecision)
			fmt.Printf("    Actual:   signal_match=%v (matched_rules=%v), decision=%s\n",
				result.SignalTriggered, result.MatchedEmbeddingRules, result.ActualDecision)
		}
		if result.Error != "" {
			fmt.Printf("  - ERROR: %s - %s\n", result.Description, result.Error)
		}
	}
}
