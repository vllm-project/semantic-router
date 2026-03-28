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

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const authzUserIDHeader = "x-authz-user-id"

func init() {
	pkgtestcases.Register("semantic-cache", pkgtestcases.TestCase{
		Description: "Test semantic cache hit rate with similar questions",
		Tags:        []string{"kubernetes", "semantic-cache", "performance"},
		Fn:          testCache,
	})
}

// CacheTestCase represents a test case for cache testing
type CacheTestCase struct {
	Description      string   `json:"description"`
	Category         string   `json:"category"`
	OriginalQuestion string   `json:"original_question"`
	SimilarQuestions []string `json:"similar_questions"`
}

// CacheResult tracks the result of a cache test
type CacheResult struct {
	Description      string
	Category         string
	OriginalQuestion string
	SimilarQuestion  string
	CacheHit         bool
	Error            string
}

func testCache(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing semantic cache functionality")
	}

	// Setup service connection and get local port
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward() // Ensure port forwarding is stopped when test completes

	// Load test cases from JSON file
	testCases, err := loadCacheCases("e2e/testcases/testdata/cache_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	// Run cache tests
	var results []CacheResult
	totalRequests := 0
	cacheHits := 0

	for _, testCase := range testCases {
		// Send original question first (should not hit cache)
		if opts.Verbose {
			fmt.Printf("[Test] Sending original question: %s\n", testCase.OriginalQuestion)
		}
		_, err := sendChatRequest(ctx, testCase.OriginalQuestion, localPort, opts.Verbose)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Error sending original question: %v\n", err)
			}
			continue
		}

		// Wait a bit to ensure cache is populated
		time.Sleep(1 * time.Second)

		// Send similar questions (should hit cache)
		for _, similarQuestion := range testCase.SimilarQuestions {
			totalRequests++
			result := testSingleCacheRequest(ctx, testCase, similarQuestion, localPort, opts.Verbose)
			results = append(results, result)
			if result.CacheHit {
				cacheHits++
			}
		}
	}

	// Calculate hit rate
	hitRate := float64(0)
	if totalRequests > 0 {
		hitRate = float64(cacheHits) / float64(totalRequests) * 100
	}

	if err := verifyUserScopedCacheBehavior(ctx, localPort, opts.Verbose); err != nil {
		return err
	}

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests":           totalRequests,
			"cache_hits":               cacheHits,
			"cache_misses":             totalRequests - cacheHits,
			"hit_rate":                 fmt.Sprintf("%.2f%%", hitRate),
			"user_scoped_cache_status": "pass",
		})
	}

	// Print results
	printCacheResults(results, totalRequests, cacheHits, hitRate)

	if opts.Verbose {
		fmt.Printf("[Test] Cache test completed: %d/%d cache hits (%.2f%% hit rate)\n",
			cacheHits, totalRequests, hitRate)
	}

	return nil
}

func verifyUserScopedCacheBehavior(ctx context.Context, localPort string, verbose bool) error {
	queryID := time.Now().UnixNano()
	question := fmt.Sprintf("User scoped semantic cache isolation probe %d: explain mitosis versus meiosis.", queryID)
	firstUserID := fmt.Sprintf("cache-user-a-%d", queryID)
	secondUserID := fmt.Sprintf("cache-user-b-%d", queryID)

	if verbose {
		fmt.Printf("[Test] Verifying user-scoped cache behavior for %s and %s\n", firstUserID, secondUserID)
	}

	firstResponse, err := sendChatRequestForUser(ctx, question, localPort, firstUserID, verbose)
	if err != nil {
		return fmt.Errorf("failed to send initial scoped cache request: %w", err)
	}
	drainAndClose(firstResponse)

	sameUserCacheHit, err := pollForCacheHit(ctx, question, localPort, firstUserID, verbose)
	if err != nil {
		return err
	}
	if sameUserCacheHit != "true" {
		return fmt.Errorf("expected same-user cache hit for scoped query, got cache-hit=%q", sameUserCacheHit)
	}

	// Cross-user request should NOT hit cache (different user scope).
	thirdResponse, err := sendChatRequestForUser(ctx, question, localPort, secondUserID, verbose)
	if err != nil {
		return fmt.Errorf("failed to send cross-user scoped cache request: %w", err)
	}
	crossUserCacheHit := thirdResponse.Header.Get("x-vsr-cache-hit")
	drainAndClose(thirdResponse)

	if crossUserCacheHit == "true" {
		return fmt.Errorf("expected cross-user cache miss for scoped query, got cache-hit=%q", crossUserCacheHit)
	}

	if verbose {
		fmt.Printf("[Test] ✓ User scoped cache isolated responses across %s and %s\n", firstUserID, secondUserID)
	}

	return nil
}

// drainAndClose reads remaining bytes from the response body and closes it so
// that the underlying connection is properly released.
func drainAndClose(resp *http.Response) {
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
}

// pollForCacheHit retries the same-user request until a cache hit is observed
// or the retry budget is exhausted. Returns the final cache-hit header value.
func pollForCacheHit(ctx context.Context, question, localPort, userID string, verbose bool) (string, error) {
	const maxRetries = 10
	const retryInterval = time.Second
	var cacheHit string
	for attempt := 1; attempt <= maxRetries; attempt++ {
		time.Sleep(retryInterval)
		resp, err := sendChatRequestForUser(ctx, question, localPort, userID, verbose)
		if err != nil {
			if attempt == maxRetries {
				return "", fmt.Errorf("failed to send same-user scoped cache request: %w", err)
			}
			continue
		}
		cacheHit = resp.Header.Get("x-vsr-cache-hit")
		drainAndClose(resp)
		if cacheHit == "true" {
			break
		}
		if verbose {
			fmt.Printf("[Test] Retry %d/%d: cache not yet populated for same-user query\n", attempt, maxRetries)
		}
	}
	return cacheHit, nil
}

func loadCacheCases(filepath string) ([]CacheTestCase, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}

	var cases []CacheTestCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}

	return cases, nil
}

func testSingleCacheRequest(ctx context.Context, testCase CacheTestCase, question, localPort string, verbose bool) CacheResult {
	result := CacheResult{
		Description:      testCase.Description,
		Category:         testCase.Category,
		OriginalQuestion: testCase.OriginalQuestion,
		SimilarQuestion:  question,
	}

	resp, err := sendChatRequest(ctx, question, localPort, verbose)
	if err != nil {
		result.Error = fmt.Sprintf("failed to send request: %v", err)
		return result
	}
	defer resp.Body.Close()

	// Check for cache hit header
	cacheHitHeader := resp.Header.Get("x-vsr-cache-hit")
	result.CacheHit = (cacheHitHeader == "true")

	if verbose {
		if result.CacheHit {
			fmt.Printf("[Test] ✓ Cache HIT for: %s\n", question)
		} else {
			fmt.Printf("[Test] ✗ Cache MISS for: %s\n", question)
		}
	}

	return result
}

func sendChatRequest(ctx context.Context, question, localPort string, verbose bool) (*http.Response, error) {
	return sendChatRequestForUser(ctx, question, localPort, "", verbose)
}

func sendChatRequestForUser(ctx context.Context, question, localPort, userID string, verbose bool) (*http.Response, error) {
	// Create chat completion request
	requestBody := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{"role": "user", "content": question},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Send request
	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if userID != "" {
		req.Header.Set(authzUserIDHeader, userID)
	}

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	// Check response status
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	return resp, nil
}

func printCacheResults(results []CacheResult, totalRequests, cacheHits int, hitRate float64) {
	separator := "================================================================================"
	fmt.Println("\n" + separator)
	fmt.Println("CACHE TEST RESULTS")
	fmt.Println(separator)
	fmt.Printf("Total Requests: %d\n", totalRequests)
	fmt.Printf("Cache Hits: %d\n", cacheHits)
	fmt.Printf("Hit Rate: %.2f%%\n", hitRate)
	fmt.Println(separator)

	// Group results by category
	categoryStats := make(map[string]struct {
		total int
		hits  int
	})

	for _, result := range results {
		stats := categoryStats[result.Category]
		stats.total++
		if result.CacheHit {
			stats.hits++
		}
		categoryStats[result.Category] = stats
	}

	// Print per-category results
	fmt.Println("\nPer-Category Results:")
	for category, stats := range categoryStats {
		categoryHitRate := float64(stats.hits) / float64(stats.total) * 100
		fmt.Printf("  - %-20s: %d/%d (%.2f%%)\n", category, stats.hits, stats.total, categoryHitRate)
	}

	// Print cache misses
	missCount := 0
	for _, result := range results {
		if !result.CacheHit && result.Error == "" {
			missCount++
		}
	}

	if missCount > 0 {
		fmt.Println("\nCache Misses:")
		for _, result := range results {
			if !result.CacheHit && result.Error == "" {
				fmt.Printf("  - Original: %s\n", result.OriginalQuestion)
				fmt.Printf("    Similar:  %s\n", result.SimilarQuestion)
				fmt.Printf("    Category: %s\n", result.Category)
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
				fmt.Printf("  - Question: %s\n", result.SimilarQuestion)
				fmt.Printf("    Error: %s\n", result.Error)
			}
		}
	}

	fmt.Println(separator + "\n")
}
