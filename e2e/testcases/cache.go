package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// cacheCancelWindow is how long the cancellation probe lets a request run before
// abandoning it. Long enough that the router has started the ext_proc exchange,
// short enough that a real upstream answer is unlikely.
const cacheCancelWindow = 250 * time.Millisecond

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
	Similarity       float64 // per-request similarity surfaced on the response (0 when absent)
	Error            string
}

//nolint:cyclop,gocognit // Existing E2E orchestration branches by request outcome.
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
	var setupFailures []string
	totalRequests := 0
	cacheHits := 0

	for _, testCase := range testCases {
		// Send original question first (should not hit cache)
		if opts.Verbose {
			fmt.Printf("[Test] Sending original question: %s\n", testCase.OriginalQuestion)
		}
		origResp, err := sendChatRequest(ctx, testCase.OriginalQuestion, localPort, opts.Verbose)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Error sending original question: %v\n", err)
			}
			// A failed priming request means the similar questions below can
			// never legitimately hit; record it so the test fails instead of
			// silently skipping the case (#2473).
			setupFailures = append(setupFailures,
				fmt.Sprintf("original question %q: %v", testCase.OriginalQuestion, err))
			continue
		}
		origResp.Body.Close()

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

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests": totalRequests,
			"cache_hits":     cacheHits,
			"cache_misses":   totalRequests - cacheHits,
			"hit_rate":       fmt.Sprintf("%.2f%%", hitRate),
		})
	}

	// Print results
	printCacheResults(results, totalRequests, cacheHits, hitRate)

	if opts.Verbose {
		fmt.Printf("[Test] Cache test completed: %d/%d cache hits (%.2f%% hit rate)\n",
			cacheHits, totalRequests, hitRate)
	}

	// Turn the collected results into an explicit pass/fail verdict (#2473).
	verdict := evaluateCacheAssertions(results, totalRequests, cacheHits, setupFailures)

	// Cancellation acceptance (#2473) runs last so it cannot disturb the hit-rate
	// measurement above.
	return errors.Join(verdict, runCacheCancellationCheck(ctx, localPort, opts.Verbose))
}

// runCacheCancellationCheck exercises the #2473 cancellation contract at the E2E
// surface: a request abandoned mid-flight must not publish cache state, so the
// same question asked again must still miss.
//
// Scope of what E2E can observe: cancellation reaches the router as an aborted
// ext_proc stream, which has no response of its own to assert on. Its observable
// consequence is the absence of cached state — including the pending entry the
// request path writes before the upstream answers, which must never be served as
// a hit. Context-error propagation and the write-path guards themselves are
// pinned at unit level (pkg/cache cancellation specs).
//
// The cancel window is timing-dependent, so a request that completes before the
// deadline is reported and skipped rather than failed: this assertion must not
// add a new flake class to the profile matrix.
func runCacheCancellationCheck(ctx context.Context, localPort string, verbose bool) error {
	// Unique question: nothing else in the corpus may be semantically close to
	// it, or a pre-existing entry would answer the follow-up lookup.
	question := fmt.Sprintf("cancellation probe %d: describe the ripening of a persimmon", time.Now().UnixNano())

	cancelCtx, cancel := context.WithTimeout(ctx, cacheCancelWindow)
	defer cancel()

	resp, err := sendChatRequest(cancelCtx, question, localPort, verbose)
	if err == nil {
		resp.Body.Close()
		fmt.Printf("[Test] cancellation probe: upstream answered within %s; no mid-flight cancel to assert on\n",
			cacheCancelWindow)
		return nil
	}
	if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
		return fmt.Errorf("cancellation probe failed for an unrelated reason: %w", err)
	}
	if verbose {
		fmt.Printf("[Test] cancellation probe: request abandoned after %s\n", cacheCancelWindow)
	}

	// Give the router the same settling time the priming requests get, so a
	// wrongly published entry would be visible by now.
	time.Sleep(1 * time.Second)

	followUp, err := sendChatRequest(ctx, question, localPort, verbose)
	if err != nil {
		return fmt.Errorf("cancellation probe follow-up request failed: %w", err)
	}
	defer followUp.Body.Close()

	if followUp.Header.Get("x-vsr-cache-hit") == "true" {
		return fmt.Errorf("cancelled request published cache state: the same question hit with similarity %q",
			followUp.Header.Get("x-vsr-cache-similarity"))
	}
	if _, simErr := parseCacheSimilarity(followUp.Header.Get("x-vsr-cache-similarity"), false); simErr != "" {
		return fmt.Errorf("cancellation probe follow-up: %s", simErr)
	}

	return nil
}

// evaluateCacheAssertions converts the collected per-request results into an
// explicit pass/fail verdict. Previously testCache returned nil unconditionally,
// so a malformed/missing similarity header (recorded in CacheResult.Error) or a
// run that executed zero requests still passed CI (#2473).
//
// The per-request acceptance condition itself lives in parseCacheSimilarity,
// which is applied to every response: a hit must carry its own score in (0,1]
// and a miss must carry none. This function is what makes those per-request
// verdicts affect the testcase result.
//
// Hit *rate* is deliberately not gated here. It is a measurement, reported
// through SetDetails, and the repository already enforces a floor for it in
// e2e/pkg/testcases/acceptance_contracts.go ("semantic-cache" -> minimum
// hit_rate) for the profile that owns that contract. Failing every profile on
// zero hits would turn a baseline contract into a matrix-wide gate.
func evaluateCacheAssertions(results []CacheResult, totalRequests, cacheHits int, setupFailures []string) error {
	assertionFailures := append([]string{}, setupFailures...)
	for _, r := range results {
		if r.Error != "" {
			assertionFailures = append(assertionFailures,
				fmt.Sprintf("%q: %s", r.SimilarQuestion, r.Error))
		}
	}

	if totalRequests == 0 {
		if len(setupFailures) > 0 {
			return fmt.Errorf("cache test executed zero similar-question requests; every priming request failed:\n  %s",
				strings.Join(setupFailures, "\n  "))
		}
		return errors.New("cache test executed zero similar-question requests; nothing was asserted")
	}
	if len(assertionFailures) > 0 {
		return fmt.Errorf("cache per-request similarity assertions failed (%d of %d requests):\n  %s",
			len(assertionFailures), totalRequests, strings.Join(assertionFailures, "\n  "))
	}
	if cacheHits == 0 {
		fmt.Printf("[Test] cache test observed zero hits across %d requests; "+
			"the hit path's similarity contract was not exercised in this run\n", totalRequests)
	}

	return nil
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

	// Per-request similarity assertion (#2473): validate the request-owned
	// x-vsr-cache-similarity header (see parseCacheSimilarity for the contract).
	sim, simErr := parseCacheSimilarity(resp.Header.Get("x-vsr-cache-similarity"), result.CacheHit)
	if simErr != "" {
		result.Error = simErr
		return result
	}
	result.Similarity = sim

	if verbose {
		if result.CacheHit {
			fmt.Printf("[Test] ✓ Cache HIT for: %s (similarity=%.4f)\n", question, result.Similarity)
		} else {
			fmt.Printf("[Test] ✗ Cache MISS for: %s\n", question)
		}
	}

	return result
}

// parseCacheSimilarity validates the request-owned x-vsr-cache-similarity header
// for one lookup and returns the parsed score, or a non-empty error message.
//
// The header rides the x-vsr-debug surface (opted into by sendChatRequest) and
// is per-request (#2473): a hit surfaces a positive score that, because a hit is
// returned only when similarity >= the configured threshold, lands in (0,1]. A
// miss may omit the header or surface zero, but must not publish a candidate
// score.
func parseCacheSimilarity(simHeader string, cacheHit bool) (float64, string) {
	if simHeader == "" {
		if cacheHit {
			return 0, "cache hit missing x-vsr-cache-similarity header (per-request score not surfaced)"
		}
		return 0, ""
	}
	sim, err := strconv.ParseFloat(simHeader, 64)
	if err != nil {
		return 0, fmt.Sprintf("unparsable x-vsr-cache-similarity %q: %v", simHeader, err)
	}
	if math.IsNaN(sim) || math.IsInf(sim, 0) {
		return 0, fmt.Sprintf("non-finite x-vsr-cache-similarity %q", simHeader)
	}
	if cacheHit {
		if sim <= 0.0 || sim > 1.0 {
			return 0, fmt.Sprintf("cache-hit similarity %.4f out of expected (0,1] range", sim)
		}
		return sim, ""
	}
	if sim != 0 {
		return 0, fmt.Sprintf("cache miss surfaced similarity %.4f; misses must not publish a candidate score", sim)
	}
	return 0, ""
}

func sendChatRequest(ctx context.Context, question, localPort string, verbose bool) (*http.Response, error) {
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

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	// Opt into the x-vsr-debug surface so the router emits the demoted
	// x-vsr-cache-similarity header we assert on for #2473.
	req.Header.Set("x-vsr-debug", "true")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	return resp, nil
}

//nolint:cyclop,funlen,gocognit // Existing reporting branches by category, miss, and error.
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
