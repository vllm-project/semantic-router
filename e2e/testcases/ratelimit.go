package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("ratelimit-limitor", pkgtestcases.TestCase{
		Description: "Verify local-limiter rate limiting: per-user RPM enforcement, 429 responses with headers, tier isolation",
		Tags:        []string{"ratelimit", "functional"},
		Fn:          testRateLimitLimitor,
	})
}

func testRateLimitLimitor(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[RateLimit] Starting local-limiter rate limit tests")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	results := &rateLimitResults{}

	// Test 1: Free-tier user — requests within 3 RPM limit succeed
	if err := testFreeTierWithinLimit(ctx, baseURL, opts, results); err != nil {
		return fmt.Errorf("test 1 (free-tier within limit) failed: %w", err)
	}

	// Test 2: Free-tier user — 4th request exceeds limit → 429
	if err := testFreeTierExceedsLimit(ctx, baseURL, opts, results); err != nil {
		return fmt.Errorf("test 2 (free-tier exceeds limit) failed: %w", err)
	}

	// Test 3: 429 response has correct rate limit headers
	if err := testRateLimitHeaders(ctx, baseURL, opts, results); err != nil {
		return fmt.Errorf("test 3 (rate limit headers) failed: %w", err)
	}

	// Test 4: 429 response body is valid JSON error
	if err := testRateLimitErrorBody(ctx, baseURL, opts, results); err != nil {
		return fmt.Errorf("test 4 (error body format) failed: %w", err)
	}

	// Test 5: Premium-tier user — higher limit, 5 requests all succeed
	if err := testPremiumTierHigherLimit(ctx, baseURL, opts, results); err != nil {
		return fmt.Errorf("test 5 (premium-tier higher limit) failed: %w", err)
	}

	// Test 6: Per-user isolation — separate users have independent buckets
	if err := testPerUserIsolation(ctx, baseURL, opts, results); err != nil {
		return fmt.Errorf("test 6 (per-user isolation) failed: %w", err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"tests_passed": results.passed,
			"tests_total":  results.total,
		})
	}

	if opts.Verbose {
		fmt.Printf("[RateLimit] All %d tests passed\n", results.total)
	}

	return nil
}

type rateLimitResults struct {
	passed int
	total  int
}

type rlResponse struct {
	statusCode int
	body       []byte
	headers    http.Header
}

func sendRateLimitRequest(ctx context.Context, baseURL, userID, groups, prompt string, maxTokens int) (*rlResponse, error) {
	requestBody := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"max_tokens": maxTokens,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if userID != "" {
		req.Header.Set("x-authz-user-id", userID)
	}
	if groups != "" {
		req.Header.Set("x-authz-user-groups", groups)
	}

	httpClient := &http.Client{Timeout: 60 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	return &rlResponse{
		statusCode: resp.StatusCode,
		body:       body,
		headers:    resp.Header,
	}, nil
}

// uniqueUser generates a unique user ID per test run to avoid cross-test interference.
func uniqueUser(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}

func testFreeTierWithinLimit(ctx context.Context, baseURL string, opts pkgtestcases.TestCaseOptions, results *rateLimitResults) error {
	user := uniqueUser("e2e-free")
	if opts.Verbose {
		fmt.Printf("[RateLimit] Test 1: Free-tier within limit (user=%s)\n", user)
	}

	for i := 1; i <= 3; i++ {
		resp, err := sendRateLimitRequest(ctx, baseURL, user, "free-tier", fmt.Sprintf("hello %d", i), 5)
		if err != nil {
			return fmt.Errorf("request %d: %w", i, err)
		}
		if resp.statusCode != http.StatusOK {
			return fmt.Errorf("request %d: expected 200, got %d (body: %s)", i, resp.statusCode, truncateString(string(resp.body), 200))
		}
		results.passed++
		results.total++
		if opts.Verbose {
			fmt.Printf("[RateLimit]   Request %d: HTTP %d ✓\n", i, resp.statusCode)
		}
	}

	return nil
}

func testFreeTierExceedsLimit(ctx context.Context, baseURL string, opts pkgtestcases.TestCaseOptions, results *rateLimitResults) error {
	// Reuse a known-exhausted user: send 3 to fill, then check 4th
	user := uniqueUser("e2e-free-burst")
	if opts.Verbose {
		fmt.Printf("[RateLimit] Test 2: Free-tier exceeds limit (user=%s)\n", user)
	}

	for i := 1; i <= 3; i++ {
		resp, err := sendRateLimitRequest(ctx, baseURL, user, "free-tier", fmt.Sprintf("fill %d", i), 5)
		if err != nil {
			return fmt.Errorf("fill request %d: %w", i, err)
		}
		if resp.statusCode != http.StatusOK {
			return fmt.Errorf("fill request %d: expected 200, got %d", i, resp.statusCode)
		}
	}

	resp, err := sendRateLimitRequest(ctx, baseURL, user, "free-tier", "this should be blocked", 5)
	if err != nil {
		return fmt.Errorf("overflow request: %w", err)
	}
	if resp.statusCode != http.StatusTooManyRequests {
		return fmt.Errorf("expected 429, got %d (body: %s)", resp.statusCode, truncateString(string(resp.body), 200))
	}

	results.passed++
	results.total++
	if opts.Verbose {
		fmt.Printf("[RateLimit]   4th request: HTTP %d ✓ (rate limited as expected)\n", resp.statusCode)
	}

	return nil
}

func testRateLimitHeaders(ctx context.Context, baseURL string, opts pkgtestcases.TestCaseOptions, results *rateLimitResults) error {
	user := uniqueUser("e2e-free-hdr")
	if opts.Verbose {
		fmt.Printf("[RateLimit] Test 3: Rate limit headers (user=%s)\n", user)
	}

	// Exhaust the limit
	for i := 1; i <= 3; i++ {
		if _, err := sendRateLimitRequest(ctx, baseURL, user, "free-tier", fmt.Sprintf("fill %d", i), 5); err != nil {
			return fmt.Errorf("fill request %d: %w", i, err)
		}
	}

	// The 4th request should be 429 with headers
	resp, err := sendRateLimitRequest(ctx, baseURL, user, "free-tier", "check headers", 5)
	if err != nil {
		return err
	}
	if resp.statusCode != http.StatusTooManyRequests {
		return fmt.Errorf("expected 429, got %d", resp.statusCode)
	}

	requiredHeaders := []string{"Retry-After", "X-Ratelimit-Limit", "X-Ratelimit-Remaining", "X-Ratelimit-Reset"}
	for _, hdr := range requiredHeaders {
		val := resp.headers.Get(hdr)
		if val == "" {
			return fmt.Errorf("missing required header: %s", hdr)
		}
		results.passed++
		results.total++
		if opts.Verbose {
			fmt.Printf("[RateLimit]   %s: %s ✓\n", hdr, val)
		}
	}

	// Validate X-Ratelimit-Limit = 3
	limitVal := resp.headers.Get("X-Ratelimit-Limit")
	if limitVal != "3" {
		return fmt.Errorf("X-Ratelimit-Limit: expected 3, got %s", limitVal)
	}
	results.passed++
	results.total++

	// Validate X-Ratelimit-Remaining = 0
	remainingVal := resp.headers.Get("X-Ratelimit-Remaining")
	if remainingVal != "0" {
		return fmt.Errorf("X-Ratelimit-Remaining: expected 0, got %s", remainingVal)
	}
	results.passed++
	results.total++

	return nil
}

func testRateLimitErrorBody(ctx context.Context, baseURL string, opts pkgtestcases.TestCaseOptions, results *rateLimitResults) error {
	user := uniqueUser("e2e-free-body")
	if opts.Verbose {
		fmt.Printf("[RateLimit] Test 4: Error body format (user=%s)\n", user)
	}

	for i := 1; i <= 3; i++ {
		if _, err := sendRateLimitRequest(ctx, baseURL, user, "free-tier", fmt.Sprintf("fill %d", i), 5); err != nil {
			return fmt.Errorf("fill request %d: %w", i, err)
		}
	}

	resp, err := sendRateLimitRequest(ctx, baseURL, user, "free-tier", "check body", 5)
	if err != nil {
		return err
	}
	if resp.statusCode != http.StatusTooManyRequests {
		return fmt.Errorf("expected 429, got %d", resp.statusCode)
	}

	var errorResp struct {
		Error struct {
			Message string      `json:"message"`
			Type    string      `json:"type"`
			Code    json.Number `json:"code"`
		} `json:"error"`
	}

	if err := json.Unmarshal(resp.body, &errorResp); err != nil {
		return fmt.Errorf("failed to parse 429 body as JSON: %w (body: %s)", err, truncateString(string(resp.body), 200))
	}

	if errorResp.Error.Type != "rate_limit_error" {
		return fmt.Errorf("error.type: expected rate_limit_error, got %s", errorResp.Error.Type)
	}
	results.passed++
	results.total++

	codeInt, err := strconv.Atoi(string(errorResp.Error.Code))
	if err != nil || codeInt != 429 {
		return fmt.Errorf("error.code: expected 429, got %s", errorResp.Error.Code)
	}
	results.passed++
	results.total++

	if !strings.Contains(errorResp.Error.Message, "Rate limit") {
		return fmt.Errorf("error.message: expected to contain 'Rate limit', got %s", errorResp.Error.Message)
	}
	results.passed++
	results.total++

	if opts.Verbose {
		fmt.Printf("[RateLimit]   Body: type=%s, code=%s, message=%s ✓\n",
			errorResp.Error.Type, errorResp.Error.Code, truncateString(errorResp.Error.Message, 60))
	}

	return nil
}

func testPremiumTierHigherLimit(ctx context.Context, baseURL string, opts pkgtestcases.TestCaseOptions, results *rateLimitResults) error {
	user := uniqueUser("e2e-premium")
	if opts.Verbose {
		fmt.Printf("[RateLimit] Test 5: Premium-tier higher limit (user=%s)\n", user)
	}

	for i := 1; i <= 5; i++ {
		resp, err := sendRateLimitRequest(ctx, baseURL, user, "premium-tier", fmt.Sprintf("premium %d", i), 5)
		if err != nil {
			return fmt.Errorf("request %d: %w", i, err)
		}
		if resp.statusCode != http.StatusOK {
			return fmt.Errorf("request %d: expected 200, got %d (body: %s)", i, resp.statusCode, truncateString(string(resp.body), 200))
		}
		results.passed++
		results.total++
		if opts.Verbose {
			fmt.Printf("[RateLimit]   Request %d: HTTP %d ✓\n", i, resp.statusCode)
		}
	}

	return nil
}

func testPerUserIsolation(ctx context.Context, baseURL string, opts pkgtestcases.TestCaseOptions, results *rateLimitResults) error {
	isolatedA := uniqueUser("e2e-free-A")
	isolatedB := uniqueUser("e2e-free-B")
	if opts.Verbose {
		fmt.Printf("[RateLimit] Test 6: Per-user isolation (A=%s, B=%s)\n", isolatedA, isolatedB)
	}

	// Exhaust isolated user A's budget
	for i := 1; i <= 3; i++ {
		if _, err := sendRateLimitRequest(ctx, baseURL, isolatedA, "free-tier", fmt.Sprintf("a %d", i), 5); err != nil {
			return fmt.Errorf("user A request %d: %w", i, err)
		}
	}

	// User A's 4th request should be 429
	respA, err := sendRateLimitRequest(ctx, baseURL, isolatedA, "free-tier", "a overflow", 5)
	if err != nil {
		return fmt.Errorf("user A overflow: %w", err)
	}
	if respA.statusCode != http.StatusTooManyRequests {
		return fmt.Errorf("user A: expected 429, got %d", respA.statusCode)
	}
	results.passed++
	results.total++
	if opts.Verbose {
		fmt.Printf("[RateLimit]   User A 4th request: HTTP %d ✓ (blocked)\n", respA.statusCode)
	}

	// User B should still have full budget
	respB, err := sendRateLimitRequest(ctx, baseURL, isolatedB, "free-tier", "b first", 5)
	if err != nil {
		return fmt.Errorf("user B request: %w", err)
	}
	if respB.statusCode != http.StatusOK {
		return fmt.Errorf("user B: expected 200, got %d (body: %s)", respB.statusCode, truncateString(string(respB.body), 200))
	}
	results.passed++
	results.total++
	if opts.Verbose {
		fmt.Printf("[RateLimit]   User B 1st request: HTTP %d ✓ (independent bucket)\n", respB.statusCode)
	}

	return nil
}
