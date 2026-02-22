package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const (
	responseAPILargeInputSize        = 16000
	responseAPIConcurrentRequests    = 20
	responseAPIConcurrentConcurrency = 5
)

func init() {
	pkgtestcases.Register("response-api-edge-empty-input", pkgtestcases.TestCase{
		Description: "Edge case - Empty input",
		Tags:        []string{"response-api", "edge-case"},
		Fn:          testResponseAPIEdgeEmptyInput,
	})
	pkgtestcases.Register("response-api-edge-large-input", pkgtestcases.TestCase{
		Description: "Edge case - Large input payload",
		Tags:        []string{"response-api", "edge-case"},
		Fn:          testResponseAPIEdgeLargeInput,
	})
	pkgtestcases.Register("response-api-edge-special-characters", pkgtestcases.TestCase{
		Description: "Edge case - Special characters in input",
		Tags:        []string{"response-api", "edge-case"},
		Fn:          testResponseAPIEdgeSpecialCharacters,
	})
	pkgtestcases.Register("response-api-edge-concurrent-requests", pkgtestcases.TestCase{
		Description: "Edge case - Concurrent requests",
		Tags:        []string{"response-api", "edge-case"},
		Fn:          testResponseAPIEdgeConcurrentRequests,
	})
}

func testResponseAPIEdgeEmptyInput(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API edge case: empty input")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	storeTrue := true
	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, raw, err := postResponseAPI(ctx, httpClient, localPort, ResponseAPIRequest{
		Model:    "openai/gpt-oss-20b",
		Input:    "",
		Store:    &storeTrue,
		Metadata: map[string]string{"test": "response-api-edge-empty-input"},
	})
	if err != nil {
		return fmt.Errorf("empty input request failed: %w", err)
	}

	echo, err := parseMockEcho(resp, raw)
	if err != nil {
		return fmt.Errorf("empty input echo parse failed: %w", err)
	}
	if len(echo.User) != 1 || echo.User[0] != "" {
		return fmt.Errorf("empty input should reach backend as single empty user message, got user=%v", echo.User)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"response_id":   resp.ID,
			"user_messages": len(echo.User),
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Empty input handled successfully (id=%s)\n", resp.ID)
	}

	return nil
}

func testResponseAPIEdgeLargeInput(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API edge case: large input")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	largeInput := strings.Repeat("The quick brown fox jumps over the lazy dog. ", responseAPILargeInputSize/46+1)
	largeInput = largeInput[:responseAPILargeInputSize]
	storeFalse := false
	httpClient := &http.Client{Timeout: 60 * time.Second}
	resp, raw, err := postResponseAPI(ctx, httpClient, localPort, ResponseAPIRequest{
		Model:    "openai/gpt-oss-20b",
		Input:    largeInput,
		Store:    &storeFalse,
		Metadata: map[string]string{"test": "response-api-edge-large-input"},
	})
	if err != nil {
		return fmt.Errorf("large input request failed: %w", err)
	}

	echo, err := parseMockEcho(resp, raw)
	if err != nil {
		return fmt.Errorf("large input echo parse failed: %w", err)
	}
	actualLen := 0
	if len(echo.User) == 1 {
		actualLen = len(echo.User[0])
	}
	if len(echo.User) != 1 || echo.User[0] != largeInput {
		return fmt.Errorf("large input should be preserved in backend echo (len=%d), got user_count=%d user_len=%d", len(largeInput), len(echo.User), actualLen)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"response_id": resp.ID,
			"input_len":   len(largeInput),
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Large input handled successfully (len=%d, id=%s)\n", len(largeInput), resp.ID)
	}

	return nil
}

func testResponseAPIEdgeSpecialCharacters(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API edge case: special characters in input")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	specialInput := "Line1\nLine2\tTabbed \"quote\" \\ backslash / slash <tag> [array] {json} | pipe ^ caret ~ tilde"
	storeTrue := true
	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, raw, err := postResponseAPI(ctx, httpClient, localPort, ResponseAPIRequest{
		Model:    "openai/gpt-oss-20b",
		Input:    specialInput,
		Store:    &storeTrue,
		Metadata: map[string]string{"test": "response-api-edge-special-characters"},
	})
	if err != nil {
		return fmt.Errorf("special characters request failed: %w", err)
	}

	echo, err := parseMockEcho(resp, raw)
	if err != nil {
		return fmt.Errorf("special characters echo parse failed: %w", err)
	}
	if len(echo.User) != 1 || echo.User[0] != specialInput {
		return fmt.Errorf("special characters input should be preserved, got user=%v", echo.User)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"response_id": resp.ID,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Special characters handled successfully (id=%s)\n", resp.ID)
	}

	return nil
}

func testResponseAPIEdgeConcurrentRequests(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API edge case: concurrent requests")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	httpClient := &http.Client{Timeout: 30 * time.Second}
	storeTrue := true

	work := make(chan int, responseAPIConcurrentRequests)
	for i := 0; i < responseAPIConcurrentRequests; i++ {
		work <- i + 1
	}
	close(work)

	var (
		wg             sync.WaitGroup
		mu             sync.Mutex
		errorCount     int
		duplicateCount int
		firstErr       string
		ids            = make(map[string]struct{}, responseAPIConcurrentRequests)
	)

	wg.Add(responseAPIConcurrentConcurrency)
	for i := 0; i < responseAPIConcurrentConcurrency; i++ {
		go func() {
			defer wg.Done()
			for id := range work {
				if ctx.Err() != nil {
					return
				}
				input := fmt.Sprintf("concurrent-request-%d", id)
				resp, _, err := postResponseAPI(ctx, httpClient, localPort, ResponseAPIRequest{
					Model:    "openai/gpt-oss-20b",
					Input:    input,
					Store:    &storeTrue,
					Metadata: map[string]string{"test": "response-api-edge-concurrent-requests"},
				})

				mu.Lock()
				if err != nil {
					errorCount++
					if firstErr == "" {
						firstErr = err.Error()
					}
					mu.Unlock()
					continue
				}

				if resp.ID == "" || !strings.HasPrefix(resp.ID, "resp_") {
					errorCount++
					if firstErr == "" {
						firstErr = fmt.Sprintf("invalid response id: %q", resp.ID)
					}
					mu.Unlock()
					continue
				}
				if _, exists := ids[resp.ID]; exists {
					duplicateCount++
					if firstErr == "" {
						firstErr = fmt.Sprintf("duplicate response id: %q", resp.ID)
					}
					mu.Unlock()
					continue
				}
				ids[resp.ID] = struct{}{}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests":   responseAPIConcurrentRequests,
			"concurrency":      responseAPIConcurrentConcurrency,
			"success_count":    responseAPIConcurrentRequests - errorCount,
			"errors":           errorCount,
			"duplicate_ids":    duplicateCount,
			"unique_responses": len(ids),
		})
	}

	if errorCount > 0 || duplicateCount > 0 {
		return fmt.Errorf("concurrent requests had %d errors and %d duplicate IDs (first error: %s)", errorCount, duplicateCount, firstErr)
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Concurrent requests handled successfully (total=%d, unique_ids=%d)\n", responseAPIConcurrentRequests, len(ids))
	}

	return nil
}
