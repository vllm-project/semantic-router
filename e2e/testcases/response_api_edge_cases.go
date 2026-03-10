package testcases

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
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
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	storeTrue := true
	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	resp, raw, err := apiClient.Create(ctx, fixtures.ResponseAPIRequest{
		Model:    "openai/gpt-oss-20b",
		Input:    "",
		Store:    &storeTrue,
		Metadata: map[string]string{"test": "response-api-edge-empty-input"},
	})
	if err != nil {
		return fmt.Errorf("empty input request failed: %w", err)
	}

	echo, err := parseMockEcho(resp, raw.Body)
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
	return nil
}

func testResponseAPIEdgeLargeInput(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	sentence := "The quick brown fox jumps over the lazy dog. "
	largeInput := strings.Repeat(sentence, responseAPILargeInputSize/len(sentence)+1)
	largeInput = largeInput[:responseAPILargeInputSize]
	storeFalse := false
	apiClient := fixtures.NewResponseAPIClient(session, 60*time.Second)
	resp, raw, err := apiClient.Create(ctx, fixtures.ResponseAPIRequest{
		Model:    "openai/gpt-oss-20b",
		Input:    largeInput,
		Store:    &storeFalse,
		Metadata: map[string]string{"test": "response-api-edge-large-input"},
	})
	if err != nil {
		return fmt.Errorf("large input request failed: %w", err)
	}

	echo, err := parseMockEcho(resp, raw.Body)
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
	return nil
}

func testResponseAPIEdgeSpecialCharacters(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	specialInput := "Line1\nLine2\tTabbed \"quote\" \\ backslash / slash <tag> [array] {json} | pipe ^ caret ~ tilde"
	storeTrue := true
	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	resp, raw, err := apiClient.Create(ctx, fixtures.ResponseAPIRequest{
		Model:    "openai/gpt-oss-20b",
		Input:    specialInput,
		Store:    &storeTrue,
		Metadata: map[string]string{"test": "response-api-edge-special-characters"},
	})
	if err != nil {
		return fmt.Errorf("special characters request failed: %w", err)
	}

	echo, err := parseMockEcho(resp, raw.Body)
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
	return nil
}

func testResponseAPIEdgeConcurrentRequests(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	storeTrue := true

	summary := newConcurrentResponseSummary(responseAPIConcurrentRequests)
	work := newConcurrentRequestQueue()
	var wg sync.WaitGroup
	wg.Add(responseAPIConcurrentConcurrency)
	for i := 0; i < responseAPIConcurrentConcurrency; i++ {
		go runConcurrentResponseWorker(ctx, apiClient, &storeTrue, work, summary, &wg)
	}
	wg.Wait()

	if err := summary.Validate(); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"requests":    responseAPIConcurrentRequests,
			"concurrency": responseAPIConcurrentConcurrency,
			"responses":   summary.ResponseCount(),
			"duplicates":  summary.duplicateCount,
			"error_count": summary.errorCount,
		})
	}
	return nil
}

type concurrentResponseSummary struct {
	mu             sync.Mutex
	errorCount     int
	duplicateCount int
	firstErr       string
	expected       int
	ids            map[string]struct{}
}

func newConcurrentRequestQueue() chan int {
	work := make(chan int, responseAPIConcurrentRequests)
	for i := 0; i < responseAPIConcurrentRequests; i++ {
		work <- i + 1
	}
	close(work)
	return work
}

func newConcurrentResponseSummary(expected int) *concurrentResponseSummary {
	return &concurrentResponseSummary{
		expected: expected,
		ids:      make(map[string]struct{}, expected),
	}
}

func runConcurrentResponseWorker(
	ctx context.Context,
	apiClient *fixtures.ResponseAPIClient,
	storeTrue *bool,
	work <-chan int,
	summary *concurrentResponseSummary,
	wg *sync.WaitGroup,
) {
	defer wg.Done()
	for id := range work {
		if ctx.Err() != nil {
			return
		}
		resp, _, err := apiClient.Create(ctx, fixtures.ResponseAPIRequest{
			Model:    "openai/gpt-oss-20b",
			Input:    fmt.Sprintf("concurrent-request-%d", id),
			Store:    storeTrue,
			Metadata: map[string]string{"test": "response-api-edge-concurrent-requests"},
		})
		summary.Record(resp, err)
	}
}

func (s *concurrentResponseSummary) Record(resp *fixtures.ResponseAPIResponse, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err != nil {
		s.errorCount++
		s.recordFirstError(err.Error())
		return
	}
	if resp == nil || resp.ID == "" || !strings.HasPrefix(resp.ID, "resp_") {
		s.errorCount++
		if resp == nil {
			s.recordFirstError("nil response")
			return
		}
		s.recordFirstError(fmt.Sprintf("invalid response id: %q", resp.ID))
		return
	}
	if _, exists := s.ids[resp.ID]; exists {
		s.duplicateCount++
		s.recordFirstError(fmt.Sprintf("duplicate response id: %q", resp.ID))
		return
	}
	s.ids[resp.ID] = struct{}{}
}

func (s *concurrentResponseSummary) Validate() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.errorCount > 0 || s.duplicateCount > 0 {
		return fmt.Errorf("concurrent requests failed: errors=%d duplicates=%d first=%s", s.errorCount, s.duplicateCount, s.firstErr)
	}
	if len(s.ids) != s.expected {
		return fmt.Errorf("expected %d successful responses, got %d", s.expected, len(s.ids))
	}
	return nil
}

func (s *concurrentResponseSummary) ResponseCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.ids)
}

func (s *concurrentResponseSummary) recordFirstError(message string) {
	if s.firstErr == "" {
		s.firstErr = message
	}
}
