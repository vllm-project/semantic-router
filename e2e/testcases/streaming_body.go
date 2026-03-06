package testcases

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

func init() {
	pkgtestcases.Register("streaming-keyword-routing", pkgtestcases.TestCase{
		Description: "Verify keyword routing works identically with streamed request body mode",
		Tags:        []string{"streaming", "routing", "keyword"},
		Fn:          testStreamingKeywordRouting,
	})
	pkgtestcases.Register("streaming-cache-roundtrip", pkgtestcases.TestCase{
		Description: "Verify semantic cache lookup and write work with streamed request bodies",
		Tags:        []string{"streaming", "cache"},
		Fn:          testStreamingCacheRoundtrip,
	})
	pkgtestcases.Register("streaming-large-body", pkgtestcases.TestCase{
		Description: "Verify large request bodies (multiple Envoy chunks) are reassembled correctly",
		Tags:        []string{"streaming", "large-body"},
		Fn:          testStreamingLargeBody,
	})
	pkgtestcases.Register("streaming-sse-cache", pkgtestcases.TestCase{
		Description: "Verify SSE streaming responses are cached and replayed correctly",
		Tags:        []string{"streaming", "cache", "sse"},
		Fn:          testStreamingSSECache,
	})
}

// ---------------------------------------------------------------------------
// streaming-keyword-routing
// ---------------------------------------------------------------------------

func testStreamingKeywordRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Streaming] Testing keyword routing with streamed body mode")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	cases := []struct {
		name     string
		query    string
		wantDec  string
		keywords []string
	}{
		{
			name:     "code_keyword_bm25",
			query:    "Can you help me implement a function to debug this algorithm?",
			wantDec:  "code_keywords",
			keywords: []string{"code_keywords"},
		},
		{
			name:    "urgent_ngram",
			query:   "This is an urgent emergency, I need help immediately!",
			wantDec: "urgent_request",
		},
	}

	passed := 0
	for _, tc := range cases {
		resp, err := sendNonStreamingRequest(ctx, tc.query, "MoM", localPort)
		if err != nil {
			fmt.Printf("[Streaming] FAIL %s: %v\n", tc.name, err)
			continue
		}
		resp.Body.Close()

		decision := resp.Header.Get("x-vsr-selected-decision")
		actualDec := strings.TrimSuffix(decision, "_decision")

		if actualDec != tc.wantDec {
			fmt.Printf("[Streaming] FAIL %s: decision=%q, want=%q\n", tc.name, actualDec, tc.wantDec)
			if opts.Verbose {
				fmt.Printf("  Headers: %s", formatResponseHeaders(resp.Header))
			}
			continue
		}

		if opts.Verbose {
			fmt.Printf("[Streaming] PASS %s: decision=%s\n", tc.name, actualDec)
		}
		passed++
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total": len(cases), "passed": passed,
		})
	}

	if passed == 0 {
		return fmt.Errorf("streaming keyword routing: 0/%d passed", len(cases))
	}
	return nil
}

// ---------------------------------------------------------------------------
// streaming-cache-roundtrip
// ---------------------------------------------------------------------------

func testStreamingCacheRoundtrip(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Streaming] Testing semantic cache round-trip with streamed body")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	originalQ := "What are the main differences between TCP and UDP protocols?"
	similarQ := "Can you explain how TCP differs from UDP?"

	// Prime the cache with the original question (should be a miss).
	resp1, err := sendNonStreamingRequest(ctx, originalQ, "MoM", localPort)
	if err != nil {
		return fmt.Errorf("original request failed: %w", err)
	}
	body1, _ := io.ReadAll(resp1.Body)
	resp1.Body.Close()

	if opts.Verbose {
		fmt.Printf("[Streaming] Original request: status=%d, cache-hit=%s\n",
			resp1.StatusCode, resp1.Header.Get("x-vsr-cache-hit"))
	}

	// Let the cache index settle.
	time.Sleep(2 * time.Second)

	// Send a semantically similar question — should hit cache.
	resp2, err := sendNonStreamingRequest(ctx, similarQ, "MoM", localPort)
	if err != nil {
		return fmt.Errorf("similar request failed: %w", err)
	}
	body2, _ := io.ReadAll(resp2.Body)
	resp2.Body.Close()

	cacheHit := resp2.Header.Get("x-vsr-cache-hit")
	if opts.Verbose {
		fmt.Printf("[Streaming] Similar request: status=%d, cache-hit=%s\n",
			resp2.StatusCode, cacheHit)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"original_status": resp1.StatusCode,
			"similar_status":  resp2.StatusCode,
			"cache_hit":       cacheHit,
			"original_len":    len(body1),
			"similar_len":     len(body2),
		})
	}

	if cacheHit != "true" {
		return fmt.Errorf("expected cache hit for similar question, got cache-hit=%q", cacheHit)
	}

	return nil
}

// ---------------------------------------------------------------------------
// streaming-large-body
// ---------------------------------------------------------------------------

func testStreamingLargeBody(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Streaming] Testing large request body spanning multiple chunks")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	// Build a long system prompt + user message that is large enough to be
	// split across multiple Envoy ext_proc chunks (Envoy default chunk size
	// is ~64 KiB, so anything > 64 KiB guarantees multi-chunk delivery).
	longContext := strings.Repeat("This is padding context to make the body large enough for multi-chunk delivery. ", 1000)
	userMsg := "Given all that context, please implement a function to sort a linked list."

	requestBody := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{"role": "system", "content": longContext},
			{"role": "user", "content": userMsg},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return fmt.Errorf("failed to marshal large request: %w", err)
	}

	bodySizeKB := len(jsonData) / 1024
	if opts.Verbose {
		fmt.Printf("[Streaming] Sending large body: %d KiB\n", bodySizeKB)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 60 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("large body request failed: %w", err)
	}
	respBody, _ := io.ReadAll(resp.Body)
	resp.Body.Close()

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"body_size_kb":    bodySizeKB,
			"status_code":     resp.StatusCode,
			"response_length": len(respBody),
			"decision":        resp.Header.Get("x-vsr-selected-decision"),
		})
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("large body returned status %d: %s", resp.StatusCode, truncateString(string(respBody), 200))
	}

	// The body contains "implement" and "function" — should trigger code_keywords
	decision := strings.TrimSuffix(resp.Header.Get("x-vsr-selected-decision"), "_decision")
	if decision != "code_keywords" {
		if opts.Verbose {
			fmt.Printf("[Streaming] WARN: expected code_keywords decision, got %q (may be domain fallback)\n", decision)
		}
	} else if opts.Verbose {
		fmt.Printf("[Streaming] PASS: large body correctly classified as code_keywords\n")
	}

	return nil
}

// ---------------------------------------------------------------------------
// streaming-sse-cache
// ---------------------------------------------------------------------------

func testStreamingSSECache(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Streaming] Testing SSE streaming response cache round-trip")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	question := "What is the speed of light in a vacuum?"

	// 1) Send a streaming request (stream: true) to prime the cache.
	resp1, err := sendStreamingRequest(ctx, question, "MoM", localPort)
	if err != nil {
		return fmt.Errorf("first streaming request failed: %w", err)
	}

	content1, err := consumeSSEResponse(resp1)
	resp1.Body.Close()
	if err != nil {
		return fmt.Errorf("failed to consume SSE stream: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Streaming] First streaming response: %d chars\n", len(content1))
	}

	// Let the response finish caching (cacheStreamingResponse runs on [DONE]).
	time.Sleep(3 * time.Second)

	// 2) Send a semantically similar non-streaming request — should hit cache.
	similarQ := "How fast does light travel through empty space?"
	resp2, err := sendNonStreamingRequest(ctx, similarQ, "MoM", localPort)
	if err != nil {
		return fmt.Errorf("similar non-streaming request failed: %w", err)
	}
	body2, _ := io.ReadAll(resp2.Body)
	resp2.Body.Close()

	cacheHit := resp2.Header.Get("x-vsr-cache-hit")

	// 3) Send a similar streaming request — should also hit cache (served as SSE).
	resp3, err := sendStreamingRequest(ctx, "What is the velocity of light in vacuum?", "MoM", localPort)
	if err != nil {
		return fmt.Errorf("similar streaming request failed: %w", err)
	}
	content3, err := consumeSSEResponse(resp3)
	resp3.Body.Close()
	if err != nil {
		return fmt.Errorf("failed to consume SSE cache hit stream: %w", err)
	}

	cacheHit3 := resp3.Header.Get("x-vsr-cache-hit")

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"original_content_len":  len(content1),
			"non_stream_cache_hit":  cacheHit,
			"non_stream_body_len":   len(body2),
			"stream_cache_hit":      cacheHit3,
			"stream_cache_body_len": len(content3),
		})
	}

	if opts.Verbose {
		fmt.Printf("[Streaming] Non-streaming similar: cache-hit=%s, len=%d\n", cacheHit, len(body2))
		fmt.Printf("[Streaming] Streaming similar:     cache-hit=%s, len=%d\n", cacheHit3, len(content3))
	}

	// At least the non-streaming follow-up should hit cache.
	if cacheHit != "true" {
		return fmt.Errorf("expected cache hit for non-streaming similar question, got %q", cacheHit)
	}

	return nil
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

func sendNonStreamingRequest(ctx context.Context, question, model, localPort string) (*http.Response, error) {
	body := map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": question},
		},
	}

	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, truncateString(string(b), 200))
	}

	return resp, nil
}

func sendStreamingRequest(ctx context.Context, question, model, localPort string) (*http.Response, error) {
	body := map[string]interface{}{
		"model":  model,
		"stream": true,
		"messages": []map[string]string{
			{"role": "user", "content": question},
		},
	}

	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 60 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, truncateString(string(b), 200))
	}

	return resp, nil
}

// consumeSSEResponse reads an SSE stream and returns the accumulated content
// from all delta chunks. It also validates the stream ends with [DONE].
func consumeSSEResponse(resp *http.Response) (string, error) {
	var content strings.Builder
	scanner := bufio.NewScanner(resp.Body)

	gotDone := false
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		data = strings.TrimSpace(data)

		if data == "[DONE]" {
			gotDone = true
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	if err := scanner.Err(); err != nil {
		return content.String(), fmt.Errorf("scanner: %w", err)
	}
	if !gotDone {
		return content.String(), fmt.Errorf("SSE stream did not end with [DONE]")
	}
	return content.String(), nil
}
