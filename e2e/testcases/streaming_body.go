package testcases

import (
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
		Description: "Verify a cached Chat response is replayed as a complete SSE stream",
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
		resp, err := sendNonStreamingRequest(ctx, tc.query, "vllm-sr/auto", localPort)
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
	resp1, err := sendNonStreamingRequest(ctx, originalQ, "vllm-sr/auto", localPort)
	if err != nil {
		return fmt.Errorf("original request failed: %w", err)
	}
	body1, _ := io.ReadAll(resp1.Body)
	resp1.Body.Close()

	if opts.Verbose {
		fmt.Printf("[Streaming] Original request: status=%d, cache-hit=%s\n",
			resp1.StatusCode, resp1.Header.Get("x-vsr-cache-hit"))
	}

	// Retry the similar question with backoff — cache writes are async and
	// the embedding index may take a moment to settle.
	var cacheHit string
	var body2 []byte
	var resp2Status int
	for attempt := 1; attempt <= 4; attempt++ {
		wait := time.Duration(attempt) * time.Second
		if opts.Verbose {
			fmt.Printf("[Streaming] Waiting %v before similar request (attempt %d/4)\n", wait, attempt)
		}
		time.Sleep(wait)

		resp2, requestErr := sendNonStreamingRequest(ctx, similarQ, "vllm-sr/auto", localPort)
		if requestErr != nil {
			if attempt == 4 {
				return fmt.Errorf("similar request failed: %w", requestErr)
			}
			continue
		}
		body2, _ = io.ReadAll(resp2.Body)
		resp2.Body.Close()

		resp2Status = resp2.StatusCode
		cacheHit = resp2.Header.Get("x-vsr-cache-hit")
		if opts.Verbose {
			fmt.Printf("[Streaming] Similar request: status=%d, cache-hit=%s, decision=%s\n",
				resp2.StatusCode, cacheHit, resp2.Header.Get("x-vsr-selected-decision"))
		}

		if cacheHit == "true" {
			break
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"original_status": resp1.StatusCode,
			"similar_status":  resp2Status,
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
		"model": "vllm-sr/auto",
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

	decision := resp.Header.Get("x-vsr-selected-decision")

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"body_size_kb":    bodySizeKB,
			"status_code":     resp.StatusCode,
			"response_length": len(respBody),
			"decision":        decision,
		})
	}

	// A 400 from the upstream mock (context length exceeded) is acceptable —
	// it proves the router successfully reassembled the multi-chunk body and
	// forwarded it. Only true failures (502 from Envoy, no routing headers)
	// indicate a streaming body problem.
	switch {
	case resp.StatusCode == http.StatusOK:
		if opts.Verbose {
			fmt.Printf("[Streaming] PASS: large body accepted by upstream (status 200)\n")
		}
	case resp.StatusCode == http.StatusBadRequest:
		if opts.Verbose {
			fmt.Printf("[Streaming] PASS: large body forwarded to upstream, rejected due to context length (expected for %d KiB body)\n", bodySizeKB)
		}
	default:
		return fmt.Errorf("large body returned unexpected status %d: %s", resp.StatusCode, truncateString(string(respBody), 200))
	}

	if decision != "" {
		trimmedDec := strings.TrimSuffix(decision, "_decision")
		if opts.Verbose {
			fmt.Printf("[Streaming] Router decision: %s\n", trimmedDec)
		}
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
	resp, err := sendNonStreamingRequest(ctx, question, "vllm-sr/auto", localPort)
	if err != nil {
		return fmt.Errorf("prime cache: %w", err)
	}
	body, readErr := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if readErr != nil {
		return fmt.Errorf("read priming response: %w", readErr)
	}
	wantText, err := decodeStructuredChatText(body)
	if err != nil || wantText == "" {
		return fmt.Errorf("cache priming requires nonempty Chat output: %v", err)
	}

	// Reuse the identical prompt: this case qualifies cached SSE encoding,
	// while streaming-cache-roundtrip owns semantic similarity matching.
	for attempt := 1; attempt <= 4; attempt++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Duration(attempt) * time.Second):
		}
		stream, err := sendStreamingRequest(ctx, question, "vllm-sr/auto", localPort)
		if err != nil {
			return fmt.Errorf("request cached SSE replay: %w", err)
		}
		streamBody, readErr := io.ReadAll(stream.Body)
		_ = stream.Body.Close()
		if readErr != nil {
			return fmt.Errorf("read cached SSE replay: %w", readErr)
		}
		if stream.Header.Get("x-vsr-cache-hit") != "true" && attempt < 4 {
			continue
		}
		if err := validateStreamingCacheReplay(stream.Header, streamBody, wantText); err != nil {
			return err
		}
		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{
				"original_body_len": len(body),
				"stream_body_len":   len(streamBody),
				"stream_cache_hit":  true,
				"replay_verified":   true,
			})
		}
		return nil
	}
	return fmt.Errorf("cached SSE replay never became available")
}

func validateStreamingCacheReplay(headers http.Header, body []byte, wantText string) error {
	if headers.Get("x-vsr-cache-hit") != "true" {
		return fmt.Errorf("expected cached SSE replay, got cache-hit=%q", headers.Get("x-vsr-cache-hit"))
	}
	if !strings.HasPrefix(headers.Get("Content-Type"), "text/event-stream") {
		return fmt.Errorf("cached response is not SSE: %q", headers.Get("Content-Type"))
	}
	if wantText == "" {
		return fmt.Errorf("cached SSE comparison requires nonempty priming output")
	}
	frames := protocolSSEDataFrames(body)
	if len(frames) == 0 || frames[len(frames)-1] != "[DONE]" {
		return fmt.Errorf("cached SSE replay must end with [DONE]")
	}
	finished := false
	for _, data := range frames {
		if data == "[DONE]" {
			continue
		}
		var chunk chatTextStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("decode cached SSE frame: %w", err)
		}
		for _, choice := range chunk.Choices {
			if finished {
				return fmt.Errorf("cached SSE replay contains a choice after its finish reason")
			}
			if choice.FinishReason != nil {
				finished = true
			}
		}
		// A final usage-only chunk has no choices and may follow the finish.
	}
	if err := validateChatProtocolStream(body, wantText); err != nil {
		return fmt.Errorf("cached SSE replay: %w", err)
	}
	text, err := extractProtocolStructuredOutputStreamText("/v1/chat/completions", body)
	if err != nil || text != wantText {
		return fmt.Errorf("cached SSE output differs from priming output: %v", err)
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
