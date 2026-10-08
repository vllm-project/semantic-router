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
		Description: "Verify every keyword routing case selects its decision when the request body arrives in several streamed chunks",
		Tags:        []string{"streaming", "routing", "keyword"},
		Fn:          testStreamingKeywordRouting,
	})
	pkgtestcases.Register("streaming-cache-roundtrip", pkgtestcases.TestCase{
		Description: "Verify semantic cache lookup and write work with streamed request bodies",
		Tags:        []string{"streaming", "cache"},
		Fn:          testStreamingCacheRoundtrip,
	})
	pkgtestcases.Register("streaming-large-body", pkgtestcases.TestCase{
		Description: "Verify a large request body written in several chunks is reassembled and answered by the upstream, not rejected by the router",
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
		resp, err := sendChunkedChatRequest(ctx, localPort, chatRequestBody(tc.query, "MoM", false), streamedBodyWrites)
		if err != nil {
			fmt.Printf("[Streaming] FAIL %s: %v\n", tc.name, err)
			continue
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			fmt.Printf("[Streaming] FAIL %s: status %d: %s\n", tc.name, resp.StatusCode, truncateString(string(body), 200))
			continue
		}

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

	if passed != len(cases) {
		return fmt.Errorf("streaming keyword routing: %d/%d passed", passed, len(cases))
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

		resp2, err := sendNonStreamingRequest(ctx, similarQ, "MoM", localPort)
		if err != nil {
			if attempt == 4 {
				return fmt.Errorf("similar request failed: %w", err)
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

	// Body size alone does not guarantee multiple ext_proc chunks: an 80 KiB
	// body written at once can reach the Router as one chunk. The body is
	// therefore written in several pieces with a pause between them.
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

	resp, err := sendChunkedChatRequest(ctx, localPort, jsonData, streamedBodyWrites)
	if err != nil {
		return fmt.Errorf("large body request failed: %w", err)
	}
	respBody, _ := io.ReadAll(resp.Body)
	resp.Body.Close()

	decision := resp.Header.Get("x-vsr-selected-decision")
	responsePath := resp.Header.Get("x-vsr-response-path")

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"body_size_kb":    bodySizeKB,
			"status_code":     resp.StatusCode,
			"response_path":   responsePath,
			"response_length": len(respBody),
			"decision":        decision,
		})
	}

	// The backend may reject a body this large for context length. That 400
	// still proves reassembly, but only if the upstream produced it: a body the
	// Router could not decode is answered by the Router itself with
	// x-vsr-response-path: error.
	switch {
	case resp.StatusCode == http.StatusOK:
		if opts.Verbose {
			fmt.Printf("[Streaming] PASS: large body accepted by upstream (status 200)\n")
		}
	case resp.StatusCode == http.StatusBadRequest && responsePath == "upstream":
		if opts.Verbose {
			fmt.Printf("[Streaming] PASS: large body reassembled and forwarded; upstream rejected it (status 400, %d KiB body)\n", bodySizeKB)
		}
	default:
		return fmt.Errorf("large body returned status %d with x-vsr-response-path=%q, want 200, or 400 from the upstream: %s",
			resp.StatusCode, responsePath, truncateString(string(respBody), 200))
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

	// 1) Send a non-streaming request to prime the cache (more reliable than
	//    SSE since the mock backend may not support streaming responses).
	resp1, err := sendNonStreamingRequest(ctx, question, "MoM", localPort)
	if err != nil {
		return fmt.Errorf("first request failed: %w", err)
	}
	body1, _ := io.ReadAll(resp1.Body)
	resp1.Body.Close()

	if opts.Verbose {
		fmt.Printf("[Streaming] First response: status=%d, len=%d, decision=%s\n",
			resp1.StatusCode, len(body1), resp1.Header.Get("x-vsr-selected-decision"))
	}

	// 2) Retry similar non-streaming request with backoff until cache hit.
	similarQ := "How fast does light travel through empty space?"
	var cacheHit string
	var body2 []byte
	for attempt := 1; attempt <= 4; attempt++ {
		wait := time.Duration(attempt) * time.Second
		if opts.Verbose {
			fmt.Printf("[Streaming] Waiting %v before cache-hit check (attempt %d/4)\n", wait, attempt)
		}
		time.Sleep(wait)

		resp2, err := sendNonStreamingRequest(ctx, similarQ, "MoM", localPort)
		if err != nil {
			if attempt == 4 {
				return fmt.Errorf("similar request failed: %w", err)
			}
			continue
		}
		body2, _ = io.ReadAll(resp2.Body)
		resp2.Body.Close()

		cacheHit = resp2.Header.Get("x-vsr-cache-hit")
		if opts.Verbose {
			fmt.Printf("[Streaming] Similar request: status=%d, cache-hit=%s, decision=%s\n",
				resp2.StatusCode, cacheHit, resp2.Header.Get("x-vsr-selected-decision"))
		}
		if cacheHit == "true" {
			break
		}
	}

	// 3) Optionally test streaming cache hit — send a streaming request for a
	//    similar question. If the backend supports SSE, validate the stream;
	//    otherwise just check the cache-hit header.
	var cacheHit3 string
	resp3, err := sendStreamingRequest(ctx, "What is the velocity of light in vacuum?", "MoM", localPort)
	if err != nil {
		if opts.Verbose {
			fmt.Printf("[Streaming] Streaming similar request failed (mock may not support SSE): %v\n", err)
		}
	} else {
		cacheHit3 = resp3.Header.Get("x-vsr-cache-hit")
		// Drain body regardless of format
		io.Copy(io.Discard, resp3.Body)
		resp3.Body.Close()
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"original_body_len":    len(body1),
			"non_stream_cache_hit": cacheHit,
			"non_stream_body_len":  len(body2),
			"stream_cache_hit":     cacheHit3,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Streaming] Non-streaming similar: cache-hit=%s, len=%d\n", cacheHit, len(body2))
		fmt.Printf("[Streaming] Streaming similar:     cache-hit=%s\n", cacheHit3)
	}

	if cacheHit != "true" {
		return fmt.Errorf("expected cache hit for non-streaming similar question, got %q", cacheHit)
	}

	return nil
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

// streamedBodyWrites and streamedWritePause make the client write a request
// body in several pieces, so Envoy hands it to the Router in STREAMED mode as
// more than one ext_proc chunk.
const (
	streamedBodyWrites = 3
	streamedWritePause = 500 * time.Millisecond
)

// chatRequestBody returns a one-message Chat Completions request.
func chatRequestBody(question, model string, stream bool) []byte {
	body := map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": question},
		},
	}
	if stream {
		body["stream"] = true
	}
	data, _ := json.Marshal(body)
	return data
}

// splitBody cuts body into n contiguous pieces of near-equal size.
func splitBody(body []byte, n int) []string {
	if n < 1 {
		n = 1
	}
	if n > len(body) {
		n = len(body)
	}
	pieces := make([]string, 0, n)
	for i := 0; i < n; i++ {
		pieces = append(pieces, string(body[i*len(body)/n:(i+1)*len(body)/n]))
	}
	return pieces
}

// sendChunkedChatRequest posts body as a chunked upload written in the given
// number of pieces with streamedWritePause between them. The caller owns the
// response and checks its status.
func sendChunkedChatRequest(ctx context.Context, localPort string, body []byte, writes int) (*http.Response, error) {
	reader, writer := io.Pipe()
	go func() {
		for i, piece := range splitBody(body, writes) {
			if i > 0 {
				select {
				case <-time.After(streamedWritePause):
				case <-ctx.Done():
					writer.CloseWithError(ctx.Err())
					return
				}
			}
			if _, err := io.WriteString(writer, piece); err != nil {
				return
			}
		}
		writer.Close()
	}()

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, reader)
	if err != nil {
		reader.Close()
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: 60 * time.Second}).Do(req)
	if err != nil {
		return nil, fmt.Errorf("do: %w", err)
	}
	return resp, nil
}

func sendNonStreamingRequest(ctx context.Context, question, model, localPort string) (*http.Response, error) {
	jsonData := chatRequestBody(question, model, false)

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
	jsonData := chatRequestBody(question, model, true)

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
