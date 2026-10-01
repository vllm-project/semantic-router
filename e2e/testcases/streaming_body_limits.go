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

// These bounds match the streaming E2E profile's test-only configuration.
const streamingE2EMaxBodyBytes = 128 * 1024

func init() {
	pkgtestcases.Register("streaming-body-size-limit", pkgtestcases.TestCase{
		Description: "Verify an over-limit streamed upload gets HTTP 413 from the router",
		Tags:        []string{"streaming", "body-limit", "regression"},
		Fn:          testStreamingBodySizeLimit,
	})
	pkgtestcases.Register("streaming-body-deadline", pkgtestcases.TestCase{
		Description: "Verify a stalled streamed upload gets HTTP 408 from the router",
		Tags:        []string{"streaming", "body-timeout", "regression"},
		Fn:          testStreamingBodyDeadline,
	})
}

func testStreamingBodySizeLimit(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	// Send more than the configured 128 KiB limit through Envoy's STREAMED
	// ExtProc mode. This is also larger than Envoy's usual body chunk size.
	body, err := json.Marshal(map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{"role": "user", "content": strings.Repeat("a", 192*1024)},
		},
	})
	if err != nil {
		return fmt.Errorf("marshal over-limit request: %w", err)
	}
	if len(body) <= streamingE2EMaxBodyBytes {
		return fmt.Errorf("test request is only %d bytes; expected more than %d", len(body), streamingE2EMaxBodyBytes)
	}

	resp, err := sendStreamedBodyLimitRequest(ctx, localPort, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("send over-limit request: %w", err)
	}
	if assertionErr := assertStreamedBodyRejection(resp, http.StatusRequestEntityTooLarge, "request_too_large"); assertionErr != nil {
		return assertionErr
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"request_bytes": len(body),
			"limit_bytes":   streamingE2EMaxBodyBytes,
			"status_code":   resp.StatusCode,
		})
	}
	return nil
}

func testStreamingBodyDeadline(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	// The first part exceeds Envoy's usual chunk size, so the router sees a
	// non-EOS chunk and starts its three-second accumulation deadline. Delay
	// the final part for six seconds; the total body remains below 128 KiB.
	first := `{"model":"MoM","messages":[{"role":"user","content":"` + strings.Repeat("a", 80*1024)
	last := `"}]}`
	if len(first)+len(last) >= streamingE2EMaxBodyBytes {
		return fmt.Errorf("timed request exceeds the size limit: %d bytes", len(first)+len(last))
	}
	body := io.MultiReader(
		strings.NewReader(first),
		&delayedStreamedBodyReader{ctx: ctx, delay: 6 * time.Second, body: strings.NewReader(last)},
	)
	resp, err := sendStreamedBodyLimitRequest(ctx, localPort, body)
	if err != nil {
		return fmt.Errorf("send delayed request: %w", err)
	}
	if assertionErr := assertStreamedBodyRejection(resp, http.StatusRequestTimeout, "request_timeout"); assertionErr != nil {
		return assertionErr
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"request_bytes": len(first) + len(last),
			"pause_seconds": 6,
			"status_code":   resp.StatusCode,
		})
	}
	return nil
}

type delayedStreamedBodyReader struct {
	ctx   context.Context
	delay time.Duration
	body  io.Reader
	ready bool
}

func (r *delayedStreamedBodyReader) Read(p []byte) (int, error) {
	if !r.ready {
		r.ready = true
		timer := time.NewTimer(r.delay)
		defer timer.Stop()
		select {
		case <-timer.C:
		case <-r.ctx.Done():
			return 0, r.ctx.Err()
		}
	}
	return r.body.Read(p)
}

func sendStreamedBodyLimitRequest(ctx context.Context, localPort string, body io.Reader) (*http.Response, error) {
	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return (&http.Client{Timeout: 25 * time.Second}).Do(req)
}

func assertStreamedBodyRejection(resp *http.Response, wantStatus int, wantCode string) error {
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if err != nil {
		return fmt.Errorf("read body-limit response: %w", err)
	}
	if resp.StatusCode != wantStatus {
		return fmt.Errorf("streamed body: status %d, want %d: %s", resp.StatusCode, wantStatus, data)
	}
	var envelope struct {
		Error struct {
			Type string `json:"type"`
			Code string `json:"code"`
		} `json:"error"`
	}
	if decodeErr := json.Unmarshal(data, &envelope); decodeErr != nil {
		return fmt.Errorf("decode streamed body rejection: %w: %s", decodeErr, data)
	}
	if envelope.Error.Type != "invalid_request_error" || envelope.Error.Code != wantCode {
		return fmt.Errorf("streamed body: error type=%q code=%q, want invalid_request_error/%s: %s",
			envelope.Error.Type, envelope.Error.Code, wantCode, data)
	}
	if decision := resp.Header.Get("x-vsr-selected-decision"); decision != "" {
		return fmt.Errorf("streamed body rejection selected a routing decision %q", decision)
	}
	return nil
}
