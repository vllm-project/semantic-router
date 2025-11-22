package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

type chatResult struct {
	headers  http.Header
	duration time.Duration
}

// doLLMDChat sends a chat completion request to the forwarded service and returns headers + latency.
func doLLMDChat(ctx context.Context, port, model, content string, timeout time.Duration) (chatResult, error) {
	body := map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": content},
		},
	}
	data, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://localhost:%s/v1/chat/completions", port), bytes.NewBuffer(data))
	if err != nil {
		return chatResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: timeout}
	start := time.Now()
	resp, err := client.Do(req)
	duration := time.Since(start)
	if err != nil {
		return chatResult{}, err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return chatResult{}, fmt.Errorf("chat failed: %d %s", resp.StatusCode, truncateString(string(b), 120))
	}
	return chatResult{headers: resp.Header, duration: duration}, nil
}

func getInferencePod(headers http.Header) string {
	return strings.TrimSpace(headers.Get("x-inference-pod"))
}

func getSelectedModel(headers http.Header) string {
	v := strings.TrimSpace(headers.Get("x-vsr-selected-model"))
	if v != "" {
		return v
	}
	return strings.TrimSpace(headers.Get("x-selected-model"))
}

func percentileDuration(ds []time.Duration, p float64) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	sorted := make([]time.Duration, len(ds))
	copy(sorted, ds)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	idx := int(float64(len(sorted)-1) * p)
	return sorted[idx]
}
