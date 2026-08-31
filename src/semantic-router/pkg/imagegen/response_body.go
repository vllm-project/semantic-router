package imagegen

import (
	"fmt"
	"net/http"

	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

const (
	defaultImageGenMaxResponseBytes int64 = 64 * 1024 * 1024
	maxImageGenErrorBodyBytes       int64 = 10 * 1024
)

func resolveImageGenMaxResponseBytes(maxResponseBytes int64) (int64, error) {
	if maxResponseBytes < 0 {
		return 0, fmt.Errorf("image_gen max_response_bytes must be non-negative")
	}
	if maxResponseBytes == 0 {
		return defaultImageGenMaxResponseBytes, nil
	}
	return maxResponseBytes, nil
}

func readImageGenResponse(resp *http.Response, maxResponseBytes int64) ([]byte, error) {
	if resp.StatusCode != http.StatusOK {
		body, truncated := httputil.ReadTruncatedBody(resp.Body, maxImageGenErrorBodyBytes)
		return nil, fmt.Errorf("request failed with status %d: %s (truncated=%t)", resp.StatusCode, string(body), truncated)
	}

	body, err := httputil.ReadLimitedBody(resp.Body, maxResponseBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}
	return body, nil
}
