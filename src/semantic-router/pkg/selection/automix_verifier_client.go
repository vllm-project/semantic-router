/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package selection

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// AutoMixVerifierClient is a client for the AutoMix self-verification server
// This implements few-shot entailment verification from arXiv:2310.12963
type AutoMixVerifierClient struct {
	serverURL        string
	httpClient       *http.Client
	maxResponseBytes int64
}

const maxSelectionErrorBodyBytes int64 = 8 * 1024

// AutoMixVerifyResponse is the response from the AutoMix verifier
type AutoMixVerifyResponse struct {
	Confidence     float64  `json:"confidence"`
	ShouldEscalate bool     `json:"should_escalate"`
	VerifiedCount  int      `json:"verified_count"`
	TotalSamples   int      `json:"total_samples"`
	Samples        []string `json:"samples"`
	Threshold      float64  `json:"threshold"`
}

// NewAutoMixVerifierClient creates a new AutoMix verifier client
func NewAutoMixVerifierClient(serverURL string) *AutoMixVerifierClient {
	return &AutoMixVerifierClient{
		serverURL: serverURL,
		httpClient: &http.Client{
			Timeout: 60 * time.Second, // Verification can take longer with multiple samples
		},
		maxResponseBytes: config.DefaultMaxResponseBytes,
	}
}

// SetTimeout overrides the verifier HTTP client timeout. Useful for callers
// that load timeout values from configuration after client construction.
// Non-positive durations are ignored.
func (c *AutoMixVerifierClient) SetTimeout(timeout time.Duration) {
	if timeout <= 0 {
		return
	}
	c.httpClient.Timeout = timeout
}

func (c *AutoMixVerifierClient) SetMaxResponseBytes(maxResponseBytes int64) {
	if maxResponseBytes > 0 {
		c.maxResponseBytes = maxResponseBytes
	}
}

// Verify sends a question/answer pair for verification
func (c *AutoMixVerifierClient) Verify(ctx context.Context, question, answer, optionalContext string, threshold float64) (*AutoMixVerifyResponse, error) {
	reqBody := map[string]interface{}{
		"question":  question,
		"answer":    answer,
		"threshold": threshold,
	}
	if optionalContext != "" {
		reqBody["context"] = optionalContext
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.serverURL+"/verify", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errorBody, truncated := httputil.ReadTruncatedBody(resp.Body, maxSelectionErrorBodyBytes)
		return nil, fmt.Errorf(
			"server returned status %d (error_body_bytes=%d, truncated=%t)",
			resp.StatusCode,
			len(errorBody),
			truncated,
		)
	}

	body, err := httputil.ReadLimitedBody(resp.Body, c.maxResponseBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	var result AutoMixVerifyResponse
	if err := json.NewDecoder(bytes.NewReader(body)).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	logging.Debugf("[AutoMixVerifier] Confidence=%.2f, ShouldEscalate=%v, Samples=%d/%d",
		result.Confidence, result.ShouldEscalate, result.VerifiedCount, result.TotalSamples)
	return &result, nil
}

// HealthCheck checks if the AutoMix verifier server is healthy
func (c *AutoMixVerifierClient) HealthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.serverURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server returned status %d", resp.StatusCode)
	}

	return nil
}
