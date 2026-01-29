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
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// RouterR1Client is a client for the Router-R1 LLM-as-Router server
// This implements the think/route pattern from arXiv:2506.09033
// Requires external server: src/training/rl_model_selection/router_r1_server.py
type RouterR1Client struct {
	serverURL  string
	httpClient *http.Client
}

// RouterR1Response is the response from the Router-R1 server
type RouterR1Response struct {
	SelectedModel string `json:"selected_model"`
	Thinking      string `json:"thinking"`
	FullResponse  string `json:"full_response"`
}

// NewRouterR1Client creates a new Router-R1 client
func NewRouterR1Client(serverURL string) *RouterR1Client {
	return &RouterR1Client{
		serverURL: serverURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Route sends a query to the Router-R1 server for routing decision
func (c *RouterR1Client) Route(ctx context.Context, query string) (*RouterR1Response, error) {
	reqBody := map[string]string{
		"query": query,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.serverURL+"/route", bytes.NewBuffer(jsonBody))
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
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("server returned status %d: %s", resp.StatusCode, string(body))
	}

	var result RouterR1Response
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	logging.Debugf("[RouterR1Client] Routed query to %s: %s", result.SelectedModel, result.Thinking)
	return &result, nil
}

// HealthCheck checks if the Router-R1 server is healthy
func (c *RouterR1Client) HealthCheck(ctx context.Context) error {
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

// AutoMixVerifierClient is a client for the AutoMix self-verification server
// This implements few-shot entailment verification from arXiv:2310.12963
type AutoMixVerifierClient struct {
	serverURL  string
	httpClient *http.Client
}

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
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("server returned status %d: %s", resp.StatusCode, string(body))
	}

	var result AutoMixVerifyResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
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
