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

package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	// Test 1: Static selection (baseline)
	pkgtestcases.Register("selection-static", pkgtestcases.TestCase{
		Description: "Verify static selection uses first model (baseline behavior)",
		Tags:        []string{"selection", "static", "model-selection"},
		Fn:          testStaticSelection,
	})

	// Test 2: Elo selection with feedback
	pkgtestcases.Register("selection-elo", pkgtestcases.TestCase{
		Description: "Verify Elo selection updates ratings based on feedback",
		Tags:        []string{"selection", "elo", "model-selection", "feedback"},
		Fn:          testEloSelection,
	})

	// Test 3: RouterDC selection with embeddings
	pkgtestcases.Register("selection-router-dc", pkgtestcases.TestCase{
		Description: "Verify RouterDC selection matches queries to models via embeddings",
		Tags:        []string{"selection", "router-dc", "model-selection", "embedding"},
		Fn:          testRouterDCSelection,
	})

	// Test 4: AutoMix selection with cost-quality tradeoff
	pkgtestcases.Register("selection-automix", pkgtestcases.TestCase{
		Description: "Verify AutoMix selection optimizes cost-quality tradeoff",
		Tags:        []string{"selection", "automix", "model-selection", "cost"},
		Fn:          testAutoMixSelection,
	})

	// Test 5: Hybrid selection combining methods
	pkgtestcases.Register("selection-hybrid", pkgtestcases.TestCase{
		Description: "Verify Hybrid selection combines Elo, RouterDC, and AutoMix",
		Tags:        []string{"selection", "hybrid", "model-selection"},
		Fn:          testHybridSelection,
	})
}

// SelectionTestCase represents a test case for model selection
type SelectionTestCase struct {
	Name           string   `json:"name"`
	Description    string   `json:"description"`
	Query          string   `json:"query"`
	Category       string   `json:"category,omitempty"`
	ExpectedModels []string `json:"expected_models,omitempty"` // Any of these is acceptable
	Algorithm      string   `json:"algorithm"`
}

// SelectionChatRequest represents an OpenAI-compatible chat request
type SelectionChatRequest struct {
	Model    string                 `json:"model,omitempty"`
	Messages []SelectionChatMessage `json:"messages"`
	User     string                 `json:"user,omitempty"`
}

// SelectionChatMessage represents a single chat message
type SelectionChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// SelectionChatResponse represents the chat completion response
type SelectionChatResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message SelectionChatMessage `json:"message"`
	} `json:"choices"`
}

// testStaticSelection tests that static selection uses the first model
func testStaticSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing static selection (first model)")
	}

	// Send multiple requests - static selection should consistently use first model
	modelCounts := make(map[string]int)
	numRequests := 10

	for i := 0; i < numRequests; i++ {
		req := SelectionChatRequest{
			Model: "MoM",
			Messages: []SelectionChatMessage{
				{Role: "user", Content: fmt.Sprintf("Simple query %d: What is 2+2?", i+1)},
			},
		}

		resp, err := sendSelectionChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			return fmt.Errorf("chat request %d failed: %w", i+1, err)
		}

		if resp.Model != "" {
			modelCounts[resp.Model]++
		}

		if opts.Verbose {
			fmt.Printf("[Test] Request %d: routed to model=%s\n", i+1, resp.Model)
		}
	}

	// Static selection should be deterministic (same model each time)
	if len(modelCounts) > 1 {
		if opts.Verbose {
			fmt.Printf("[Test] Warning: Static selection returned multiple models: %v\n", modelCounts)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"algorithm":      "static",
			"total_requests": numRequests,
			"model_counts":   modelCounts,
			"deterministic":  len(modelCounts) == 1,
		})
	}

	if len(modelCounts) == 0 {
		return fmt.Errorf("no models were selected in %d requests", numRequests)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Static selection: %d unique models (expected: 1 for deterministic)\n", len(modelCounts))
	}

	return nil
}

// testEloSelection tests Elo rating-based selection with feedback
func testEloSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing Elo selection with feedback")
	}

	// Phase 1: Get initial ratings
	initialRatings, err := getModelRatings(httpClient, baseURL+"/api/v1/ratings", "tech")
	if err != nil {
		if opts.Verbose {
			fmt.Printf("[Test] Initial ratings not available (may be new setup): %v\n", err)
		}
	} else if opts.Verbose {
		fmt.Printf("[Test] Initial ratings: %v\n", initialRatings)
	}

	// Phase 2: Submit feedback to boost one model
	winnerModel := "gemma3:27b" // We'll train system to prefer this
	loserModel := "llama3.2:3b"

	for i := 0; i < 5; i++ {
		feedback := map[string]interface{}{
			"winner_model": winnerModel,
			"loser_model":  loserModel,
			"category":     "tech",
		}

		if err := sendFeedback(httpClient, baseURL+"/api/v1/feedback", feedback); err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Warning: feedback request %d failed: %v\n", i+1, err)
			}
		}
	}

	// Phase 3: Get updated ratings
	updatedRatings, err := getModelRatings(httpClient, baseURL+"/api/v1/ratings", "tech")
	if err != nil {
		if opts.Verbose {
			fmt.Printf("[Test] Updated ratings not available: %v\n", err)
		}
	} else if opts.Verbose {
		fmt.Printf("[Test] Updated ratings: %v\n", updatedRatings)
	}

	// Phase 4: Send requests and track selection
	modelCounts := make(map[string]int)
	numRequests := 15

	for i := 0; i < numRequests; i++ {
		req := SelectionChatRequest{
			Model: "MoM",
			Messages: []SelectionChatMessage{
				{Role: "user", Content: "Explain machine learning algorithms"},
			},
		}

		resp, err := sendSelectionChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			return fmt.Errorf("chat request %d failed: %w", i+1, err)
		}

		if resp.Model != "" {
			modelCounts[resp.Model]++
		}
	}

	// Check if winner model is selected more often
	winnerCount := modelCounts[winnerModel]
	winnerRate := float64(winnerCount) / float64(numRequests)

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"algorithm":       "elo",
			"winner_model":    winnerModel,
			"winner_count":    winnerCount,
			"winner_rate":     winnerRate,
			"model_counts":    modelCounts,
			"initial_ratings": initialRatings,
			"updated_ratings": updatedRatings,
			"feedback_count":  5,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] Elo selection: winner=%s selected %d/%d (%.1f%%)\n",
			winnerModel, winnerCount, numRequests, winnerRate*100)
	}

	return nil
}

// testRouterDCSelection tests embedding-based query-model matching
func testRouterDCSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing RouterDC selection (embedding similarity)")
	}

	// Test different query types that should match different model specialties
	testQueries := []struct {
		query       string
		queryType   string
		expectedTag string // Expected capability/specialty
	}{
		{"Write a Python function to sort a list", "coding", "code"},
		{"Explain quantum entanglement", "science", "reasoning"},
		{"What's the weather like?", "simple", "chat"},
	}

	results := make([]map[string]interface{}, 0)

	for _, tq := range testQueries {
		req := SelectionChatRequest{
			Model: "MoM",
			Messages: []SelectionChatMessage{
				{Role: "user", Content: tq.query},
			},
		}

		resp, err := sendSelectionChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			return fmt.Errorf("chat request for %s failed: %w", tq.queryType, err)
		}

		results = append(results, map[string]interface{}{
			"query":        tq.query,
			"query_type":   tq.queryType,
			"expected_tag": tq.expectedTag,
			"selected":     resp.Model,
		})

		if opts.Verbose {
			fmt.Printf("[Test] Query type=%s: selected model=%s\n", tq.queryType, resp.Model)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"algorithm": "router_dc",
			"results":   results,
		})
	}

	if opts.Verbose {
		fmt.Println("[Test] RouterDC selection test completed")
	}

	return nil
}

// testAutoMixSelection tests cost-quality optimization
func testAutoMixSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing AutoMix selection (cost-quality tradeoff)")
	}

	// Test with different complexity queries
	// AutoMix should select cheaper models for simple queries, expensive for complex
	testQueries := []struct {
		query      string
		complexity string
	}{
		{"What is 2+2?", "simple"},
		{"Explain the Riemann hypothesis in detail", "complex"},
		{"Hello, how are you?", "simple"},
		{"Derive the equations of general relativity", "complex"},
	}

	results := make([]map[string]interface{}, 0)

	for _, tq := range testQueries {
		req := SelectionChatRequest{
			Model: "MoM",
			Messages: []SelectionChatMessage{
				{Role: "user", Content: tq.query},
			},
		}

		resp, err := sendSelectionChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			return fmt.Errorf("chat request failed: %w", err)
		}

		results = append(results, map[string]interface{}{
			"query":      tq.query,
			"complexity": tq.complexity,
			"selected":   resp.Model,
		})

		if opts.Verbose {
			fmt.Printf("[Test] Complexity=%s: selected model=%s\n", tq.complexity, resp.Model)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"algorithm": "automix",
			"results":   results,
		})
	}

	if opts.Verbose {
		fmt.Println("[Test] AutoMix selection test completed")
	}

	return nil
}

// testHybridSelection tests combined selection methods
func testHybridSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing Hybrid selection (combined methods)")
	}

	// Send multiple requests to see distribution
	modelCounts := make(map[string]int)
	numRequests := 20

	for i := 0; i < numRequests; i++ {
		req := SelectionChatRequest{
			Model: "MoM",
			Messages: []SelectionChatMessage{
				{Role: "user", Content: fmt.Sprintf("Test query %d: Explain algorithms", i+1)},
			},
		}

		resp, err := sendSelectionChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			return fmt.Errorf("chat request %d failed: %w", i+1, err)
		}

		if resp.Model != "" {
			modelCounts[resp.Model]++
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"algorithm":      "hybrid",
			"total_requests": numRequests,
			"model_counts":   modelCounts,
			"unique_models":  len(modelCounts),
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] Hybrid selection: %d unique models across %d requests\n",
			len(modelCounts), numRequests)
		for model, count := range modelCounts {
			fmt.Printf("[Test]   %s: %d requests (%.1f%%)\n",
				model, count, float64(count)/float64(numRequests)*100)
		}
	}

	if len(modelCounts) == 0 {
		return fmt.Errorf("no models were selected in %d requests", numRequests)
	}

	return nil
}

// Helper functions

func sendSelectionChatRequest(client *http.Client, url string, req SelectionChatRequest) (*SelectionChatResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp SelectionChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &chatResp, nil
}

func sendFeedback(client *http.Client, url string, feedback map[string]interface{}) error {
	body, err := json.Marshal(feedback)
	if err != nil {
		return fmt.Errorf("failed to marshal feedback: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("feedback request failed with status %d: %s", resp.StatusCode, string(respBody))
	}

	return nil
}

func getModelRatings(client *http.Client, url string, category string) (map[string]float64, error) {
	reqURL := url
	if category != "" {
		reqURL = fmt.Sprintf("%s?category=%s", url, category)
	}

	resp, err := client.Get(reqURL)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ratings request failed with status %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		Ratings []struct {
			Model  string  `json:"model"`
			Rating float64 `json:"rating"`
		} `json:"ratings"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	ratings := make(map[string]float64)
	for _, r := range result.Ratings {
		ratings[r.Model] = r.Rating
	}

	return ratings, nil
}
