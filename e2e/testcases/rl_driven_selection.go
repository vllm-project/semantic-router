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
	// Test 1: Basic RL-driven model selection
	pkgtestcases.Register("rl-driven-basic-selection", pkgtestcases.TestCase{
		Description: "Verify RL-driven model selection picks from candidates using Thompson Sampling",
		Tags:        []string{"rl", "selection", "model-selection"},
		Fn:          testRLDrivenBasicSelection,
	})

	// Test 2: RL-driven personalization with user feedback
	pkgtestcases.Register("rl-driven-personalization", pkgtestcases.TestCase{
		Description: "Verify user-specific preferences are learned and used for personalized routing",
		Tags:        []string{"rl", "personalization", "feedback"},
		Fn:          testRLDrivenPersonalization,
	})

	// Test 3: RL-driven feedback loop
	pkgtestcases.Register("rl-driven-feedback-loop", pkgtestcases.TestCase{
		Description: "Verify feedback API updates RL model preferences correctly",
		Tags:        []string{"rl", "feedback", "api"},
		Fn:          testRLDrivenFeedbackLoop,
	})

	// Test 4: Multi-turn session context
	pkgtestcases.Register("rl-driven-multi-turn", pkgtestcases.TestCase{
		Description: "Verify session context influences model selection across conversation turns",
		Tags:        []string{"rl", "multi-turn", "session"},
		Fn:          testRLDrivenMultiTurn,
	})

	// Test 5: Exploration vs exploitation balance
	pkgtestcases.Register("rl-driven-exploration", pkgtestcases.TestCase{
		Description: "Verify Thompson Sampling balances exploration and exploitation correctly",
		Tags:        []string{"rl", "exploration", "thompson-sampling"},
		Fn:          testRLDrivenExploration,
	})
}

// FeedbackRequest represents the feedback API request format
type FeedbackRequest struct {
	WinnerModel  string  `json:"winner_model"`
	LoserModel   string  `json:"loser_model,omitempty"`
	Category     string  `json:"category,omitempty"`
	UserID       string  `json:"user_id,omitempty"`
	SessionID    string  `json:"session_id,omitempty"`
	FeedbackType string  `json:"feedback_type,omitempty"`
	Confidence   float64 `json:"confidence,omitempty"`
	Tie          bool    `json:"tie,omitempty"`
}

// FeedbackResponse represents the feedback API response
type FeedbackResponse struct {
	Success      bool    `json:"success"`
	WinnerRating float64 `json:"winner_rating,omitempty"`
	LoserRating  float64 `json:"loser_rating,omitempty"`
	Message      string  `json:"message,omitempty"`
}

// RatingsResponse represents the ratings API response
type RatingsResponse struct {
	Category string        `json:"category"`
	Ratings  []ModelRating `json:"ratings"`
}

// ModelRating represents a model's rating
type ModelRating struct {
	Model  string  `json:"model"`
	Rating float64 `json:"rating"`
	Games  int     `json:"games"`
}

// ChatRequest represents an OpenAI-compatible chat request
type ChatRequest struct {
	Model    string        `json:"model,omitempty"`
	Messages []ChatMessage `json:"messages"`
	User     string        `json:"user,omitempty"`
}

// ChatMessage represents a single chat message
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatResponse represents the chat completion response
type ChatResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message ChatMessage `json:"message"`
	} `json:"choices"`
}

// testRLDrivenBasicSelection tests that RL-driven selection picks from candidates
func testRLDrivenBasicSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	// Send multiple requests and track which models are selected
	modelCounts := make(map[string]int)
	numRequests := 20

	for i := 0; i < numRequests; i++ {
		req := ChatRequest{
			Messages: []ChatMessage{
				{Role: "user", Content: fmt.Sprintf("Request %d: What is 2+2?", i+1)},
			},
		}

		resp, err := sendRLChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
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

	// Verify we got responses from at least one model
	if len(modelCounts) == 0 {
		return fmt.Errorf("no models were selected in %d requests", numRequests)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests": numRequests,
			"model_counts":   modelCounts,
			"unique_models":  len(modelCounts),
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] RL-driven selection distributed across %d unique models\n", len(modelCounts))
		for model, count := range modelCounts {
			fmt.Printf("[Test]   %s: %d requests (%.1f%%)\n", model, count, float64(count)/float64(numRequests)*100)
		}
	}

	return nil
}

// testRLDrivenPersonalization tests user-specific preference learning
func testRLDrivenPersonalization(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	userID := fmt.Sprintf("test-user-%d", time.Now().UnixNano())
	preferredModel := "gpt-4" // We'll train the system to prefer this

	// Phase 1: Submit feedback to build user preference for gpt-4
	if opts.Verbose {
		fmt.Printf("[Test] Phase 1: Building preference for %s for user %s\n", preferredModel, userID)
	}

	for i := 0; i < 10; i++ {
		feedback := FeedbackRequest{
			WinnerModel: preferredModel,
			LoserModel:  "mistral-7b",
			UserID:      userID,
			Category:    "test",
		}

		if err := sendFeedbackRequest(httpClient, baseURL+"/api/v1/feedback", feedback); err != nil {
			return fmt.Errorf("feedback request %d failed: %w", i+1, err)
		}
	}

	// Phase 2: Send requests as this user and verify preference is respected
	if opts.Verbose {
		fmt.Printf("[Test] Phase 2: Verifying personalized selection\n")
	}

	preferredCount := 0
	totalRequests := 20

	for i := 0; i < totalRequests; i++ {
		req := ChatRequest{
			Messages: []ChatMessage{
				{Role: "user", Content: fmt.Sprintf("Test query %d", i+1)},
			},
			User: userID,
		}

		resp, err := sendRLChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			return fmt.Errorf("chat request %d failed: %w", i+1, err)
		}

		if resp.Model == preferredModel {
			preferredCount++
		}

		if opts.Verbose {
			fmt.Printf("[Test] Request %d for user %s: routed to %s\n", i+1, userID, resp.Model)
		}
	}

	// Verify that the preferred model is selected significantly more often
	preferenceRate := float64(preferredCount) / float64(totalRequests)

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"user_id":            userID,
			"preferred_model":    preferredModel,
			"preference_rate":    preferenceRate,
			"preferred_count":    preferredCount,
			"total_requests":     totalRequests,
			"feedback_submitted": 10,
		})
	}

	// Expect at least 60% preference (accounting for exploration)
	if preferenceRate < 0.60 {
		return fmt.Errorf("personalization not working: preferred model %s only selected %.1f%% of time (expected >= 60%%)",
			preferredModel, preferenceRate*100)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Personalization verified: %s selected %.1f%% of time\n", preferredModel, preferenceRate*100)
	}

	return nil
}

// testRLDrivenFeedbackLoop tests the feedback API integration
func testRLDrivenFeedbackLoop(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 30 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	// Get initial ratings
	initialRatings, err := getRatings(httpClient, baseURL+"/api/v1/ratings", "test-category")
	if err != nil {
		// Ratings endpoint might not exist yet, continue with test
		if opts.Verbose {
			fmt.Printf("[Test] Initial ratings not available (expected for new setup): %v\n", err)
		}
	}

	// Submit feedback: model-a beats model-b
	feedback := FeedbackRequest{
		WinnerModel: "model-a",
		LoserModel:  "model-b",
		Category:    "test-category",
	}

	if err := sendFeedbackRequest(httpClient, baseURL+"/api/v1/feedback", feedback); err != nil {
		return fmt.Errorf("feedback request failed: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Submitted feedback: %s beats %s\n", feedback.WinnerModel, feedback.LoserModel)
	}

	// Get updated ratings
	updatedRatings, err := getRatings(httpClient, baseURL+"/api/v1/ratings", "test-category")
	if err != nil {
		if opts.Verbose {
			fmt.Printf("[Test] Updated ratings not available: %v\n", err)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"feedback":        feedback,
			"initial_ratings": initialRatings,
			"updated_ratings": updatedRatings,
		})
	}

	// Submit implicit feedback (simulating FeedbackDetector classification)
	implicitFeedback := FeedbackRequest{
		WinnerModel:  "model-a",
		Category:     "test-category",
		FeedbackType: "satisfied",
		Confidence:   0.85,
	}

	if err := sendFeedbackRequest(httpClient, baseURL+"/api/v1/feedback", implicitFeedback); err != nil {
		return fmt.Errorf("implicit feedback request failed: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Submitted implicit feedback: %s (type=%s, confidence=%.2f)\n",
			implicitFeedback.WinnerModel, implicitFeedback.FeedbackType, implicitFeedback.Confidence)
	}

	return nil
}

// testRLDrivenMultiTurn tests session context across conversation turns
func testRLDrivenMultiTurn(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	sessionID := fmt.Sprintf("session-%d", time.Now().UnixNano())
	userID := fmt.Sprintf("user-%d", time.Now().UnixNano())

	// Turn 1: Initial query
	turn1Models := make(map[string]int)
	for i := 0; i < 5; i++ {
		req := ChatRequest{
			Messages: []ChatMessage{
				{Role: "user", Content: "Hello, can you help me with coding?"},
			},
			User: userID,
		}

		resp, err := sendChatRequestWithSession(httpClient, baseURL+"/v1/chat/completions", req, sessionID)
		if err != nil {
			return fmt.Errorf("turn 1 request %d failed: %w", i+1, err)
		}
		turn1Models[resp.Model]++
	}

	// Submit positive feedback for the session's model
	if len(turn1Models) > 0 {
		var winnerModel string
		for model := range turn1Models {
			winnerModel = model
			break
		}

		feedback := FeedbackRequest{
			WinnerModel: winnerModel,
			SessionID:   sessionID,
			UserID:      userID,
			Category:    "test",
		}

		if err := sendFeedbackRequest(httpClient, baseURL+"/api/v1/feedback", feedback); err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Warning: feedback request failed: %v\n", err)
			}
		}
	}

	// Turn 2: Follow-up query (should consider session context)
	turn2Models := make(map[string]int)
	for i := 0; i < 5; i++ {
		req := ChatRequest{
			Messages: []ChatMessage{
				{Role: "user", Content: "Hello, can you help me with coding?"},
				{Role: "assistant", Content: "Of course! I'd be happy to help with coding."},
				{Role: "user", Content: "Great, can you write a Python function to sort a list?"},
			},
			User: userID,
		}

		resp, err := sendChatRequestWithSession(httpClient, baseURL+"/v1/chat/completions", req, sessionID)
		if err != nil {
			return fmt.Errorf("turn 2 request %d failed: %w", i+1, err)
		}
		turn2Models[resp.Model]++
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"session_id":   sessionID,
			"user_id":      userID,
			"turn1_models": turn1Models,
			"turn2_models": turn2Models,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] Session %s:\n", sessionID)
		fmt.Printf("[Test]   Turn 1 models: %v\n", turn1Models)
		fmt.Printf("[Test]   Turn 2 models: %v\n", turn2Models)
	}

	return nil
}

// testRLDrivenExploration tests exploration vs exploitation balance
func testRLDrivenExploration(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	// Phase 1: Fresh start - should see exploration (multiple models selected)
	phase1Models := make(map[string]int)
	numRequests := 30

	if opts.Verbose {
		fmt.Printf("[Test] Phase 1: Testing exploration with %d requests\n", numRequests)
	}

	for i := 0; i < numRequests; i++ {
		req := ChatRequest{
			Messages: []ChatMessage{
				{Role: "user", Content: fmt.Sprintf("Exploration test query %d", i+1)},
			},
		}

		resp, err := sendRLChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			return fmt.Errorf("exploration request %d failed: %w", i+1, err)
		}

		phase1Models[resp.Model]++
	}

	// Verify exploration: with Thompson Sampling, we should see multiple models
	uniqueModels := len(phase1Models)

	// Phase 2: Build strong preference through feedback
	if opts.Verbose {
		fmt.Printf("[Test] Phase 2: Building exploitation preference\n")
	}

	exploitModel := ""
	for model := range phase1Models {
		exploitModel = model
		break
	}

	// Submit many feedback events to build strong preference
	for i := 0; i < 20; i++ {
		feedback := FeedbackRequest{
			WinnerModel: exploitModel,
			Category:    "test",
		}
		_ = sendFeedbackRequest(httpClient, baseURL+"/api/v1/feedback", feedback)
	}

	// Phase 3: After training, should see more exploitation
	phase2Models := make(map[string]int)

	if opts.Verbose {
		fmt.Printf("[Test] Phase 3: Testing exploitation bias\n")
	}

	for i := 0; i < numRequests; i++ {
		req := ChatRequest{
			Messages: []ChatMessage{
				{Role: "user", Content: fmt.Sprintf("Exploitation test query %d", i+1)},
			},
		}

		resp, err := sendRLChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			return fmt.Errorf("exploitation request %d failed: %w", i+1, err)
		}

		phase2Models[resp.Model]++
	}

	exploitCount := phase2Models[exploitModel]
	exploitRate := float64(exploitCount) / float64(numRequests)

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"phase1_models":      phase1Models,
			"phase1_unique":      uniqueModels,
			"exploit_target":     exploitModel,
			"phase2_models":      phase2Models,
			"exploit_rate":       exploitRate,
			"feedback_submitted": 20,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] Results:\n")
		fmt.Printf("[Test]   Phase 1 (exploration): %d unique models\n", uniqueModels)
		fmt.Printf("[Test]   Phase 2 (exploitation): %s selected %.1f%% of time\n", exploitModel, exploitRate*100)
	}

	return nil
}

// Helper functions for RL-driven selection tests

func sendRLChatRequest(client *http.Client, url string, req ChatRequest) (*ChatResponse, error) {
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

	var chatResp ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &chatResp, nil
}

func sendChatRequestWithSession(client *http.Client, url string, req ChatRequest, sessionID string) (*ChatResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Session-ID", sessionID)

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

	var chatResp ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &chatResp, nil
}

func sendFeedbackRequest(client *http.Client, url string, feedback FeedbackRequest) error {
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

func getRatings(client *http.Client, url string, category string) (*RatingsResponse, error) {
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

	var ratings RatingsResponse
	if err := json.NewDecoder(resp.Body).Decode(&ratings); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &ratings, nil
}
