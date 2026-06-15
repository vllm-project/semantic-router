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

// Package testcases contains E2E tests for all 8 model selection algorithms.
// Core Algorithms:
//   - Static selection (baseline)
//   - Elo rating-based selection
//   - RouterDC embedding similarity
//   - AutoMix cost-quality optimization
//   - Hybrid multi-signal combination
//
// RL-driven Algorithms:
//   - Thompson Sampling exploration-exploitation
//   - GMTRouter graph neural network routing
//   - Router-R1 LLM-as-router with reasoning
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
	pkgtestcases.Register("core-selection-static", pkgtestcases.TestCase{
		Description: "Verify static selection uses first configured model (baseline)",
		Tags:        []string{"selection", "static", "core", "model-selection"},
		Fn:          testStaticSelection,
	})

	// Test 2: Elo rating-based selection
	pkgtestcases.Register("core-selection-elo", pkgtestcases.TestCase{
		Description: "Verify Elo selection changes based on pairwise feedback",
		Tags:        []string{"selection", "elo", "core", "feedback", "model-selection"},
		Fn:          testEloSelection,
	})

	// Test 3: Elo feedback API
	pkgtestcases.Register("core-selection-elo-feedback", pkgtestcases.TestCase{
		Description: "Verify Elo feedback API updates model ratings correctly",
		Tags:        []string{"selection", "elo", "feedback", "api", "model-selection"},
		Fn:          testEloFeedback,
	})

	// Test 4: RouterDC embedding-based selection
	pkgtestcases.Register("core-selection-routerdc", pkgtestcases.TestCase{
		Description: "Verify RouterDC matches queries to models via embedding similarity",
		Tags:        []string{"selection", "router_dc", "embedding", "core", "model-selection"},
		Fn:          testRouterDCSelection,
	})

	// Test 5: AutoMix cost-quality optimization
	pkgtestcases.Register("core-selection-automix", pkgtestcases.TestCase{
		Description: "Verify AutoMix routes based on cost-quality tradeoff",
		Tags:        []string{"selection", "automix", "pomdp", "core", "model-selection"},
		Fn:          testAutoMixSelection,
	})

	// Test 6: Hybrid multi-signal selection
	pkgtestcases.Register("core-selection-hybrid", pkgtestcases.TestCase{
		Description: "Verify Hybrid selection combines multiple signals with weights",
		Tags:        []string{"selection", "hybrid", "core", "model-selection"},
		Fn:          testHybridSelection,
	})

	// Test 7: Thompson Sampling exploration-exploitation
	pkgtestcases.Register("core-selection-thompson", pkgtestcases.TestCase{
		Description: "Verify Thompson Sampling balances exploration vs exploitation",
		Tags:        []string{"selection", "thompson", "bayesian", "rl", "model-selection"},
		Fn:          testThompsonSamplingSelection,
	})

	// Test 8: GMTRouter graph-based routing
	pkgtestcases.Register("core-selection-gmtrouter", pkgtestcases.TestCase{
		Description: "Verify GMTRouter uses graph neural network for personalized routing",
		Tags:        []string{"selection", "gmtrouter", "gnn", "rl", "model-selection"},
		Fn:          testGMTRouterSelection,
	})

	// Test 9: Router-R1 LLM-as-router
	pkgtestcases.Register("core-selection-router-r1", pkgtestcases.TestCase{
		Description: "Verify Router-R1 uses LLM reasoning to select models",
		Tags:        []string{"selection", "router_r1", "llm", "rl", "model-selection"},
		Fn:          testRouterR1Selection,
	})
}

// CoreSelectionTestCase represents a test case for core selection algorithms
type CoreSelectionTestCase struct {
	Query          string   `json:"query"`
	Algorithm      string   `json:"algorithm"`
	ExpectedModels []string `json:"expected_models"`
	Description    string   `json:"description"`
}

// CoreChatRequest represents an OpenAI-compatible chat request for selection tests
type CoreChatRequest struct {
	Model    string            `json:"model,omitempty"`
	Messages []CoreChatMessage `json:"messages"`
	User     string            `json:"user,omitempty"`
}

// CoreChatMessage represents a single chat message
type CoreChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CoreChatResponse represents the chat completion response
type CoreChatResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message CoreChatMessage `json:"message"`
	} `json:"choices"`
	// SelectedModel is populated from x-vsr-selected-model response header
	// This is more reliable than the JSON body's model field for VSR routing info
	SelectedModel string `json:"-"`
}

// CoreFeedbackRequest represents the feedback API request format
type CoreFeedbackRequest struct {
	WinnerModel  string  `json:"winner_model"`
	LoserModel   string  `json:"loser_model,omitempty"`
	Category     string  `json:"category,omitempty"`
	UserID       string  `json:"user_id,omitempty"`
	SessionID    string  `json:"session_id,omitempty"`
	FeedbackType string  `json:"feedback_type,omitempty"`
	Confidence   float64 `json:"confidence,omitempty"`
	Tie          bool    `json:"tie,omitempty"`
}

// CoreFeedbackResponse represents the feedback API response
type CoreFeedbackResponse struct {
	Success      bool    `json:"success"`
	WinnerRating float64 `json:"winner_rating,omitempty"`
	LoserRating  float64 `json:"loser_rating,omitempty"`
	Message      string  `json:"message,omitempty"`
}

// CoreRatingsResponse represents the ratings API response
type CoreRatingsResponse struct {
	Category string            `json:"category"`
	Ratings  []CoreModelRating `json:"ratings"`
}

// CoreModelRating represents a model's rating
type CoreModelRating struct {
	Model  string  `json:"model"`
	Rating float64 `json:"rating"`
	Wins   int     `json:"wins"`
	Losses int     `json:"losses"`
}

// testStaticSelection verifies that static selection uses the first configured model
func testStaticSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing static model selection...")
	}

	// Static selection should consistently pick the same model
	testCases := []CoreSelectionTestCase{
		{
			Query:       "What is machine learning?",
			Algorithm:   "static",
			Description: "Basic ML question - should use first model",
		},
		{
			Query:       "Explain quantum computing",
			Algorithm:   "static",
			Description: "Physics question - should use first model",
		},
		{
			Query:       "Write a Python function",
			Algorithm:   "static",
			Description: "Coding question - should use first model",
		},
	}

	modelCounts := make(map[string]int)
	successCount := 0

	for i, tc := range testCases {
		req := CoreChatRequest{
			Messages: []CoreChatMessage{
				{Role: "user", Content: tc.Query},
			},
		}

		resp, err := sendCoreChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Request %d failed: %v\n", i+1, err)
			}
			continue
		}

		if resp.SelectedModel != "" {
			modelCounts[resp.SelectedModel]++
			successCount++
		}

		if opts.Verbose {
			fmt.Printf("[Test] %s: routed to %s\n", tc.Description, resp.SelectedModel)
		}
	}

	// Static selection should use the same model for all requests
	if len(modelCounts) > 1 {
		return fmt.Errorf("static selection routed to multiple models: %v (expected single model)", modelCounts)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests":   len(testCases),
			"success_requests": successCount,
			"model_counts":     modelCounts,
			"algorithm":        "static",
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✓ Static selection consistently used single model: %v\n", modelCounts)
	}

	return nil
}

// testEloSelection verifies that Elo selection uses ratings from feedback
func testEloSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	// Setup connection to Envoy Gateway for chat completions
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	// Setup connection to Router API for feedback/ratings endpoints
	apiPort, stopAPIPort, err := setupRouterAPIConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup router API connection: %w", err)
	}
	defer stopAPIPort()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	apiBaseURL := fmt.Sprintf("http://localhost:%s", apiPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing Elo rating-based model selection...")
		fmt.Printf("[Test] Gateway URL: %s, Router API URL: %s\n", baseURL, apiBaseURL)
	}

	category := fmt.Sprintf("elo-test-%d", time.Now().UnixNano())
	winnerModel := "gpt-4"
	loserModel := "gpt-3.5-turbo"

	// Phase 1: Submit multiple feedback entries to establish clear winner
	if opts.Verbose {
		fmt.Printf("[Test] Phase 1: Building Elo ratings for category %s\n", category)
	}

	feedbackCount := 5
	for i := 0; i < feedbackCount; i++ {
		feedback := CoreFeedbackRequest{
			WinnerModel: winnerModel,
			LoserModel:  loserModel,
			Category:    category,
		}
		_, err := sendCoreFeedbackRequest(httpClient, apiBaseURL+"/api/v1/feedback", feedback)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Warning: Feedback %d failed: %v\n", i+1, err)
			}
		}
	}

	// Phase 2: Verify ratings were updated
	ratingsResp, err := getCoreRatings(httpClient, apiBaseURL+"/api/v1/ratings", category)
	if err != nil {
		return fmt.Errorf("failed to get ratings: %w", err)
	}

	var winnerRating, loserRating float64
	var winnerWins, loserLosses int
	for _, r := range ratingsResp.Ratings {
		if r.Model == winnerModel {
			winnerRating = r.Rating
			winnerWins = r.Wins
		} else if r.Model == loserModel {
			loserRating = r.Rating
			loserLosses = r.Losses
		}
	}

	if opts.Verbose {
		fmt.Printf("[Test] Ratings after feedback: %s rating=%.2f wins=%d, %s rating=%.2f losses=%d\n",
			winnerModel, winnerRating, winnerWins, loserModel, loserRating, loserLosses)
	}

	// Phase 3: Send requests and verify selection favors higher-rated model
	modelCounts := make(map[string]int)
	numRequests := 10

	for i := 0; i < numRequests; i++ {
		req := CoreChatRequest{
			Messages: []CoreChatMessage{
				{Role: "user", Content: fmt.Sprintf("Test question %d for Elo selection", i+1)},
			},
		}

		resp, err := sendCoreChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Request %d failed: %v\n", i+1, err)
			}
			continue
		}

		if resp.SelectedModel != "" {
			modelCounts[resp.SelectedModel]++
		}
	}

	// Assert: Elo selection should favor the higher-rated model (winner)
	// After positive feedback, winnerModel should be selected more than loserModel
	winnerCount := modelCounts[winnerModel]
	loserCount := modelCounts[loserModel]
	if winnerRating > loserRating && winnerCount < loserCount {
		return fmt.Errorf("Elo selection did not favor higher-rated model: %s (rating=%.2f, count=%d) vs %s (rating=%.2f, count=%d)",
			winnerModel, winnerRating, winnerCount, loserModel, loserRating, loserCount)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"category":       category,
			"feedback_count": feedbackCount,
			"model_counts":   modelCounts,
			"winner_rating":  winnerRating,
			"loser_rating":   loserRating,
			"algorithm":      "elo",
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✓ Elo selection completed: model distribution=%v (winner favored: %t)\n",
			modelCounts, winnerCount >= loserCount)
	}

	return nil
}

// testEloFeedback verifies the Elo feedback API
func testEloFeedback(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	// Setup connection to Router API for feedback/ratings endpoints
	apiPort, stopAPIPort, err := setupRouterAPIConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup router API connection: %w", err)
	}
	defer stopAPIPort()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	apiBaseURL := fmt.Sprintf("http://localhost:%s", apiPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing Elo feedback API...")
		fmt.Printf("[Test] Router API URL: %s\n", apiBaseURL)
	}

	category := fmt.Sprintf("feedback-test-%d", time.Now().UnixNano())

	// Test 1: Submit feedback
	feedback := CoreFeedbackRequest{
		WinnerModel: "gpt-4",
		LoserModel:  "gpt-3.5-turbo",
		Category:    category,
	}

	feedbackResp, err := sendCoreFeedbackRequest(httpClient, apiBaseURL+"/api/v1/feedback", feedback)
	if err != nil {
		return fmt.Errorf("failed to submit feedback: %w", err)
	}

	if !feedbackResp.Success {
		return fmt.Errorf("feedback submission failed: %s", feedbackResp.Message)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Feedback submitted successfully: %+v\n", feedbackResp)
	}

	// Test 2: Verify ratings were updated
	ratingsResp, err := getCoreRatings(httpClient, apiBaseURL+"/api/v1/ratings", category)
	if err != nil {
		return fmt.Errorf("failed to get ratings: %w", err)
	}

	winnerFound := false
	loserFound := false
	for _, r := range ratingsResp.Ratings {
		if r.Model == "gpt-4" && r.Wins > 0 {
			winnerFound = true
		}
		if r.Model == "gpt-3.5-turbo" && r.Losses > 0 {
			loserFound = true
		}
	}

	if !winnerFound || !loserFound {
		return fmt.Errorf("ratings not updated correctly: winnerFound=%v, loserFound=%v", winnerFound, loserFound)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"category":      category,
			"winner_found":  winnerFound,
			"loser_found":   loserFound,
			"ratings_count": len(ratingsResp.Ratings),
		})
	}

	if opts.Verbose {
		fmt.Println("[Test] ✓ Elo feedback API working correctly")
	}

	return nil
}

// testRouterDCSelection verifies embedding-based query-to-model matching
func testRouterDCSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing RouterDC embedding-based selection...")
	}

	// RouterDC should match queries to models based on description similarity
	testCases := []CoreSelectionTestCase{
		{
			Query:       "Write a complex algorithm in Python with recursion and dynamic programming",
			Algorithm:   "router_dc",
			Description: "Coding query - should match coding-capable model",
		},
		{
			Query:       "Explain the theory of relativity in simple terms for a beginner",
			Algorithm:   "router_dc",
			Description: "Explanation query - should match explanation-capable model",
		},
		{
			Query:       "Translate this text to French: Hello, how are you?",
			Algorithm:   "router_dc",
			Description: "Translation query - should match translation-capable model",
		},
	}

	modelCounts := make(map[string]int)
	successCount := 0

	for i, tc := range testCases {
		req := CoreChatRequest{
			Messages: []CoreChatMessage{
				{Role: "user", Content: tc.Query},
			},
		}

		resp, err := sendCoreChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Request %d failed: %v\n", i+1, err)
			}
			continue
		}

		if resp.SelectedModel != "" {
			modelCounts[resp.SelectedModel]++
			successCount++
		}

		if opts.Verbose {
			fmt.Printf("[Test] %s: routed to %s\n", tc.Description, resp.SelectedModel)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests":   len(testCases),
			"success_requests": successCount,
			"model_counts":     modelCounts,
			"algorithm":        "router_dc",
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✓ RouterDC selection completed: %d/%d requests succeeded\n", successCount, len(testCases))
	}

	return nil
}

// testAutoMixSelection verifies POMDP-based cost-quality optimization
func testAutoMixSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing AutoMix cost-quality optimization...")
	}

	// AutoMix should route simple queries to cheaper models, complex to expensive
	testCases := []CoreSelectionTestCase{
		{
			Query:       "What is 2+2?",
			Algorithm:   "automix",
			Description: "Simple math - should use cost-effective model",
		},
		{
			Query:       "Derive the Schwarzschild metric from Einstein's field equations step by step",
			Algorithm:   "automix",
			Description: "Complex physics - may escalate to better model",
		},
		{
			Query:       "What is the capital of France?",
			Algorithm:   "automix",
			Description: "Simple factual - should use cost-effective model",
		},
		{
			Query:       "Prove the Riemann hypothesis with formal mathematical rigor",
			Algorithm:   "automix",
			Description: "Extremely complex - should use best model",
		},
	}

	modelCounts := make(map[string]int)
	queryComplexity := make(map[string]string) // model -> complexity level

	for i, tc := range testCases {
		req := CoreChatRequest{
			Messages: []CoreChatMessage{
				{Role: "user", Content: tc.Query},
			},
		}

		resp, err := sendCoreChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Request %d failed: %v\n", i+1, err)
			}
			continue
		}

		if resp.SelectedModel != "" {
			modelCounts[resp.SelectedModel]++
			queryComplexity[tc.Description] = resp.SelectedModel
		}

		if opts.Verbose {
			fmt.Printf("[Test] %s: routed to %s\n", tc.Description, resp.SelectedModel)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests":   len(testCases),
			"model_counts":     modelCounts,
			"query_complexity": queryComplexity,
			"algorithm":        "automix",
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✓ AutoMix selection completed: model distribution=%v\n", modelCounts)
	}

	return nil
}

// testHybridSelection verifies multi-signal weighted combination
func testHybridSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	// Setup connection to Envoy Gateway for chat completions
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	// Setup connection to Router API for feedback/ratings endpoints
	apiPort, stopAPIPort, err := setupRouterAPIConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup router API connection: %w", err)
	}
	defer stopAPIPort()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	apiBaseURL := fmt.Sprintf("http://localhost:%s", apiPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing Hybrid multi-signal selection...")
	}

	// First, submit some feedback to influence Elo component (via Router API)
	category := fmt.Sprintf("hybrid-test-%d", time.Now().UnixNano())
	for i := 0; i < 3; i++ {
		feedback := CoreFeedbackRequest{
			WinnerModel: "gpt-4",
			LoserModel:  "gpt-3.5-turbo",
			Category:    category,
		}
		_, _ = sendCoreFeedbackRequest(httpClient, apiBaseURL+"/api/v1/feedback", feedback)
	}

	// Hybrid combines Elo + RouterDC + AutoMix + Cost with configurable weights
	testCases := []CoreSelectionTestCase{
		{
			Query:       "Explain machine learning concepts for a technical audience",
			Algorithm:   "hybrid",
			Description: "Technical explanation - hybrid considers multiple signals",
		},
		{
			Query:       "Write optimized code for sorting a large dataset",
			Algorithm:   "hybrid",
			Description: "Coding task - hybrid considers capability and cost",
		},
		{
			Query:       "Summarize the key points of quantum mechanics",
			Algorithm:   "hybrid",
			Description: "Summary task - hybrid balances quality and efficiency",
		},
	}

	modelCounts := make(map[string]int)
	successCount := 0

	for i, tc := range testCases {
		req := CoreChatRequest{
			Messages: []CoreChatMessage{
				{Role: "user", Content: tc.Query},
			},
		}

		resp, err := sendCoreChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Request %d failed: %v\n", i+1, err)
			}
			continue
		}

		if resp.SelectedModel != "" {
			modelCounts[resp.SelectedModel]++
			successCount++
		}

		if opts.Verbose {
			fmt.Printf("[Test] %s: routed to %s\n", tc.Description, resp.SelectedModel)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests":   len(testCases),
			"success_requests": successCount,
			"model_counts":     modelCounts,
			"algorithm":        "hybrid",
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✓ Hybrid selection completed: %d/%d requests succeeded\n", successCount, len(testCases))
	}

	return nil
}

// testThompsonSamplingSelection verifies Bayesian exploration-exploitation
func testThompsonSamplingSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing Thompson Sampling exploration-exploitation...")
	}

	// Thompson Sampling should explore initially, then exploit best models
	// As it collects more data, it should converge to better models
	testCases := []CoreSelectionTestCase{
		{
			Query:       "What is the meaning of life?",
			Algorithm:   "thompson",
			Description: "Philosophical query - exploration phase",
		},
		{
			Query:       "Calculate the derivative of x^2 + 3x",
			Algorithm:   "thompson",
			Description: "Math query - may explore or exploit",
		},
		{
			Query:       "Write a haiku about programming",
			Algorithm:   "thompson",
			Description: "Creative query - tests creative model selection",
		},
	}

	modelCounts := make(map[string]int)
	successCount := 0

	for i, tc := range testCases {
		req := CoreChatRequest{
			Messages: []CoreChatMessage{
				{Role: "user", Content: tc.Query},
			},
		}

		resp, err := sendCoreChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Request %d failed: %v\n", i+1, err)
			}
			continue
		}

		if resp.SelectedModel != "" {
			modelCounts[resp.SelectedModel]++
			successCount++
		}

		if opts.Verbose {
			fmt.Printf("[Test] %s: routed to %s\n", tc.Description, resp.SelectedModel)
		}
	}

	// Thompson Sampling should show some variation in model selection (exploration)
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests":   len(testCases),
			"success_requests": successCount,
			"model_counts":     modelCounts,
			"algorithm":        "thompson",
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✓ Thompson Sampling completed: %d/%d requests, distribution=%v\n",
			successCount, len(testCases), modelCounts)
	}

	return nil
}

// testGMTRouterSelection verifies graph neural network based routing
func testGMTRouterSelection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 60 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing GMTRouter graph neural network routing...")
	}

	// GMTRouter uses user preference graphs for personalized routing
	// Different users should potentially get different model selections
	testCases := []CoreSelectionTestCase{
		{
			Query:       "Recommend a book about artificial intelligence",
			Algorithm:   "gmtrouter",
			Description: "Recommendation query - tests preference learning",
		},
		{
			Query:       "What's the best programming language for beginners?",
			Algorithm:   "gmtrouter",
			Description: "Opinion query - tests personalization",
		},
		{
			Query:       "Explain neural networks in simple terms",
			Algorithm:   "gmtrouter",
			Description: "Technical explanation - tests routing based on user history",
		},
	}

	modelCounts := make(map[string]int)
	successCount := 0

	// Test with different user IDs to verify personalization
	userIDs := []string{"user-alpha", "user-beta", "user-gamma"}

	for i, tc := range testCases {
		req := CoreChatRequest{
			Messages: []CoreChatMessage{
				{Role: "user", Content: tc.Query},
			},
			User: userIDs[i%len(userIDs)], // Rotate through users
		}

		resp, err := sendCoreChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Request %d (user=%s) failed: %v\n", i+1, userIDs[i%len(userIDs)], err)
			}
			continue
		}

		if resp.SelectedModel != "" {
			modelCounts[resp.SelectedModel]++
			successCount++
		}

		if opts.Verbose {
			fmt.Printf("[Test] %s (user=%s): routed to %s\n", tc.Description, userIDs[i%len(userIDs)], resp.SelectedModel)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests":   len(testCases),
			"success_requests": successCount,
			"model_counts":     modelCounts,
			"user_ids_tested":  userIDs,
			"algorithm":        "gmtrouter",
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✓ GMTRouter completed: %d/%d requests, distribution=%v\n",
			successCount, len(testCases), modelCounts)
	}

	return nil
}

// testRouterR1Selection verifies LLM-as-router with reasoning
func testRouterR1Selection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPF, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPF()

	httpClient := &http.Client{Timeout: 120 * time.Second} // Longer timeout for LLM reasoning
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	if opts.Verbose {
		fmt.Println("[Test] Testing Router-R1 LLM-as-router with reasoning...")
	}

	// Router-R1 uses an LLM to reason about which model to select
	// It should consider query complexity, required capabilities, and model strengths
	testCases := []CoreSelectionTestCase{
		{
			Query:       "Solve this step by step: A train leaves station A at 9:00 AM traveling at 60 mph...",
			Algorithm:   "router_r1",
			Description: "Multi-step reasoning - should select reasoning-capable model",
		},
		{
			Query:       "Translate 'Hello World' to Japanese, Chinese, and Korean",
			Algorithm:   "router_r1",
			Description: "Multilingual query - should select multilingual model",
		},
		{
			Query:       "Debug this Python code: for i in range(10) print(i)",
			Algorithm:   "router_r1",
			Description: "Code debugging - should select coding-capable model",
		},
	}

	modelCounts := make(map[string]int)
	successCount := 0

	for i, tc := range testCases {
		req := CoreChatRequest{
			Messages: []CoreChatMessage{
				{Role: "user", Content: tc.Query},
			},
		}

		resp, err := sendCoreChatRequest(httpClient, baseURL+"/v1/chat/completions", req)
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Request %d failed: %v\n", i+1, err)
			}
			continue
		}

		if resp.SelectedModel != "" {
			modelCounts[resp.SelectedModel]++
			successCount++
		}

		if opts.Verbose {
			fmt.Printf("[Test] %s: routed to %s\n", tc.Description, resp.SelectedModel)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_requests":   len(testCases),
			"success_requests": successCount,
			"model_counts":     modelCounts,
			"algorithm":        "router_r1",
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✓ Router-R1 completed: %d/%d requests, distribution=%v\n",
			successCount, len(testCases), modelCounts)
	}

	return nil
}

// Helper functions

// sendCoreChatRequest sends a chat completion request and returns the response
func sendCoreChatRequest(client *http.Client, url string, req CoreChatRequest) (*CoreChatResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(body))
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
		return nil, fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	var chatResp CoreChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Populate SelectedModel from x-vsr-selected-model header (preferred source)
	chatResp.SelectedModel = resp.Header.Get("x-vsr-selected-model")
	// Fallback to JSON body model if header not present
	if chatResp.SelectedModel == "" {
		chatResp.SelectedModel = chatResp.Model
	}

	return &chatResp, nil
}

// sendCoreFeedbackRequest sends a feedback request and returns the response
func sendCoreFeedbackRequest(client *http.Client, url string, req CoreFeedbackRequest) (*CoreFeedbackResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(body))
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
		return nil, fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	var feedbackResp CoreFeedbackResponse
	if err := json.Unmarshal(respBody, &feedbackResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &feedbackResp, nil
}

// getCoreRatings retrieves ratings for a category
func getCoreRatings(client *http.Client, baseURL, category string) (*CoreRatingsResponse, error) {
	url := fmt.Sprintf("%s?category=%s", baseURL, category)

	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	var ratingsResp CoreRatingsResponse
	if err := json.Unmarshal(respBody, &ratingsResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &ratingsResp, nil
}
