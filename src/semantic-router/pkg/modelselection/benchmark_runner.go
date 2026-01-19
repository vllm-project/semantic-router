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

package modelselection

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// nilIfEmpty returns nil if the string is empty, otherwise returns the string
// Used to preserve null values in JSON output for empty fields
func nilIfEmpty(s string) interface{} {
	if s == "" {
		return nil
	}
	return s
}

// QueryData represents a query extracted from training data
type QueryData struct {
	Query      string `json:"query"`
	Category   string `json:"category"`
	TaskName   string `json:"task_name"`
	QueryIndex int    `json:"query_index"`
	// Preserve original metadata from training data
	GroundTruth string `json:"ground_truth,omitempty"`
	Metric      string `json:"metric,omitempty"`
	Choices     string `json:"choices,omitempty"`
	TaskID      string `json:"task_id,omitempty"`
}

// BenchmarkResult represents the result of running a model on a query
type BenchmarkResult struct {
	Query       string  `json:"query"`
	Category    string  `json:"category"`
	TaskName    string  `json:"task_name"`
	ModelName   string  `json:"model_name"`
	Response    string  `json:"response"`
	LatencyMs   float64 `json:"latency_ms"`
	Performance float64 `json:"performance"`
	Success     bool    `json:"success"`
	Error       string  `json:"error,omitempty"`
	// Preserve original metadata
	GroundTruth string `json:"ground_truth,omitempty"`
	Metric      string `json:"metric,omitempty"`
	Choices     string `json:"choices,omitempty"`
	TaskID      string `json:"task_id,omitempty"`
	EmbeddingID int    `json:"embedding_id,omitempty"`
}

// BenchmarkRunner runs benchmarks on configured LLMs
type BenchmarkRunner struct {
	Config           *config.RouterConfig
	Models           []string
	ModelEndpoints   map[string]string // model name -> endpoint URL
	Queries          []QueryData
	Results          []BenchmarkResult
	Concurrency      int
	TimeoutSeconds   int
	QueryLimit       int // Limit number of queries (0 = no limit)
	RateLimitDelayMs int // Delay between requests in ms (0 = no delay, use for paid APIs)
	mu               sync.Mutex
}

// NewBenchmarkRunner creates a new benchmark runner
func NewBenchmarkRunner(cfg *config.RouterConfig, models []string) *BenchmarkRunner {
	runner := &BenchmarkRunner{
		Config:           cfg,
		Models:           models,
		ModelEndpoints:   make(map[string]string),
		Queries:          []QueryData{},
		Results:          []BenchmarkResult{},
		Concurrency:      1, // Sequential for free tier (set to 4+ for paid APIs)
		TimeoutSeconds:   60,
		RateLimitDelayMs: 500, // 500ms delay for free tier (set to 0 for paid APIs)
	}

	// Build model endpoint map
	for _, modelName := range models {
		endpoint := runner.resolveModelEndpoint(modelName)
		if endpoint != "" {
			runner.ModelEndpoints[modelName] = endpoint
		}
	}

	return runner
}

// resolveModelEndpoint finds the endpoint URL for a model
func (r *BenchmarkRunner) resolveModelEndpoint(modelName string) string {
	// Check model_config for preferred endpoints
	if modelCfg, ok := r.Config.ModelConfig[modelName]; ok {
		if len(modelCfg.PreferredEndpoints) > 0 {
			epName := modelCfg.PreferredEndpoints[0]
			for _, ep := range r.Config.VLLMEndpoints {
				if ep.Name == epName {
					return r.buildEndpointURL(ep, modelName)
				}
			}
		}
	}

	// Use first available endpoint
	if len(r.Config.VLLMEndpoints) > 0 {
		ep := r.Config.VLLMEndpoints[0]
		return r.buildEndpointURL(ep, modelName)
	}

	return ""
}

// buildEndpointURL builds the correct endpoint URL based on endpoint type and configuration
func (r *BenchmarkRunner) buildEndpointURL(ep config.VLLMEndpoint, modelName string) string {
	// Determine endpoint type from config, or infer from address
	endpointType := r.getEndpointType(ep)

	// Use HTTPS for port 443 or known HTTPS domains
	scheme := "http"
	if ep.Port == 443 || strings.Contains(ep.Address, "nvidia.com") ||
		strings.Contains(ep.Address, "openai.com") || strings.Contains(ep.Address, "api.") ||
		strings.Contains(ep.Address, "huggingface.co") || strings.Contains(ep.Address, "openrouter.ai") {
		scheme = "https"
	}

	switch endpointType {
	case "ollama":
		// Ollama uses a different API format
		return fmt.Sprintf("http://%s:%d/api/chat", ep.Address, ep.Port)

	case "huggingface":
		// HuggingFace now uses router.huggingface.co (api-inference is deprecated)
		// Model is specified in request body, not URL path
		return fmt.Sprintf("%s://router.huggingface.co/v1/chat/completions", scheme)

	case "openrouter":
		// OpenRouter uses standard OpenAI format
		return fmt.Sprintf("%s://openrouter.ai/api/v1/chat/completions", scheme)

	case "nvidia":
		// NVIDIA NIM
		return fmt.Sprintf("%s://%s/v1/chat/completions", scheme, ep.Address)

	default:
		// Standard vLLM/OpenAI endpoint
		if ep.Port == 443 || ep.Port == 80 {
			return fmt.Sprintf("%s://%s/v1/chat/completions", scheme, ep.Address)
		}
		return fmt.Sprintf("%s://%s:%d/v1/chat/completions", scheme, ep.Address, ep.Port)
	}
}

// getEndpointType determines the endpoint type from config or infers from address
func (r *BenchmarkRunner) getEndpointType(ep config.VLLMEndpoint) string {
	// Check if type is explicitly set in config
	if ep.Type != "" {
		return ep.Type
	}

	// Infer from address/name patterns
	if strings.Contains(ep.Address, "huggingface.co") {
		return "huggingface"
	}
	if strings.Contains(ep.Address, "openrouter.ai") {
		return "openrouter"
	}
	if strings.Contains(ep.Address, "nvidia.com") {
		return "nvidia"
	}
	if ep.Name == "ollama" || (strings.Contains(ep.Address, "localhost") && ep.Port == 11434) {
		return "ollama"
	}

	// Default to vLLM/OpenAI compatible
	return "vllm"
}

// getExternalModelID returns the external model ID for a given endpoint type
// First checks config.ModelParams.ExternalModelIDs, then falls back to internal name
func (r *BenchmarkRunner) getExternalModelID(modelName, endpointType string) string {
	// Check if model has external ID mapping in config
	if r.Config != nil {
		if modelParams, ok := r.Config.ModelConfig[modelName]; ok {
			if modelParams.ExternalModelIDs != nil {
				if externalID, ok := modelParams.ExternalModelIDs[endpointType]; ok {
					return externalID
				}
			}
		}
	}
	// Fall back to using internal model name
	return modelName
}

// getHuggingFaceModel maps internal model names to HuggingFace model IDs
func (r *BenchmarkRunner) getHuggingFaceModel(modelName string) string {
	return r.getExternalModelID(modelName, "huggingface")
}

// getOllamaModel maps internal model names to Ollama model names
func (r *BenchmarkRunner) getOllamaModel(modelName string) string {
	return r.getExternalModelID(modelName, "ollama")
}

// LoadQueriesFromTrainingData loads unique queries from existing training data
// Preserves embedding_id from original data for consistent grouping
func (r *BenchmarkRunner) LoadQueriesFromTrainingData(trainingDataPath string) error {
	file, err := os.Open(trainingDataPath)
	if err != nil {
		return fmt.Errorf("failed to open training data: %w", err)
	}
	defer file.Close()

	// Track unique queries by embedding_id (preserves original grouping)
	uniqueQueries := make(map[int]QueryData)

	decoder := json.NewDecoder(file)
	lineNum := 0

	for {
		var record struct {
			Query       string `json:"query"`
			Category    string `json:"category"`
			TaskName    string `json:"task_name"`
			EmbeddingID int    `json:"embedding_id"`
			GroundTruth string `json:"ground_truth"`
			Metric      string `json:"metric"`
			Choices     string `json:"choices"`
			TaskID      string `json:"task_id"`
		}

		if err := decoder.Decode(&record); err != nil {
			if err == io.EOF {
				break
			}
			// Skip malformed lines
			lineNum++
			continue
		}
		lineNum++

		// Use embedding_id as key for uniqueness (preserves original grouping)
		if _, exists := uniqueQueries[record.EmbeddingID]; !exists {
			uniqueQueries[record.EmbeddingID] = QueryData{
				Query:       record.Query,
				Category:    record.Category,
				TaskName:    record.TaskName,
				QueryIndex:  record.EmbeddingID, // Preserve original embedding_id
				GroundTruth: record.GroundTruth,
				Metric:      record.Metric,
				Choices:     record.Choices,
				TaskID:      record.TaskID,
			}
		}
	}

	// Convert map to slice
	r.Queries = make([]QueryData, 0, len(uniqueQueries))
	for _, qd := range uniqueQueries {
		r.Queries = append(r.Queries, qd)
		// Apply query limit if set
		if r.QueryLimit > 0 && len(r.Queries) >= r.QueryLimit {
			break
		}
	}

	logging.Infof("Loaded %d unique queries from training data", len(r.Queries))
	if r.QueryLimit > 0 {
		logging.Infof("Query limit applied: %d", r.QueryLimit)
	}
	return nil
}

// ChatCompletionRequest represents an OpenAI-compatible chat request
type ChatCompletionRequest struct {
	Model     string        `json:"model"`
	Messages  []ChatMessage `json:"messages"`
	MaxTokens int           `json:"max_tokens,omitempty"`
}

// ChatMessage represents a chat message
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionResponse represents an OpenAI-compatible response
type ChatCompletionResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// OllamaChatRequest represents Ollama's chat API request format
type OllamaChatRequest struct {
	Model    string              `json:"model"`
	Messages []OllamaChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
}

// OllamaChatMessage represents an Ollama chat message
type OllamaChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OllamaChatResponse represents Ollama's chat API response
type OllamaChatResponse struct {
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	Done bool `json:"done"`
}

// callModel sends a query to a model endpoint
func (r *BenchmarkRunner) callModel(ctx context.Context, modelName, endpoint string, queryData QueryData) (string, float64, error) {
	startTime := time.Now()

	// Format query with appropriate prompt based on task type
	formattedQuery := r.formatQueryForTask(queryData)

	// Detect endpoint type and use appropriate request format
	if strings.Contains(endpoint, "/api/chat") {
		// Ollama API format
		return r.callOllamaModel(ctx, modelName, endpoint, formattedQuery, startTime)
	}

	// OpenAI-compatible API format (vLLM, HuggingFace, OpenRouter, NVIDIA NIM)
	return r.callOpenAICompatibleModel(ctx, modelName, endpoint, formattedQuery, startTime)
}

// formatQueryForTask formats the query with appropriate prompt based on metric/task type
func (r *BenchmarkRunner) formatQueryForTask(q QueryData) string {
	metric := q.Metric

	switch metric {
	case "em_mc":
		// Multiple choice - add instruction to answer with just the letter
		return fmt.Sprintf("Answer with ONLY the letter of the correct choice (A, B, C, or D). Do not explain.\n\nQuestion: %s\n\nChoices: %s\n\nAnswer:", q.Query, q.Choices)

	case "MATH", "GSM8K":
		// Math problems - ask for boxed answer
		return fmt.Sprintf("%s\n\nPut your final answer in \\boxed{...} format.", q.Query)

	case "f1_score":
		// QA tasks - ask for concise answer
		return fmt.Sprintf("Answer the following question concisely:\n\n%s", q.Query)

	case "code_eval":
		// Code generation - ask for code only
		return fmt.Sprintf("Write code to solve this problem. Output ONLY the code, no explanations:\n\n%s", q.Query)

	default:
		// Default - return as-is
		return q.Query
	}
}

// callOllamaModel handles Ollama-specific API calls
func (r *BenchmarkRunner) callOllamaModel(ctx context.Context, modelName, endpoint, query string, startTime time.Time) (string, float64, error) {
	ollamaModel := r.getOllamaModel(modelName)

	reqBody := OllamaChatRequest{
		Model: ollamaModel,
		Messages: []OllamaChatMessage{
			{Role: "user", Content: query},
		},
		Stream: false,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", 0, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", 0, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: time.Duration(r.TimeoutSeconds) * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	latencyMs := float64(time.Since(startTime).Milliseconds())

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", latencyMs, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", latencyMs, fmt.Errorf("non-200 status: %d - %s", resp.StatusCode, string(body))
	}

	var ollamaResp OllamaChatResponse
	if err := json.Unmarshal(body, &ollamaResp); err != nil {
		return "", latencyMs, fmt.Errorf("failed to parse response: %w", err)
	}

	return ollamaResp.Message.Content, latencyMs, nil
}

// callOpenAICompatibleModel handles OpenAI-compatible API calls (vLLM, HuggingFace, NVIDIA NIM, etc.)
func (r *BenchmarkRunner) callOpenAICompatibleModel(ctx context.Context, modelName, endpoint, query string, startTime time.Time) (string, float64, error) {
	// Determine which model name to use in the request based on endpoint type
	requestModel := modelName
	if strings.Contains(endpoint, "huggingface.co") {
		requestModel = r.getHuggingFaceModel(modelName)
	} else if strings.Contains(endpoint, "openrouter.ai") {
		requestModel = r.getOpenRouterModel(modelName)
	} else if strings.Contains(endpoint, "nvidia.com") {
		// NVIDIA NIM uses model IDs like "meta/llama-3.1-8b-instruct"
		requestModel = r.getExternalModelID(modelName, "nvidia")
	}

	reqBody := ChatCompletionRequest{
		Model: requestModel,
		Messages: []ChatMessage{
			{Role: "user", Content: query},
		},
		MaxTokens: 512,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", 0, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", 0, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Add API keys for various providers
	r.addAuthHeaders(req, endpoint, modelName)

	client := &http.Client{Timeout: time.Duration(r.TimeoutSeconds) * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	latencyMs := float64(time.Since(startTime).Milliseconds())

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", latencyMs, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", latencyMs, fmt.Errorf("non-200 status: %d - %s", resp.StatusCode, string(body))
	}

	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return "", latencyMs, fmt.Errorf("failed to parse response: %w", err)
	}

	if chatResp.Error != nil {
		return "", latencyMs, fmt.Errorf("API error: %s", chatResp.Error.Message)
	}

	if len(chatResp.Choices) == 0 {
		return "", latencyMs, fmt.Errorf("no choices in response")
	}

	return chatResp.Choices[0].Message.Content, latencyMs, nil
}

// addAuthHeaders adds appropriate authorization headers based on endpoint
// Uses API key from config if available, otherwise falls back to environment variables
func (r *BenchmarkRunner) addAuthHeaders(req *http.Request, endpoint string, modelName string) {
	// Try to get API key from endpoint config first
	apiKey := r.getAPIKeyForEndpoint(endpoint, modelName)

	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)

		// Add OpenRouter-specific headers
		if strings.Contains(endpoint, "openrouter.ai") {
			req.Header.Set("HTTP-Referer", "https://github.com/vllm-project/semantic-router")
			req.Header.Set("X-Title", "VSR Model Selection Benchmark")
		}
		return
	}

	// Fall back to environment variables
	if strings.Contains(endpoint, "huggingface.co") {
		if envKey := os.Getenv("HF_API_KEY"); envKey != "" {
			req.Header.Set("Authorization", "Bearer "+envKey)
		} else if envKey := os.Getenv("HUGGINGFACE_API_KEY"); envKey != "" {
			req.Header.Set("Authorization", "Bearer "+envKey)
		}
		return
	}

	if strings.Contains(endpoint, "openrouter.ai") {
		if envKey := os.Getenv("OPENROUTER_API_KEY"); envKey != "" {
			req.Header.Set("Authorization", "Bearer "+envKey)
			req.Header.Set("HTTP-Referer", "https://github.com/vllm-project/semantic-router")
			req.Header.Set("X-Title", "VSR Model Selection Benchmark")
		}
		return
	}

	if strings.Contains(endpoint, "nvidia.com") {
		if envKey := os.Getenv("NVIDIA_API_KEY"); envKey != "" {
			req.Header.Set("Authorization", "Bearer "+envKey)
		}
		return
	}

	// Generic OpenAI-compatible API
	if envKey := os.Getenv("OPENAI_API_KEY"); envKey != "" {
		req.Header.Set("Authorization", "Bearer "+envKey)
	}
}

// getAPIKeyForEndpoint retrieves API key from endpoint config, model config, or environment
func (r *BenchmarkRunner) getAPIKeyForEndpoint(endpoint string, modelName string) string {
	if r.Config != nil {
		// Check model's access key first
		if modelParams, ok := r.Config.ModelConfig[modelName]; ok {
			if modelParams.AccessKey != "" {
				return modelParams.AccessKey
			}
		}

		// Check endpoint's API key
		for _, ep := range r.Config.VLLMEndpoints {
			if strings.Contains(endpoint, ep.Address) && ep.APIKey != "" {
				return ep.APIKey
			}
		}
	}

	// Fall back to environment variables based on endpoint type
	if strings.Contains(endpoint, "huggingface.co") {
		if envKey := os.Getenv("HF_API_KEY"); envKey != "" {
			return envKey
		}
		if envKey := os.Getenv("HUGGINGFACE_API_KEY"); envKey != "" {
			return envKey
		}
	}
	if strings.Contains(endpoint, "openrouter.ai") {
		if envKey := os.Getenv("OPENROUTER_API_KEY"); envKey != "" {
			return envKey
		}
	}
	if strings.Contains(endpoint, "nvidia.com") {
		if envKey := os.Getenv("NVIDIA_API_KEY"); envKey != "" {
			return envKey
		}
	}

	return ""
}

// getOpenRouterModel maps internal model names to OpenRouter model IDs
func (r *BenchmarkRunner) getOpenRouterModel(modelName string) string {
	return r.getExternalModelID(modelName, "openrouter")
}

// evaluateResponse evaluates the quality of a response (simplified)
func (r *BenchmarkRunner) evaluateResponse(query, response, taskName, groundTruth, metric string) float64 {
	// Evaluate response using ground truth and metric when available

	if response == "" {
		return 0.0
	}

	// If we have ground_truth and metric, use proper evaluation
	if groundTruth != "" && metric != "" {
		return r.evaluateWithGroundTruth(response, groundTruth, metric)
	}

	// Fallback to heuristic-based evaluation
	return r.evaluateWithHeuristics(response, taskName)
}

// evaluateWithGroundTruth evaluates response against ground truth using the specified metric
// This implementation matches standard LLM evaluation metrics
// Based on RouteLLM methodology (arXiv:2406.18665)
func (r *BenchmarkRunner) evaluateWithGroundTruth(response, groundTruth, metric string) float64 {
	respLower := strings.ToLower(strings.TrimSpace(response))
	gtLower := strings.ToLower(strings.TrimSpace(groundTruth))

	switch metric {
	case "em_mc":
		// Exact match for multiple choice
		// Ground truth is typically a single letter: "A", "B", "C", "D"
		// Response must contain the correct answer letter
		return r.evaluateExactMatchMC(respLower, gtLower)

	case "f1_score":
		// F1 score based on word overlap
		return r.evaluateF1Score(respLower, gtLower)

	case "MATH", "GSM8K":
		// Math evaluation - extract answer from \boxed{} or find number
		return r.evaluateMath(response, groundTruth)

	case "code_eval":
		// Code evaluation requires execution - check if response contains key elements
		// Code execution metrics require actual runtime evaluation
		// We approximate by checking if response looks like valid code
		return r.evaluateCodeApprox(response, groundTruth)

	case "commongen_coverage":
		// CommonGen coverage - check word coverage
		return r.evaluateCommonGenCoverage(respLower, gtLower)

	case "em", "exact_match":
		// Strict exact match
		if respLower == gtLower {
			return 1.0
		}
		return 0.0

	case "contains":
		// Check if response contains the ground truth
		if strings.Contains(respLower, gtLower) {
			return 1.0
		}
		return 0.0

	default:
		// Unknown metric - try contains check
		if strings.Contains(respLower, gtLower) {
			return 1.0
		}
		return 0.0
	}
}

// evaluateExactMatchMC evaluates multiple choice responses (em_mc metric)
func (r *BenchmarkRunner) evaluateExactMatchMC(response, groundTruth string) float64 {
	// Ground truth is a single letter like "A", "B", "C", "D"
	// Check if response starts with or clearly indicates the answer
	gt := strings.TrimSpace(groundTruth)

	// Check common patterns where the answer letter appears
	patterns := []string{
		gt,                        // Just the letter
		"(" + gt + ")",            // (A)
		gt + ".",                  // A.
		gt + ":",                  // A:
		gt + ")",                  // A)
		"answer is " + gt,         // answer is A
		"answer: " + gt,           // answer: A
		"the answer is " + gt,     // the answer is A
		"correct answer is " + gt, // correct answer is A
	}

	// First check if response starts with the answer
	if strings.HasPrefix(response, gt) {
		return 1.0
	}

	// Check all patterns
	for _, pattern := range patterns {
		if strings.Contains(response, pattern) {
			return 1.0
		}
	}

	return 0.0
}

// evaluateF1Score calculates F1 score based on word overlap (f1_score metric)
func (r *BenchmarkRunner) evaluateF1Score(response, groundTruth string) float64 {
	// Tokenize into words
	respWords := tokenizeWords(response)
	gtWords := tokenizeWords(groundTruth)

	if len(gtWords) == 0 {
		return 0.0
	}
	if len(respWords) == 0 {
		return 0.0
	}

	// Count matching words (case-insensitive)
	gtWordSet := make(map[string]bool)
	for _, w := range gtWords {
		gtWordSet[w] = true
	}

	matches := 0
	for _, w := range respWords {
		if gtWordSet[w] {
			matches++
		}
	}

	if matches == 0 {
		return 0.0
	}

	precision := float64(matches) / float64(len(respWords))
	recall := float64(matches) / float64(len(gtWords))

	if precision+recall == 0 {
		return 0.0
	}

	return 2 * precision * recall / (precision + recall)
}

// evaluateMath evaluates math responses (MATH/GSM8K metrics)
func (r *BenchmarkRunner) evaluateMath(response, groundTruth string) float64 {
	gt := strings.TrimSpace(groundTruth)

	// Try to extract answer from \boxed{...}
	boxedAnswer := extractBoxedAnswer(response)
	if boxedAnswer != "" {
		if strings.EqualFold(boxedAnswer, gt) {
			return 1.0
		}
		// Try numeric comparison
		if compareNumeric(boxedAnswer, gt) {
			return 1.0
		}
	}

	// Check if ground truth number appears in response
	if strings.Contains(strings.ToLower(response), strings.ToLower(gt)) {
		return 1.0
	}

	// Try to find the answer at the end of response
	lastAnswer := extractLastNumber(response)
	if lastAnswer != "" && compareNumeric(lastAnswer, gt) {
		return 1.0
	}

	return 0.0
}

// evaluateCodeApprox approximates code evaluation
func (r *BenchmarkRunner) evaluateCodeApprox(response, groundTruth string) float64 {
	// Without actual code execution, we can only approximate
	// Check if response contains function definition
	if strings.Contains(response, "def ") || strings.Contains(response, "function ") {
		// Basic check - response has code structure
		if strings.Contains(response, "return") {
			return 0.5 // Partial credit for having a return statement
		}
		return 0.3
	}
	return 0.0
}

// evaluateCommonGenCoverage evaluates CommonGen coverage
func (r *BenchmarkRunner) evaluateCommonGenCoverage(response, groundTruth string) float64 {
	// CommonGen: check how many ground truth concepts appear in response
	gtWords := strings.Fields(groundTruth)
	if len(gtWords) == 0 {
		return 0.0
	}

	covered := 0
	for _, word := range gtWords {
		if strings.Contains(response, word) {
			covered++
		}
	}

	return float64(covered) / float64(len(gtWords))
}

// tokenizeWords splits text into lowercase words, removing punctuation
func tokenizeWords(text string) []string {
	// Remove common punctuation
	text = strings.ToLower(text)
	replacer := strings.NewReplacer(
		",", " ", ".", " ", "!", " ", "?", " ",
		"(", " ", ")", " ", "[", " ", "]", " ",
		"\"", " ", "'", " ", ":", " ", ";", " ",
	)
	text = replacer.Replace(text)

	words := strings.Fields(text)
	result := make([]string, 0, len(words))
	for _, w := range words {
		w = strings.TrimSpace(w)
		if len(w) > 0 {
			result = append(result, w)
		}
	}
	return result
}

// extractBoxedAnswer extracts the answer from \boxed{...} format
func extractBoxedAnswer(response string) string {
	// Look for \boxed{...}
	idx := strings.Index(response, "\\boxed{")
	if idx == -1 {
		return ""
	}

	start := idx + 7 // len("\\boxed{")
	depth := 1
	end := start

	for end < len(response) && depth > 0 {
		switch response[end] {
		case '{':
			depth++
		case '}':
			depth--
		}
		if depth > 0 {
			end++
		}
	}

	if end > start {
		return strings.TrimSpace(response[start:end])
	}
	return ""
}

// extractLastNumber finds the last number in the response
func extractLastNumber(response string) string {
	// Find the last sequence of digits (possibly with decimal point)
	var lastNum string
	var current strings.Builder
	inNumber := false

	for _, ch := range response {
		if ch >= '0' && ch <= '9' {
			current.WriteRune(ch)
			inNumber = true
		} else if ch == '.' && inNumber {
			current.WriteRune(ch)
		} else if inNumber {
			lastNum = current.String()
			current.Reset()
			inNumber = false
		}
	}
	if inNumber {
		lastNum = current.String()
	}

	return strings.TrimSuffix(lastNum, ".")
}

// compareNumeric compares two strings as numbers
func compareNumeric(a, b string) bool {
	// Clean up the strings
	a = strings.TrimSpace(a)
	b = strings.TrimSpace(b)

	// Remove trailing .0 for integer comparison
	a = strings.TrimSuffix(a, ".0")
	b = strings.TrimSuffix(b, ".0")

	return a == b
}

// evaluateWithHeuristics provides fallback heuristic-based evaluation
func (r *BenchmarkRunner) evaluateWithHeuristics(response, taskName string) float64 {
	// Basic heuristics when ground truth is not available
	score := 0.5 // Base score for non-empty response

	// Longer responses for complex queries
	if len(response) > 100 {
		score += 0.1
	}
	if len(response) > 500 {
		score += 0.1
	}

	// Check for common quality indicators
	lowResp := strings.ToLower(response)
	if strings.Contains(lowResp, "error") || strings.Contains(lowResp, "sorry") ||
		strings.Contains(lowResp, "i don't know") || strings.Contains(lowResp, "cannot") {
		score -= 0.2
	}

	// Task-specific adjustments
	switch taskName {
	case "math", "gsm8k":
		// Math responses should have numbers
		if strings.ContainsAny(response, "0123456789") {
			score += 0.1
		}
	case "code", "human_eval", "mbpp":
		// Code responses should have code-like content
		if strings.Contains(response, "def ") || strings.Contains(response, "function") ||
			strings.Contains(response, "{") {
			score += 0.1
		}
	}

	// Clamp to [0, 1]
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return score
}

// RunBenchmarks runs all models on all queries
func (r *BenchmarkRunner) RunBenchmarks(progressCallback func(completed, total int)) error {
	if len(r.Queries) == 0 {
		return fmt.Errorf("no queries loaded")
	}
	if len(r.Models) == 0 {
		return fmt.Errorf("no models to benchmark")
	}

	totalTasks := len(r.Queries) * len(r.Models)
	completedTasks := 0

	logging.Infof("Running benchmarks: %d queries × %d models = %d total tasks",
		len(r.Queries), len(r.Models), totalTasks)

	// Create work channel
	type workItem struct {
		query QueryData
		model string
	}
	workChan := make(chan workItem, totalTasks)

	// Fill work channel
	for _, model := range r.Models {
		for _, query := range r.Queries {
			workChan <- workItem{query: query, model: model}
		}
	}
	close(workChan)

	// Create worker pool
	var wg sync.WaitGroup
	for i := 0; i < r.Concurrency; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			for work := range workChan {
				ctx, cancel := context.WithTimeout(context.Background(),
					time.Duration(r.TimeoutSeconds)*time.Second)

				endpoint := r.ModelEndpoints[work.model]
				if endpoint == "" {
					r.addResult(BenchmarkResult{
						Query:     work.query.Query,
						Category:  work.query.Category,
						TaskName:  work.query.TaskName,
						ModelName: work.model,
						Success:   false,
						Error:     "no endpoint configured",
					})
					cancel()
					continue
				}

				response, latencyMs, err := r.callModel(ctx, work.model, endpoint, work.query)
				cancel()

				result := BenchmarkResult{
					Query:       work.query.Query,
					Category:    work.query.Category,
					TaskName:    work.query.TaskName,
					ModelName:   work.model,
					LatencyMs:   latencyMs,
					GroundTruth: work.query.GroundTruth,
					Metric:      work.query.Metric,
					Choices:     work.query.Choices,
					TaskID:      work.query.TaskID,
					EmbeddingID: work.query.QueryIndex,
				}

				if err != nil {
					result.Success = false
					result.Error = err.Error()
				} else {
					result.Success = true
					result.Response = response
					result.Performance = r.evaluateResponse(work.query.Query, response, work.query.TaskName, work.query.GroundTruth, work.query.Metric)
				}

				r.addResult(result)

				// Optional delay between requests (for rate-limited free tier APIs)
				if r.RateLimitDelayMs > 0 {
					time.Sleep(time.Duration(r.RateLimitDelayMs) * time.Millisecond)
				}

				r.mu.Lock()
				completedTasks++
				if progressCallback != nil && completedTasks%100 == 0 {
					progressCallback(completedTasks, totalTasks)
				}
				r.mu.Unlock()
			}
		}(i)
	}

	wg.Wait()

	logging.Infof("Benchmark complete: %d results collected", len(r.Results))
	return nil
}

// addResult adds a result to the results slice (thread-safe)
func (r *BenchmarkRunner) addResult(result BenchmarkResult) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.Results = append(r.Results, result)
}

// SaveTrainingData saves benchmark results as training data
// Output format matches BenchmarkRecord for compatibility with LoadBenchmarkData
// Fields are identical to training_data_with_category.jsonl
func (r *BenchmarkRunner) SaveTrainingData(outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	successCount := 0

	for _, result := range r.Results {
		if !result.Success {
			continue
		}

		// Convert to BenchmarkRecord format - identical fields to training_data_with_category.jsonl
		record := map[string]interface{}{
			"task_name":     result.TaskName,
			"query":         result.Query,
			"ground_truth":  nilIfEmpty(result.GroundTruth), // Preserved from original data
			"metric":        nilIfEmpty(result.Metric),      // Preserved from original data
			"choices":       nilIfEmpty(result.Choices),     // Preserved from original data
			"task_id":       nilIfEmpty(result.TaskID),
			"model_name":    result.ModelName,
			"response":      result.Response,           // Include the actual LLM response
			"token_num":     nil,                       // Not tracked in current implementation
			"input_tokens":  nil,                       // Not tracked in current implementation
			"output_tokens": nil,                       // Not tracked in current implementation
			"response_time": result.LatencyMs / 1000.0, // Convert ms to seconds
			"api_key_used":  nil,                       // Not exposed for security
			"performance":   result.Performance,
			"embedding_id":  result.EmbeddingID, // Use from result (copied from query)
			"user_id":       nil,
			"fig_id":        nil,
			"category":      result.Category,
		}

		if err := encoder.Encode(record); err != nil {
			logging.Warnf("Failed to encode record: %v", err)
			continue
		}
		successCount++
	}

	logging.Infof("Saved %d training records to %s", successCount, outputPath)
	return nil
}

// GetStatistics returns benchmark statistics
func (r *BenchmarkRunner) GetStatistics() map[string]interface{} {
	stats := make(map[string]interface{})

	// Count successes and failures per model
	modelStats := make(map[string]map[string]int)
	for _, result := range r.Results {
		if _, ok := modelStats[result.ModelName]; !ok {
			modelStats[result.ModelName] = map[string]int{"success": 0, "failure": 0}
		}
		if result.Success {
			modelStats[result.ModelName]["success"]++
		} else {
			modelStats[result.ModelName]["failure"]++
		}
	}

	stats["models"] = modelStats
	stats["total_queries"] = len(r.Queries)
	stats["total_models"] = len(r.Models)
	stats["total_results"] = len(r.Results)

	return stats
}
