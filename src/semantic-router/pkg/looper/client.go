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

package looper

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Client handles HTTP requests to OpenAI-compatible endpoints
type Client struct {
	httpClient       *http.Client
	endpoint         string
	headers          map[string]string
	decisionName     string // Decision name to pass in looper requests
	fusionDepth      int    // Recursion guard for Fusion requests
	maxResponseBytes int64  // Ceiling for a single upstream response body
}

// NewClient creates a new looper HTTP client
func NewClient(cfg *config.LooperConfig) *Client {
	c := &Client{
		httpClient: &http.Client{
			Timeout: time.Duration(cfg.GetTimeout()) * time.Second,
		},
		endpoint:         cfg.Endpoint,
		headers:          cfg.Headers,
		maxResponseBytes: cfg.GetMaxResponseBytes(),
	}
	return c
}

// SetDecisionName sets the decision name for this client
func (c *Client) SetDecisionName(name string) {
	c.decisionName = name
}

// SetFusionDepth sets the Fusion recursion depth marker for internal requests.
func (c *Client) SetFusionDepth(depth int) {
	c.fusionDepth = depth
}

// resolveEndpoint returns the configured looper endpoint.
func (c *Client) resolveEndpoint() string {
	return c.endpoint
}

// ModelResponse contains the parsed response from a model call
type ModelResponse struct {
	// Raw is the raw response body
	Raw []byte

	// Semantic is the decoded neutral backend response used by every Looper
	// algorithm. Raw remains private workflow/debug evidence only.
	Semantic *llmprotocol.Response

	// Content is the extracted text content from the response
	Content string

	// ReasoningContent is the extracted reasoning/thinking content from vLLM models
	// This field is populated when vLLM returns reasoning in extra response fields
	// (e.g., reasoning_content, reasoning)
	ReasoningContent string

	// Model is the model name from the response
	Model string

	// Logprobs contains token logprobs if available
	Logprobs []float64

	// AverageLogprob is the average logprob across all tokens (for confidence assessment)
	// Range: negative values, closer to 0 = more confident
	AverageLogprob float64

	// TopLogprobMargins contains the margin (top1 - top2) for each token position
	// Higher margin = model is more certain about the chosen token
	TopLogprobMargins []float64

	// AverageMargin is the average margin across all tokens
	// Range: positive values, higher = more confident
	AverageMargin float64

	// MarginEvidenceComplete is true only when every generated token has at
	// least two top-logprob entries, so AverageMargin was computed from real
	// alternatives rather than an invented fallback margin.
	MarginEvidenceComplete bool

	// Tokens contains the text of each generated token (for token filtering)
	Tokens []string

	// FilteredAverageLogprob is the average logprob computed only over semantic tokens
	// (e.g., argument values in tool calls, excluding JSON boilerplate)
	FilteredAverageLogprob float64

	// FilteredAverageMargin is the average margin computed only over semantic tokens
	FilteredAverageMargin float64

	// FilteredTokenCount records how many semantic tokens contributed to the
	// filtered averages. Presence cannot be inferred from either average because
	// zero is valid evidence for both logprob and margin.
	FilteredTokenCount int

	// HasToolCalls indicates the response contained tool_calls (not just content)
	HasToolCalls bool

	// IsStreaming indicates if this was a streaming response
	IsStreaming bool

	// StreamingChunks contains the raw SSE chunks for streaming responses
	StreamingChunks []string

	// Usage holds the token counts reported by the backend for this single
	// call. It is zero when the backend omits usage (e.g. streaming responses
	// without stream_options.include_usage).
	Usage TokenUsage

	// LatencyMs is the wall-clock duration in milliseconds of the upstream
	// round-trip (request + read + parse) for this single call.
	LatencyMs int64
}

// LogprobsConfig controls logprobs behavior for model calls
type LogprobsConfig struct {
	Enabled     bool // Whether to request logprobs from the model
	TopLogprobs int  // Number of top logprobs to return (0-5, default 1 for margin calculation)
}

// CallSemanticModel accepts the Router's neutral request contract. The current
// Looper transport remains an internal Chat-compatible endpoint, but ExtProc
// never manufactures or mutates that wire DTO.
func (c *Client) CallSemanticModel(
	ctx context.Context,
	request *llmprotocol.Request,
	modelName string,
	streaming bool,
	iteration int,
	logprobsCfg *LogprobsConfig,
) (*ModelResponse, error) {
	looperRequest, err := NewRequestFromSemantic(request)
	if err != nil {
		return nil, err
	}
	return c.CallModel(ctx, looperRequest.executionRequest, modelName, streaming, iteration, logprobsCfg)
}

// CallModel sends a request to the configured endpoint with a specific model
// Parameters:
//   - iteration: 1-based iteration number for tracking
//   - logprobsCfg: controls whether to enable logprobs and top_logprobs (nil = disabled)
func (c *Client) CallModel(ctx context.Context, req *openai.ChatCompletionNewParams, modelName string, streaming bool, iteration int, logprobsCfg *LogprobsConfig) (*ModelResponse, error) {
	// Clone and modify the request with the target model
	modifiedReq := cloneRequest(req)
	modifiedReq.Model = modelName

	// Configure logprobs based on config
	if logprobsCfg != nil && logprobsCfg.Enabled {
		modifiedReq.Logprobs = openai.Bool(true)
		topLogprobs := logprobsCfg.TopLogprobs
		if topLogprobs < 1 {
			topLogprobs = 1 // Need at least 1 for margin calculation
		}
		if topLogprobs > 5 {
			topLogprobs = 5 // API limit
		}
		modifiedReq.TopLogprobs = openai.Int(int64(topLogprobs))
	}

	// Marshal request to JSON first
	body, err := json.Marshal(modifiedReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Add stream parameter via JSON manipulation (SDK doesn't expose Stream field)
	body, err = setStreamParam(body, streaming)
	if err != nil {
		return nil, fmt.Errorf("failed to set stream param: %w", err)
	}

	logprobsEnabled := logprobsCfg != nil && logprobsCfg.Enabled
	endpoint := c.resolveEndpoint()
	logging.ComponentDebugEvent("looper", "model_call_started", map[string]interface{}{
		"decision":  c.decisionName,
		"model_ref": modelName,
		"endpoint":  endpoint,
		"streaming": streaming,
		"iteration": iteration,
		"logprobs":  logprobsEnabled,
	})

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	for k, v := range c.headers {
		httpReq.Header.Set(k, v)
	}

	observer := dispatchObserverFromContext(ctx)
	authorization := DispatchAuthorization{}
	if observer != nil {
		authorization, err = observer.Started(ctx, DispatchStart{Model: modelName, Iteration: iteration})
		if err != nil {
			return nil, fmt.Errorf("dispatch admission failed: %w", err)
		}
		if authorization.DispatchID == "" || authorization.Grant == "" {
			return nil, fmt.Errorf("dispatch admission returned an incomplete authorization")
		}
	}
	c.setInternalRequestHeaders(httpReq, ctx, iteration, authorization.Grant, authorization.RequestID)
	httpStarted := false
	completed := false
	complete := func(failureCode string) {
		if observer == nil || completed {
			return
		}
		completed = true
		observer.Completed(ctx, DispatchCompletion{
			DispatchID: authorization.DispatchID, Model: modelName, Iteration: iteration,
			HTTPStarted: httpStarted, FailureCode: failureCode,
		})
	}

	// Execute request
	start := time.Now()
	httpStarted = true
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		complete("transport_error")
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := c.readResponseBody(resp)
	if err != nil {
		complete("response_read_error")
		return nil, err
	}

	// Parse response based on streaming mode
	var result *ModelResponse
	if streaming {
		result, err = c.parseStreamingResponse(respBody, modelName)
	} else {
		result, err = c.parseNonStreamingResponse(respBody, modelName)
	}
	if err != nil {
		complete("response_parse_error")
		return nil, err
	}
	result.LatencyMs = time.Since(start).Milliseconds()
	complete("")
	return result, nil
}

// parseNonStreamingResponse parses a non-streaming JSON response
func (c *Client) parseNonStreamingResponse(body []byte, modelName string) (*ModelResponse, error) {
	semantic, _, _, err := protocolcodec.NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		return nil, fmt.Errorf("decode neutral model response: %w", err)
	}
	if semantic.Error != nil {
		return nil, semantic.Error
	}
	neutral := semanticModelResponse(semantic, modelName)
	content, reasoning, hasToolCalls := semanticResponseText(neutral)

	result := &ModelResponse{
		Raw:              body,
		Semantic:         neutral,
		Model:            modelName, // Use the requested model name, not the backend's response
		IsStreaming:      false,
		Usage:            tokenUsageFromSemantic(neutral.Usage),
		Content:          content,
		ReasoningContent: reasoning,
		HasToolCalls:     hasToolCalls,
	}

	// Logprobs remain execution evidence for the confidence algorithm. All
	// publishable content and usage above came from the neutral codec.
	if len(neutral.Evidence.TokenLogprobs) > 0 {
		analysis := extractLogprobs(neutral.Evidence.TokenLogprobs)
		result.Tokens = analysis.Tokens
		result.Logprobs = analysis.Logprobs
		result.AverageLogprob = analysis.AverageLogprob
		result.TopLogprobMargins = analysis.Margins
		result.AverageMargin = analysis.AverageMargin
		result.MarginEvidenceComplete = analysis.MarginEvidenceComplete
	}

	logging.ComponentDebugEvent("looper", "model_call_completed", map[string]interface{}{
		"decision":      c.decisionName,
		"model_ref":     modelName,
		"content_len":   len(result.Content),
		"reasoning_len": len(result.ReasoningContent),
		"avg_logprob":   result.AverageLogprob,
		"avg_margin":    result.AverageMargin,
		"streaming":     false,
	})

	return result, nil
}

// parseStreamingResponse parses SSE streaming response
func (c *Client) parseStreamingResponse(body []byte, modelName string) (*ModelResponse, error) {
	semantic, _, err := protocolcodec.NewBuiltinEngine().DecodeResponseStream(
		llmprotocol.OpenAIChatV1,
		body,
		llmprotocol.StreamContext{PublicModel: modelName},
	)
	if err != nil {
		return nil, fmt.Errorf("decode neutral model stream: %w", err)
	}
	if semantic.Error != nil {
		return nil, semantic.Error
	}
	neutral := semanticModelResponse(semantic, modelName)
	content, reasoning, hasToolCalls := semanticResponseText(neutral)
	result := &ModelResponse{
		Raw:              body,
		Semantic:         neutral,
		Model:            modelName,
		Content:          content,
		ReasoningContent: reasoning,
		HasToolCalls:     hasToolCalls,
		IsStreaming:      true,
		Usage:            tokenUsageFromSemantic(neutral.Usage),
	}

	logging.ComponentDebugEvent("looper", "model_call_completed", map[string]interface{}{
		"decision":      c.decisionName,
		"model_ref":     modelName,
		"content_len":   len(result.Content),
		"reasoning_len": len(result.ReasoningContent),
		"total_tokens":  result.Usage.TotalTokens,
		"streaming":     true,
	})

	return result, nil
}

// LogprobAnalysis contains analyzed logprob data from a response
type LogprobAnalysis struct {
	// Tokens contains the text of each generated token
	Tokens []string
	// Logprobs contains the logprob for each token (the chosen token's logprob)
	Logprobs []float64
	// AverageLogprob is the average logprob across all tokens
	// Range: negative, closer to 0 = more confident
	AverageLogprob float64

	// Margins contains the margin (top1 - top2) for each token position
	// Higher margin = model was more certain about the chosen token
	Margins []float64
	// AverageMargin is the average margin across all tokens
	// Range: positive, higher = more confident
	AverageMargin float64
	// MarginEvidenceComplete reports whether every token had a real
	// alternative from which its margin could be calculated.
	MarginEvidenceComplete bool
}

// extractLogprobs analyzes protocol-neutral token evidence produced by the
// backend response codec.
func extractLogprobs(tokens []llmprotocol.TokenLogprob) *LogprobAnalysis {
	result := &LogprobAnalysis{}
	if len(tokens) == 0 {
		return result
	}

	var logprobSum float64
	var marginSum float64
	result.MarginEvidenceComplete = true

	for _, tokenLogprob := range tokens {
		result.Tokens = append(result.Tokens, tokenLogprob.Token)
		result.Logprobs = append(result.Logprobs, tokenLogprob.Logprob)
		logprobSum += tokenLogprob.Logprob

		margin, hasAlternative := calculateMargin(tokenLogprob.Alternatives)
		result.Margins = append(result.Margins, margin)
		if hasAlternative {
			marginSum += margin
		} else {
			result.MarginEvidenceComplete = false
		}
	}

	// Calculate averages
	if len(result.Logprobs) > 0 {
		result.AverageLogprob = logprobSum / float64(len(result.Logprobs))
	}
	if result.MarginEvidenceComplete && len(result.Margins) > 0 {
		result.AverageMargin = marginSum / float64(len(result.Margins))
	}

	return result
}

// calculateMargin calculates the margin between the chosen token and the next best alternative
// A large margin indicates the model was very confident in its choice
// A small margin indicates the model was uncertain between multiple options
func calculateMargin(topLogprobs []llmprotocol.TokenLogprobAlternative) (float64, bool) {
	if len(topLogprobs) < 2 {
		// A chosen-token logprob alone contains no evidence about the nearest
		// alternative. Returning zero keeps token/margin indexes aligned; the
		// completeness flag prevents callers from treating it as confidence.
		return 0, false
	}

	// topLogprobs[0] is the chosen token (should match chosenLogprob)
	// topLogprobs[1] is the second-best alternative
	// Margin = logprob(top1) - logprob(top2)
	// Since logprobs are negative, a positive margin means top1 > top2 in probability
	top1 := topLogprobs[0].Logprob
	top2 := topLogprobs[1].Logprob

	// Margin: how much better is top1 than top2
	// Example: top1=-0.1, top2=-2.0 => margin=1.9 (high confidence)
	// Example: top1=-0.5, top2=-0.6 => margin=0.1 (low confidence, model is uncertain)
	return top1 - top2, true
}

// ApplyTokenFilter computes filtered logprob/margin averages on a ModelResponse
// using only "semantic" tokens identified by the given filter strategy.
// If the filter finds no semantic tokens or doesn't apply, the response is unchanged.
func ApplyTokenFilter(resp *ModelResponse, filter string) {
	if resp == nil || len(resp.Tokens) == 0 || filter == "" || filter == "all" {
		return
	}
	if filter == "tool_call_args" {
		filterToolCallArgTokens(resp)
	}
}

// filterToolCallArgTokens identifies tokens that represent argument VALUES in
// a JSON tool call and computes filtered averages excluding structural
// boilerplate (braces, colons, field names, quotes).
//
// Supports optional <tool_call> XML wrapper around the JSON object.
func filterToolCallArgTokens(resp *ModelResponse) {
	fullText := strings.Join(resp.Tokens, "")
	semantic := classifyToolCallChars(fullText)
	if semantic == nil {
		return
	}

	var filteredLP, filteredM []float64
	charPos := 0
	for i, tok := range resp.Tokens {
		tokenLen := len(tok)
		isSemantic := false
		for j := 0; j < tokenLen && charPos+j < len(semantic); j++ {
			if semantic[charPos+j] {
				isSemantic = true
				break
			}
		}
		if isSemantic {
			filteredLP = append(filteredLP, resp.Logprobs[i])
			if i < len(resp.TopLogprobMargins) {
				filteredM = append(filteredM, resp.TopLogprobMargins[i])
			}
		}
		charPos += tokenLen
	}

	if len(filteredLP) == 0 {
		return
	}
	resp.FilteredTokenCount = len(filteredLP)

	var lpSum float64
	for _, v := range filteredLP {
		lpSum += v
	}
	resp.FilteredAverageLogprob = lpSum / float64(len(filteredLP))

	if len(filteredM) > 0 {
		var mSum float64
		for _, v := range filteredM {
			mSum += v
		}
		resp.FilteredAverageMargin = mSum / float64(len(filteredM))
	}

	logging.Infof("[TokenFilter] tool_call_args: %d/%d tokens semantic, filtered_avg_logprob=%.4f, filtered_avg_margin=%.4f",
		len(filteredLP), len(resp.Tokens), resp.FilteredAverageLogprob, resp.FilteredAverageMargin)
}

// classifyToolCallChars returns a per-byte boolean slice indicating which
// characters are part of argument VALUES inside a tool-call JSON object.
//
// The function walks the text with a minimal JSON state machine, looking for
// the top-level "arguments" key.  All values (strings, numbers, booleans)
// directly inside the arguments object — including array elements — are
// marked as semantic.
//
// Returns nil when the text is not a recognisable tool call.
func classifyToolCallChars(text string) []bool {
	jsonStart := strings.Index(text, "{")
	if jsonStart < 0 {
		return nil
	}

	semantic := make([]bool, len(text))

	depth := 0
	argsDepth := -1 // depth of the "arguments" object; -1 = not inside
	inString := false
	escaped := false
	expectingValue := false
	buildingKey := false
	inArgValue := false

	// Track whether each depth level is an array (true) or object (false)
	// so commas inside arrays keep expecting values.
	depthIsArray := make(map[int]bool)

	var keyBuf strings.Builder
	lastKey := ""

	for i := jsonStart; i < len(text); i++ {
		c := text[i]

		// Handle escape sequences inside strings
		if escaped {
			escaped = false
			if inArgValue {
				semantic[i] = true
			}
			continue
		}
		if c == '\\' && inString {
			escaped = true
			if inArgValue {
				semantic[i] = true
			}
			continue
		}

		if inString {
			if c == '"' {
				inString = false
				if buildingKey {
					lastKey = keyBuf.String()
					keyBuf.Reset()
					buildingKey = false
				}
				if inArgValue {
					inArgValue = false // closing quote is structural
				}
			} else {
				if buildingKey {
					keyBuf.WriteByte(c)
				}
				if inArgValue {
					semantic[i] = true
				}
			}
			continue
		}

		// Not inside a string
		switch c {
		case '"':
			inString = true
			if expectingValue {
				expectingValue = false
				if argsDepth > 0 && depth >= argsDepth {
					inArgValue = true
				}
			} else if !depthIsArray[depth] {
				buildingKey = true
			} else if argsDepth > 0 && depth >= argsDepth {
				// String element inside an array that is an arg value
				inArgValue = true
			}

		case ':':
			expectingValue = true
			if lastKey == "arguments" && argsDepth < 0 {
				argsDepth = depth
			}

		case '{':
			depth++
			depthIsArray[depth] = false
			if expectingValue {
				if lastKey == "arguments" && argsDepth < 0 {
					argsDepth = depth
				}
				expectingValue = false
			}

		case '[':
			depth++
			depthIsArray[depth] = true
			// Don't clear expectingValue — first array element is a value

		case '}':
			if inArgValue {
				inArgValue = false
			}
			if argsDepth > 0 && depth == argsDepth {
				argsDepth = -1
			}
			delete(depthIsArray, depth)
			depth--

		case ']':
			if inArgValue {
				inArgValue = false
			}
			delete(depthIsArray, depth)
			depth--

		case ',':
			if inArgValue {
				inArgValue = false
			}
			// In arrays within arguments, next element is still a value
			if depthIsArray[depth] && argsDepth > 0 && depth >= argsDepth {
				expectingValue = true
			} else {
				expectingValue = false
			}

		default:
			if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
				continue
			}
			if expectingValue && argsDepth > 0 && depth >= argsDepth {
				inArgValue = true
				semantic[i] = true
				expectingValue = false
			} else if inArgValue {
				semantic[i] = true
			}
		}
	}

	for _, s := range semantic {
		if s {
			return semantic
		}
	}
	return nil
}

// setStreamParam adds or updates the stream parameter in a JSON request body
func setStreamParam(body []byte, streaming bool) ([]byte, error) {
	var reqMap map[string]interface{}
	if err := json.Unmarshal(body, &reqMap); err != nil {
		return nil, err
	}
	reqMap["stream"] = streaming
	if streaming {
		// Ask the backend to emit a trailing usage chunk so token accounting
		// works for streamed calls; preserve any caller-set stream_options.
		opts, _ := reqMap["stream_options"].(map[string]interface{})
		if opts == nil {
			opts = map[string]interface{}{}
		}
		opts["include_usage"] = true
		reqMap["stream_options"] = opts
	} else {
		delete(reqMap, "stream_options")
	}
	return json.Marshal(reqMap)
}

// cloneRequest creates a shallow copy of the request
func cloneRequest(req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
	// Create a new params with the same values
	clone := *req
	return &clone
}
