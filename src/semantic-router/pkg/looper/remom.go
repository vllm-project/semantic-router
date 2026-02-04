package looper

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"sort"
	"text/template"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ReMoMLooper implements the ReMoM (Reasoning for Mixture of Models) algorithm
// This algorithm performs multi-round parallel reasoning with intelligent synthesis
// Inspired by PaCoRe (arXiv:2601.05593) but extended to support mixture of models
type ReMoMLooper struct {
	*BaseLooper
}

// NewReMoMLooper creates a new ReMoM looper
func NewReMoMLooper(cfg *config.LooperConfig) *ReMoMLooper {
	return &ReMoMLooper{
		BaseLooper: NewBaseLooper(cfg),
	}
}

// getDefaultReMoMConfig returns default configuration
func getDefaultReMoMConfig() *config.ReMoMAlgorithmConfig {
	return &config.ReMoMAlgorithmConfig{
		BreadthSchedule:              []int{4},
		ModelDistribution:            "weighted",
		Temperature:                  1.0,
		IncludeReasoning:             false,
		CompactionStrategy:           "full",
		CompactionTokens:             1000,
		OnError:                      "skip",
		ShuffleSeed:                  42,
		IncludeIntermediateResponses: true,
	}
}

// ModelCall represents a single model call with its configuration
type ModelCall struct {
	Model    string
	LoRAName string
}

// ReferenceResponse represents a response used as reference in synthesis
type ReferenceResponse struct {
	Content   string
	Reasoning string
	Model     string
}

// SynthesisData contains data for template rendering
type SynthesisData struct {
	OriginalContent    string
	ReferenceResponses []ReferenceResponse
}

// RoundResponse represents responses from a single round (for visualization)
type RoundResponse struct {
	Round     int                `json:"round"`
	Breadth   int                `json:"breadth"`
	Responses []IntermediateResp `json:"responses"`
}

// IntermediateResp represents a single intermediate response
type IntermediateResp struct {
	Model            string `json:"model"`
	Content          string `json:"content"`
	Reasoning        string `json:"reasoning,omitempty"`
	CompactedContent string `json:"compacted_content,omitempty"`
	TokenCount       int    `json:"token_count,omitempty"`
}

// Default synthesis templates
const defaultSynthesisTemplate = `You are given a problem and a list of reference responses. Your job is to analyze these references and provide your own response.

Original Problem:
{{.OriginalContent}}

Reference Responses:
{{range $i, $resp := .ReferenceResponses}}
Reference {{add $i 1}}{{if $resp.Model}} ({{$resp.Model}}){{end}}:
{{$resp.Content}}
{{end}}

Now, based on the original problem and reference responses above, please provide your own comprehensive solution.`

const defaultSynthesisTemplateWithReasoning = `You are given a problem and a list of reference responses with their reasoning processes. Your job is to analyze these reasoning processes and provide your own response.

Original Problem:
{{.OriginalContent}}

Reference Responses:
{{range $i, $resp := .ReferenceResponses}}
Reference {{add $i 1}}{{if $resp.Model}} ({{$resp.Model}}){{end}}:
{{if $resp.Reasoning}}
Reasoning:
{{$resp.Reasoning}}

Answer:
{{end}}
{{$resp.Content}}
{{end}}

Now, analyze these reasoning processes and reference responses, then provide your own comprehensive solution with clear reasoning.`

// Execute implements the Looper interface for ReMoM
func (l *ReMoMLooper) Execute(ctx context.Context, req *Request) (*Response, error) {
	// 1. Set decision name in client for header transmission
	l.client.SetDecisionName(req.DecisionName)

	// 2. Get config from request
	var cfg *config.ReMoMAlgorithmConfig
	if req.Algorithm != nil && req.Algorithm.ReMoM != nil {
		cfg = req.Algorithm.ReMoM
	} else {
		cfg = getDefaultReMoMConfig()
	}

	// 3. Validate
	// Note: ReMoM internally uses non-streaming calls even if client expects streaming
	// This is because we need complete responses for synthesis across rounds
	// The final response will be returned as a complete (non-streaming) response
	if len(req.ModelRefs) == 0 {
		return nil, fmt.Errorf("no models configured")
	}
	if len(cfg.BreadthSchedule) == 0 {
		return nil, fmt.Errorf("breadth_schedule cannot be empty")
	}

	// 3. Build schedule (append [1] for final round)
	schedule := append([]int{}, cfg.BreadthSchedule...)
	schedule = append(schedule, 1)

	// 4. Extract original content
	originalContent := extractOriginalContent(req.OriginalRequest)

	// 5. Initialize tracking
	var allRoundResponses []RoundResponse
	modelsUsed := make(map[string]bool)
	totalIterations := 0

	// 6. Execute multi-round iteration
	currentMessages := cloneMessages(req.OriginalRequest)

	for roundIdx, numCalls := range schedule {
		logging.Infof("[ReMoM] Round %d/%d: %d parallel calls", roundIdx+1, len(schedule), numCalls)

		// Build synthesis prompt if not first round
		if roundIdx > 0 {
			// Get previous round responses
			prevRound := allRoundResponses[roundIdx-1]
			var prevResponses []*ModelResponse
			for _, ir := range prevRound.Responses {
				prevResponses = append(prevResponses, &ModelResponse{
					Content:          ir.Content,
					ReasoningContent: ir.Reasoning,
					Model:            ir.Model,
				})
			}

			synthesisPrompt, err := l.buildSynthesisPrompt(cfg, originalContent, prevResponses)
			if err != nil {
				return nil, fmt.Errorf("failed to build synthesis prompt for round %d: %w", roundIdx+1, err)
			}

			// Replace last message with synthesis prompt
			currentMessages = replaceLastMessage(currentMessages, synthesisPrompt)
		}

		// Distribute calls to models
		modelCalls := l.distributeCallsToModels(cfg, numCalls, req.ModelRefs)

		// Execute parallel calls
		responses, err := l.executeParallelCalls(ctx, cfg, modelCalls, currentMessages)
		if err != nil {
			if cfg.OnError == "fail" {
				return nil, fmt.Errorf("round %d failed: %w", roundIdx+1, err)
			}
			logging.Warnf("[ReMoM] Round %d had errors but continuing (on_error=skip)", roundIdx+1)
		}

		if len(responses) == 0 {
			return nil, fmt.Errorf("round %d: all model calls failed", roundIdx+1)
		}

		// Sort and shuffle responses
		responses = l.sortAndShuffle(cfg, responses)

		// Collect intermediate responses for visualization
		roundResp := RoundResponse{
			Round:   roundIdx + 1,
			Breadth: numCalls,
		}

		maxResponses := len(responses)
		if cfg.MaxResponsesPerRound > 0 && cfg.MaxResponsesPerRound < maxResponses {
			maxResponses = cfg.MaxResponsesPerRound
		}

		for i := 0; i < maxResponses; i++ {
			resp := responses[i]
			compacted := l.compactResponse(cfg, resp.Content)
			roundResp.Responses = append(roundResp.Responses, IntermediateResp{
				Model:            resp.Model,
				Content:          resp.Content,
				Reasoning:        resp.ReasoningContent,
				CompactedContent: compacted,
				TokenCount:       estimateTokens(resp.Content),
			})
		}

		allRoundResponses = append(allRoundResponses, roundResp)

		// Track models used
		for _, resp := range responses {
			modelsUsed[resp.Model] = true
		}
		totalIterations += len(responses)

		logging.Infof("[ReMoM] Round %d completed: %d responses", roundIdx+1, len(responses))
	}

	// 7. Build final response
	finalResponse := allRoundResponses[len(allRoundResponses)-1].Responses[0]

	// Convert models used to slice
	var modelsUsedSlice []string
	for model := range modelsUsed {
		modelsUsedSlice = append(modelsUsedSlice, model)
	}

	// Format response based on streaming preference
	if req.IsStreaming {
		return l.formatReMoMStreamingResponse(finalResponse, allRoundResponses, modelsUsedSlice, totalIterations, cfg)
	}
	return l.formatReMoMJSONResponse(finalResponse, allRoundResponses, modelsUsedSlice, totalIterations, cfg)
}

// formatReMoMJSONResponse creates a non-streaming JSON response
func (l *ReMoMLooper) formatReMoMJSONResponse(
	finalResponse IntermediateResp,
	allRoundResponses []RoundResponse,
	modelsUsed []string,
	iterations int,
	cfg *config.ReMoMAlgorithmConfig,
) (*Response, error) {
	completion := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-remom-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   finalResponse.Model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": finalResponse.Content,
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     0,
			"completion_tokens": 0,
			"total_tokens":      0,
		},
	}

	// Add intermediate responses if enabled
	if cfg.IncludeIntermediateResponses {
		completion["reasoning_mom_responses"] = allRoundResponses
	}

	responseBody, err := json.Marshal(completion)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal response: %w", err)
	}

	return &Response{
		Body:                  responseBody,
		ContentType:           "application/json",
		Model:                 finalResponse.Model,
		ModelsUsed:            modelsUsed,
		Iterations:            iterations,
		AlgorithmType:         "remom",
		IntermediateResponses: allRoundResponses,
	}, nil
}

// formatReMoMStreamingResponse creates an SSE streaming response
func (l *ReMoMLooper) formatReMoMStreamingResponse(
	finalResponse IntermediateResp,
	allRoundResponses []RoundResponse,
	modelsUsed []string,
	iterations int,
	cfg *config.ReMoMAlgorithmConfig,
) (*Response, error) {
	timestamp := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-remom-%d", timestamp)

	var sseBody []byte

	// First chunk: send reasoning_mom_responses if enabled
	if cfg.IncludeIntermediateResponses {
		firstChunk := map[string]interface{}{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": timestamp,
			"model":   finalResponse.Model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"role": "assistant",
					},
					"finish_reason": nil,
				},
			},
			"reasoning_mom_responses": allRoundResponses,
		}
		firstChunkJSON, _ := json.Marshal(firstChunk)
		sseBody = append(sseBody, []byte(fmt.Sprintf("data: %s\n\n", firstChunkJSON))...)
	} else {
		// Standard first chunk with role
		firstChunk := map[string]interface{}{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": timestamp,
			"model":   finalResponse.Model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"role": "assistant",
					},
					"finish_reason": nil,
				},
			},
		}
		firstChunkJSON, _ := json.Marshal(firstChunk)
		sseBody = append(sseBody, []byte(fmt.Sprintf("data: %s\n\n", firstChunkJSON))...)
	}

	// Content chunk (send all content at once since we have the complete response)
	contentChunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": timestamp,
		"model":   finalResponse.Model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"delta": map[string]interface{}{
					"content": finalResponse.Content,
				},
				"finish_reason": nil,
			},
		},
	}
	contentChunkJSON, _ := json.Marshal(contentChunk)
	sseBody = append(sseBody, []byte(fmt.Sprintf("data: %s\n\n", contentChunkJSON))...)

	// Final chunk with finish_reason
	finalChunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": timestamp,
		"model":   finalResponse.Model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"delta":         map[string]interface{}{},
				"finish_reason": "stop",
			},
		},
	}
	finalChunkJSON, _ := json.Marshal(finalChunk)
	sseBody = append(sseBody, []byte(fmt.Sprintf("data: %s\n\n", finalChunkJSON))...)

	// Add [DONE] marker
	sseBody = append(sseBody, []byte("data: [DONE]\n\n")...)

	return &Response{
		Body:                  sseBody,
		ContentType:           "text/event-stream",
		Model:                 finalResponse.Model,
		ModelsUsed:            modelsUsed,
		Iterations:            iterations,
		AlgorithmType:         "remom",
		IntermediateResponses: allRoundResponses,
	}, nil
}

// distributeCallsToModels distributes K calls among models based on strategy
func (l *ReMoMLooper) distributeCallsToModels(cfg *config.ReMoMAlgorithmConfig, numCalls int, modelRefs []config.ModelRef) []ModelCall {
	strategy := cfg.ModelDistribution
	if strategy == "" {
		strategy = "weighted"
	}

	switch strategy {
	case "weighted":
		return distributeByWeight(numCalls, modelRefs, cfg.ShuffleSeed)
	case "equal":
		return distributeEqually(numCalls, modelRefs, cfg.ShuffleSeed)
	case "first_only":
		return distributeFirstOnly(numCalls, modelRefs)
	default:
		logging.Warnf("[ReMoM] Unknown distribution strategy %s, using weighted", strategy)
		return distributeByWeight(numCalls, modelRefs, cfg.ShuffleSeed)
	}
}

// distributeByWeight distributes calls proportionally based on model weights
// Since ModelRef doesn't have Weight field, this falls back to equal distribution
func distributeByWeight(numCalls int, modelRefs []config.ModelRef, seed int) []ModelCall {
	// For now, treat weighted distribution as equal distribution
	// In the future, we could add weight support to ModelRef if needed
	return distributeEqually(numCalls, modelRefs, seed)
}

// distributeEqually distributes calls evenly among all models
func distributeEqually(numCalls int, modelRefs []config.ModelRef, seed int) []ModelCall {
	if len(modelRefs) == 0 {
		return nil
	}

	var calls []ModelCall
	callsPerModel := numCalls / len(modelRefs)
	remainder := numCalls % len(modelRefs)

	for i, ref := range modelRefs {
		count := callsPerModel
		if i < remainder {
			count++ // Distribute remainder to first N models
		}

		for j := 0; j < count; j++ {
			calls = append(calls, ModelCall{
				Model:    ref.Model,
				LoRAName: ref.LoRAName,
			})
		}
	}

	// Shuffle
	r := rand.New(rand.NewSource(int64(seed)))
	r.Shuffle(len(calls), func(i, j int) {
		calls[i], calls[j] = calls[j], calls[i]
	})

	return calls
}

// distributeFirstOnly uses only the first model (PaCoRe-compatible)
func distributeFirstOnly(numCalls int, modelRefs []config.ModelRef) []ModelCall {
	if len(modelRefs) == 0 {
		return nil
	}

	ref := modelRefs[0]
	calls := make([]ModelCall, numCalls)
	for i := 0; i < numCalls; i++ {
		calls[i] = ModelCall{
			Model:    ref.Model,
			LoRAName: ref.LoRAName,
		}
	}

	return calls
}

// executeParallelCalls executes model calls in parallel with concurrency control
func (l *ReMoMLooper) executeParallelCalls(ctx context.Context, cfg *config.ReMoMAlgorithmConfig, modelCalls []ModelCall, messages *openai.ChatCompletionNewParams) ([]*ModelResponse, error) {
	numCalls := len(modelCalls)

	// Determine max concurrent
	maxConcurrent := numCalls
	if cfg.MaxConcurrent > 0 && cfg.MaxConcurrent < numCalls {
		maxConcurrent = cfg.MaxConcurrent
	}

	// Semaphore for concurrency control
	sem := make(chan struct{}, maxConcurrent)

	// Result channel
	type result struct {
		resp  *ModelResponse
		err   error
		index int
	}
	results := make(chan result, numCalls)

	// Launch goroutines
	for i, call := range modelCalls {
		go func(idx int, mc ModelCall) {
			modelName := mc.Model
			if mc.LoRAName != "" {
				modelName = mc.LoRAName
			}

			logging.Infof("[ReMoM] Goroutine %d/%d started for model %s", idx+1, numCalls, modelName)

			sem <- struct{}{}
			defer func() { <-sem }()

			// Clone request and set temperature
			msgCopy := cloneRequest(messages)
			if cfg.Temperature > 0 {
				msgCopy.Temperature = openai.Float(cfg.Temperature)
			}

			startTime := time.Now()
			resp, err := l.client.CallModel(
				ctx,
				msgCopy,
				modelName,
				false, // streaming
				idx+1, // iteration
				nil,   // logprobs config
				"",    // accessKey - not used in ReMoM
			)
			elapsed := time.Since(startTime)

			if err != nil {
				logging.Warnf("[ReMoM] Goroutine %d/%d failed for model %s after %v: %v", idx+1, numCalls, modelName, elapsed, err)
			} else {
				logging.Infof("[ReMoM] Goroutine %d/%d completed for model %s in %v", idx+1, numCalls, modelName, elapsed)
			}

			results <- result{resp: resp, err: err, index: idx}
		}(i, call)
	}

	// Collect results
	var responses []*ModelResponse
	var errs []error

	for i := 0; i < numCalls; i++ {
		select {
		case res := <-results:
			if res.err != nil {
				errs = append(errs, res.err)
				if cfg.OnError == "fail" {
					return nil, fmt.Errorf("model call %d failed: %w", res.index, res.err)
				}
			} else {
				responses = append(responses, res.resp)
			}
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	if len(responses) == 0 {
		return nil, fmt.Errorf("all %d model calls failed: %v", numCalls, errs)
	}

	logging.Infof("[ReMoM] Collected %d/%d successful responses", len(responses), numCalls)
	return responses, nil
}

// buildSynthesisPrompt builds the synthesis prompt using template
func (l *ReMoMLooper) buildSynthesisPrompt(cfg *config.ReMoMAlgorithmConfig, originalContent string, prevResponses []*ModelResponse) (string, error) {
	// Prepare reference responses
	var refResponses []ReferenceResponse
	for _, resp := range prevResponses {
		compacted := l.compactResponse(cfg, resp.Content)
		refResp := ReferenceResponse{
			Content: compacted,
			Model:   resp.Model,
		}
		if cfg.IncludeReasoning && resp.ReasoningContent != "" {
			refResp.Reasoning = resp.ReasoningContent
		}
		refResponses = append(refResponses, refResp)
	}

	data := SynthesisData{
		OriginalContent:    originalContent,
		ReferenceResponses: refResponses,
	}

	// Choose template
	templateStr := cfg.SynthesisTemplate
	if templateStr == "" {
		if cfg.IncludeReasoning {
			templateStr = defaultSynthesisTemplateWithReasoning
		} else {
			templateStr = defaultSynthesisTemplate
		}
	}

	// Parse and execute template
	tmpl, err := template.New("synthesis").Funcs(template.FuncMap{
		"add": func(a, b int) int { return a + b },
	}).Parse(templateStr)
	if err != nil {
		return "", fmt.Errorf("failed to parse template: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("failed to execute template: %w", err)
	}

	return buf.String(), nil
}

// compactResponse compacts a response based on strategy
func (l *ReMoMLooper) compactResponse(cfg *config.ReMoMAlgorithmConfig, content string) string {
	strategy := cfg.CompactionStrategy
	if strategy == "" {
		strategy = "full"
	}

	switch strategy {
	case "last_n_tokens":
		maxTokens := cfg.CompactionTokens
		if maxTokens <= 0 {
			maxTokens = 1000
		}
		// Rough heuristic: ~4 chars per token
		maxChars := maxTokens * 4
		if len(content) <= maxChars {
			return content
		}
		return content[len(content)-maxChars:]
	case "full":
		fallthrough
	default:
		return content
	}
}

// sortAndShuffle sorts responses by length and shuffles
func (l *ReMoMLooper) sortAndShuffle(cfg *config.ReMoMAlgorithmConfig, responses []*ModelResponse) []*ModelResponse {
	// Sort by content length (descending)
	sort.Slice(responses, func(i, j int) bool {
		return len(responses[i].Content) > len(responses[j].Content)
	})

	// Shuffle with seed for reproducibility
	r := rand.New(rand.NewSource(int64(cfg.ShuffleSeed)))
	r.Shuffle(len(responses), func(i, j int) {
		responses[i], responses[j] = responses[j], responses[i]
	})

	return responses
}

// Helper functions

// extractOriginalContent extracts the last user message content
func extractOriginalContent(req *openai.ChatCompletionNewParams) string {
	if req == nil {
		return ""
	}

	// Marshal to JSON and parse to extract messages
	data, err := json.Marshal(req)
	if err != nil {
		return ""
	}

	var reqMap map[string]interface{}
	if err := json.Unmarshal(data, &reqMap); err != nil {
		return ""
	}

	messages, ok := reqMap["messages"].([]interface{})
	if !ok || len(messages) == 0 {
		return ""
	}

	// Find last user message
	for i := len(messages) - 1; i >= 0; i-- {
		msg, ok := messages[i].(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if role == "user" {
			if content, ok := msg["content"].(string); ok {
				return content
			}
		}
	}

	return ""
}

// cloneMessages creates a deep copy of messages
func cloneMessages(req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
	return cloneRequest(req)
}

// replaceLastMessage replaces the last user message with new content
func replaceLastMessage(req *openai.ChatCompletionNewParams, newContent string) *openai.ChatCompletionNewParams {
	if req == nil {
		return nil
	}

	// Marshal to JSON
	data, err := json.Marshal(req)
	if err != nil {
		return req
	}

	var reqMap map[string]interface{}
	if unmarshalErr := json.Unmarshal(data, &reqMap); unmarshalErr != nil {
		return req
	}

	messages, ok := reqMap["messages"].([]interface{})
	if !ok || len(messages) == 0 {
		return req
	}

	// Replace last message
	messages[len(messages)-1] = map[string]string{
		"role":    "user",
		"content": newContent,
	}
	reqMap["messages"] = messages

	// Unmarshal back to ChatCompletionNewParams
	modifiedData, err := json.Marshal(reqMap)
	if err != nil {
		return req
	}

	var result openai.ChatCompletionNewParams
	if err := json.Unmarshal(modifiedData, &result); err != nil {
		return req
	}

	return &result
}

// estimateTokens estimates token count from text (rough heuristic)
func estimateTokens(text string) int {
	// Rough heuristic: ~4 chars per token
	return len(text) / 4
}
