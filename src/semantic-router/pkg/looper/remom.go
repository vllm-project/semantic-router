package looper

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"sort"
	"strings"
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

const (
	remomDistributionWeighted   = "weighted"
	remomDistributionEqual      = "equal"
	remomDistributionRoundRobin = "round_robin"
	remomDistributionFirstOnly  = "first_only"
)

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
		ModelDistribution:            remomDistributionWeighted,
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

// remomParallelResult is one completed parallel model call in a ReMoM round.
type remomParallelResult struct {
	resp  *ModelResponse
	err   error
	index int
}

func remomParallelMaxConcurrent(numCalls, maxFromCfg int) int {
	if maxFromCfg > 0 && maxFromCfg < numCalls {
		return maxFromCfg
	}
	return numCalls
}

func remomParallelMinSuccessful(numCalls, minFromCfg int) int {
	if minFromCfg > 0 && minFromCfg < numCalls {
		return minFromCfg
	}
	return numCalls
}

// remomRunOneParallelCall performs a single gated CallModel for ReMoM parallel rounds.
func (l *ReMoMLooper) remomRunOneParallelCall(
	ctx context.Context,
	idx, numCalls int,
	mc ModelCall,
	req *Request,
	messages *openai.ChatCompletionNewParams,
	cfg *config.ReMoMAlgorithmConfig,
	streaming bool,
	sem chan struct{},
) remomParallelResult {
	modelName := mc.Model
	if mc.LoRAName != "" {
		modelName = mc.LoRAName
	}

	logging.Infof("[ReMoM] Goroutine %d/%d started for model %s", idx+1, numCalls, modelName)

	select {
	case sem <- struct{}{}:
	case <-ctx.Done():
		return remomParallelResult{err: ctx.Err(), index: idx}
	}
	defer func() { <-sem }()

	msgCopy := cloneRequest(messages)
	if cfg.Temperature > 0 {
		msgCopy.Temperature = openai.Float(cfg.Temperature)
	}

	startTime := time.Now()
	resp, err := l.client.CallModel(
		ctx,
		msgCopy,
		modelName,
		streaming,
		idx+1,
		nil,
		accessKeyForModel(req, modelName),
	)
	elapsed := time.Since(startTime)

	if err != nil {
		logging.Warnf("[ReMoM] Goroutine %d/%d failed for model %s after %v: %v", idx+1, numCalls, modelName, elapsed, err)
	} else {
		logging.Infof("[ReMoM] Goroutine %d/%d completed for model %s in %v", idx+1, numCalls, modelName, elapsed)
	}

	return remomParallelResult{resp: resp, err: err, index: idx}
}

func collectRemomParallelResults(
	ctx context.Context,
	numCalls int,
	minSuccessful int,
	results <-chan remomParallelResult,
	cfg *config.ReMoMAlgorithmConfig,
) ([]*ModelResponse, error) {
	collector := newRemomResultCollector(numCalls, minSuccessful, cfg)

	for completed := 0; completed < numCalls; completed++ {
		select {
		case res := <-results:
			responses, err, done := collector.handleResult(res)
			if done {
				return responses, err
			}
		case <-ctx.Done():
			return collector.handleContextDone(ctx.Err())
		}
	}
	return collector.finalize()
}

type remomResultCollector struct {
	numCalls      int
	minSuccessful int
	onError       string
	responses     []*ModelResponse
	errs          []error
}

func newRemomResultCollector(numCalls int, minSuccessful int, cfg *config.ReMoMAlgorithmConfig) *remomResultCollector {
	return &remomResultCollector{
		numCalls:      numCalls,
		minSuccessful: minSuccessful,
		onError:       cfg.OnError,
	}
}

func (c *remomResultCollector) handleResult(res remomParallelResult) ([]*ModelResponse, error, bool) {
	if res.err != nil {
		c.errs = append(c.errs, res.err)
		if c.onError == config.ReMoMOnErrorFail {
			return nil, fmt.Errorf("model call %d failed: %w", res.index, res.err), true
		}
		return nil, nil, false
	}
	c.responses = append(c.responses, res.resp)
	if len(c.responses) < c.minSuccessful {
		return nil, nil, false
	}
	if len(c.responses) < c.numCalls {
		logging.Infof("[ReMoM] Quorum reached with %d/%d successful responses", len(c.responses), c.numCalls)
	}
	return c.responses, nil, true
}

func (c *remomResultCollector) handleContextDone(err error) ([]*ModelResponse, error) {
	if len(c.responses) > 0 && c.onError != config.ReMoMOnErrorFail {
		return c.responses, err
	}
	return nil, err
}

func (c *remomResultCollector) finalize() ([]*ModelResponse, error) {
	if len(c.responses) == 0 {
		return nil, fmt.Errorf("all %d model calls failed: %v", c.numCalls, c.errs)
	}
	logging.Infof("[ReMoM] Collected %d/%d successful responses", len(c.responses), c.numCalls)
	return c.responses, nil
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

type remomScheduleResult struct {
	allRoundResponses []RoundResponse
	modelsUsed        map[string]bool
	totalIterations   int
	usage             TokenUsage
}

// Execute implements the Looper interface for ReMoM
func (l *ReMoMLooper) Execute(ctx context.Context, req *Request) (*Response, error) {
	l.client.SetDecisionName(req.DecisionName)

	var cfg *config.ReMoMAlgorithmConfig
	if req.Algorithm != nil && req.Algorithm.ReMoM != nil {
		cfg = req.Algorithm.ReMoM
	} else {
		cfg = getDefaultReMoMConfig()
	}

	// Note: ReMoM internally uses non-streaming calls even if client expects streaming
	// because we need complete responses for synthesis across rounds.
	if len(req.ModelRefs) == 0 {
		return nil, fmt.Errorf("no models configured")
	}
	if len(cfg.BreadthSchedule) == 0 {
		return nil, fmt.Errorf("breadth_schedule cannot be empty")
	}

	schedule := append([]int{}, cfg.BreadthSchedule...)
	schedule = append(schedule, 1)
	originalContent := extractOriginalContent(req.OriginalRequest)
	originalWithOutputContract := requestTextWithOutputContract(originalContent, req.OriginalRequest, req.OutputContract)

	result, err := l.runReMoMSchedule(ctx, req, cfg, schedule, originalWithOutputContract)
	if err != nil {
		return nil, err
	}

	finalResponse := result.allRoundResponses[len(result.allRoundResponses)-1].Responses[0]
	finalModelResp := &ModelResponse{
		Content:          finalResponse.Content,
		ReasoningContent: finalResponse.Reasoning,
		Model:            finalResponse.Model,
	}
	candidateResponses := remomRoundResponsesToModelResponses(result.allRoundResponses)
	applyJSONActionOutputContract(
		req.OutputContractSpec,
		finalModelResp,
		candidateResponses,
	)
	applyReferenceSelectionOutputContract(
		req.OutputContractSpec,
		finalModelResp,
		candidateResponses,
	)
	finalResponse.Content = finalModelResp.Content
	modelsUsedSlice := make([]string, 0, len(result.modelsUsed))
	for model := range result.modelsUsed {
		modelsUsedSlice = append(modelsUsedSlice, model)
	}

	if req.IsStreaming {
		return l.formatReMoMStreamingResponse(finalResponse, result.allRoundResponses, modelsUsedSlice, result.totalIterations, result.usage, cfg)
	}
	return l.formatReMoMJSONResponse(finalResponse, result.allRoundResponses, modelsUsedSlice, result.totalIterations, result.usage, cfg)
}

func (l *ReMoMLooper) runReMoMSchedule(
	ctx context.Context,
	req *Request,
	cfg *config.ReMoMAlgorithmConfig,
	schedule []int,
	originalWithOutputContract string,
) (*remomScheduleResult, error) {
	var allRoundResponses []RoundResponse
	modelsUsed := make(map[string]bool)
	totalIterations := 0
	var usage TokenUsage
	currentMessages := cloneMessages(req.OriginalRequest)
	if requestsJSONAction(req.OutputContractSpec) {
		currentMessages = replaceLastMessage(currentMessages, originalWithOutputContract)
	}

	for roundIdx, numCalls := range schedule {
		logging.Infof("[ReMoM] Round %d/%d: %d parallel calls", roundIdx+1, len(schedule), numCalls)

		updatedMessages, err := l.prepareReMoMRoundMessages(cfg, originalWithOutputContract, roundIdx, allRoundResponses, currentMessages)
		if err != nil {
			return nil, err
		}
		currentMessages = updatedMessages

		isFinalRound := roundIdx == len(schedule)-1
		roundResp, responses, err := l.executeReMoMRound(ctx, req, cfg, roundIdx, numCalls, currentMessages, isFinalRound)
		if err != nil {
			if canFallbackToPreviousReMoMRound(cfg, allRoundResponses) {
				logging.Warnf("[ReMoM] Round %d failed; using previous round responses as fallback: %v", roundIdx+1, err)
				break
			}
			return nil, err
		}

		allRoundResponses = append(allRoundResponses, roundResp)
		trackReMoMModelsUsed(modelsUsed, responses)
		usage = usage.Add(responses...)
		totalIterations += len(responses)
		logging.Infof("[ReMoM] Round %d completed: %d responses", roundIdx+1, len(responses))
	}

	return &remomScheduleResult{
		allRoundResponses: allRoundResponses,
		modelsUsed:        modelsUsed,
		totalIterations:   totalIterations,
		usage:             usage,
	}, nil
}

func canFallbackToPreviousReMoMRound(cfg *config.ReMoMAlgorithmConfig, allRoundResponses []RoundResponse) bool {
	if cfg.OnError == config.ReMoMOnErrorFail || len(allRoundResponses) == 0 {
		return false
	}
	lastRound := allRoundResponses[len(allRoundResponses)-1]
	return len(lastRound.Responses) > 0
}

func (l *ReMoMLooper) prepareReMoMRoundMessages(
	cfg *config.ReMoMAlgorithmConfig,
	originalContent string,
	roundIdx int,
	allRoundResponses []RoundResponse,
	currentMessages *openai.ChatCompletionNewParams,
) (*openai.ChatCompletionNewParams, error) {
	if roundIdx == 0 {
		return currentMessages, nil
	}

	prevRound := allRoundResponses[roundIdx-1]
	prevResponses := intermediateResponsesToModelResponses(prevRound.Responses)
	synthesisPrompt, err := l.buildSynthesisPrompt(cfg, originalContent, prevResponses)
	if err != nil {
		return nil, fmt.Errorf("failed to build synthesis prompt for round %d: %w", roundIdx+1, err)
	}
	return replaceLastMessage(currentMessages, synthesisPrompt), nil
}

func intermediateResponsesToModelResponses(responses []IntermediateResp) []*ModelResponse {
	prevResponses := make([]*ModelResponse, 0, len(responses))
	for _, ir := range responses {
		prevResponses = append(prevResponses, &ModelResponse{
			Content:          ir.Content,
			ReasoningContent: ir.Reasoning,
			Model:            ir.Model,
		})
	}
	return prevResponses
}

func remomRoundResponsesToModelResponses(rounds []RoundResponse) []*ModelResponse {
	var responses []*ModelResponse
	for _, round := range rounds {
		responses = append(responses, intermediateResponsesToModelResponses(round.Responses)...)
	}
	return responses
}

func (l *ReMoMLooper) executeReMoMRound(
	ctx context.Context,
	req *Request,
	cfg *config.ReMoMAlgorithmConfig,
	roundIdx, numCalls int,
	currentMessages *openai.ChatCompletionNewParams,
	isFinalRound bool,
) (RoundResponse, []*ModelResponse, error) {
	modelCalls := l.distributeCallsToModels(cfg, numCalls, req.ModelRefs)
	if isFinalRound {
		modelCalls = remomFinalRoundModelCalls(cfg, modelCalls, req.ModelRefs)
	}
	responses, err := l.executeParallelCalls(ctx, req, cfg, modelCalls, currentMessages, req.IsStreaming)
	if err != nil {
		if cfg.OnError == "fail" {
			return RoundResponse{}, nil, fmt.Errorf("round %d failed: %w", roundIdx+1, err)
		}
		logging.Warnf("[ReMoM] Round %d had errors but continuing (on_error=skip)", roundIdx+1)
	}
	if len(responses) == 0 {
		return RoundResponse{}, nil, fmt.Errorf("round %d: all model calls failed", roundIdx+1)
	}

	responses = l.sortAndShuffle(cfg, responses)
	return l.buildReMoMRoundResponse(cfg, roundIdx+1, numCalls, responses), responses, nil
}

func remomFinalRoundModelCalls(cfg *config.ReMoMAlgorithmConfig, defaultCalls []ModelCall, modelRefs []config.ModelRef) []ModelCall {
	if cfg == nil || strings.TrimSpace(cfg.SynthesisModel) == "" {
		return defaultCalls
	}
	synthesisModel := strings.TrimSpace(cfg.SynthesisModel)
	for _, ref := range modelRefs {
		if ref.Model == synthesisModel {
			return []ModelCall{{
				Model:    ref.Model,
				LoRAName: ref.LoRAName,
			}}
		}
	}
	return []ModelCall{{Model: synthesisModel}}
}

func (l *ReMoMLooper) buildReMoMRoundResponse(
	cfg *config.ReMoMAlgorithmConfig,
	round, breadth int,
	responses []*ModelResponse,
) RoundResponse {
	roundResp := RoundResponse{Round: round, Breadth: breadth}
	maxResponses := len(responses)
	if cfg.MaxResponsesPerRound > 0 && cfg.MaxResponsesPerRound < maxResponses {
		maxResponses = cfg.MaxResponsesPerRound
	}
	for i := 0; i < maxResponses; i++ {
		resp := responses[i]
		roundResp.Responses = append(roundResp.Responses, IntermediateResp{
			Model:            resp.Model,
			Content:          resp.Content,
			Reasoning:        resp.ReasoningContent,
			CompactedContent: l.compactResponse(cfg, resp.Content),
			TokenCount:       estimateTokens(resp.Content),
		})
	}
	return roundResp
}

func trackReMoMModelsUsed(modelsUsed map[string]bool, responses []*ModelResponse) {
	for _, resp := range responses {
		modelsUsed[resp.Model] = true
	}
}

// formatReMoMJSONResponse creates a non-streaming JSON response
func (l *ReMoMLooper) formatReMoMJSONResponse(
	finalResponse IntermediateResp,
	allRoundResponses []RoundResponse,
	modelsUsed []string,
	iterations int,
	usage TokenUsage,
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
		"usage": usage.Map(),
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
		Usage:                 usage,
	}, nil
}

// formatReMoMStreamingResponse creates an SSE streaming response
func (l *ReMoMLooper) formatReMoMStreamingResponse(
	finalResponse IntermediateResp,
	allRoundResponses []RoundResponse,
	modelsUsed []string,
	iterations int,
	usage TokenUsage,
	cfg *config.ReMoMAlgorithmConfig,
) (*Response, error) {
	timestamp := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-remom-%d", timestamp)
	sseBody := buildReMoMStreamingSSE(id, timestamp, finalResponse, allRoundResponses, cfg.IncludeIntermediateResponses)

	resp := streamingLooperResponse(sseBody, finalResponse.Model, modelsUsed, iterations, "remom")
	resp.IntermediateResponses = allRoundResponses
	resp.Usage = usage
	return resp, nil
}

// executeParallelCalls executes model calls in parallel with concurrency control.
func (l *ReMoMLooper) executeParallelCalls(
	ctx context.Context,
	req *Request,
	cfg *config.ReMoMAlgorithmConfig,
	modelCalls []ModelCall,
	messages *openai.ChatCompletionNewParams,
	streaming bool,
) ([]*ModelResponse, error) {
	numCalls := len(modelCalls)
	maxConcurrent := remomParallelMaxConcurrent(numCalls, cfg.MaxConcurrent)
	minSuccessful := remomParallelMinSuccessful(numCalls, cfg.MinSuccessfulResponses)
	roundCtx := ctx
	cancel := func() {}
	if cfg.RoundTimeoutSeconds > 0 {
		roundCtx, cancel = context.WithTimeout(ctx, time.Duration(cfg.RoundTimeoutSeconds)*time.Second)
	}
	defer cancel()

	sem := make(chan struct{}, maxConcurrent)
	results := make(chan remomParallelResult, numCalls)

	for i, call := range modelCalls {
		go func(idx int, mc ModelCall) {
			results <- l.remomRunOneParallelCall(roundCtx, idx, numCalls, mc, req, messages, cfg, streaming, sem)
		}(i, call)
	}

	return collectRemomParallelResults(roundCtx, numCalls, minSuccessful, results, cfg)
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

	return appendOutputContractForPrompt(buf.String(), embeddedOutputContract(originalContent)), nil
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
