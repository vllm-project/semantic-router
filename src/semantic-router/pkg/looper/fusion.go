package looper

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// FusionLooper implements Fusion-style multi-model deliberation:
// parallel panel responses, judge analysis, then a final synthesized answer.
type FusionLooper struct {
	*BaseLooper
}

func NewFusionLooper(cfg *config.LooperConfig) *FusionLooper {
	return &FusionLooper{BaseLooper: NewBaseLooper(cfg)}
}

type fusionExecutionConfig struct {
	Model                        string
	AnalysisModels               []string
	MaxConcurrent                int
	MaxCompletionTokens          int
	Temperature                  *float64
	IncludeAnalysis              bool
	IncludeIntermediateResponses bool
	OnError                      string
	AnalysisTemplate             string
	SynthesisTemplate            string
	JudgePromptVersion           string
}

type FusionAnalysis struct {
	Consensus       []string `json:"consensus,omitempty"`
	Contradictions  []string `json:"contradictions,omitempty"`
	PartialCoverage []string `json:"partial_coverage,omitempty"`
	UniqueInsights  []string `json:"unique_insights,omitempty"`
	BlindSpots      []string `json:"blind_spots,omitempty"`
	Raw             string   `json:"raw,omitempty"`
	ParseFailed     bool     `json:"parse_failed,omitempty"`
}

type FusionPanelResponse struct {
	Model     string `json:"model"`
	Content   string `json:"content"`
	Reasoning string `json:"reasoning,omitempty"`
}

type FusionFailedModel struct {
	Model string `json:"model"`
	Error string `json:"error"`
}

type FusionTrace struct {
	Analysis       *FusionAnalysis       `json:"analysis,omitempty"`
	Responses      []FusionPanelResponse `json:"responses,omitempty"`
	FailedModels   []FusionFailedModel   `json:"failed_models,omitempty"`
	JudgeModel     string                `json:"judge_model,omitempty"`
	AnalysisModels []string              `json:"analysis_models,omitempty"`
	PromptVersion  string                `json:"prompt_version,omitempty"`
}

type fusionPanelResult struct {
	index int
	model string
	resp  *ModelResponse
	err   error
}

func (l *FusionLooper) Execute(ctx context.Context, req *Request) (*Response, error) {
	l.client.SetDecisionName(req.DecisionName)
	l.client.SetFusionDepth(1)
	defer l.client.SetFusionDepth(0)

	cfg := l.resolveFusionExecutionConfig(req)
	if len(cfg.AnalysisModels) == 0 {
		return nil, fmt.Errorf("fusion analysis_models cannot be empty")
	}
	if cfg.Model == "" {
		cfg.Model = cfg.AnalysisModels[0]
	}
	if err := validateFusionExecutionConfig(cfg); err != nil {
		return nil, err
	}
	if err := l.validateFusionModels(cfg); err != nil {
		return nil, err
	}

	logging.ComponentEvent("looper", "fusion_execution_started", map[string]interface{}{
		"decision":        req.DecisionName,
		"judge_model":     cfg.Model,
		"analysis_models": len(cfg.AnalysisModels),
		"streaming":       req.IsStreaming,
	})

	panelResponses, failedModels, err := l.executeFusionPanel(ctx, req, cfg)
	if err != nil {
		return nil, err
	}

	analysis := l.runFusionAnalysis(ctx, req, cfg, panelResponses)
	finalResp, err := l.runFusionFinal(ctx, req, cfg, panelResponses, analysis)
	if err != nil {
		return nil, err
	}

	trace := buildFusionTrace(cfg, panelResponses, failedModels, analysis)
	modelsUsed := orderedFusionModelsUsed(cfg.AnalysisModels, cfg.Model)
	iterations := len(cfg.AnalysisModels) + 2

	if req.IsStreaming {
		return l.formatFusionStreamingResponse(finalResp, modelsUsed, iterations, cfg, trace)
	}
	return l.formatFusionJSONResponse(finalResp, modelsUsed, iterations, cfg, trace)
}

func (l *FusionLooper) resolveFusionExecutionConfig(req *Request) fusionExecutionConfig {
	cfg := fusionExecutionConfig{
		IncludeAnalysis:              true,
		IncludeIntermediateResponses: true,
	}

	algorithmHasAnalysisModels := false
	if req.Algorithm != nil && req.Algorithm.Fusion != nil {
		algorithmHasAnalysisModels = len(req.Algorithm.Fusion.AnalysisModels) > 0
		mergeFusionAlgorithmConfig(&cfg, req.Algorithm.Fusion)
	}
	if len(req.ModelRefs) > 0 && !algorithmHasAnalysisModels {
		cfg.AnalysisModels = modelRefsToNames(req.ModelRefs)
	}
	if len(cfg.AnalysisModels) == 0 {
		cfg.AnalysisModels = modelRefsToNames(req.ModelRefs)
	}
	if req.Fusion != nil && fusionRequestEnabled(req.Fusion) {
		mergeFusionRequestConfig(&cfg, req.Fusion)
	}
	cfg.OnError = strings.TrimSpace(cfg.OnError)
	if cfg.OnError == "" {
		cfg.OnError = config.FusionOnErrorSkip
	}
	if cfg.JudgePromptVersion == "" {
		cfg.JudgePromptVersion = config.DefaultFusionJudgePromptVersion
	}
	cfg.AnalysisModels = normalizeModelNames(cfg.AnalysisModels)
	if cfg.MaxConcurrent <= 0 || cfg.MaxConcurrent > len(cfg.AnalysisModels) {
		cfg.MaxConcurrent = len(cfg.AnalysisModels)
	}
	return cfg
}

func validateFusionExecutionConfig(cfg fusionExecutionConfig) error {
	switch cfg.OnError {
	case config.FusionOnErrorSkip, config.FusionOnErrorFail:
		return nil
	default:
		return fmt.Errorf("fusion on_error must be %q or %q, got %q", config.FusionOnErrorSkip, config.FusionOnErrorFail, cfg.OnError)
	}
}

func mergeFusionAlgorithmConfig(dst *fusionExecutionConfig, src *config.FusionAlgorithmConfig) {
	if src.Model != "" {
		dst.Model = src.Model
	}
	if len(src.AnalysisModels) > 0 {
		dst.AnalysisModels = append([]string(nil), src.AnalysisModels...)
	}
	if src.MaxConcurrent > 0 {
		dst.MaxConcurrent = src.MaxConcurrent
	}
	if src.MaxCompletionTokens > 0 {
		dst.MaxCompletionTokens = src.MaxCompletionTokens
	}
	if src.Temperature != nil {
		dst.Temperature = src.Temperature
	}
	if src.IncludeAnalysis != nil {
		dst.IncludeAnalysis = *src.IncludeAnalysis
	}
	if src.IncludeIntermediateResponses != nil {
		dst.IncludeIntermediateResponses = *src.IncludeIntermediateResponses
	}
	if src.OnError != "" {
		dst.OnError = src.OnError
	}
	if src.AnalysisTemplate != "" {
		dst.AnalysisTemplate = src.AnalysisTemplate
	}
	if src.SynthesisTemplate != "" {
		dst.SynthesisTemplate = src.SynthesisTemplate
	}
	if src.JudgePromptVersion != "" {
		dst.JudgePromptVersion = src.JudgePromptVersion
	}
}

func mergeFusionRequestConfig(dst *fusionExecutionConfig, src *config.FusionRequestConfig) {
	if src.Model != "" {
		dst.Model = src.Model
	}
	if len(src.AnalysisModels) > 0 {
		dst.AnalysisModels = append([]string(nil), src.AnalysisModels...)
	}
	if src.MaxConcurrent > 0 {
		dst.MaxConcurrent = src.MaxConcurrent
	}
	if src.MaxCompletionTokens > 0 {
		dst.MaxCompletionTokens = src.MaxCompletionTokens
	}
	if src.Temperature != nil {
		dst.Temperature = src.Temperature
	}
	if src.OnError != "" {
		dst.OnError = src.OnError
	}
}

func fusionRequestEnabled(req *config.FusionRequestConfig) bool {
	return req.Enabled == nil || *req.Enabled
}

func modelRefsToNames(modelRefs []config.ModelRef) []string {
	names := make([]string, 0, len(modelRefs))
	for _, ref := range modelRefs {
		if ref.LoRAName != "" {
			names = append(names, ref.LoRAName)
			continue
		}
		names = append(names, ref.Model)
	}
	return names
}

func normalizeModelNames(names []string) []string {
	seen := make(map[string]bool, len(names))
	result := make([]string, 0, len(names))
	for _, name := range names {
		trimmed := strings.TrimSpace(name)
		if trimmed == "" || seen[trimmed] {
			continue
		}
		seen[trimmed] = true
		result = append(result, trimmed)
	}
	return result
}

func (l *FusionLooper) validateFusionModels(cfg fusionExecutionConfig) error {
	for _, model := range append(append([]string{}, cfg.AnalysisModels...), cfg.Model) {
		for _, fusionName := range l.cfg.Fusion.EffectiveModelNames() {
			if model == fusionName {
				return fmt.Errorf("fusion model %q cannot be used as a judge or analysis model", model)
			}
		}
	}
	return nil
}

func (l *FusionLooper) executeFusionPanel(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
) ([]*ModelResponse, []FusionFailedModel, error) {
	results := make(chan fusionPanelResult, len(cfg.AnalysisModels))
	sem := make(chan struct{}, cfg.MaxConcurrent)
	for i, model := range cfg.AnalysisModels {
		go func(index int, modelName string) {
			sem <- struct{}{}
			defer func() { <-sem }()
			resp, err := l.callFusionModel(ctx, req, cfg, modelName, false, index+1)
			results <- fusionPanelResult{index: index, model: modelName, resp: resp, err: err}
		}(i, model)
	}

	ordered := make([]*ModelResponse, len(cfg.AnalysisModels))
	var failed []FusionFailedModel
	for range cfg.AnalysisModels {
		select {
		case result := <-results:
			if result.err != nil {
				failed = append(failed, FusionFailedModel{Model: result.model, Error: result.err.Error()})
				if cfg.OnError == config.FusionOnErrorFail {
					return nil, failed, fmt.Errorf("fusion panel model %q failed: %w", result.model, result.err)
				}
				continue
			}
			ordered[result.index] = result.resp
		case <-ctx.Done():
			return nil, failed, ctx.Err()
		}
	}

	responses := make([]*ModelResponse, 0, len(ordered))
	for _, resp := range ordered {
		if resp != nil {
			responses = append(responses, resp)
		}
	}
	if len(responses) == 0 {
		return nil, failed, fmt.Errorf("fusion panel failed: all %d analysis models failed", len(cfg.AnalysisModels))
	}
	return responses, failed, nil
}

func (l *FusionLooper) callFusionModel(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	modelName string,
	streaming bool,
	iteration int,
) (*ModelResponse, error) {
	callReq := cloneRequest(req.OriginalRequest)
	if cfg.Temperature != nil {
		callReq.Temperature = openai.Float(*cfg.Temperature)
	}
	if cfg.MaxCompletionTokens > 0 {
		callReq.MaxCompletionTokens = openai.Int(int64(cfg.MaxCompletionTokens))
	}
	return l.client.CallModel(ctx, callReq, modelName, streaming, iteration, nil, accessKeyForModel(req, modelName))
}

func accessKeyForModel(req *Request, modelName string) string {
	if req == nil || req.ModelParams == nil {
		return ""
	}
	if params, ok := req.ModelParams[modelName]; ok {
		return params.AccessKey
	}
	for _, params := range req.ModelParams {
		for _, extID := range params.ExternalModelIDs {
			if extID == modelName {
				return params.AccessKey
			}
		}
	}
	return ""
}

func (l *FusionLooper) runFusionAnalysis(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	panelResponses []*ModelResponse,
) *FusionAnalysis {
	prompt := buildFusionAnalysisPrompt(cfg, extractOriginalContent(req.OriginalRequest), panelResponses)
	analysisReq := replaceLastMessage(req.OriginalRequest, prompt)
	resp, err := l.callFusionModel(ctx, &Request{OriginalRequest: analysisReq, ModelParams: req.ModelParams}, cfg, cfg.Model, false, len(panelResponses)+1)
	if err != nil {
		logging.ComponentWarnEvent("looper", "fusion_analysis_failed", map[string]interface{}{
			"judge_model": cfg.Model,
			"error":       err.Error(),
		})
		return nil
	}
	analysis, parseErr := parseFusionAnalysis(resp.Content)
	if parseErr != nil {
		logging.ComponentWarnEvent("looper", "fusion_analysis_parse_failed", map[string]interface{}{
			"judge_model": cfg.Model,
			"error":       parseErr.Error(),
		})
		return &FusionAnalysis{Raw: resp.Content, ParseFailed: true}
	}
	return analysis
}

func (l *FusionLooper) runFusionFinal(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	panelResponses []*ModelResponse,
	analysis *FusionAnalysis,
) (*ModelResponse, error) {
	prompt := buildFusionFinalPrompt(cfg, extractOriginalContent(req.OriginalRequest), panelResponses, analysis)
	finalReq := replaceLastMessage(req.OriginalRequest, prompt)
	resp, err := l.callFusionModel(ctx, &Request{OriginalRequest: finalReq, ModelParams: req.ModelParams}, cfg, cfg.Model, false, len(panelResponses)+2)
	if err != nil {
		return nil, fmt.Errorf("fusion final synthesis failed for judge model %q: %w", cfg.Model, err)
	}
	return resp, nil
}

func buildFusionAnalysisPrompt(cfg fusionExecutionConfig, original string, responses []*ModelResponse) string {
	if cfg.AnalysisTemplate != "" {
		return renderFusionPrompt(cfg.AnalysisTemplate, original, responses, nil)
	}
	return fmt.Sprintf(`You are the Fusion analysis judge. Compare the panel responses and return only valid JSON with these keys: consensus, contradictions, partial_coverage, unique_insights, blind_spots. Return compact JSON only: no markdown, no code fences, no prose before or after the JSON. Each value must be an array with at most two concise strings.

Original prompt:
%s

Panel responses:
%s`, original, formatPanelResponses(responses))
}

func buildFusionFinalPrompt(
	cfg fusionExecutionConfig,
	original string,
	responses []*ModelResponse,
	analysis *FusionAnalysis,
) string {
	if cfg.SynthesisTemplate != "" {
		return renderFusionPrompt(cfg.SynthesisTemplate, original, responses, analysis)
	}
	analysisBlock := "No structured analysis is available. Synthesize directly from the panel responses."
	if analysis != nil && !analysis.ParseFailed {
		if data, err := json.MarshalIndent(analysis, "", "  "); err == nil {
			analysisBlock = string(data)
		}
	}
	return fmt.Sprintf(`You are the Fusion calling model. Produce the final answer for the user using the panel responses and structured analysis. Resolve contradictions explicitly and do not mention internal model names unless the user asks.

Original prompt:
%s

Structured analysis:
%s

Panel responses:
%s

Final answer:`, original, analysisBlock, formatPanelResponses(responses))
}

func renderFusionPrompt(template string, original string, responses []*ModelResponse, analysis *FusionAnalysis) string {
	replacer := strings.NewReplacer(
		"{{original}}", original,
		"{{responses}}", formatPanelResponses(responses),
		"{{analysis}}", formatFusionAnalysisForPrompt(analysis),
	)
	return replacer.Replace(template)
}

func formatPanelResponses(responses []*ModelResponse) string {
	var b strings.Builder
	for i, resp := range responses {
		if resp == nil {
			continue
		}
		fmt.Fprintf(&b, "Response %d (%s):\n%s\n\n", i+1, resp.Model, resp.Content)
		if resp.ReasoningContent != "" {
			fmt.Fprintf(&b, "Reasoning %d (%s):\n%s\n\n", i+1, resp.Model, resp.ReasoningContent)
		}
	}
	return strings.TrimSpace(b.String())
}

func formatFusionAnalysisForPrompt(analysis *FusionAnalysis) string {
	if analysis == nil {
		return ""
	}
	data, err := json.MarshalIndent(analysis, "", "  ")
	if err != nil {
		return analysis.Raw
	}
	return string(data)
}

func parseFusionAnalysis(content string) (*FusionAnalysis, error) {
	candidate := strings.TrimSpace(content)
	if strings.HasPrefix(candidate, "```") {
		candidate = strings.TrimPrefix(candidate, "```json")
		candidate = strings.TrimPrefix(candidate, "```")
		candidate = strings.TrimSuffix(candidate, "```")
		candidate = strings.TrimSpace(candidate)
	}
	if !json.Valid([]byte(candidate)) {
		start := strings.Index(candidate, "{")
		end := strings.LastIndex(candidate, "}")
		if start >= 0 && end > start {
			candidate = candidate[start : end+1]
		}
	}
	var analysis FusionAnalysis
	if err := json.Unmarshal([]byte(candidate), &analysis); err != nil {
		return nil, err
	}
	return &analysis, nil
}

func buildFusionTrace(
	cfg fusionExecutionConfig,
	panelResponses []*ModelResponse,
	failedModels []FusionFailedModel,
	analysis *FusionAnalysis,
) *FusionTrace {
	trace := &FusionTrace{
		JudgeModel:     cfg.Model,
		AnalysisModels: append([]string(nil), cfg.AnalysisModels...),
		FailedModels:   failedModels,
		PromptVersion:  cfg.JudgePromptVersion,
	}
	if cfg.IncludeAnalysis {
		trace.Analysis = analysis
	}
	if cfg.IncludeIntermediateResponses {
		trace.Responses = make([]FusionPanelResponse, 0, len(panelResponses))
		for _, resp := range panelResponses {
			trace.Responses = append(trace.Responses, FusionPanelResponse{
				Model:     resp.Model,
				Content:   resp.Content,
				Reasoning: resp.ReasoningContent,
			})
		}
	}
	return trace
}

func orderedFusionModelsUsed(analysisModels []string, judge string) []string {
	seen := map[string]bool{}
	models := make([]string, 0, len(analysisModels)+1)
	add := func(model string) {
		if model == "" || seen[model] {
			return
		}
		seen[model] = true
		models = append(models, model)
	}
	for _, model := range analysisModels {
		add(model)
	}
	add(judge)
	return models
}

func (l *FusionLooper) formatFusionJSONResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	cfg fusionExecutionConfig,
	trace *FusionTrace,
) (*Response, error) {
	completion := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-fusion-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   finalResp.Model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": finalResp.Content,
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
	if cfg.IncludeAnalysis || cfg.IncludeIntermediateResponses || len(trace.FailedModels) > 0 {
		completion["fusion"] = trace
	}
	body, err := json.Marshal(completion)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal fusion response: %w", err)
	}
	return &Response{
		Body:                  body,
		ContentType:           "application/json",
		Model:                 finalResp.Model,
		ModelsUsed:            modelsUsed,
		Iterations:            iterations,
		AlgorithmType:         "fusion",
		IntermediateResponses: trace,
	}, nil
}

func (l *FusionLooper) formatFusionStreamingResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	cfg fusionExecutionConfig,
	trace *FusionTrace,
) (*Response, error) {
	timestamp := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-fusion-%d", timestamp)
	body := buildFusionStreamingSSE(id, timestamp, finalResp.Model, finalResp.Content, cfg, trace)
	resp := streamingLooperResponse(body, finalResp.Model, modelsUsed, iterations, "fusion")
	resp.IntermediateResponses = trace
	return resp, nil
}

func buildFusionStreamingSSE(
	id string,
	created int64,
	model string,
	content string,
	cfg fusionExecutionConfig,
	trace *FusionTrace,
) []byte {
	var body []byte
	roleChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{"role": "assistant"},
		"finish_reason": nil,
	}
	var extra map[string]interface{}
	if cfg.IncludeAnalysis || cfg.IncludeIntermediateResponses || len(trace.FailedModels) > 0 {
		extra = map[string]interface{}{"fusion": trace}
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, roleChoice, extra))
	for _, chunk := range splitIntoChunks(content, 50) {
		contentChoice := map[string]interface{}{
			"index":         0,
			"delta":         map[string]interface{}{"content": chunk},
			"finish_reason": nil,
		}
		body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, contentChoice, nil))
	}
	finalChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{},
		"finish_reason": "stop",
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, finalChoice, nil))
	return appendSSEDone(body)
}
