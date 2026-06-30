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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
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

	GroundingEnabled                 bool
	GroundingReference               string
	GroundingPolicy                  string
	GroundingMinScore                float64
	GroundingMinKeep                 int
	GroundingNLIContradictionPenalty float64
	GroundingOnError                 string
	GroundingEarlyExitEnabled        bool
	GroundingEarlyExitMinConsistency float64

	EscalationEnabled   bool
	EscalationHardRules []string
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
	Grounding      *FusionGroundingTrace `json:"grounding,omitempty"`
}

type fusionPanelResult struct {
	index int
	model string
	resp  *ModelResponse
	err   error
}

func (l *FusionLooper) Execute(ctx context.Context, req *Request) (resp *Response, err error) {
	l.client.SetDecisionName(req.DecisionName)
	l.client.SetFusionDepth(1)
	defer l.client.SetFusionDepth(0)

	cfg, err := l.prepareFusionExecution(req)
	if err != nil {
		return nil, err
	}

	logging.ComponentEvent("looper", "fusion_execution_started", map[string]interface{}{
		"decision":        req.DecisionName,
		"judge_model":     cfg.Model,
		"analysis_models": len(cfg.AnalysisModels),
		"streaming":       req.IsStreaming,
	})

	// Per-request metrics. Timed from here so config/validation errors above (not
	// real traffic) are excluded; the named return err lets the defer classify
	// the final status.
	defer recordFusionOutcome(req.DecisionName, time.Now(), &err)

	// Adaptive escalation: when enabled and the request did not match a configured
	// hard-complexity rule, the query is easy — answer with a single judge-model
	// call instead of paying the N+2 panel cost. Eval (CachedPanel) always runs the
	// full panel so arms stay comparable.
	if shouldEscalateToSingleModel(cfg, req) {
		metrics.RecordFusionEscalationBypass(req.DecisionName)
		logging.ComponentEvent("looper", "fusion_escalation_bypass", map[string]interface{}{
			"decision":           req.DecisionName,
			"judge_model":        cfg.Model,
			"matched_complexity": req.MatchedComplexity,
		})
		return l.runFusionSingleModel(ctx, req, cfg)
	}

	panelStart := time.Now()
	panelResponses, failedModels, err := l.executeFusionPanel(ctx, req, cfg)
	metrics.RecordFusionStageDuration("panel", time.Since(panelStart).Seconds())
	if err != nil {
		return nil, err
	}
	recordFusionPanelMetrics(panelResponses, failedModels)

	// Grounding (optional) ranks/filters the panel before the judge. It makes no
	// model calls, so usage is summed from the full panel (the real cost paid).
	groundingStart := time.Now()
	groundedPanel, groundingScores, groundingMode, err := l.applyGrounding(req, cfg, panelResponses)
	metrics.RecordFusionStageDuration("grounding", time.Since(groundingStart).Seconds())
	if err != nil {
		return nil, err
	}
	recordFusionGroundingMetrics(cfg, groundingMode, groundingScores)

	// Panel-agreement early-exit: when the panel is unanimous, skip the separate
	// analysis judge call and synthesize directly (saves one judge call). Only
	// fires in panel mode with no dissenter, so it never deletes a minority view.
	analysis, analysisResp, earlyExit := l.runFusionAnalysisStage(ctx, req, cfg, groundedPanel, groundingScores, groundingMode)

	synthesisStart := time.Now()
	finalResp, err := l.runFusionFinal(ctx, req, cfg, groundedPanel, analysis, groundingScores)
	metrics.RecordFusionStageDuration("synthesis", time.Since(synthesisStart).Seconds())
	if err != nil {
		return nil, err
	}
	usage := SumUsage(panelResponses...).Add(analysisResp, finalResp)
	trace := buildFusionTrace(cfg, groundedPanel, failedModels, analysis, groundingMode, groundingScores)
	return l.finalizeFusionResponse(req, cfg, finalResp, trace, usage, earlyExit)
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
	applyGroundingDefaults(&cfg)
	return cfg
}

// applyGroundingDefaults fills in grounding defaults when grounding is enabled.
func applyGroundingDefaults(cfg *fusionExecutionConfig) {
	if !cfg.GroundingEnabled {
		return
	}
	if cfg.GroundingReference == "" {
		cfg.GroundingReference = config.FusionGroundingReferenceHybrid
	}
	if cfg.GroundingPolicy == "" {
		// Default to soft down-weighting: hard-dropping the least mutually
		// consistent response regresses quality on contested factual items
		// (see bench/grounded_fusion/FINDINGS.md). Set policy=filter for the
		// prior drop behavior.
		cfg.GroundingPolicy = config.FusionGroundingPolicyWeight
	}
	if cfg.GroundingMinKeep <= 0 {
		cfg.GroundingMinKeep = 1
	}
	if cfg.GroundingNLIContradictionPenalty <= 0 {
		cfg.GroundingNLIContradictionPenalty = 1.0
	}
	if strings.TrimSpace(cfg.GroundingOnError) == "" {
		cfg.GroundingOnError = cfg.OnError
	}
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
	mergeFusionPromptConfig(dst, src)
	mergeFusionGroundingConfig(dst, src.Grounding)
	mergeFusionEscalationConfig(dst, src.Escalation)
}

func mergeFusionGroundingConfig(dst *fusionExecutionConfig, src *config.FusionGroundingConfig) {
	if src == nil {
		return
	}
	dst.GroundingEnabled = src.Enabled
	dst.GroundingReference = src.Reference
	dst.GroundingPolicy = src.Policy
	dst.GroundingMinScore = src.MinScore
	dst.GroundingMinKeep = src.MinKeep
	dst.GroundingNLIContradictionPenalty = src.NLIContradictionPenalty
	dst.GroundingOnError = src.OnError
	dst.GroundingEarlyExitEnabled = src.EarlyExitEnabled
	dst.GroundingEarlyExitMinConsistency = src.EarlyExitMinConsistency
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
	// Paired multi-arm evaluation supplies the panel verbatim so every arm
	// synthesizes from a byte-identical panel (see bench/grounded_fusion). Skip
	// the live model calls and feed the cached panel straight into grounding +
	// synthesis, which are source-agnostic over []*ModelResponse.
	if len(req.CachedPanel) > 0 {
		return req.CachedPanel, nil, nil
	}

	results := make(chan fusionPanelResult, len(cfg.AnalysisModels))
	sem := make(chan struct{}, cfg.MaxConcurrent)
	for i, model := range cfg.AnalysisModels {
		go func(index int, modelName string) {
			sem <- struct{}{}
			defer func() { <-sem }()
			resp, err := l.callFusionModel(ctx, req, cfg, modelName, false, false, index+1)
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
	allowTools bool,
	streaming bool,
	iteration int,
) (*ModelResponse, error) {
	callReq := cloneRequest(req.OriginalRequest)
	if !allowTools {
		callReq = stripFusionToolUse(callReq)
	}
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
	groundingScores []groundingScore,
) (*FusionAnalysis, *ModelResponse) {
	prompt := buildFusionAnalysisPrompt(cfg, extractOriginalContent(req.OriginalRequest), panelResponses)
	if notes := formatGroundingNotes(groundingScores); notes != "" {
		prompt = prompt + "\n\n" + notes
	}
	analysisReq := appendFusionStageMessage(req.OriginalRequest, prompt)
	resp, err := l.callFusionModel(ctx, &Request{OriginalRequest: analysisReq, ModelParams: req.ModelParams}, cfg, cfg.Model, false, false, len(panelResponses)+1)
	if err != nil {
		logging.ComponentWarnEvent("looper", "fusion_analysis_failed", map[string]interface{}{
			"judge_model": cfg.Model,
			"error":       err.Error(),
		})
		return nil, nil
	}
	analysis, parseErr := parseFusionAnalysis(resp.Content)
	if parseErr != nil {
		logging.ComponentWarnEvent("looper", "fusion_analysis_parse_failed", map[string]interface{}{
			"judge_model": cfg.Model,
			"error":       parseErr.Error(),
		})
		return &FusionAnalysis{Raw: resp.Content, ParseFailed: true}, resp
	}
	return analysis, resp
}

func (l *FusionLooper) runFusionFinal(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	panelResponses []*ModelResponse,
	analysis *FusionAnalysis,
	groundingScores []groundingScore,
) (*ModelResponse, error) {
	prompt := buildFusionFinalPrompt(cfg, extractOriginalContent(req.OriginalRequest), panelResponses, analysis)
	// Under weight/annotate policies the panel was not pruned, so the judge needs
	// the per-response groundedness signal at synthesis time to soft-weight.
	if notes := groundingSynthesisNotes(groundingScores, cfg.GroundingPolicy); notes != "" {
		prompt = prompt + "\n\n" + notes
	}
	finalReq := appendFusionStageMessage(req.OriginalRequest, prompt)
	resp, err := l.callFusionModel(ctx, &Request{OriginalRequest: finalReq, ModelParams: req.ModelParams}, cfg, cfg.Model, true, false, len(panelResponses)+2)
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
	groundingMode string,
	groundingScores []groundingScore,
) *FusionTrace {
	trace := &FusionTrace{
		JudgeModel:     cfg.Model,
		AnalysisModels: append([]string(nil), cfg.AnalysisModels...),
		FailedModels:   failedModels,
		PromptVersion:  cfg.JudgePromptVersion,
	}
	if len(groundingScores) > 0 {
		trace.Grounding = &FusionGroundingTrace{
			ReferenceMode: groundingMode,
			Policy:        cfg.GroundingPolicy,
			Scores:        groundingScores,
		}
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
	usage TokenUsage,
) (*Response, error) {
	if finalResp.HasToolCalls {
		return l.formatFusionToolCallJSONResponse(finalResp, modelsUsed, iterations, cfg, trace, usage)
	}

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
		"usage": usage.Map(),
	}
	if cfg.IncludeAnalysis || cfg.IncludeIntermediateResponses || len(trace.FailedModels) > 0 || trace.Grounding != nil {
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
		Usage:                 usage,
	}, nil
}

func (l *FusionLooper) formatFusionToolCallJSONResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	cfg fusionExecutionConfig,
	trace *FusionTrace,
	usage TokenUsage,
) (*Response, error) {
	var completion map[string]interface{}
	if err := json.Unmarshal(finalResp.Raw, &completion); err != nil {
		return nil, fmt.Errorf("failed to parse fusion tool-call response: %w", err)
	}
	completion["id"] = fmt.Sprintf("chatcmpl-fusion-%d", time.Now().UnixNano())
	completion["model"] = finalResp.Model
	completion["usage"] = usage.Map()
	if cfg.IncludeAnalysis || cfg.IncludeIntermediateResponses || len(trace.FailedModels) > 0 || trace.Grounding != nil {
		completion["fusion"] = trace
	}
	body, err := json.Marshal(completion)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal fusion tool-call response: %w", err)
	}
	return &Response{
		Body:                  body,
		ContentType:           "application/json",
		Model:                 finalResp.Model,
		ModelsUsed:            modelsUsed,
		Iterations:            iterations,
		AlgorithmType:         "fusion",
		IntermediateResponses: trace,
		Usage:                 usage,
	}, nil
}

func (l *FusionLooper) formatFusionStreamingResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	cfg fusionExecutionConfig,
	trace *FusionTrace,
	usage TokenUsage,
) (*Response, error) {
	timestamp := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-fusion-%d", timestamp)
	var (
		body []byte
		err  error
	)
	if finalResp.HasToolCalls {
		body, err = buildFusionStreamingToolCallSSE(id, timestamp, finalResp.Model, finalResp.Raw, cfg, trace)
		if err != nil {
			return nil, err
		}
	} else {
		body = buildFusionStreamingSSE(id, timestamp, finalResp.Model, finalResp.Content, cfg, trace)
	}
	resp := streamingLooperResponse(body, finalResp.Model, modelsUsed, iterations, "fusion")
	resp.IntermediateResponses = trace
	resp.Usage = usage
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
	if cfg.IncludeAnalysis || cfg.IncludeIntermediateResponses || len(trace.FailedModels) > 0 || trace.Grounding != nil {
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
