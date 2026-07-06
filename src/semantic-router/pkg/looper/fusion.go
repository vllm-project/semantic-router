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
	RoundTimeoutSeconds          int
	MinSuccessfulResponses       int
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
		if cfg.OnError == config.FusionOnErrorFail || len(panelResponses) == 0 {
			return nil, err
		}
		logging.ComponentWarnEvent("looper", "fusion_panel_partial", map[string]interface{}{
			"decision":  req.DecisionName,
			"responses": len(panelResponses),
			"error":     err.Error(),
		})
	}

	// Grounding (optional) ranks/filters the panel before the judge. It makes no
	// model calls, so usage is summed from the full panel (the real cost paid).
	groundedPanel, groundingScores, groundingMode, err := l.applyGrounding(req, cfg, panelResponses)
	if err != nil {
		return nil, err
	}

	analysis, analysisResp := l.runFusionAnalysis(ctx, req, cfg, groundedPanel, groundingScores)
	finalResp, err := l.runFusionFinal(ctx, req, cfg, groundedPanel, analysis, groundingScores)
	if err != nil {
		return nil, err
	}
	usage := SumUsage(panelResponses...).Add(analysisResp, finalResp)

	trace := buildFusionTrace(cfg, groundedPanel, failedModels, analysis, groundingMode, groundingScores)
	modelsUsed := orderedFusionModelsUsed(cfg.AnalysisModels, cfg.Model)
	iterations := len(cfg.AnalysisModels) + 2

	if req.IsStreaming {
		return l.formatFusionStreamingResponse(finalResp, modelsUsed, iterations, cfg, trace, usage)
	}
	return l.formatFusionJSONResponse(finalResp, modelsUsed, iterations, cfg, trace, usage)
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

	panelCtx := ctx
	cancel := func() {}
	if cfg.RoundTimeoutSeconds > 0 {
		panelCtx, cancel = context.WithTimeout(ctx, time.Duration(cfg.RoundTimeoutSeconds)*time.Second)
	}
	defer cancel()

	results := make(chan fusionPanelResult, len(cfg.AnalysisModels))
	sem := make(chan struct{}, cfg.MaxConcurrent)
	for i, model := range cfg.AnalysisModels {
		go func(index int, modelName string) {
			select {
			case sem <- struct{}{}:
			case <-panelCtx.Done():
				results <- fusionPanelResult{index: index, model: modelName, err: panelCtx.Err()}
				return
			}
			defer func() { <-sem }()
			resp, err := l.callFusionModel(panelCtx, req, cfg, modelName, false, false, index+1)
			results <- fusionPanelResult{index: index, model: modelName, resp: resp, err: err}
		}(i, model)
	}

	collector := newFusionPanelCollector(cfg, cancel)
	for range cfg.AnalysisModels {
		select {
		case result := <-results:
			responses, err, done := collector.handleResult(result)
			if done {
				return responses, collector.failed, err
			}
		case <-panelCtx.Done():
			responses, err := collector.handleTimeout(panelCtx.Err())
			return responses, collector.failed, err
		}
	}

	responses := collector.responses()
	if len(responses) == 0 {
		return nil, collector.failed, fmt.Errorf("fusion panel failed: all %d analysis models failed", len(cfg.AnalysisModels))
	}
	return responses, collector.failed, nil
}

type fusionPanelCollector struct {
	cfg       fusionExecutionConfig
	cancel    context.CancelFunc
	ordered   []*ModelResponse
	failed    []FusionFailedModel
	successes int
}

func newFusionPanelCollector(cfg fusionExecutionConfig, cancel context.CancelFunc) *fusionPanelCollector {
	return &fusionPanelCollector{
		cfg:     cfg,
		cancel:  cancel,
		ordered: make([]*ModelResponse, len(cfg.AnalysisModels)),
	}
}

func (c *fusionPanelCollector) handleResult(result fusionPanelResult) ([]*ModelResponse, error, bool) {
	if result.err != nil {
		c.failed = append(c.failed, FusionFailedModel{Model: result.model, Error: result.err.Error()})
		if c.cfg.OnError == config.FusionOnErrorFail {
			c.cancel()
			return nil, fmt.Errorf("fusion panel model %q failed: %w", result.model, result.err), true
		}
		return nil, nil, false
	}
	c.ordered[result.index] = result.resp
	c.successes++
	if c.successes < c.cfg.MinSuccessfulResponses {
		return nil, nil, false
	}
	c.logQuorum()
	c.cancel()
	return c.responses(), nil, true
}

func (c *fusionPanelCollector) handleTimeout(err error) ([]*ModelResponse, error) {
	responses := c.responses()
	if len(responses) > 0 && c.cfg.OnError != config.FusionOnErrorFail {
		c.failed = append(c.failed, FusionFailedModel{Model: "panel", Error: err.Error()})
		return responses, err
	}
	return nil, err
}

func (c *fusionPanelCollector) responses() []*ModelResponse {
	return compactFusionPanelResponses(c.ordered)
}

func (c *fusionPanelCollector) logQuorum() {
	if c.successes >= len(c.cfg.AnalysisModels) {
		return
	}
	logging.ComponentEvent("looper", "fusion_panel_quorum_reached", map[string]interface{}{
		"responses": c.successes,
		"panel":     len(c.cfg.AnalysisModels),
	})
}

func compactFusionPanelResponses(ordered []*ModelResponse) []*ModelResponse {
	responses := make([]*ModelResponse, 0, len(ordered))
	for _, resp := range ordered {
		if resp != nil {
			responses = append(responses, resp)
		}
	}
	return responses
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
	original := extractOriginalContent(req.OriginalRequest)
	outputContract := requestOutputContract(req.OriginalRequest, req.OutputContract)
	prompt := buildFusionFinalPrompt(cfg, original, outputContract, panelResponses, analysis)
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
	applyJSONActionOutputContract(req.OutputContractSpec, resp, panelResponses)
	applyFinalOutputContract(req.OutputContractSpec, resp)
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
	outputContract string,
	responses []*ModelResponse,
	analysis *FusionAnalysis,
) string {
	if cfg.SynthesisTemplate != "" {
		return appendOutputContractForPrompt(
			renderFusionPrompt(cfg.SynthesisTemplate, original, responses, analysis),
			outputContract,
		)
	}
	analysisBlock := "No structured analysis is available. Synthesize directly from the panel responses."
	if analysis != nil && !analysis.ParseFailed {
		if data, err := json.MarshalIndent(analysis, "", "  "); err == nil {
			analysisBlock = string(data)
		}
	}
	prompt := fmt.Sprintf(`You are the Fusion calling model. Produce the final answer for the user using the panel responses and structured analysis. Resolve contradictions explicitly and do not mention internal model names unless the user asks.

Rules:
- Preserve the original output contract exactly.
- Do not reveal hidden reasoning, scratch work, panel reasoning, tool traces, or internal deliberation.
- Provide a concise explanation only when the original output contract asks for one.

Original prompt:
%s

Structured analysis:
%s

Panel responses:
%s

Final answer:`, original, analysisBlock, formatPanelResponses(responses))

	return appendOutputContractForPrompt(prompt, outputContract)
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
	candidates := jsonObjectParseCandidates(content)
	if len(candidates) == 0 {
		return nil, fmt.Errorf("empty fusion analysis response")
	}
	var failures []string
	for _, candidate := range candidates {
		var analysis FusionAnalysis
		if err := json.Unmarshal([]byte(candidate), &analysis); err == nil {
			return &analysis, nil
		} else {
			failures = append(failures, err.Error())
		}
	}
	return nil, fmt.Errorf("%s", strings.Join(failures, "; "))
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
