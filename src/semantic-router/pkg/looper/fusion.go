package looper

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
	AnalysisOverrides            map[string]config.FusionModelOverride
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

	panel, err := l.executeFusionPanel(ctx, req, cfg)
	panelResponses := panel.responses
	failedModels := panel.failed
	if err != nil {
		if cfg.OnError == config.FusionOnErrorFail || len(panelResponses) == 0 {
			return nil, newPartialExecutionError(err, fusionExecutionEvidence(panel.accountingResponses, nil, nil))
		}
		logging.ComponentWarnEvent("looper", "fusion_panel_partial", map[string]interface{}{
			"decision":  req.DecisionName,
			"responses": len(panelResponses),
			"error":     err.Error(),
		})
	}

	// Grounding (optional) ranks/filters only the quorum snapshot. Accounting
	// separately retains any sibling result already queued when quorum canceled
	// the remaining workers.
	groundedPanel, groundingScores, groundingMode, err := l.applyGrounding(req, cfg, panelResponses)
	if err != nil {
		return nil, newPartialExecutionError(err, fusionExecutionEvidence(panel.accountingResponses, nil, nil))
	}

	analysis, analysisResp := l.runFusionAnalysis(ctx, req, cfg, groundedPanel, groundingScores)
	finalResp, err := l.runFusionFinal(ctx, req, cfg, groundedPanel, analysis, groundingScores)
	if err != nil {
		return nil, newPartialExecutionError(err, fusionExecutionEvidence(panel.accountingResponses, analysisResp, nil))
	}
	usage := SumUsage(panel.accountingResponses...).Add(analysisResp, finalResp)

	trace := buildFusionTrace(cfg, groundedPanel, failedModels, analysis, groundingMode, groundingScores)
	modelsUsed := orderedFusionModelsUsed(cfg.AnalysisModels, cfg.Model)
	iterations := len(cfg.AnalysisModels) + 2

	var response *Response
	if req.IsStreaming {
		response, err = l.formatFusionStreamingResponse(finalResp, modelsUsed, iterations, cfg, trace, usage, streamUsageRequested(req))
	} else {
		response, err = l.formatFusionJSONResponse(finalResp, modelsUsed, iterations, cfg, trace, usage)
	}
	if err != nil {
		return nil, newPartialExecutionError(err, ExecutionEvidence{
			ModelsUsed: modelsUsed,
			Iterations: iterations,
			Usage:      usage,
		})
	}
	return response, nil
}

func fusionExecutionEvidence(
	panelResponses []*ModelResponse,
	analysisResp *ModelResponse,
	finalResp *ModelResponse,
) ExecutionEvidence {
	responses := append([]*ModelResponse(nil), panelResponses...)
	responses = append(responses, analysisResp, finalResp)
	modelsUsed := make([]string, 0, len(responses))
	iterations := 0
	for _, response := range responses {
		if response == nil {
			continue
		}
		iterations++
		if response.Model != "" {
			modelsUsed = appendUniqueModel(modelsUsed, response.Model)
		}
	}
	return executionEvidenceFromResponses(responses, modelsUsed, iterations)
}

func appendUniqueModel(models []string, model string) []string {
	for _, existing := range models {
		if existing == model {
			return models
		}
	}
	return append(models, model)
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
) (fusionPanelExecution, error) {
	// Paired multi-arm evaluation supplies the panel verbatim so every arm
	// synthesizes from a byte-identical panel (see bench/grounded_fusion). Skip
	// the live model calls and feed the cached panel straight into grounding +
	// synthesis, which are source-agnostic over []*ModelResponse.
	if len(req.CachedPanel) > 0 {
		return fusionPanelExecution{responses: req.CachedPanel, accountingResponses: req.CachedPanel}, nil
	}

	var panelCtx context.Context
	var cancel context.CancelFunc
	if cfg.RoundTimeoutSeconds > 0 {
		panelCtx, cancel = context.WithTimeout(ctx, time.Duration(cfg.RoundTimeoutSeconds)*time.Second)
	} else {
		panelCtx, cancel = context.WithCancel(ctx)
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
			resp, err := l.callFusionModel(panelCtx, req, cfg, modelName, false, false, index+1, cfg.AnalysisOverrides[modelName])
			results <- fusionPanelResult{index: index, model: modelName, resp: resp, err: err}
		}(i, model)
	}

	collector := newFusionPanelCollector(cfg, cancel)
	remaining := len(cfg.AnalysisModels)
	for remaining > 0 {
		select {
		case result := <-results:
			remaining--
			collector.handleResult(result)
			if collector.quorumReached || collector.terminalErr != nil {
				drainFusionPanelResults(results, collector, &remaining)
				return collector.execution(), collector.terminalErr
			}
		case <-panelCtx.Done():
			responses, err := collector.handleTimeout(panelCtx.Err())
			execution := collector.execution()
			execution.responses = responses
			return execution, err
		}
	}

	execution := collector.execution()
	responses := execution.responses
	if collector.terminalErr != nil {
		return execution, collector.terminalErr
	}
	if len(responses) == 0 {
		return execution, fmt.Errorf("fusion panel failed: all %d analysis models failed", len(cfg.AnalysisModels))
	}
	return execution, nil
}

type fusionPanelExecution struct {
	responses           []*ModelResponse
	accountingResponses []*ModelResponse
	failed              []FusionFailedModel
}

func drainFusionPanelResults(results <-chan fusionPanelResult, collector *fusionPanelCollector, remaining *int) {
	for *remaining > 0 {
		select {
		case result := <-results:
			(*remaining)--
			collector.handleResult(result)
		default:
			return
		}
	}
}

type fusionPanelCollector struct {
	cfg           fusionExecutionConfig
	cancel        context.CancelFunc
	ordered       []*ModelResponse
	quorumOrdered []*ModelResponse
	failed        []FusionFailedModel
	successes     int
	quorumReached bool
	terminalErr   error
}

func newFusionPanelCollector(cfg fusionExecutionConfig, cancel context.CancelFunc) *fusionPanelCollector {
	return &fusionPanelCollector{
		cfg:     cfg,
		cancel:  cancel,
		ordered: make([]*ModelResponse, len(cfg.AnalysisModels)),
	}
}

func (c *fusionPanelCollector) handleResult(result fusionPanelResult) {
	if result.err != nil {
		c.failed = append(c.failed, FusionFailedModel{Model: result.model, Error: result.err.Error()})
		if c.cfg.OnError == config.FusionOnErrorFail && !c.quorumReached && c.terminalErr == nil {
			c.cancel()
			c.terminalErr = fmt.Errorf("fusion panel model %q failed: %w", result.model, result.err)
		}
		return
	}
	c.ordered[result.index] = result.resp
	c.successes++
	if c.quorumReached || c.terminalErr != nil || c.successes < c.cfg.MinSuccessfulResponses {
		return
	}
	c.quorumReached = true
	c.quorumOrdered = append([]*ModelResponse(nil), c.ordered...)
	c.logQuorum()
	c.cancel()
}

func (c *fusionPanelCollector) handleTimeout(err error) ([]*ModelResponse, error) {
	responses := c.responses()
	if len(responses) > 0 && c.cfg.OnError != config.FusionOnErrorFail {
		c.failed = append(c.failed, FusionFailedModel{Model: "panel", Error: err.Error()})
		return responses, err
	}
	return responses, err
}

func (c *fusionPanelCollector) responses() []*ModelResponse {
	if c.quorumReached {
		return compactFusionPanelResponses(c.quorumOrdered)
	}
	return compactFusionPanelResponses(c.ordered)
}

func (c *fusionPanelCollector) execution() fusionPanelExecution {
	return fusionPanelExecution{
		responses:           c.responses(),
		accountingResponses: compactFusionPanelResponses(c.ordered),
		failed:              c.failed,
	}
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
	override config.FusionModelOverride,
) (*ModelResponse, error) {
	callReq := cloneRequest(req.executionRequest)
	if !allowTools {
		callReq = stripFusionToolUse(callReq)
	}
	if override.Temperature != nil {
		callReq.Temperature = openai.Float(*override.Temperature)
	} else if cfg.Temperature != nil {
		callReq.Temperature = openai.Float(*cfg.Temperature)
	}
	if override.MaxCompletionTokens > 0 {
		callReq.MaxCompletionTokens = openai.Int(int64(override.MaxCompletionTokens))
	} else if cfg.MaxCompletionTokens > 0 {
		callReq.MaxCompletionTokens = openai.Int(int64(cfg.MaxCompletionTokens))
	}
	return l.client.CallModel(ctx, callReq, modelName, streaming, iteration, nil)
}

func (l *FusionLooper) runFusionAnalysis(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	panelResponses []*ModelResponse,
	groundingScores []groundingScore,
) (*FusionAnalysis, *ModelResponse) {
	prompt := buildFusionAnalysisPrompt(cfg, extractOriginalContent(req.executionRequest), panelResponses)
	if notes := formatGroundingNotes(groundingScores); notes != "" {
		prompt = prompt + "\n\n" + notes
	}
	analysisReq := appendFusionStageMessage(req.executionRequest, prompt)
	resp, err := l.callFusionModel(ctx, &Request{executionRequest: analysisReq, ModelParams: req.ModelParams}, cfg, cfg.Model, false, false, len(panelResponses)+1, config.FusionModelOverride{})
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
	original := extractOriginalContent(req.executionRequest)
	outputContract := requestOutputContract(req.executionRequest, req.OutputContract)
	prompt := buildFusionFinalPrompt(cfg, original, outputContract, panelResponses, analysis)
	// Under weight/annotate policies the panel was not pruned, so the judge needs
	// the per-response groundedness signal at synthesis time to soft-weight.
	if notes := groundingSynthesisNotes(groundingScores, cfg.GroundingPolicy); notes != "" {
		prompt = prompt + "\n\n" + notes
	}
	finalReq := appendFusionStageMessage(req.executionRequest, prompt)
	resp, err := l.callFusionModel(ctx, &Request{executionRequest: finalReq, ModelParams: req.ModelParams}, cfg, cfg.Model, true, false, len(panelResponses)+2, config.FusionModelOverride{})
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
	semantic := newTextSemanticResponse("response-fusion", finalResp.Model, finalResp.Content, usage)
	return newLooperResponse(semantic, false, true, finalResp.Model, modelsUsed, iterations, "fusion", usage, trace), nil
}

func (l *FusionLooper) formatFusionToolCallJSONResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	cfg fusionExecutionConfig,
	trace *FusionTrace,
	usage TokenUsage,
) (*Response, error) {
	_ = cfg
	semantic, err := newModelSemanticResponse("response-fusion", finalResp, finalResp.Model, usage)
	if err != nil {
		return nil, fmt.Errorf("build neutral fusion tool-call response: %w", err)
	}
	return newLooperResponse(semantic, false, true, finalResp.Model, modelsUsed, iterations, "fusion", usage, trace), nil
}

func (l *FusionLooper) formatFusionStreamingResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	cfg fusionExecutionConfig,
	trace *FusionTrace,
	usage TokenUsage,
	includeUsage bool,
) (*Response, error) {
	_ = cfg
	var semantic llmprotocol.Response
	if finalResp.HasToolCalls {
		var err error
		semantic, err = newModelSemanticResponse("response-fusion", finalResp, finalResp.Model, usage)
		if err != nil {
			return nil, err
		}
	} else {
		semantic = newTextSemanticResponse("response-fusion", finalResp.Model, finalResp.Content, usage)
	}
	return newLooperResponse(semantic, true, includeUsage, finalResp.Model, modelsUsed, iterations, "fusion", usage, trace), nil
}
