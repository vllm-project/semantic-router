package extproc

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// This reserve covers provider chat framing that the neutral request cannot
// tokenize. UTF-8 bytes bound text tokens for byte-based tokenizers; media and
// provider-specific templates remain estimates, not exact tokenizer counts.
const contextOverflowTemplateReserve = 1024

type overflowTokenCounter struct{ decision *config.Decision }

func (overflowTokenCounter) CountText(_ string, text string) (int, string) {
	return len(text), "utf8_byte_upper_bound"
}

func (counter overflowTokenCounter) CountRequest(_ string, request *contextcompression.RequestIR) (int, string) {
	if request == nil || request.Semantic == nil {
		return 0, "utf8_byte_upper_bound"
	}
	effective, err := selection.EffectiveCandidateRequest(request.Semantic, counter.decision)
	if err != nil {
		return int(^uint(0) >> 1), "utf8_byte_upper_bound"
	}
	estimate := llmprotocol.EstimateInput(effective)
	estimatedText := estimate.TextBytes / llmprotocol.InputTextBytesPerToken
	if estimate.TextBytes%llmprotocol.InputTextBytesPerToken != 0 {
		estimatedText++
	}
	return llmprotocol.SaturatingTokenSum(estimate.Tokens-estimatedText, estimate.TextBytes, overflowFramingReserve(effective)), "utf8_byte_upper_bound"
}

// Reserve per-item framing as well as a fixed template allowance. A fixed
// allowance alone would undercount many tiny messages or tool definitions.
func overflowFramingReserve(request *llmprotocol.Request) int {
	reserve := contextOverflowTemplateReserve
	for range request.Messages {
		reserve = llmprotocol.SaturatingTokenSum(reserve, 32)
	}
	for range request.Instructions {
		reserve = llmprotocol.SaturatingTokenSum(reserve, 32)
	}
	for range request.Tools {
		reserve = llmprotocol.SaturatingTokenSum(reserve, 64)
	}
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentToolCall || content.Kind == llmprotocol.ContentToolResult {
				reserve = llmprotocol.SaturatingTokenSum(reserve, 64)
			}
		}
	}
	return reserve
}

func contextOverflowConfig(ctx *RequestContext) *config.ContextCompressionPluginConfig {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return nil
	}
	cfg := ctx.VSRSelectedDecision.GetContextCompressionConfig()
	if cfg == nil || !cfg.Enabled || cfg.Targets == nil || cfg.Targets.CurrentUser.Mode != config.ContextCompressionTargetTruncate {
		return nil
	}
	applyContextCompressionRequestControls(ctx, cfg)
	if ctx.ContextCompressionSkipReason == "request_bypass" {
		return nil
	}
	return cfg
}

// prepareDecisionContextOverflow runs after signals and the decision, before
// budget eligibility. It does not choose a backend: the widest compatible
// assigned budget only bounds the request that normal selection will inspect.
func (r *OpenAIRouter) prepareDecisionContextOverflow(ctx *RequestContext, originalModel string) error {
	cfg := contextOverflowConfig(ctx)
	if cfg == nil || ctx.SemanticRequest == nil || ctx.VSRSelectedDecision.GetFastResponseConfig() != nil {
		return nil
	}
	decision := ctx.VSRSelectedDecision
	refs := append([]config.ModelRef(nil), decision.ModelRefs...)
	if decision.Action != nil && decision.Action.Type == config.DecisionActionRoute {
		refs = append(refs, config.ModelRef{Model: decision.Action.Destination})
	} else if !r.requestModelActsAsAuto(originalModel) {
		refs = []config.ModelRef{{Model: originalModel}}
	}
	demand, err := selection.EffectiveCandidateDemand(ctx.SemanticRequest, decision)
	if err != nil {
		return err
	}
	// Retain capability, codec, known-limit and output-limit admission. Only
	// the input-budget question is answered after the configured transformation.
	demand.InputTokens = 0
	model, available := "", 0
	for _, ref := range refs {
		if err := r.validateModelDemand(r.candidateRequirements(ctx), ref.Model, demand); err != nil {
			continue
		}
		budget := r.contextOverflowInputBudget(ref.Model, demand, cfg)
		if budget > available {
			model, available = ref.Model, budget
		}
	}
	if available == 0 {
		return nil
	} // Ordinary eligibility reports missing metadata/capabilities.
	if err := r.applyContextOverflow(ctx, ctx.SemanticRequest, cfg, decision, model, available); err != nil {
		return err
	}
	return nil
}

func (r *OpenAIRouter) contextOverflowInputBudget(model string, demand selection.CandidateDemand, cfg *config.ContextCompressionPluginConfig) int {
	params, _ := selection.CandidateModelParams(r.Config.ModelConfig, nil, model)
	if params.ContextWindowSize <= 0 || params.MaxOutputTokens <= 0 {
		return 0
	}
	reserve := params.MaxOutputTokens
	if demand.MaxOutputTokens != nil {
		if *demand.MaxOutputTokens <= 0 || *demand.MaxOutputTokens > int64(params.MaxOutputTokens) {
			return 0
		}
		reserve = int(*demand.MaxOutputTokens)
	}
	if cfg.Budget != nil && cfg.Budget.ReserveOutputTokens != nil && !cfg.Budget.ReserveOutputTokens.Auto {
		reserve = max(reserve, cfg.Budget.ReserveOutputTokens.Value)
	}
	return max(0, params.ContextWindowSize-reserve)
}

// Repeat the bound after enrichment and deterministic provider preparation.
// This checks the actual dispatch body rather than trusting the earlier view.
func (r *OpenAIRouter) prepareDispatchContextOverflow(ctx *RequestContext, request *llmprotocol.Request, model string) error {
	cfg := contextOverflowConfig(ctx)
	if cfg == nil {
		return nil
	}
	available := r.contextOverflowInputBudget(model, selection.DemandForRequest(request), cfg)
	if available == 0 {
		if err := r.validateModelDemand(r.candidateRequirements(ctx), model, selection.DemandForRequest(request)); err != nil {
			return err
		}
		return fmt.Errorf("%w: selected model has no usable known input budget for context compression", selection.ErrNoEligibleCandidates)
	}
	return r.applyContextOverflow(ctx, request, cfg, nil, model, available)
}

func overflowBudgetError(message string) error {
	return &selection.RequestBudgetError{Code: "context_length_exceeded", Message: message}
}

func (r *OpenAIRouter) applyContextOverflow(ctx *RequestContext, request *llmprotocol.Request, cfg *config.ContextCompressionPluginConfig, decision *config.Decision, model string, available int) error {
	start := time.Now()
	counter := overflowTokenCounter{decision: decision}
	before, _ := counter.CountRequest(model, &contextcompression.RequestIR{Semantic: request})
	if before <= available {
		return nil
	}
	// Never partially rewrite a request that will be rejected. The detached
	// copy keeps codec metadata and immutable tool schemas, with owned text.
	detached, err := selection.EffectiveCandidateRequest(request, nil)
	if err != nil {
		return err
	}
	ir := contextcompression.ParseSemanticRequest(detached, contextcompression.Provenance{
		ProtectedMessages: ctx.ProtectedContextMessages, OriginalHistory: ctx.OriginalContextHistory,
		RAGToolCallIDs: ctx.RAGToolCallIDs, MemoryMessageIndexes: ctx.MemoryMessageIndexes,
	})
	policy := contextcompression.PolicyFromConfig(cfg, ctx.ContextCompressionTargetTokens)
	policy.Mode = contextcompression.ModeAlways
	policy.Budget.TargetTokens = available
	if cfg.Budget != nil && cfg.Budget.TargetTokens != nil && !cfg.Budget.TargetTokens.Auto && cfg.Budget.TargetTokens.Value > 0 {
		policy.Budget.TargetTokens = min(available, cfg.Budget.TargetTokens.Value)
	}
	if ctx.ContextCompressionTargetTokens != nil {
		policy.Budget.TargetTokens = min(policy.Budget.TargetTokens, *ctx.ContextCompressionTargetTokens)
	}
	policy.Budget.TargetAuto = false
	policy.Budget.ReserveOutputTokens, policy.Budget.ReserveAuto = 0, false
	// Recovery is a selected-route side effect. The admission pass cannot write
	// recovery records or add retrieval tools for candidates not yet selected.
	for _, target := range []*contextcompression.TargetPolicy{&policy.Targets.ToolOutputs, &policy.Targets.History, &policy.Targets.RAG, &policy.Targets.Memory} {
		if target.Mode == contextcompression.TargetRecoverable {
			target.Mode = contextcompression.TargetPreserve
		}
	}
	callContext := ctx.TraceContext
	if callContext == nil {
		callContext = context.Background()
	}
	result := r.contextCompressionService().Apply(callContext, contextcompression.Request{
		Model: model, Scope: r.contextCompressionScope(ctx), Request: ir, Policy: policy,
		Capabilities: contextcompression.ModelContextCapabilities{ContextWindow: available}, TokenCounter: counter,
		Scorer: r.contextCompressionScorer(callContext, cfg, ctx),
	})
	after, _ := counter.CountRequest(model, ir)
	if result.Failure != nil || !result.Applied || after > available {
		return overflowBudgetError(fmt.Sprintf("Request exceeds the model input budget after configured compression (%d > %d conservative tokens); protected instructions, tool structure, schemas or preserved content cannot be truncated", after, available))
	}
	*request = *detached
	request.Generation++
	ctx.ContextRequestIR = nil
	ctx.VSRContextTokenCount = llmprotocol.EstimateInput(request).Tokens
	result.Plan.TriggerReason = "context_window"
	r.recordCompressionPlan(ctx, result)
	if fingerprint, fingerprintErr := cache.FingerprintValue(cfg); fingerprintErr == nil && len(fingerprint) >= 12 {
		ctx.ContextCompressionRevision = fingerprint[:12]
	}
	ctx.ContextCompressionTrigger = "context_window"
	stats := contextCompressionStats{beforeTokens: before, afterTokens: after, appliedMessages: result.MessagesCompressed, omittedChunks: result.OmittedChunks, jsonBlocks: result.JSONBlocks, appliedBlocks: result.BlocksCompressed}
	if ctx.ContextCompressionApplied {
		stats.beforeTokens = ctx.ContextCompressionBefore
		stats.appliedMessages += ctx.ContextCompressionMessages
		stats.omittedChunks += ctx.ContextCompressionOmitted
	}
	recordContextCompressionApplied(ctx, stats, start)
	return nil
}
