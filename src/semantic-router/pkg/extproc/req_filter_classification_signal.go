package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/promptcompression"
)

type signalEvaluationInput struct {
	evaluationText         string
	allMessagesText        string
	compressedText         string
	skipCompressionSignals map[string]bool
	currentUserText        string
	priorUserMessages      []string
	hasAssistantReply      bool
	conversationFacts      classification.ConversationFacts
}

func (r *OpenAIRouter) prepareSignalEvaluationInput(history signalConversationHistory) signalEvaluationInput {
	input := signalEvaluationInput{
		evaluationText:    history.currentUserMessage,
		compressedText:    history.currentUserMessage,
		allMessagesText:   strings.Join(history.nonUserMessages, " "),
		currentUserText:   history.currentUserMessage,
		priorUserMessages: append([]string(nil), history.priorUserMessages...),
		hasAssistantReply: history.hasAssistantReply,
		conversationFacts: classification.ConversationFacts{
			HasDeveloperMessage:    history.hasDeveloperMessage,
			UserMessageCount:       history.userMessageCount,
			AssistantMessageCount:  history.assistantMessageCount,
			SystemMessageCount:     history.systemMessageCount,
			ToolMessageCount:       history.toolMessageCount,
			ToolDefinitionCount:    history.toolDefinitionCount,
			AssistantToolCallCount: history.assistantToolCallCount,
			CompletedToolCycles:    history.completedToolCycles,
		},
	}

	if input.evaluationText == "" && len(history.nonUserMessages) > 0 {
		input.evaluationText = strings.Join(history.nonUserMessages, " ")
		input.compressedText = input.evaluationText
	}

	if history.currentUserMessage != "" && len(history.nonUserMessages) > 0 {
		allMessages := make([]string, 0, len(history.nonUserMessages)+1)
		allMessages = append(allMessages, history.nonUserMessages...)
		allMessages = append(allMessages, history.currentUserMessage)
		input.allMessagesText = strings.Join(allMessages, " ")
	} else if history.currentUserMessage != "" {
		input.allMessagesText = history.currentUserMessage
	}

	if input.evaluationText == "" {
		return input
	}

	// Hard safety limit: truncate before any signal processing, independently of
	// prompt_compression. Keeps embedding and classifier inference bounded even
	// when compression is disabled, preventing super-linear latency blowup on
	// giant prompts (see issue #1454).
	input.evaluationText = r.applyMaxEvaluationChars(input.evaluationText)
	input.compressedText = input.evaluationText

	input.compressedText, input.skipCompressionSignals = r.compressSignalEvaluationText(input.evaluationText)
	return input
}

// applyMaxEvaluationChars truncates evaluationText to the configured hard limit.
// A limit <= 0 means no truncation. Only the routing-decision text is affected;
// the actual request body forwarded to the model is never modified.
func (r *OpenAIRouter) applyMaxEvaluationChars(text string) string {
	limit := r.Config.PromptCompression.MaxEvaluationChars
	if limit <= 0 || len(text) <= limit {
		return text
	}
	logging.Infof("[SignalEval] evaluationText truncated by max_evaluation_chars: %d -> %d chars", len(text), limit)
	return text[:limit]
}

func (r *OpenAIRouter) compressSignalEvaluationText(evaluationText string) (string, map[string]bool) {
	compressedText := evaluationText
	var skipCompressionSignals map[string]bool

	if !r.Config.PromptCompression.Enabled || r.Config.PromptCompression.MaxTokens <= 0 {
		return compressedText, skipCompressionSignals
	}

	cfg := buildCompressionConfig(r.Config.PromptCompression)
	origTokens := promptcompression.CountTokensApprox(evaluationText)
	if r.Config.PromptCompression.MinLength > 0 && len(evaluationText) <= r.Config.PromptCompression.MinLength {
		logging.Infof("[PromptCompression] Skipped: %d chars <= min_length threshold %d", len(evaluationText), r.Config.PromptCompression.MinLength)
		return compressedText, skipCompressionSignals
	}
	if origTokens <= cfg.MaxTokens {
		return compressedText, skipCompressionSignals
	}

	result := promptcompression.Compress(evaluationText, cfg)
	logging.Infof("[PromptCompression] Compressed evaluationText: %d -> %d tokens (ratio=%.2f, kept %d sentences)",
		result.OriginalTokens, result.CompressedTokens, result.Ratio, len(result.KeptIndices))
	return result.Compressed, r.Config.PromptCompression.SkipSignalsSet()
}

func (r *OpenAIRouter) applySignalResultsToContext(ctx *RequestContext, signals *classification.SignalResults) {
	ctx.VSRMatchedKeywords = signals.MatchedKeywordRules
	ctx.VSRMatchedEmbeddings = signals.MatchedEmbeddingRules
	ctx.VSRMatchedDomains = signals.MatchedDomainRules
	ctx.VSRMatchedFactCheck = signals.MatchedFactCheckRules
	ctx.VSRMatchedUserFeedback = signals.MatchedUserFeedbackRules
	ctx.VSRMatchedReask = signals.MatchedReaskRules
	ctx.VSRMatchedPreference = signals.MatchedPreferenceRules
	ctx.VSRMatchedLanguage = signals.MatchedLanguageRules
	ctx.VSRMatchedContext = signals.MatchedContextRules
	ctx.VSRContextTokenCount = signals.TokenCount
	ctx.VSRMatchedStructure = signals.MatchedStructureRules
	ctx.VSRMatchedComplexity = signals.MatchedComplexityRules
	ctx.VSRMatchedModality = signals.MatchedModalityRules
	ctx.VSRMatchedAuthz = signals.MatchedAuthzRules
	ctx.VSRMatchedJailbreak = signals.MatchedJailbreakRules
	ctx.VSRMatchedPII = signals.MatchedPIIRules
	ctx.VSRMatchedKB = signals.MatchedKBRules
	ctx.VSRMatchedConversation = signals.MatchedConversationRules
	ctx.VSRMatchedProjection = signals.MatchedProjectionRules
	ctx.VSRProjectionScores = cloneReplayFloat64Map(signals.ProjectionScores)
	ctx.VSRSignalConfidences = cloneReplayFloat64Map(signals.SignalConfidences)
	ctx.VSRSignalValues = cloneReplayFloat64Map(signals.SignalValues)

	if signals.JailbreakDetected {
		ctx.JailbreakDetected = signals.JailbreakDetected
		ctx.JailbreakType = signals.JailbreakType
		ctx.JailbreakConfidence = signals.JailbreakConfidence
	}
	if signals.PIIDetected {
		ctx.PIIDetected = signals.PIIDetected
		ctx.PIIEntities = signals.PIIEntities
	}

	r.setFactCheckFromSignals(ctx, signals.MatchedFactCheckRules)
	r.setModalityFromSignals(ctx, signals.MatchedModalityRules)
}

func cloneReplayFloat64Map(values map[string]float64) map[string]float64 {
	if values == nil {
		return nil
	}
	cloned := make(map[string]float64, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

func collectMatchedSignalRules(signals *classification.SignalResults) []string {
	allMatchedRules := []string{}
	allMatchedRules = append(allMatchedRules, signals.MatchedKeywordRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedEmbeddingRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedDomainRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedFactCheckRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedUserFeedbackRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedReaskRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedPreferenceRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedLanguageRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedContextRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedStructureRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedComplexityRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedModalityRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedAuthzRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedJailbreakRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedPIIRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedKBRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedConversationRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedProjectionRules...)
	return allMatchedRules
}

// buildCompressionConfig translates the YAML config into the promptcompression
// package's Config struct, applying defaults for omitted fields.
func buildCompressionConfig(pc config.PromptCompressionConfig) promptcompression.Config {
	cfg := promptcompression.DefaultConfig(pc.MaxTokens)
	if pc.TextRankWeight > 0 {
		cfg.TextRankWeight = pc.TextRankWeight
	}
	if pc.PositionWeight > 0 {
		cfg.PositionWeight = pc.PositionWeight
	}
	if pc.TFIDFWeight > 0 {
		cfg.TFIDFWeight = pc.TFIDFWeight
	}
	if pc.PositionDepth > 0 {
		cfg.PositionDepth = pc.PositionDepth
	}
	return cfg
}
