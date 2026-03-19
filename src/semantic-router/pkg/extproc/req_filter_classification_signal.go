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
}

func (r *OpenAIRouter) prepareSignalEvaluationInput(userContent string, nonUserMessages []string) signalEvaluationInput {
	input := signalEvaluationInput{
		evaluationText:  userContent,
		compressedText:  userContent,
		allMessagesText: strings.Join(nonUserMessages, " "),
	}

	if input.evaluationText == "" && len(nonUserMessages) > 0 {
		input.evaluationText = strings.Join(nonUserMessages, " ")
		input.compressedText = input.evaluationText
	}

	if userContent != "" && len(nonUserMessages) > 0 {
		allMessages := make([]string, 0, len(nonUserMessages)+1)
		allMessages = append(allMessages, nonUserMessages...)
		allMessages = append(allMessages, userContent)
		input.allMessagesText = strings.Join(allMessages, " ")
	} else if userContent != "" {
		input.allMessagesText = userContent
	}

	if input.evaluationText == "" {
		return input
	}

	input.compressedText, input.skipCompressionSignals = r.compressSignalEvaluationText(input.evaluationText)
	return input
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
	ctx.VSRMatchedPreference = signals.MatchedPreferenceRules
	ctx.VSRMatchedLanguage = signals.MatchedLanguageRules
	ctx.VSRMatchedContext = signals.MatchedContextRules
	ctx.VSRContextTokenCount = signals.TokenCount
	ctx.VSRMatchedComplexity = signals.MatchedComplexityRules
	ctx.VSRMatchedModality = signals.MatchedModalityRules
	ctx.VSRMatchedAuthz = signals.MatchedAuthzRules
	ctx.VSRMatchedJailbreak = signals.MatchedJailbreakRules
	ctx.VSRMatchedPII = signals.MatchedPIIRules

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

func collectMatchedSignalRules(signals *classification.SignalResults) []string {
	allMatchedRules := []string{}
	allMatchedRules = append(allMatchedRules, signals.MatchedKeywordRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedEmbeddingRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedDomainRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedFactCheckRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedUserFeedbackRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedPreferenceRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedLanguageRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedModalityRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedJailbreakRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedPIIRules...)
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
