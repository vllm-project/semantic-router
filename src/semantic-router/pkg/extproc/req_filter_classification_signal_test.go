package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestPrepareSignalEvaluationInput_CombinesMessagesWithoutCompression(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
	}

	input := router.prepareSignalEvaluationInput(signalConversationHistory{
		currentUserMessage: "user question",
		nonUserMessages:    []string{"system setup", "assistant reply"},
		hasAssistantReply:  true,
	})

	assert.Equal(t, "user question", input.evaluationText)
	assert.Equal(t, "system setup assistant reply user question", input.allMessagesText)
	assert.Equal(t, "user question", input.compressedText)
	assert.True(t, input.hasAssistantReply)
	assert.Nil(t, input.skipCompressionSignals)
}

func TestPrepareSignalEvaluationInput_UsesNonUserMessagesWhenUserContentMissing(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
	}

	input := router.prepareSignalEvaluationInput(signalConversationHistory{
		nonUserMessages: []string{"system setup", "assistant reply"},
	})

	assert.Equal(t, "system setup assistant reply", input.evaluationText)
	assert.Equal(t, "system setup assistant reply", input.allMessagesText)
	assert.Equal(t, "system setup assistant reply", input.compressedText)
	assert.False(t, input.hasAssistantReply)
}

func TestApplySignalResultsToContext_PropagatesSignalState(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{}
	signals := &classification.SignalResults{
		MatchedKeywordRules:      []string{"keyword:math"},
		MatchedEmbeddingRules:    []string{"embedding:math"},
		MatchedDomainRules:       []string{"domain:math"},
		MatchedFactCheckRules:    []string{"needs_fact_check"},
		MatchedUserFeedbackRules: []string{"satisfied"},
		MatchedReaskRules:        []string{"likely_dissatisfied"},
		MatchedPreferenceRules:   []string{"pref:code"},
		MatchedLanguageRules:     []string{"en"},
		MatchedContextRules:      []string{"context:short"},
		TokenCount:               42,
		MatchedComplexityRules:   []string{"complexity:medium"},
		MatchedModalityRules:     []string{"AR"},
		MatchedAuthzRules:        []string{"authz:team-a"},
		MatchedJailbreakRules:    []string{"jailbreak:block"},
		MatchedPIIRules:          []string{"pii:email"},
		MatchedEventRules:        []string{"critical_payment_event"},
		MatchedProjectionRules:   []string{"balance_reasoning"},
		JailbreakDetected:        true,
		JailbreakType:            "prompt_injection",
		JailbreakConfidence:      0.91,
		PIIDetected:              true,
		PIIEntities:              []string{"EMAIL_ADDRESS"},
	}

	router.applySignalResultsToContext(ctx, signals)

	assert.Equal(t, []string{"keyword:math"}, ctx.VSRMatchedKeywords)
	assert.Equal(t, []string{"embedding:math"}, ctx.VSRMatchedEmbeddings)
	assert.Equal(t, []string{"domain:math"}, ctx.VSRMatchedDomains)
	assert.Equal(t, []string{"needs_fact_check"}, ctx.VSRMatchedFactCheck)
	assert.Equal(t, []string{"satisfied"}, ctx.VSRMatchedUserFeedback)
	assert.Equal(t, []string{"likely_dissatisfied"}, ctx.VSRMatchedReask)
	assert.Equal(t, []string{"pref:code"}, ctx.VSRMatchedPreference)
	assert.Equal(t, []string{"en"}, ctx.VSRMatchedLanguage)
	assert.Equal(t, []string{"context:short"}, ctx.VSRMatchedContext)
	assert.Equal(t, 42, ctx.VSRContextTokenCount)
	assert.Equal(t, []string{"complexity:medium"}, ctx.VSRMatchedComplexity)
	assert.Equal(t, []string{"AR"}, ctx.VSRMatchedModality)
	assert.Equal(t, []string{"authz:team-a"}, ctx.VSRMatchedAuthz)
	assert.Equal(t, []string{"jailbreak:block"}, ctx.VSRMatchedJailbreak)
	assert.Equal(t, []string{"pii:email"}, ctx.VSRMatchedPII)
	assert.Equal(t, []string{"critical_payment_event"}, ctx.VSRMatchedEvent)
	assert.Equal(t, []string{"balance_reasoning"}, ctx.VSRMatchedProjection)
	assert.True(t, ctx.JailbreakDetected)
	assert.Equal(t, "prompt_injection", ctx.JailbreakType)
	assert.Equal(t, float32(0.91), ctx.JailbreakConfidence)
	assert.True(t, ctx.PIIDetected)
	assert.Equal(t, []string{"EMAIL_ADDRESS"}, ctx.PIIEntities)
	assert.True(t, ctx.FactCheckNeeded)
	require.NotNil(t, ctx.ModalityClassification)
	assert.Equal(t, "AR", ctx.ModalityClassification.Modality)
}

func TestEnsureContextTokenCount_FillsFallbackWhenContextSignalSkipped(t *testing.T) {
	ctx := &RequestContext{}
	ensureContextTokenCount(ctx, signalEvaluationInput{
		allMessagesText: "system prompt assistant state continue the task",
		evaluationText:  "continue the task",
	})

	assert.Positive(t, ctx.VSRContextTokenCount)
}

func TestEnsureContextTokenCount_PreservesSignalTokenCount(t *testing.T) {
	ctx := &RequestContext{VSRContextTokenCount: 2048}
	ensureContextTokenCount(ctx, signalEvaluationInput{
		allMessagesText: "short",
	})

	assert.Equal(t, 2048, ctx.VSRContextTokenCount)
	assert.Equal(t, len("short"), ctx.VSRContextTextBytes)
}

func TestEnsureContextTokenCountRecordsContextTextBytes(t *testing.T) {
	ctx := &RequestContext{}
	ensureContextTokenCount(ctx, signalEvaluationInput{
		allMessagesText: "system prompt and current user prompt",
		evaluationText:  "current user prompt",
	})

	assert.Equal(t, len("system prompt and current user prompt"), ctx.VSRContextTextBytes)
}

func TestEnsureContextTokenCountUsesRequestAwareFloorWithoutPollutingTextBytes(t *testing.T) {
	ctx := &RequestContext{VSRContextTokenCount: 64}
	ensureContextTokenCount(ctx, signalEvaluationInput{
		allMessagesText: "ok",
		requestFacts: classification.RequestFacts{
			ContextTokenFloor:      12_345,
			ContextTextBytes:       2,
			ContextEquivalentBytes: 49_380,
			ContextHasNonText:      true,
		},
	})

	assert.Equal(t, 12_345, ctx.VSRContextTokenCount)
	assert.Equal(t, 2, ctx.VSRContextTextBytes,
		"routing reserve must not masquerade as prose calibration bytes")
	assert.Equal(t, 49_380, ctx.VSRContextEquivalentBytes)
	assert.True(t, ctx.VSRContextHasNonText)
}

func TestPrepareSignalEvaluationInputDoesNotDoubleCountCurrentUserInFloor(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	history := signalConversationHistory{
		currentUserMessage:     "hello",
		contextTokenFloor:      7,
		contextTextBytes:       9,
		contextEquivalentBytes: 28,
		contextHasNonText:      true,
	}

	input := router.prepareSignalEvaluationInput(history)
	assert.Equal(t, "hello", input.evaluationText)
	assert.Equal(t, "hello", input.allMessagesText)
	assert.Equal(t, 7, input.requestFacts.ContextTokenFloor)
	assert.Equal(t, 9, input.requestFacts.ContextTextBytes)
	assert.Equal(t, 28, input.requestFacts.ContextEquivalentBytes)
	assert.True(t, input.requestFacts.ContextHasNonText)
}

func TestApplyFastRequestContextEstimateSetsContentFreeScalars(t *testing.T) {
	ctx := &RequestContext{}
	applyFastRequestContextEstimate(&FastExtractResult{
		ContextTokenFloor:      24_000,
		ContextTextBytes:       400,
		ContextEquivalentBytes: 96_000,
		ContextHasNonText:      true,
	}, ctx)

	assert.Equal(t, 24_000, ctx.VSRContextTokenCount)
	assert.Equal(t, 400, ctx.VSRContextTextBytes)
	assert.Equal(t, 96_000, ctx.VSRContextEquivalentBytes)
	assert.True(t, ctx.VSRContextHasNonText)
}

func TestCollectMatchedSignalRules_PreservesFamilyOrder(t *testing.T) {
	signals := &classification.SignalResults{
		MatchedKeywordRules:      []string{"keyword:a"},
		MatchedEmbeddingRules:    []string{"embedding:b"},
		MatchedDomainRules:       []string{"domain:c"},
		MatchedFactCheckRules:    []string{"fact:d"},
		MatchedUserFeedbackRules: []string{"feedback:e"},
		MatchedReaskRules:        []string{"reask:f"},
		MatchedPreferenceRules:   []string{"pref:f"},
		MatchedLanguageRules:     []string{"lang:g"},
		MatchedContextRules:      []string{"context:h"},
		MatchedComplexityRules:   []string{"complexity:i"},
		MatchedModalityRules:     []string{"modality:h"},
		MatchedAuthzRules:        []string{"authz:j"},
		MatchedJailbreakRules:    []string{"jailbreak:k"},
		MatchedPIIRules:          []string{"pii:l"},
		MatchedEventRules:        []string{"event:m"},
		MatchedProjectionRules:   []string{"projection:m"},
	}

	assert.Equal(t, []string{
		"keyword:a",
		"embedding:b",
		"domain:c",
		"fact:d",
		"feedback:e",
		"reask:f",
		"pref:f",
		"lang:g",
		"context:h",
		"complexity:i",
		"modality:h",
		"authz:j",
		"jailbreak:k",
		"pii:l",
		"event:m",
		"projection:m",
	}, collectMatchedSignalRules(signals))
}

func TestBuildCompressionConfigAppliesProfileAndOverrides(t *testing.T) {
	cfg := buildCompressionConfig(config.PromptCompressionConfig{
		Profile:        "coding",
		MaxTokens:      2048,
		NoveltyWeight:  0.25,
		PreserveLastN:  6,
		PositionDepth:  0.7,
		TextRankWeight: 0.2,
	})

	assert.Equal(t, 2048, cfg.MaxTokens)
	assert.Equal(t, 0.25, cfg.NoveltyWeight)
	assert.Equal(t, 6, cfg.PreserveLastN)
	assert.Equal(t, 0.7, cfg.PositionDepth)
	assert.Equal(t, 0.2, cfg.TextRankWeight)
	assert.Equal(t, 2, cfg.PreserveFirstN, "coding profile should apply when not explicitly overridden")
}
