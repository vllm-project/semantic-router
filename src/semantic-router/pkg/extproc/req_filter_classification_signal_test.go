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
		lastMessageRole:    "user",
		lastUserHasText:    true,
	})

	assert.Equal(t, "user question", input.evaluationText)
	assert.Equal(t, "system setup assistant reply user question", input.allMessagesText)
	assert.Equal(t, "user question", input.compressedText)
	assert.True(t, input.hasAssistantReply)
	assert.Nil(t, input.skipCompressionSignals)
}

func TestPrepareSignalEvaluationInputCopiesToolResultTexts(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	history := signalConversationHistory{
		toolResultTexts:          []string{"customer@example.com", "order result"},
		toolResultScanIncomplete: true,
	}

	input := router.prepareSignalEvaluationInput(history)
	history.toolResultTexts[0] = "mutated"

	assert.Equal(t, []string{"customer@example.com", "order result"}, input.toolResultTexts)
	assert.True(t, input.toolResultScanIncomplete)
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

func TestApplyRequestContextEstimateSetsContentFreeScalars(t *testing.T) {
	ctx := &RequestContext{}
	applyRequestContextEstimate(&requestSignalSnapshot{
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

func TestApplySignalResultsPreservesUnscoredPolicyMatch(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{}
	key := "jailbreak:guard"
	signals := &classification.SignalResults{MatchedJailbreakRules: []string{"guard"}, JailbreakDetected: true, JailbreakType: classification.JailbreakUnscannedType, SignalErrorMatches: map[string]bool{key: true}}
	router.applySignalResultsToContext(ctx, signals)
	signals.SignalErrorMatches[key] = false
	require.True(t, ctx.VSRSignalErrorMatches[key])
	require.False(t, ctx.JailbreakScoreAvailable)
}
