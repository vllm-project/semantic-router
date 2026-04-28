package extproc

import (
	"strings"
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
		"projection:m",
	}, collectMatchedSignalRules(signals))
}

// --- Tests for max_evaluation_chars (#1454) ---

func routerWithEvalLimit(limit int) *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			InlineModels: config.InlineModels{
				PromptCompression: config.PromptCompressionConfig{
					MaxEvaluationChars: limit,
				},
			},
		},
	}
}

func TestApplyMaxEvaluationCharsNoLimit(t *testing.T) {
	r := routerWithEvalLimit(0)
	text := strings.Repeat("a", 50000)
	got := r.applyMaxEvaluationChars(text)
	assert.Equal(t, 50000, len(got), "limit=0 should not truncate")
}

func TestApplyMaxEvaluationCharsNegativeNoLimit(t *testing.T) {
	r := routerWithEvalLimit(-1)
	text := strings.Repeat("b", 20000)
	got := r.applyMaxEvaluationChars(text)
	assert.Equal(t, 20000, len(got), "limit=-1 should not truncate")
}

func TestApplyMaxEvaluationCharsTruncates(t *testing.T) {
	const limit = 8192
	r := routerWithEvalLimit(limit)
	text := strings.Repeat("x", 25000)
	got := r.applyMaxEvaluationChars(text)
	assert.Equal(t, limit, len(got), "giant prompt must be capped to limit")
}

func TestApplyMaxEvaluationCharsBelowLimitUnchanged(t *testing.T) {
	r := routerWithEvalLimit(8192)
	got := r.applyMaxEvaluationChars("short prompt")
	assert.Equal(t, "short prompt", got)
}

func TestApplyMaxEvaluationCharsExactLimit(t *testing.T) {
	const limit = 100
	r := routerWithEvalLimit(limit)
	text := strings.Repeat("z", limit)
	got := r.applyMaxEvaluationChars(text)
	assert.Equal(t, limit, len(got), "text at exactly the limit should not be truncated")
}

func TestPrepareSignalEvaluationInputRespectsEvalLimit(t *testing.T) {
	const limit = 500
	r := routerWithEvalLimit(limit)
	input := r.prepareSignalEvaluationInput(signalConversationHistory{
		currentUserMessage: strings.Repeat("q", 5000),
	})
	require.Equal(t, limit, len(input.evaluationText), "evaluationText must be capped at max_evaluation_chars")
	assert.LessOrEqual(t, len(input.compressedText), limit, "compressedText must not exceed cap")
}
