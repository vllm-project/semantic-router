package extproc

import (
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestBuildResponseHeaderMutation_IncludesExtendedMatchedSignalHeaders(t *testing.T) {
	// Matched signal headers are demoted to the debug surface (#2205), so the
	// request opts into x-vsr-debug to exercise them.
	ctx := &RequestContext{
		Headers:              map[string]string{headers.VSRDebug: "true"},
		VSRMatchedKeywords:   []string{"keyword:math"},
		VSRMatchedEmbeddings: []string{"embedding:math"},
		VSRMatchedDomains:    []string{"domain:math"},
		VSRMatchedContext:    []string{"context:long"},
		VSRMatchedComplexity: []string{"complexity:hard"},
		VSRMatchedModality:   []string{"AR"},
		VSRMatchedAuthz:      []string{"authz:premium"},
		VSRMatchedJailbreak:  []string{"jailbreak:block"},
		VSRMatchedPII:        []string{"pii:email"},
		VSRMatchedReask:      []string{"likely_dissatisfied"},
		VSRMatchedEvent:      []string{"critical_payment_event"},
		VSRMatchedProjection: []string{"balance_reasoning"},
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)

	assert.Equal(t, "complexity:hard", headerMap[headers.VSRMatchedComplexity])
	assert.Equal(t, "AR", headerMap[headers.VSRMatchedModality])
	assert.Equal(t, "authz:premium", headerMap[headers.VSRMatchedAuthz])
	assert.Equal(t, "jailbreak:block", headerMap[headers.VSRMatchedJailbreak])
	assert.Equal(t, "pii:email", headerMap[headers.VSRMatchedPII])
	assert.Equal(t, "likely_dissatisfied", headerMap[headers.VSRMatchedReask])
	assert.Equal(t, "critical_payment_event", headerMap[headers.VSRMatchedEvent])
	assert.Equal(t, "balance_reasoning", headerMap[headers.VSRMatchedProjection])
}

func TestBuildResponseHeaderMutation_EmitsZeroDecisionConfidence(t *testing.T) {
	// The decision/confidence/model facts ride on the default surface; the
	// session phase and context-token count are demoted to x-vsr-debug (#2205),
	// so the request opts into debug to exercise all five fields together.
	ctx := &RequestContext{
		Headers:                       map[string]string{headers.VSRDebug: "true"},
		VSRSelectedDecisionName:       "agentic_routing",
		VSRSelectedDecisionConfidence: 0,
		VSRSelectedModel:              "qwen-small",
		VSRContextTokenCount:          42,
		VSRLearningPolicies: testLearningPolicies(
			replayTestProtectionPolicyWithTrace(&selection.SessionPolicyTrace{
				Phase: "provider_state",
			}),
		),
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)

	assert.Equal(t, "agentic_routing", headerMap[headers.VSRSelectedDecision])
	assert.Equal(t, "0.0000", headerMap[headers.VSRSelectedConfidence])
	assert.Equal(t, "qwen-small", headerMap[headers.VSRSelectedModel])
	assert.Equal(t, "provider_state", headerMap[headers.VSRSessionPhase])
	assert.Equal(t, "42", headerMap[headers.VSRContextTokenCount])
}

func TestBuildResponseHeaderMutation_SameProtocolOmitsMarkers(t *testing.T) {
	// Default ctx: client and upstream both normalize to "openai" — no
	// cross-protocol translation, so the protocol markers are omitted (#2206).
	ctx := &RequestContext{}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.NotContains(t, headerMap, headers.VSRClientProtocol)
	assert.NotContains(t, headerMap, headers.VSRUpstreamProtocol)
}

func TestBuildResponseHeaderMutation_SameProtocolAnthropicOmitsMarkers(t *testing.T) {
	// Anthropic in and anthropic out is still same-protocol (no translation),
	// so the markers are omitted (#2206).
	ctx := &RequestContext{
		ClientProtocol: config.ClientProtocolAnthropic,
		APIFormat:      config.APIFormatAnthropic,
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.NotContains(t, headerMap, headers.VSRClientProtocol)
	assert.NotContains(t, headerMap, headers.VSRUpstreamProtocol)
}

func TestBuildResponseHeaderMutation_CrossProtocolEmitsMarkers(t *testing.T) {
	// Anthropic client translated to an OpenAI upstream — cross-protocol, so
	// both markers are emitted (#2206).
	ctx := &RequestContext{
		ClientProtocol: config.ClientProtocolAnthropic,
		APIFormat:      "openai",
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "anthropic", headerMap[headers.VSRClientProtocol])
	assert.Equal(t, "openai", headerMap[headers.VSRUpstreamProtocol])
}

func TestBuildResponseHeaderMutation_DebugHeaderEmitsMarkersOnSameProtocol(t *testing.T) {
	// Same-protocol (both openai), but the request opted into debug headers via
	// x-vsr-debug, so the markers are emitted anyway (#2216).
	ctx := &RequestContext{
		Headers: map[string]string{headers.VSRDebug: "true"},
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "openai", headerMap[headers.VSRClientProtocol])
	assert.Equal(t, "openai", headerMap[headers.VSRUpstreamProtocol])
}

func TestBuildResponseHeaderMutation_ProtocolMarkersEmittedOnFailedResponse(t *testing.T) {
	ctx := &RequestContext{
		ClientProtocol: config.ClientProtocolAnthropic,
	}

	mutation := buildResponseHeaderMutation(ctx, false)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "anthropic", headerMap[headers.VSRClientProtocol])
	assert.Equal(t, "openai", headerMap[headers.VSRUpstreamProtocol])

	// Existing pre-PR3 markers stay gated by isSuccessful.
	assert.NotContains(t, headerMap, headers.VSRSelectedCategory)
}

func TestBuildResponseHeaderMutation_CacheHitSkipsNewHeaders(t *testing.T) {
	ctx := &RequestContext{
		ClientProtocol: config.ClientProtocolAnthropic,
		VSRCacheHit:    true,
		IRExtensions: &ir.IRExtensions{
			Warnings: []ir.Warning{{Field: "top_k", Reason: ir.ReasonDropped, Severity: ir.WarningSeverityLossy}},
		},
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	assert.Nil(t, mutation, "cache hit should produce no header mutation")
}

func TestBuildResponseHeaderMutation_NilContextReturnsNil(t *testing.T) {
	assert.Nil(t, buildResponseHeaderMutation(nil, true))
}

func TestBuildResponseHeaderMutation_EmitsRetentionDirectiveHeaders(t *testing.T) {
	ctx := &RequestContext{
		Headers:                 map[string]string{headers.VSRDebug: "true"},
		VSRSelectedDecisionName: "agentic_routing",
		VSRSelectedModel:        "qwen-small",
		EmittedRetention: &config.RetentionDirective{
			TTLTurns:              intPtr(3),
			KeepCurrentModel:      boolPtr(true),
			PreferPrefixRetention: boolPtr(true),
		},
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "3", headerMap[headers.VSRRetentionTTLTurns])
	assert.Equal(t, "true", headerMap[headers.VSRRetentionKeepCurrentModel])
	assert.Equal(t, "true", headerMap[headers.VSRRetentionPreferPrefix])
}

func TestBuildResponseHeaderMutation_OmitsUnsetRetentionHeaders(t *testing.T) {
	// Tri-state: only fields the directive explicitly set are emitted.
	ctx := &RequestContext{
		Headers:                 map[string]string{headers.VSRDebug: "true"},
		VSRSelectedDecisionName: "agentic_routing",
		VSRSelectedModel:        "qwen-small",
		EmittedRetention:        &config.RetentionDirective{PreferPrefixRetention: boolPtr(true)},
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "true", headerMap[headers.VSRRetentionPreferPrefix])
	assert.NotContains(t, headerMap, headers.VSRRetentionTTLTurns)
	assert.NotContains(t, headerMap, headers.VSRRetentionKeepCurrentModel)
	assert.NotContains(t, headerMap, headers.VSRRetentionDrop)
}

func TestBuildResponseHeaderMutation_EmitsExplicitZeroTTLTurns(t *testing.T) {
	// Tri-state: an explicitly set ttl_turns: 0 is still an explicit field, so
	// it must be emitted (the validator permits 0 as a no-op; only negatives
	// are rejected). Regression guard against the old "*TTLTurns > 0" gate that
	// silently dropped the header for an explicit zero.
	ctx := &RequestContext{
		Headers:                 map[string]string{headers.VSRDebug: "true"},
		VSRSelectedDecisionName: "agentic_routing",
		VSRSelectedModel:        "qwen-small",
		EmittedRetention:        &config.RetentionDirective{TTLTurns: intPtr(0)},
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "0", headerMap[headers.VSRRetentionTTLTurns])
}

func TestBuildResponseHeaderMutation_EmitsExplicitFalseRetentionBools(t *testing.T) {
	// Tri-state companion to the explicit-zero ttl_turns guard: an explicitly
	// set bool field of value false must still be emitted as "false" (not
	// suppressed), so the wire contract is symmetric across every field.
	ctx := &RequestContext{
		Headers:                 map[string]string{headers.VSRDebug: "true"},
		VSRSelectedDecisionName: "agentic_routing",
		VSRSelectedModel:        "qwen-small",
		EmittedRetention: &config.RetentionDirective{
			Drop:             boolPtr(false),
			KeepCurrentModel: boolPtr(false),
		},
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "false", headerMap[headers.VSRRetentionDrop])
	assert.Equal(t, "false", headerMap[headers.VSRRetentionKeepCurrentModel])
	// PreferPrefixRetention was left unset -> its header must be omitted.
	assert.NotContains(t, headerMap, headers.VSRRetentionPreferPrefix)
}

func TestBuildResponseHeaderMutation_CacheHitSkipsRetentionHeaders(t *testing.T) {
	ctx := &RequestContext{
		VSRCacheHit:      true,
		EmittedRetention: &config.RetentionDirective{PreferPrefixRetention: boolPtr(true)},
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	assert.Nil(t, mutation, "cache hit must not emit retention headers")
}

func TestAddLossinessWarnings_EmptyProducesNoHeader(t *testing.T) {
	ctx := &RequestContext{ClientProtocol: config.ClientProtocolAnthropic}
	builder := newResponseHeaderMutationBuilder()

	builder.addLossinessWarnings(ctx, nil)
	builder.addLossinessWarnings(ctx, []ir.Warning{})

	mutation := builder.mutation()
	assert.Nil(t, mutation, "no warnings should produce no header")
}

func TestAddLossinessWarnings_SingleEntry(t *testing.T) {
	ctx := &RequestContext{ClientProtocol: config.ClientProtocolAnthropic}
	builder := newResponseHeaderMutationBuilder()

	builder.addLossinessWarnings(ctx, []ir.Warning{{
		Field:    "top_k",
		Reason:   ir.ReasonTopKDropOnOpenAIBackend,
		Severity: ir.WarningSeverityLossy,
	}})

	headerMap := mutationToMap(builder.setHeaders)
	assert.Equal(t,
		"lossy;top_k_drop_on_openai_backend;top_k",
		headerMap[headers.VSRProtocolWarnings],
	)
}

func TestAddLossinessWarnings_MultipleEntriesCommaSeparated(t *testing.T) {
	ctx := &RequestContext{ClientProtocol: config.ClientProtocolAnthropic}
	builder := newResponseHeaderMutationBuilder()

	builder.addLossinessWarnings(ctx, []ir.Warning{
		{Field: "messages[0].content[2]", Reason: ir.ReasonUnsupportedBlockType, Severity: ir.WarningSeverityLossy},
		{Field: "top_k", Reason: ir.ReasonDropped, Severity: ir.WarningSeverityLossy},
		{Field: "tool_choice.disable_parallel_tool_use", Reason: ir.ReasonCoercedString, Severity: ir.WarningSeverityInfo},
	})

	headerMap := mutationToMap(builder.setHeaders)
	assert.Equal(t,
		"lossy;unsupported_block_type;messages[0].content[2],"+
			"lossy;dropped;top_k,"+
			"info;coerced_string;tool_choice.disable_parallel_tool_use",
		headerMap[headers.VSRProtocolWarnings],
	)
}

func TestAddLossinessWarnings_TruncationAppendsTrailer(t *testing.T) {
	ctx := &RequestContext{ClientProtocol: config.ClientProtocolAnthropic}
	builder := newResponseHeaderMutationBuilder()

	const total = 500
	warnings := make([]ir.Warning, total)
	for i := range warnings {
		warnings[i] = ir.Warning{
			Field:    "messages[0].content[" + strings.Repeat("x", 16) + "]",
			Reason:   ir.ReasonUnsupportedBlockType,
			Severity: ir.WarningSeverityLossy,
		}
	}

	builder.addLossinessWarnings(ctx, warnings)

	headerMap := mutationToMap(builder.setHeaders)
	got := headerMap[headers.VSRProtocolWarnings]

	assert.NotEmpty(t, got)
	// The encoded list lives close to 4 KB; the trailer itself adds a
	// few dozen bytes more.
	assert.LessOrEqual(t, len(got), lossinessHeaderSizeLimit+128)
	assert.Contains(t, got, "error;warnings_truncated;count=")
}

func TestAddLossinessWarnings_SanitizesSeparatorsInFieldPaths(t *testing.T) {
	ctx := &RequestContext{ClientProtocol: config.ClientProtocolAnthropic}
	builder := newResponseHeaderMutationBuilder()

	builder.addLossinessWarnings(ctx, []ir.Warning{{
		Field:    "weird;field,name",
		Reason:   ir.WarningReason("dropped;extra,bits"),
		Severity: ir.WarningSeverityLossy,
	}})

	headerMap := mutationToMap(builder.setHeaders)
	assert.Equal(t,
		"lossy;dropped%3Bextra%2Cbits;weird%3Bfield%2Cname",
		headerMap[headers.VSRProtocolWarnings],
	)
}

func TestAddLossinessWarnings_SanitizesCRLFInFieldPaths(t *testing.T) {
	ctx := &RequestContext{ClientProtocol: config.ClientProtocolAnthropic}
	builder := newResponseHeaderMutationBuilder()

	builder.addLossinessWarnings(ctx, []ir.Warning{{
		Field:    "field\r\nx-injected: pwned",
		Reason:   ir.WarningReason("rea\nson"),
		Severity: ir.WarningSeverityLossy,
	}})

	headerMap := mutationToMap(builder.setHeaders)
	got := headerMap[headers.VSRProtocolWarnings]
	assert.NotContains(t, got, "\r")
	assert.NotContains(t, got, "\n")
	assert.Contains(t, got, "%0D%0A")
	assert.Contains(t, got, "rea%0Ason")
}

func TestBuildResponseHeaderMutation_EmitsWarningsAlongsideStandardHeaders(t *testing.T) {
	ctx := &RequestContext{
		ClientProtocol: config.ClientProtocolAnthropic,
		APIFormat:      "openai",
		IRExtensions: &ir.IRExtensions{
			Warnings: []ir.Warning{{
				Field:    "top_k",
				Reason:   ir.ReasonTopKDropOnOpenAIBackend,
				Severity: ir.WarningSeverityLossy,
			}},
		},
		VSRSelectedDecisionName: "math_decision",
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "anthropic", headerMap[headers.VSRClientProtocol])
	assert.Equal(t, "openai", headerMap[headers.VSRUpstreamProtocol])
	assert.Equal(t,
		"lossy;top_k_drop_on_openai_backend;top_k",
		headerMap[headers.VSRProtocolWarnings],
	)
	// The warnings ride alongside the default-surface routing facts (#2205).
	assert.Equal(t, "math_decision", headerMap[headers.VSRSelectedDecision])
}

func TestBuildResponseHeaderMutation_EmitsSplitRouterLearningHeaders(t *testing.T) {
	// Router Learning headers are part of the intermediate decision detail and
	// are demoted to the x-vsr-debug surface (#2205), so the request opts into
	// debug to exercise them.
	ctx := &RequestContext{
		Headers: map[string]string{headers.VSRDebug: "true"},
		VSRLearningPolicy: headerTestLearningPolicy(
			routerLearningMethodProtection,
			"apply",
			routerLearningActionAllowSwitch,
			"switch_allowed",
			"conversation",
		),
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "protection", headerMap[headers.VSRLearningMethods])
	assert.Equal(t, "protection=allow_switch", headerMap[headers.VSRLearningActions])
	assert.Equal(t, "protection=conversation", headerMap[headers.VSRLearningScopes])
	assert.Equal(t, "protection=switch_allowed", headerMap[headers.VSRLearningReasons])
}

func TestBuildResponseHeaderMutation_EmitsBaselineLearningHeaders(t *testing.T) {
	// Demoted to the x-vsr-debug surface (#2205), so opt into debug.
	ctx := &RequestContext{
		Headers: map[string]string{headers.VSRDebug: "true"},
		VSRLearningPolicy: headerTestLearningPolicy(
			routerLearningMethodProtection,
			"apply",
			routerLearningActionEstablishBaseline,
			"new_conversation",
			"conversation",
		),
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "protection", headerMap[headers.VSRLearningMethods])
	assert.Equal(t, "protection=establish_baseline", headerMap[headers.VSRLearningActions])
	assert.Equal(t, "protection=conversation", headerMap[headers.VSRLearningScopes])
	assert.Equal(t, "protection=new_conversation", headerMap[headers.VSRLearningReasons])
}

func TestBuildResponseHeaderMutation_EmitsNonApplyLearningActionWithoutModeHeader(t *testing.T) {
	// Demoted to the x-vsr-debug surface (#2205), so opt into debug.
	ctx := &RequestContext{
		Headers: map[string]string{headers.VSRDebug: "true"},
		VSRLearningPolicy: headerTestLearningPolicy(
			routerLearningMethodProtection,
			"observe",
			routerLearningActionHoldCurrent,
			"tool_or_protocol_state",
			"session",
		),
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "protection=hold_current", headerMap[headers.VSRLearningActions])
	assert.Equal(t, "protection=session", headerMap[headers.VSRLearningScopes])
	assert.NotContains(t, headerMap, "x-vsr-learning-modes")
}

func TestBuildResponseHeaderMutation_EmitsMultipleLearningPolicies(t *testing.T) {
	// Demoted to the x-vsr-debug surface (#2205), so opt into debug.
	ctx := &RequestContext{
		Headers: map[string]string{headers.VSRDebug: "true"},
		VSRLearningPolicies: testLearningPolicies(
			*headerTestLearningPolicy(
				routerLearningMethodProtection,
				"apply",
				routerLearningActionHoldCurrent,
				"cache_cost_high",
				"conversation",
			),
			*headerTestLearningPolicy(
				routerLearningMethodAdaptation,
				"observe",
				routerLearningActionObserve,
				"observe_only",
				"",
			),
		),
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "adaptation,protection", headerMap[headers.VSRLearningMethods])
	assert.Equal(t, "adaptation=observe,protection=hold_current", headerMap[headers.VSRLearningActions])
	assert.Equal(t, "protection=conversation", headerMap[headers.VSRLearningScopes])
	assert.Equal(t, "adaptation=observe_only,protection=cache_cost_high", headerMap[headers.VSRLearningReasons])
	assert.NotContains(t, headerMap, "x-vsr-learning-modes")
}

func TestBuildResponseHeaderMutation_OmitsEmptyLearningPolicies(t *testing.T) {
	ctx := &RequestContext{
		VSRLearningPolicies: testLearningPolicies(
			routerLearningPolicy{},
			routerLearningPolicy{},
		),
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.NotContains(t, headerMap, headers.VSRLearningMethods)
	assert.NotContains(t, headerMap, headers.VSRLearningActions)
	assert.NotContains(t, headerMap, headers.VSRLearningScopes)
	assert.NotContains(t, headerMap, headers.VSRLearningReasons)
}

func headerTestLearningPolicy(
	method routerLearningMethod,
	mode string,
	action routerLearningAction,
	reason string,
	scope string,
) *routerLearningPolicy {
	policy := newRouterLearningPolicy(method)
	policy.Mode = mode
	policy.Action = action
	policy.Reason = reason
	policy.Scope = scope
	return &policy
}

func TestNormalizeProtocol(t *testing.T) {
	assert.Equal(t, "openai", normalizeProtocol(""))
	assert.Equal(t, "openai", normalizeProtocol("  "))
	assert.Equal(t, "anthropic", normalizeProtocol("anthropic"))
	assert.Equal(t, "anthropic", normalizeProtocol(" anthropic "))
}

func mutationToMap(setHeaders []*core.HeaderValueOption) map[string]string {
	headerMap := make(map[string]string, len(setHeaders))
	for _, h := range setHeaders {
		headerMap[h.Header.Key] = string(h.Header.RawValue)
	}
	return headerMap
}
