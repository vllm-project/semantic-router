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
)

func TestBuildResponseHeaderMutation_IncludesExtendedMatchedSignalHeaders(t *testing.T) {
	ctx := &RequestContext{
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
	assert.Equal(t, "balance_reasoning", headerMap[headers.VSRMatchedProjection])
}

func TestBuildResponseHeaderMutation_EmitsZeroDecisionConfidence(t *testing.T) {
	ctx := &RequestContext{
		VSRSelectedDecisionName:       "agentic_routing",
		VSRSelectedDecisionConfidence: 0,
		VSRSelectedModel:              "qwen-small",
		VSRContextTokenCount:          42,
		VSRSessionPolicy: map[string]interface{}{
			"phase": "provider_state",
		},
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

func TestBuildResponseHeaderMutation_ProtocolMarkersDefaultToOpenAI(t *testing.T) {
	ctx := &RequestContext{}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "openai", headerMap[headers.VSRInboundProtocol])
	assert.Equal(t, "openai", headerMap[headers.VSROutboundProtocol])
}

func TestBuildResponseHeaderMutation_ProtocolMarkersReflectAnthropicIngress(t *testing.T) {
	ctx := &RequestContext{
		ClientProtocol: config.ClientProtocolAnthropic,
		APIFormat:      config.APIFormatAnthropic,
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "anthropic", headerMap[headers.VSRInboundProtocol])
	assert.Equal(t, "anthropic", headerMap[headers.VSROutboundProtocol])
}

func TestBuildResponseHeaderMutation_ProtocolMarkersEmittedOnFailedResponse(t *testing.T) {
	ctx := &RequestContext{
		ClientProtocol: config.ClientProtocolAnthropic,
	}

	mutation := buildResponseHeaderMutation(ctx, false)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "anthropic", headerMap[headers.VSRInboundProtocol])
	assert.Equal(t, "openai", headerMap[headers.VSROutboundProtocol])

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
		headerMap[headers.VSRLossinessWarnings],
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
		headerMap[headers.VSRLossinessWarnings],
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
	got := headerMap[headers.VSRLossinessWarnings]

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
		headerMap[headers.VSRLossinessWarnings],
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
	got := headerMap[headers.VSRLossinessWarnings]
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
		VSRSelectedCategory: "math",
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := mutationToMap(mutation.SetHeaders)
	assert.Equal(t, "anthropic", headerMap[headers.VSRInboundProtocol])
	assert.Equal(t, "openai", headerMap[headers.VSROutboundProtocol])
	assert.Equal(t,
		"lossy;top_k_drop_on_openai_backend;top_k",
		headerMap[headers.VSRLossinessWarnings],
	)
	assert.Equal(t, "math", headerMap[headers.VSRSelectedCategory])
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
