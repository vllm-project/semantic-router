package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestAttachDecisionDiagnosticsOmitsMetadataWhenPluginDisabled(t *testing.T) {
	ctx := decisionDiagnosticsTestContext(false)
	response := newContinueRequestBodyResponse()

	attachDecisionDiagnostics(response, ctx)

	require.Nil(t, response.DynamicMetadata)
}

func TestAttachDecisionDiagnosticsEmitsSelectedDecisionSignalsOnly(t *testing.T) {
	ctx := decisionDiagnosticsTestContext(true)
	response := newContinueRequestBodyResponse()

	attachDecisionDiagnostics(response, ctx)

	raw := decisionDiagnosticsJSON(t, response)
	var got decisionDiagnosticsPayload
	require.NoError(t, json.Unmarshal([]byte(raw), &got))
	require.Equal(t, decisionDiagnosticsSchemaVersion, got.SchemaVersion)
	require.Equal(t, "general-route", got.Decision)
	require.Equal(t, "general", got.Category)
	require.Equal(t, "model-b", got.SelectedModel)
	require.Equal(t, "multi_factor", got.SelectionAlgorithm)
	require.Equal(t, "single", got.SelectionMethod)
	require.Equal(t, 0.91, got.DecisionConfidence)
	require.Equal(t, []decisionDiagnosticSignal{
		{Key: "authz:general-users", Type: "authz", Executed: true, Matched: true, Confidence: float64Ptr(1)},
		{Key: "keyword:routing", Type: "keyword", Executed: true, Matched: true, Confidence: float64Ptr(0.88)},
		{Key: "projection:complex-request", Type: "projection", Executed: true, Matched: true, Confidence: float64Ptr(0.73)},
	}, got.Signals)
	require.Len(t, got.Projections, 1)
	require.Equal(t, "complex-request", got.Projections[0].Name)
	require.NotContains(t, raw, "other-profile")
	require.NotContains(t, raw, "unrelated-secret-signal")
}

func TestBuildDecisionDiagnosticsAppliesStableCardinalityAndTextBounds(t *testing.T) {
	ctx := decisionDiagnosticsTestContext(true)
	ctx.VSRSelectedDecision.Rules = config.RuleCombination{
		Operator: "OR",
		Conditions: []config.RuleNode{
			{Type: "keyword", Name: strings.Repeat("a", 40)},
			{Type: "keyword", Name: "second"},
		},
	}
	cfg := &config.DecisionDiagnosticsPluginConfig{
		Enabled: true, MaxSignals: 1, MaxProjections: 1,
		MaxTextRunes: 8, MaxPayloadBytes: 4096,
	}

	got, ok := buildDecisionDiagnosticsPayload(ctx, cfg)

	require.True(t, ok)
	require.True(t, got.Truncated)
	require.Len(t, got.Signals, 1)
	require.LessOrEqual(t, len([]rune(got.Signals[0].Key)), 8)
}

func TestDecisionDiagnosticsNeverIncludesRequestContentOrCredentials(t *testing.T) {
	ctx := decisionDiagnosticsTestContext(true)
	ctx.OriginalRequestBody = []byte(`{"messages":[{"content":"private prompt"}],"authorization":"Bearer secret-token"}`)
	ctx.UserContent = "private prompt"
	response := &ext_proc.ProcessingResponse{}

	attachDecisionDiagnostics(response, ctx)
	raw := decisionDiagnosticsJSON(t, response)

	require.NotContains(t, raw, "private prompt")
	require.NotContains(t, raw, "secret-token")
	require.NotContains(t, strings.ToLower(raw), "authorization")
}

func decisionDiagnosticsTestContext(enabled bool) *RequestContext {
	return &RequestContext{
		VSRSelectedCategory:           "general",
		VSRSelectedDecisionName:       "general-route",
		VSRSelectedDecisionConfidence: 0.91,
		VSRSelectedModel:              "model-b",
		VSRSelectionMethod:            "single",
		VSRSelectedDecision: &config.Decision{
			Name:      "general-route",
			Algorithm: &config.AlgorithmConfig{Type: "multi_factor"},
			Rules: config.RuleCombination{Operator: "AND", Conditions: []config.RuleNode{
				{Type: "keyword", Name: "routing"},
				{Type: "authz", Name: "general-users"},
				{Type: "projection", Name: "complex-request"},
			}},
			Plugins: []config.DecisionPlugin{{
				Type: config.DecisionPluginDecisionDiagnostics,
				Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled": enabled,
				}),
			}},
		},
		VSRMatchedKeywords:   []string{"routing", "other-profile"},
		VSRMatchedAuthz:      []string{"general-users"},
		VSRMatchedProjection: []string{"complex-request"},
		VSRProjectionScores: map[string]float64{
			"complex-request": 0.73,
			"other-profile":   0.99,
		},
		VSRSignalConfidences: map[string]float64{
			"keyword:routing":                 0.88,
			"authz:general-users":             1,
			"projection:complex-request":      0.73,
			"keyword:unrelated-secret-signal": 0.99,
		},
	}
}

func decisionDiagnosticsJSON(t *testing.T, response *ext_proc.ProcessingResponse) string {
	t.Helper()
	require.NotNil(t, response.DynamicMetadata)
	namespace := response.DynamicMetadata.GetFields()[decisionDiagnosticsNamespace].GetStructValue()
	require.NotNil(t, namespace)
	return namespace.GetFields()[decisionDiagnosticsField].GetStringValue()
}

func float64Ptr(value float64) *float64 {
	return &value
}
