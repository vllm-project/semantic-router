package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
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

	metadata := decisionDiagnosticsStruct(t, response)
	require.Equal(t, "general-route", metadata.GetFields()["decision"].GetStringValue())
	require.Equal(t, float64(decisionDiagnosticsSchemaVersion), metadata.GetFields()["schemaVersion"].GetNumberValue())

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
	require.Equal(t, []string{
		"authz:general-users",
		"keyword:routing",
		"projection:complex-request",
	}, got.MatchedRules)
	require.Equal(t, []decisionDiagnosticSignal{
		{Key: "authz:general-users", Type: "authz", Name: "general-users", Executed: false, Matched: true, Confidence: float64Ptr(1)},
		{Key: "keyword:routing", Type: "keyword", Name: "routing", Executed: true, Matched: true, Confidence: float64Ptr(0.88)},
		{Key: "projection:complex-request", Type: "projection", Name: "complex-request", Executed: true, Matched: true, Confidence: float64Ptr(0.73)},
	}, got.Signals)
	require.Len(t, got.Projections, 1)
	require.Equal(t, "complex-request", got.Projections[0].Name)
	require.NotContains(t, raw, "other-profile")
	require.NotContains(t, raw, "unrelated-secret-signal")
}

func TestHandleRequestBodyFastResponseIncludesDecisionDiagnostics(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - name: local
          endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: model-a
      description: test model
  signals:
    keywords:
      - name: blocked
        operator: OR
        keywords: [blocked]
  decisions:
    - name: block-decision
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: blocked
      modelRefs:
        - model: model-a
      plugins:
        - type: fast_response
          configuration:
            message: blocked by policy
        - type: decision_diagnostics
          configuration:
            enabled: true
`))
	require.NoError(t, err)
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	require.NoError(t, err)
	router := &OpenAIRouter{Config: cfg, Classifier: classifier}
	ctx := &RequestContext{Headers: map[string]string{}}

	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(`{"model":"MoM","messages":[{"role":"user","content":"blocked"}]}`)},
	}, ctx)

	require.NoError(t, err)
	require.NotNil(t, response.GetImmediateResponse())
	metadata := decisionDiagnosticsStruct(t, response)
	require.Equal(t, "block-decision", metadata.GetFields()["decision"].GetStringValue())
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
		VSRMatchedDecisionRules: []string{
			"projection:complex-request",
			"keyword:routing",
			"authz:general-users",
		},
		VSRExecutedSignalTypes: map[string]bool{
			config.SignalTypeKeyword:    true,
			config.SignalTypeProjection: true,
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

func decisionDiagnosticsStruct(t *testing.T, response *ext_proc.ProcessingResponse) *structpb.Struct {
	t.Helper()
	require.NotNil(t, response.DynamicMetadata)
	namespace := response.DynamicMetadata.GetFields()[decisionDiagnosticsNamespace].GetStructValue()
	require.NotNil(t, namespace)
	metadata := namespace.GetFields()[decisionDiagnosticsField].GetStructValue()
	require.NotNil(t, metadata)
	return metadata
}

func decisionDiagnosticsJSON(t *testing.T, response *ext_proc.ProcessingResponse) string {
	t.Helper()
	raw, err := json.Marshal(decisionDiagnosticsStruct(t, response).AsMap())
	require.NoError(t, err)
	return string(raw)
}

func float64Ptr(value float64) *float64 {
	return &value
}
