package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func immediateResponseTestConfig() *config.RouterConfig {
	replay := false
	return &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			CandidateRequirements: &config.CandidateRequirements{
				Capabilities: config.CandidateCapabilitiesDeclared,
				Context:      config.CandidateContextKnownLimits,
			},
			DataPolicy: &config.RoutingDataPolicy{Replay: &replay},
			Decisions: []config.Decision{{
				Name: "immediate", Priority: 1,
				Plugins: []config.DecisionPlugin{{
					Type: config.DecisionPluginFastResponse,
					Configuration: config.MustStructuredPayload(config.FastResponsePluginConfig{
						Message: "Synthetic immediate response.",
					}),
				}},
			}},
		},
	}
}

func TestFastResponsePreviewNeedsNoCandidateOrCodec(t *testing.T) {
	cfg := immediateResponseTestConfig()
	router := &OpenAIRouter{
		Config: cfg,
		// No backend facts, selector, or registered codec can satisfy admission.
		ProtocolCodecs: &protocolcodec.Registry{},
	}
	for _, refs := range [][]config.ModelRef{nil, {{Model: "unavailable-model"}}} {
		decision := cfg.Decisions[0]
		decision.ModelRefs = refs
		input := services.EvalModelSelectionInput{
			Decision: &decision,
			Demand:   selection.CandidateDemand{InputTokens: 1_000_000},
		}
		result := router.SelectModelForEval(input)
		require.Equal(t, services.EvalSelectionNotRequired, result.Status)
		require.Equal(t, "fast_response", result.Method)
		require.Empty(t, result.SelectedModel)
		require.NotEmpty(t, result.Reason)

		// Removing the immediate action must restore ordinary strict admission.
		decision.Plugins = nil
		result = router.SelectModelForEval(input)
		require.Equal(t, services.EvalSelectionUnavailable, result.Status)
		require.Empty(t, result.SelectedModel)
	}
}

func TestFastResponseLiveDecisionAndPreviewNeedNoBackend(t *testing.T) {
	const requestedModel = "auto"
	cfg := immediateResponseTestConfig()
	cfg.DefaultModel = "immediate-unused-default"
	cfg.Decisions[0].ModelRefs = []config.ModelRef{{Model: "immediate-unused-candidate"}}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, classifier.Close()) })
	router := &OpenAIRouter{Config: cfg, Classifier: classifier}
	service := services.NewClassificationService(classifier, cfg)
	service.SetEvalModelSelector(router)

	preview, err := service.ClassifyIntentForEval(context.Background(), services.IntentRequest{
		Model: "auto", Text: "Synthetic routing contract.",
	})
	require.NoError(t, err)
	require.NotNil(t, preview.DecisionResult)
	require.Equal(t, "immediate", preview.DecisionResult.DecisionName)
	require.Equal(t, services.EvalSelectionNotRequired, preview.SelectionStatus)
	require.Equal(t, "fast_response", preview.SelectionMethod)
	require.Empty(t, preview.SelectedModel)
	encodedPreview, err := json.Marshal(preview)
	require.NoError(t, err)
	var previewFields map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(encodedPreview, &previewFields))
	require.NotContains(t, previewFields, "selected_model")

	request := &llmprotocol.Request{
		Model: requestedModel,
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentText, Text: "Synthetic routing contract.",
			}},
		}},
	}
	ctx := &RequestContext{
		RequestID: "immediate-contract", RequestModel: requestedModel,
		TraceContext: context.Background(), SemanticRequest: request,
		SourceFormat: llmprotocol.OpenAIChatV1,
	}
	countsBefore := map[string]float64{}
	for _, model := range []string{cfg.DefaultModel, cfg.Decisions[0].ModelRefs[0].Model, "unknown"} {
		countsBefore[model] = testutil.ToFloat64(metrics.ModelRequests.WithLabelValues(model))
	}
	_, response := router.runRequestPreRoutingStages(requestedModel, extractSemanticRequestSignals(request), ctx)
	require.NotNil(t, response)
	immediate := response.GetImmediateResponse()
	require.NotNil(t, immediate)
	require.EqualValues(t, 200, immediate.GetStatus().GetCode())
	require.Equal(t, "fast_response", ctx.VSRSelectionMethod)
	require.Empty(t, ctx.VSRSelectedModel)
	require.Zero(t, ctx.InflightToken)
	for model, count := range countsBefore {
		require.Equal(t, count, testutil.ToFloat64(metrics.ModelRequests.WithLabelValues(model)), model)
	}
	require.Empty(t, ctx.TargetFormat)
	require.Nil(t, ctx.VSREligibleModelRefs)
	require.Empty(t, ctx.RouterReplayID)
	var wire struct {
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(immediate.GetBody(), &wire))
	// Preserve the ingress synthetic-response identity, not a fictional backend.
	require.Equal(t, requestedModel, wire.Model)
	require.Len(t, wire.Choices, 1)
	require.Equal(t, "Synthetic immediate response.", wire.Choices[0].Message.Content)
}
