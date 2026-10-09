package classification

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func TestDecisionPIIToolResultSourceAndEvidence(t *testing.T) {
	for _, tc := range []struct {
		name       string
		tool       string
		incomplete bool
		detected   bool
	}{
		{name: "clean tool ignores private prompt", tool: "clean tool"},
		{name: "private tool is detected", tool: "private tool", detected: true},
		{name: "incomplete extraction stays unknown", tool: "clean tool", incomplete: true, detected: true},
		{name: "missing extracted text stays unknown", incomplete: true, detected: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			decider := judgmentDeciderFunc(func(_ context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
				calls.Add(1)
				require.Equal(t, tc.tool, request.State)
				p := .01
				if request.State == "private tool" {
					p = .95
				}
				answers := map[string]modelservice.Answer{}
				for _, q := range request.Questions {
					answers[q.ID] = modelservice.Answer{Type: "noul", Noul: p, InputCoverage: "complete"}
				}
				return modelservice.Response{Answers: answers}, nil
			})
			backend := testSourceDecisionPIIBackend(t, decider)
			cfg := &config.RouterConfig{}
			cfg.PIIRules = []config.PIIRule{{Name: "tool", Source: config.PIISourceToolResult, Threshold: .7, IncludeHistory: true}}
			cfg.PIIModel.OnError = config.OnErrorBlock
			classifier := &Classifier{Config: cfg, piiInference: backend}
			result := &SignalResults{Metrics: &SignalMetricsCollection{}}
			classifier.evaluatePIISignalWithToolResults(t.Context(), result, &sync.Mutex{}, "private prompt", []string{"private history"}, []string{tc.tool}, tc.incomplete)
			require.Equal(t, tc.detected, result.PIIDetected)
			require.False(t, result.PIIContentVerified, "unscanned prompt cannot be certified")
			if tc.tool != "" {
				require.EqualValues(t, 1, calls.Load())
				require.Len(t, result.PIIEvidence, 1)
				require.Equal(t, !tc.detected, result.PIIEvidence[0].CoversClean("request", tc.tool))
				require.False(t, result.PIIEvidence[0].CoversClean("request", "private prompt"))
			} else {
				require.Zero(t, calls.Load())
			}
			if tc.incomplete {
				require.Equal(t, piiEvaluationIncompleteCode, result.SignalErrors["pii:tool"])
			}
		})
	}
}

func TestDecisionPIIToolResultBudgetDoesNotConsumeLegacyScan(t *testing.T) {
	var calls atomic.Int32
	backend := testSourceDecisionPIIBackend(t, judgmentDeciderFunc(func(_ context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
		calls.Add(1)
		answers := map[string]modelservice.Answer{}
		for _, q := range request.Questions {
			answers[q.ID] = modelservice.Answer{Type: "noul", Noul: .01, InputCoverage: "complete"}
		}
		return modelservice.Response{Answers: answers}, nil
	}))
	cfg := &config.RouterConfig{}
	cfg.PIIRules = []config.PIIRule{{Name: "tool", Source: config.PIISourceToolResult, Threshold: .7}, {Name: "legacy", Threshold: .7}}
	cfg.PIIModel.OnError, cfg.PIIModel.OnUnscanned = config.OnErrorAllow, config.OnErrorAllow
	classifier := &Classifier{Config: cfg, piiInference: backend}
	tools := make([]string, maxPIIToolResultInferenceCalls+1)
	for i := range tools {
		tools[i] = fmt.Sprintf("tool item %d", i)
	}
	result := &SignalResults{Metrics: &SignalMetricsCollection{}}
	classifier.evaluatePIISignalWithToolResults(t.Context(), result, &sync.Mutex{}, tools[len(tools)-1], nil, tools, false)
	require.EqualValues(t, maxPIIToolResultInferenceCalls+1, calls.Load())
	require.NotEmpty(t, result.SignalErrors["pii:tool"])
	require.Empty(t, result.SignalErrors["pii:legacy"])
	require.False(t, result.PIIContentVerified)
}

func testSourceDecisionPIIBackend(t *testing.T, decider modelservice.Decider) *decisionPIIBackend {
	t.Helper()
	judgment := testJudgment(t, "pii_presence", decider)
	categories, _ := modelservice.BuiltinTask("pii_categories")
	plan, err := modelservice.CompileTask(categories, categories.Question, judgment.card)
	require.NoError(t, err)
	return &decisionPIIBackend{judgment: judgment, categories: plan}
}
