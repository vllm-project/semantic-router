package extproc

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

// ledgerHops charges each hop to the request's call ledger, as the upstream
// layer does before every physical exchange.
type ledgerHops struct{ inner graph.Caller }

func (h ledgerHops) Call(ctx context.Context, req *graph.HopRequest) (*graph.HopResponse, error) {
	if err := budget.Consume(ctx); err != nil {
		return nil, err
	}
	return h.inner.Call(ctx, req)
}

// meteredAnswerChecks is a decision model behind a transport that charges
// every exchange to the request's call ledger, as a model runtime client
// does, and finds no answer good enough.
type meteredAnswerChecks struct{ asked atomic.Int64 }

func (d *meteredAnswerChecks) Card(context.Context, string) (modelservice.ModelCard, error) {
	return modelservice.ModelCard{Surfaces: []string{"decisions"}, Presets: []string{"answer_check", "refusal"}}, nil
}

func (d *meteredAnswerChecks) Decide(ctx context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	if err := budget.Consume(ctx); err != nil {
		return modelservice.Response{}, err
	}
	d.asked.Add(1)
	answers := map[string]modelservice.Answer{}
	for _, question := range request.Questions {
		answers[question.ID] = modelservice.Answer{Type: "noul", Noul: 0.1, InputCoverage: "complete"}
	}
	return modelservice.Response{Answers: answers}, nil
}

// An answer check counts against the escalation's deadline but not its
// max_calls, so max_calls: 2 covers both candidates of a two-model escalation.
func TestEscalateAnswerChecksDoNotSpendMaxCalls(t *testing.T) {
	var candidateCalls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		candidateCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":"an answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`))
	}))
	defer server.Close()
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.5
providers:
  defaults: {model: small}
  models:
    - name: small
      backend_refs: [{name: small, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
    - name: large
      backend_refs: [{name: large, endpoint: "127.0.0.1:8001", protocol: http, weight: 1}]
recipes:
  - name: default
    decisions:
      - name: verified
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: small}, {model: large}]
        algorithm:
          type: escalate
          budget: {deadline: 30s, max_calls: 2}
          escalate:
            accept:
              systemone: {gte: 0.9}
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-4B
`))
	require.NoError(t, err)
	decider := &meteredAnswerChecks{}
	cards, err := prepareDecisionSelectionCards(cfg, decider)
	require.NoError(t, err)
	router := &OpenAIRouter{Config: cfg, decisionDecider: decider, decisionCards: cards, hopCaller: ledgerHops{inner: directLooperHops(server.URL)}}
	ctx := &RequestContext{TraceContext: context.Background(), Headers: map[string]string{}}
	request, rejected := router.prepareProtocolRequest([]byte(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"What is the capital of Australia?"}]}`), ctx)
	require.Nil(t, rejected)

	response, err := router.handleLooperExecution(context.Background(), request, cfg.GetDecisionByName("verified"), ctx)
	require.NoError(t, err)
	require.EqualValues(t, 200, response.GetImmediateResponse().GetStatus().GetCode(), string(response.GetImmediateResponse().GetBody()))
	require.EqualValues(t, 2, candidateCalls.Load())
	require.EqualValues(t, 2, decider.asked.Load())
}
