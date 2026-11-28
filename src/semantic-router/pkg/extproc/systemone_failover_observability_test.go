package extproc

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

const systemOneFailoverObservabilityConfig = `
version: v0.5
providers:
  defaults: {model: fast}
  models:
    - name: fast
      backend_refs: [{name: fast, endpoint: "BACKEND", protocol: http}]
    - name: deep
      backend_refs: [{name: deep, endpoint: "BACKEND", protocol: http}]
recipes:
  - name: default
    signals:
      systemone:
        - name: support_team
          deployment: failover-primary
          question: &team_question
            type: choice
            instructions: Which team should handle this ticket?
            choices:
              - {key: billing, description: Payments and refunds}
              - {key: engineering, description: Outages and bugs}
        - name: answer_team
          deployment: failover-primary
          stage: response
          input: {answer: true}
          question: *team_question
    decisions:
      - name: engineering
        priority: 200
        rules: {type: systemone, name: support_team, label: engineering}
        modelRefs: [{model: deep}]
        plugins:
          - type: response_guard
            configuration:
              rules:
                - when: {type: systemone, name: answer_team, label: engineering}
                  action: annotate
      - name: everything-else
        modelRefs: [{model: fast}]
global:
  services:
    router_replay: {store_backend: memory}
  plugins:
    - type: router_replay
      configuration: {enabled: true}
  model_catalog:
    deployments:
      failover-primary: &gateway
        provider: systemone
        schema: vllm-sr/4.0
        endpoint: PRIMARY/Unified/v1/systemone
        model: vLLM-SR-d3-27B
        auth: {header: Ocp-Apim-Subscription-Key, prefix: "", api_key_env: REMOTE_DECISION_TEST_KEY}
        fallbacks: [failover-backup]
      failover-backup:
        <<: *gateway
        endpoint: BACKUP/Unified/v1/systemone
        fallbacks: []
`

func TestSystemOneFailoverObservability(t *testing.T) {
	traces := routingTraceRecorder(t)
	t.Setenv("REMOTE_DECISION_TEST_KEY", "fake-gateway-key")
	primary, backup := &fakeSystemOneGateway{}, &fakeSystemOneGateway{}
	primary.choice.Store("unavailable")
	backup.choice.Store("engineering")
	primaryServer, backupServer := httptest.NewServer(primary), httptest.NewServer(backup)
	t.Cleanup(primaryServer.Close)
	t.Cleanup(backupServer.Close)
	backend := &servedModelBackend{}
	backendServer := httptest.NewServer(backend)
	t.Cleanup(backendServer.Close)
	document := strings.NewReplacer("PRIMARY", primaryServer.URL, "BACKUP", backupServer.URL,
		"BACKEND", strings.TrimPrefix(backendServer.URL, "http://")).Replace(systemOneFailoverObservabilityConfig)
	cfg, err := config.ParseYAMLBytes([]byte(document))
	require.NoError(t, err)
	router := newLooperRouter(t, cfg)
	h := &servedModelHarness{router: router, engine: routing.NewEngine(router, routing.DefaultOptions), set: router.upstreamSet(), backend: backend}

	var checkAttempts []func()
	for _, stage := range []string{"request", "response"} {
		primaryFailures := metrics.SystemOneSignalFailoverAttempts.WithLabelValues("failover-primary", stage, "primary", "unavailable")
		backupResponses := metrics.SystemOneSignalFailoverAttempts.WithLabelValues("failover-backup", stage, "backup", "ok")
		failedBefore, okBefore := testutil.ToFloat64(primaryFailures), testutil.ToFloat64(backupResponses)
		checkAttempts = append(checkAttempts, func() {
			require.Equal(t, 1.0, testutil.ToFloat64(primaryFailures)-failedBefore, stage)
			require.Equal(t, 1.0, testutil.ToFloat64(backupResponses)-okBefore, stage)
		})
	}

	request := h.send(t, "/v1/chat/completions", chatBody("Can you investigate the outage?", ""))
	require.Equal(t, 200, request.status)
	require.Equal(t, []string{"deep"}, request.called)
	require.EqualValues(t, 2, primary.asked.Load())
	require.EqualValues(t, 2, backup.asked.Load())
	for _, check := range checkAttempts {
		check()
	}
	require.NotNil(t, request.record.RouteDiagnostics)
	require.Equal(t, map[string]string{"systemone:support_team": "failover-backup"}, request.record.RouteDiagnostics.SignalDeployments)
	require.Equal(t, []string{"failover-backup"}, responseAnsweringDeployments(request.record.Outcomes))

	deployments := make(map[string]string)
	for _, span := range traces.Ended() {
		if value, ok := traceAttributes(span)["routing.signal.deployments"]; ok {
			deployments[span.Name()] = value.AsString()
		}
	}
	require.JSONEq(t, `{"systemone:support_team":"failover-backup"}`, deployments[tracing.SpanSignalEvaluation])
	require.JSONEq(t, `{"systemone:answer_team":"failover-backup"}`, deployments[tracing.SpanRequest])
}

func responseAnsweringDeployments(outcomes []routerreplay.Outcome) []string {
	var deployments []string
	for _, outcome := range outcomes {
		if outcome.Source == "router" && outcome.Target == "systemone:answer_team" && outcome.Metadata["stage"] == "response" {
			deployments = append(deployments, outcome.Metadata["deployment"])
		}
	}
	return deployments
}
