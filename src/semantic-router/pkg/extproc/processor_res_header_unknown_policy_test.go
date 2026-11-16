package extproc

import (
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestAppliedUnknownPoliciesAreSortedInTheTrace(t *testing.T) {
	ctx := debugRequestContext()
	ctx.VSRDecisionDiagnostics.AppliedUnknownPolicies = map[string]string{"strict": "fail_request", "guarded": "no_match"}
	if got := traceMembersOf(t, responseTrace(ctx))[headers.TraceOnUnknown]; got != `("guarded";policy=no_match "strict";policy=fail_request)` {
		t.Fatalf("on_unknown = %s", got)
	}
	if got, ok := traceMembersOf(t, responseTrace(debugRequestContext()))[headers.TraceOnUnknown]; ok {
		t.Fatalf("on_unknown without applied policies = %s", got)
	}
}

func TestRespondDecisionUnresolvedExposesPolicyAndLogsFix(t *testing.T) {
	core, logs := observer.New(zapcore.WarnLevel)
	t.Cleanup(zap.ReplaceGlobals(zap.New(core)))
	router := &OpenAIRouter{}
	ctx := debugRequestContext()
	ctx.RequestID = "unresolved"
	ctx.VSRDecisionDiagnostics.AppliedUnknownPolicies = map[string]string{"guarded": "fail_request"}

	resp := router.respondDecisionUnresolved(ctx, "model", &decision.DecisionUnresolvedError{Decision: "guarded"})

	immediate := resp.GetImmediateResponse()
	if immediate == nil {
		t.Fatal("expected an immediate response")
	}
	if body := string(immediate.GetBody()); !strings.Contains(body, `"code":"decision_unresolved"`) || strings.Contains(body, "guarded") {
		t.Fatalf("body = %q, want the reason code without the decision", body)
	}
	appendRoutedResponseHeaders(resp, ctx)
	if got := traceOf(t, immediate.GetHeaders().GetSetHeaders())[headers.TraceOnUnknown]; got != `("guarded";policy=fail_request)` {
		t.Fatalf("on_unknown = %s", got)
	}
	logged := logs.FilterField(zap.String("request_id", "unresolved")).FilterMessage("decision_unresolved").All()
	if len(logged) != 1 || !strings.Contains(logged[0].ContextMap()["error"].(string), "rules.on_unknown") {
		t.Fatalf("the log does not name the fix: %v", logged)
	}
}

func TestAppliedUnknownPolicyTraceAcrossPolicies(t *testing.T) {
	for _, policy := range config.UnknownPolicies {
		t.Run(string(policy), func(t *testing.T) {
			ctx, err := evaluateGuardedPolicy(policy)
			router := &OpenAIRouter{}
			ctx.Headers = map[string]string{headers.VSRDebug: "true"}
			want := `("guarded";policy=` + string(policy) + `)`
			var got string
			if policy == config.RuleOnUnknownFailRequest {
				if err == nil {
					t.Fatal("expected fail_request to reject the request")
				}
				response := router.respondDecisionUnresolved(ctx, "model", err)
				appendRoutedResponseHeaders(response, ctx)
				got = traceOf(t, response.GetImmediateResponse().GetHeaders().GetSetHeaders())[headers.TraceOnUnknown]
			} else {
				if err != nil {
					t.Fatal(err)
				}
				got = appliedPolicyResponseHeader(t, router, ctx)
			}
			if got != want {
				t.Fatalf("on_unknown = %s, want %s", got, want)
			}
		})
	}
}

func appliedPolicyResponseHeader(t *testing.T, router *OpenAIRouter, ctx *RequestContext) string {
	t.Helper()
	resp, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "200"}}}},
	}, ctx)
	if err != nil {
		t.Fatal(err)
	}
	return traceOf(t, resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders())[headers.TraceOnUnknown]
}

func evaluateGuardedPolicy(policy config.UnknownPolicy) (*RequestContext, error) {
	threshold := 0.5
	engine := decision.NewDecisionEngine([]config.Decision{{Name: "guarded", Rules: config.RuleNode{
		Type:      config.SignalTypeSystemOne,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: &threshold},
		OnUnknown: policy,
	}}}, config.RoutingStrategyPriority)
	_, diagnostics, err := engine.EvaluateDecisionsWithDiagnostics(&decision.SignalMatches{
		SignalErrors: map[string]string{"systemone:risk": "timeout"},
	})
	return &RequestContext{VSRDecisionDiagnostics: diagnostics}, err
}
