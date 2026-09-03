package extproc

import (
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestAppliedUnknownPolicyHeaderSortsPairs(t *testing.T) {
	ctx := &RequestContext{VSRDecisionDiagnostics: decision.EvaluationDiagnostics{
		AppliedUnknownPolicies: map[string]string{"strict": "fail_request", "guarded": "no_match"},
	}}
	if got := appliedUnknownPolicyHeader(ctx); got != "guarded=no_match,strict=fail_request" {
		t.Fatalf("header = %q", got)
	}
	if got := appliedUnknownPolicyHeader(&RequestContext{}); got != "" {
		t.Fatalf("empty header = %q", got)
	}
}

func TestRespondDecisionUnresolvedExposesPolicyAndFix(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{RequestID: "unresolved", VSRDecisionDiagnostics: decision.EvaluationDiagnostics{
		AppliedUnknownPolicies: map[string]string{"guarded": "fail_request"},
	}}

	resp := router.respondDecisionUnresolved(ctx, "model", &decision.DecisionUnresolvedError{Decision: "guarded"})

	immediate := resp.GetImmediateResponse()
	if immediate == nil {
		t.Fatal("expected an immediate response")
	}
	if body := string(immediate.GetBody()); !strings.Contains(body, "rules.on_unknown") {
		t.Fatalf("body = %q, want fix hint", body)
	}
	if got := immediateHeaderValue(resp, headers.VSRAppliedUnknownPolicy); got != "guarded=fail_request" {
		t.Fatalf("policy header = %q", got)
	}
}

func TestAppliedUnknownPolicyHeaderAcrossPolicies(t *testing.T) {
	threshold := 0.5
	for _, policy := range config.UnknownPolicies {
		t.Run(string(policy), func(t *testing.T) {
			engine := decision.NewDecisionEngine(nil, nil, nil, []config.Decision{{Name: "guarded", Rules: config.RuleNode{
				Type:      config.SignalTypeClassifier,
				Name:      "risk",
				Label:     "RISKY",
				Predicate: &config.NumericPredicate{GTE: &threshold},
				OnUnknown: policy,
			}}}, config.RoutingStrategyPriority)
			_, diagnostics, err := engine.EvaluateDecisionsWithDiagnostics(&decision.SignalMatches{
				SignalErrors: map[string]string{"classifier:risk": "timeout"},
			})
			router := &OpenAIRouter{}
			ctx := &RequestContext{VSRDecisionDiagnostics: diagnostics}
			want := "guarded=" + string(policy)

			if policy == config.RuleOnUnknownFailRequest {
				if err == nil {
					t.Fatal("expected fail_request to reject the request")
				}
				resp := router.respondDecisionUnresolved(ctx, "model", err)
				if got := immediateHeaderValue(resp, headers.VSRAppliedUnknownPolicy); got != want {
					t.Fatalf("503 header = %q, want %q", got, want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			resp, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
				ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "200"}}}},
			}, ctx)
			if err != nil {
				t.Fatal(err)
			}
			var got string
			for _, option := range resp.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders() {
				if option.GetHeader().GetKey() == headers.VSRAppliedUnknownPolicy {
					got = string(option.GetHeader().GetRawValue())
				}
			}
			if got != want {
				t.Fatalf("response header = %q, want %q", got, want)
			}
		})
	}
}
