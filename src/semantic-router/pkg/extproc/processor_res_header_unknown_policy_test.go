package extproc

import (
	"strings"
	"testing"

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
	var policyHeader string
	for _, option := range immediate.GetHeaders().GetSetHeaders() {
		if option.GetHeader().GetKey() == headers.VSRAppliedUnknownPolicy {
			policyHeader = string(option.GetHeader().GetRawValue())
		}
	}
	if policyHeader != "guarded=fail_request" {
		t.Fatalf("policy header = %q", policyHeader)
	}
}
