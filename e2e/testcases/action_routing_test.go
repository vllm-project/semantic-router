package testcases

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestActionRoutingDomainControls(t *testing.T) {
	cases, err := loadSignalRoutingCases("testdata/action_routing_cases.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, control := range []struct {
		query    string
		decision string
	}{
		{query: "What is the derivative of x squared with respect to x?", decision: "math_decision"},
		{query: "Tell me about cellular biology", decision: "biology_decision"},
	} {
		t.Run(control.decision, func(t *testing.T) {
			var testCase *SignalRoutingCase
			for i := range cases {
				if cases[i].Query == control.query {
					testCase = &cases[i]
					break
				}
			}
			if testCase == nil {
				t.Fatalf("missing domain control for %q", control.query)
			}
			for _, response := range []struct {
				name         string
				decision     string
				action       string
				wantDecision bool
				wantMatch    bool
			}{
				{name: "domain winner with explain attribution", decision: control.decision, action: "explain", wantDecision: true, wantMatch: true},
				{name: "action route displaced domain", decision: "explain_action", action: "explain", wantMatch: true},
				{name: "missing action attribution", decision: control.decision, wantDecision: true},
				{name: "wrong action attribution", decision: control.decision, action: "fix", wantDecision: true},
			} {
				t.Run(response.name, func(t *testing.T) {
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						if r.URL.Path != localChatCompletionsPath || r.Header.Get("x-vsr-debug") != "true" {
							t.Errorf("request did not use the debug chat-completions path")
						}
						w.Header().Set("x-vsr-selected-decision", response.decision)
						w.Header().Set("x-vsr-matched-action", response.action)
						w.WriteHeader(http.StatusOK)
					}))
					t.Cleanup(server.Close)
					_, port, err := net.SplitHostPort(server.Listener.Addr().String())
					if err != nil {
						t.Fatal(err)
					}
					result := testSingleSignalRouting(context.Background(), *testCase, port, true, signalRoutingConfig{
						MatchedHeader: "x-vsr-matched-action", TargetDecision: targetActionDecision,
					})
					if result.Error != "" || result.DecisionCorrect != response.wantDecision || result.MatchCorrect != response.wantMatch {
						t.Fatalf("result = %+v, want decision correct=%v, match correct=%v", result, response.wantDecision, response.wantMatch)
					}
				})
			}
		})
	}
}
