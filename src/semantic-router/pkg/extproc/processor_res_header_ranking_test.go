package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

const nameTieBreakReason = "equal priority 0, no comparable confidence, decision name ordering"

func rankedContext(debug bool, winner, runnerUp string) *RequestContext {
	ctx := &RequestContext{
		VSRSelectedDecisionName: winner,
		VSRSelectedModel:        "local",
		VSRDecisionDiagnostics: decision.EvaluationDiagnostics{Ranking: &decision.RankingTrace{
			DecidedBy:  "name",
			Winner:     winner,
			RunnerUp:   runnerUp,
			Reason:     nameTieBreakReason,
			Candidates: 2,
		}},
	}
	if debug {
		ctx.Headers = map[string]string{headers.VSRDebug: "true"}
	}
	return ctx
}

func upstreamHeaders(ctx *RequestContext) map[string]string {
	return headerValuesByName(buildResponseHeaderMutation(ctx).GetSetHeaders())
}

func looperTraceMembers(t *testing.T, ctx *RequestContext) map[string]string {
	t.Helper()
	response := (&OpenAIRouter{}).createLooperResponse(looperResponseWithLatencyAndUsage(), ctx)
	appendRoutedResponseHeaders(response, ctx)
	return traceOf(t, response.GetImmediateResponse().GetHeaders().GetSetHeaders())
}

const wantRanking = `"escalate-hard";reason="` + nameTieBreakReason + `"`

func TestDecisionRankingNamesRunnerUpInTheTrace(t *testing.T) {
	got := traceMembersOf(t, upstreamHeaders(rankedContext(true, "escalate-extreme", "escalate-hard"))[headers.VSRTrace])
	if got[headers.TraceRanking] != wantRanking {
		t.Fatalf("ranking = %s, want %s", got[headers.TraceRanking], wantRanking)
	}
}

// The default surface keeps the final decision but not the comparison.
func TestDecisionRankingOmittedWithoutDebug(t *testing.T) {
	got := upstreamHeaders(rankedContext(false, "escalate-extreme", "escalate-hard"))

	if _, ok := got[headers.VSRTrace]; ok {
		t.Fatalf("%s must stay on the debug surface, got %q", headers.VSRTrace, got[headers.VSRTrace])
	}
	if got[headers.VSRSelectedDecision] != "escalate-extreme" {
		t.Fatalf("%s = %q, want escalate-extreme", headers.VSRSelectedDecision, got[headers.VSRSelectedDecision])
	}
}

func TestDecisionRankingOmittedWithoutRankedComparison(t *testing.T) {
	singleCandidate := rankedContext(true, "escalate-extreme", "")
	noRanking := rankedContext(true, "escalate-extreme", "escalate-hard")
	noRanking.VSRDecisionDiagnostics.Ranking = nil
	// A later stage replaced the ranked winner, so the ranking no longer applies.
	servedElsewhere := rankedContext(true, "escalate-extreme", "escalate-hard")
	servedElsewhere.VSRSelectedDecisionName = "other"

	for name, ctx := range map[string]*RequestContext{
		"single candidate": singleCandidate,
		"no ranking":       noRanking,
		"served elsewhere": servedElsewhere,
	} {
		if got, ok := traceMembersOf(t, upstreamHeaders(ctx)[headers.VSRTrace])[headers.TraceRanking]; ok {
			t.Fatalf("%s: ranking = %s, want it omitted", name, got)
		}
	}
}

func TestDecisionRankingQuotesNames(t *testing.T) {
	got := traceMembersOf(t, upstreamHeaders(rankedContext(true, "a,b", `c"d`))[headers.VSRTrace])[headers.TraceRanking]

	if want := `"c\"d";reason="` + nameTieBreakReason + `"`; got != want {
		t.Fatalf("ranking = %s, want %s", got, want)
	}
}

func TestDecisionRankingOnLooperResponse(t *testing.T) {
	if got := looperTraceMembers(t, rankedContext(true, "escalate-extreme", "escalate-hard"))[headers.TraceRanking]; got != wantRanking {
		t.Fatalf("looper ranking = %s, want %s", got, wantRanking)
	}
	if got := looperTraceMembers(t, rankedContext(false, "escalate-extreme", "escalate-hard")); got != nil {
		t.Fatalf("looper trace = %v without debug, want none", got)
	}
}
