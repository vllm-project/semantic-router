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

func upstreamHeaders(ctx *RequestContext, successful bool) map[string]string {
	return headerValuesByName(buildResponseHeaderMutation(ctx, successful).GetSetHeaders())
}

func looperHeaders(ctx *RequestContext) map[string]string {
	response := (&OpenAIRouter{}).createLooperResponse(looperResponseWithLatencyAndUsage(), ctx)
	return headerValuesByName(response.GetImmediateResponse().GetHeaders().GetSetHeaders())
}

func TestDecisionRankingHeaderNamesRunnerUpUnderDebug(t *testing.T) {
	got := upstreamHeaders(rankedContext(true, "escalate-extreme", "escalate-hard"), true)[headers.VSRDecisionRanking]

	if want := "escalate-extreme over escalate-hard: " + nameTieBreakReason; got != want {
		t.Fatalf("%s = %q, want %q", headers.VSRDecisionRanking, got, want)
	}
}

// The default surface keeps the final decision but not the comparison (#2205).
func TestDecisionRankingHeaderOmittedWithoutDebug(t *testing.T) {
	got := upstreamHeaders(rankedContext(false, "escalate-extreme", "escalate-hard"), true)

	if _, ok := got[headers.VSRDecisionRanking]; ok {
		t.Fatalf("%s must stay on the debug surface, got %q", headers.VSRDecisionRanking, got[headers.VSRDecisionRanking])
	}
	if got[headers.VSRSelectedDecision] != "escalate-extreme" {
		t.Fatalf("%s = %q, want escalate-extreme", headers.VSRSelectedDecision, got[headers.VSRSelectedDecision])
	}
}

func TestDecisionRankingHeaderOmittedWithoutRankedComparison(t *testing.T) {
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
		if got, ok := upstreamHeaders(ctx, true)[headers.VSRDecisionRanking]; ok {
			t.Fatalf("%s: %s = %q, want it omitted", name, headers.VSRDecisionRanking, got)
		}
	}
}

func TestDecisionRankingHeaderEscapesNames(t *testing.T) {
	got := upstreamHeaders(rankedContext(true, "a,b", "c;d"), true)[headers.VSRDecisionRanking]

	if want := "a%2Cb over c%3Bd: " + nameTieBreakReason; got != want {
		t.Fatalf("%s = %q, want %q", headers.VSRDecisionRanking, got, want)
	}
}

func TestDecisionRankingHeaderOnLooperResponse(t *testing.T) {
	got := looperHeaders(rankedContext(true, "escalate-extreme", "escalate-hard"))[headers.VSRDecisionRanking]
	if want := "escalate-extreme over escalate-hard: " + nameTieBreakReason; got != want {
		t.Fatalf("looper %s = %q, want %q", headers.VSRDecisionRanking, got, want)
	}

	if got, ok := looperHeaders(rankedContext(false, "escalate-extreme", "escalate-hard"))[headers.VSRDecisionRanking]; ok {
		t.Fatalf("looper %s = %q without debug, want it omitted", headers.VSRDecisionRanking, got)
	}
}

// Decision headers ride only on successful responses.
func TestDecisionRankingHeaderOmittedOnErrorResponse(t *testing.T) {
	if got, ok := upstreamHeaders(rankedContext(true, "escalate-extreme", "escalate-hard"), false)[headers.VSRDecisionRanking]; ok {
		t.Fatalf("%s = %q on an error response, want it omitted", headers.VSRDecisionRanking, got)
	}
}
