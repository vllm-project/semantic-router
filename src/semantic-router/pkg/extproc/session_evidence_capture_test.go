package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestClassifyTurnOutcome(t *testing.T) {
	cases := []struct {
		name  string
		ctx   *RequestContext
		usage responseUsageMetrics
		want  sessiontelemetry.TurnOutcomeCategory
	}{
		{
			name: "nil context is unobservable",
			ctx:  nil,
			want: sessiontelemetry.TurnMissing,
		},
		{
			name:  "rate limited upstream is provider noise",
			ctx:   &RequestContext{UpstreamStatusCode: 429},
			usage: responseUsageMetrics{completionTokens: 10, completionTokensReported: true},
			want:  sessiontelemetry.TurnProviderError,
		},
		{
			name: "server error is provider noise",
			ctx:  &RequestContext{UpstreamStatusCode: 503},
			want: sessiontelemetry.TurnProviderError,
		},
		{
			name:  "client error is not provider noise",
			ctx:   &RequestContext{UpstreamStatusCode: 400},
			usage: responseUsageMetrics{completionTokens: 5, completionTokensReported: true},
			want:  sessiontelemetry.TurnProgress,
		},
		{
			name:  "invalid usage is unobservable",
			ctx:   &RequestContext{UpstreamStatusCode: 200},
			usage: responseUsageMetrics{invalid: true},
			want:  sessiontelemetry.TurnMissing,
		},
		{
			name:  "reported empty output is no progress",
			ctx:   &RequestContext{UpstreamStatusCode: 200},
			usage: responseUsageMetrics{completionTokens: 0, completionTokensReported: true},
			want:  sessiontelemetry.TurnNoProgress,
		},
		{
			name:  "unreported output cannot be judged",
			ctx:   &RequestContext{UpstreamStatusCode: 200},
			usage: responseUsageMetrics{completionTokens: 0},
			want:  sessiontelemetry.TurnMissing,
		},
		{
			name:  "output tokens are progress",
			ctx:   &RequestContext{UpstreamStatusCode: 200},
			usage: responseUsageMetrics{completionTokens: 42, completionTokensReported: true},
			want:  sessiontelemetry.TurnProgress,
		},
	}

	for _, tc := range cases {
		if got := classifyTurnOutcome(tc.ctx, tc.usage); got != tc.want {
			t.Fatalf("%s: category = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestClassifyTurnOutcomeAttributionMatchesCategory(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	ctx := &RequestContext{
		SessionID:          "attribution",
		RequestModel:       "model-a",
		UpstreamStatusCode: 503,
	}
	configureProgressEvidence(ctx, config.ProgressGateConfig{Enabled: true, WindowSize: 8, WindowTTLSeconds: 900}, time.Now())
	recordSessionTurnOutcome(ctx, responseUsageMetrics{completionTokens: 5, completionTokensReported: true})

	window := sessiontelemetry.RecentTurnOutcomes(routingSessionStateKey(ctx), time.Now())
	if len(window) != 1 {
		t.Fatalf("window = %+v, want one outcome", window)
	}
	if window[0].Category != sessiontelemetry.TurnProviderError || window[0].ModelAttributable {
		t.Fatalf("provider error must not be model attributable: %+v", window[0])
	}
}

func TestRecordSessionTurnOutcomeSkipsUnresolvableSession(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	ctx := &RequestContext{RequestModel: "model-a", UpstreamStatusCode: 200}
	configureProgressEvidence(ctx, config.ProgressGateConfig{Enabled: true, WindowSize: 8, WindowTTLSeconds: 900}, time.Now())
	recordSessionTurnOutcome(ctx, responseUsageMetrics{completionTokens: 5, completionTokensReported: true})

	if window := sessiontelemetry.RecentTurnOutcomes(routingSessionStateKey(ctx), time.Now()); len(window) != 0 {
		t.Fatalf("session-less request must not write evidence: %+v", window)
	}
}

func TestVerdictOutcomeCategory(t *testing.T) {
	cases := []struct {
		verdict routerLearningOutcomeVerdict
		want    sessiontelemetry.TurnOutcomeCategory
		mapped  bool
	}{
		{routerLearningOutcomeGoodFit, sessiontelemetry.TurnProgress, true},
		{routerLearningOutcomeUnderpowered, sessiontelemetry.TurnRegression, true},
		{routerLearningOutcomeFailed, sessiontelemetry.TurnRegression, true},
		// Cost signals must not feed the escalation streak.
		{routerLearningOutcomeOverprovisioned, "", false},
	}
	for _, tc := range cases {
		got, ok := verdictOutcomeCategory(tc.verdict)
		if ok != tc.mapped || got != tc.want {
			t.Fatalf("verdict %q = %q/%v, want %q/%v", tc.verdict, got, ok, tc.want, tc.mapped)
		}
	}
}

// The ingest path must key the window exactly like the response path, or the
// gate would read two disjoint windows for one session.
func TestRecordIngestedTurnOutcomeSharesSessionKeyWithCapture(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	const recipe = "balance"
	const sessionID = "shared-key"

	ctx := &RequestContext{
		SessionID:          sessionID,
		RequestModel:       "model-a",
		TurnIndex:          2,
		UpstreamStatusCode: 200,
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: recipe})
	configureProgressEvidence(ctx, config.ProgressGateConfig{Enabled: true, WindowSize: 8, WindowTTLSeconds: 900}, time.Now())
	recordSessionTurnOutcome(ctx, responseUsageMetrics{completionTokens: 20, completionTokensReported: true})

	recordIngestedTurnOutcome(routerreplay.RoutingRecord{
		SessionID: sessionID,
		Recipe:    recipe,
		TurnIndex: 1,
		Timestamp: time.Now().Add(-time.Minute),
	}, "model-a", routerLearningOutcomeUnderpowered, 0.9, true)

	key := config.RoutingNamespaceKey(config.RecipeName(recipe), sessionID)
	window := sessiontelemetry.RecentTurnOutcomes(key, time.Now())
	if len(window) != 2 {
		t.Fatalf("both writers must land in one window, got %+v", window)
	}
	// Ordered by event time: the ingested older turn comes first.
	if window[0].TurnIndex != 1 || window[0].Source != sessiontelemetry.TurnSourceOutcomeIngest {
		t.Fatalf("window[0] = %+v, want the ingested turn", window[0])
	}
	if window[1].TurnIndex != 2 || window[1].Source != sessiontelemetry.TurnSourceRouterObserved {
		t.Fatalf("window[1] = %+v, want the captured turn", window[1])
	}
	if window[0].Category != sessiontelemetry.TurnRegression || !window[0].ModelAttributable {
		t.Fatalf("underpowered verdict must be an attributable regression: %+v", window[0])
	}
}

func TestIngestedScoreAvailability(t *testing.T) {
	for _, provided := range []bool{false, true} {
		sessiontelemetry.ResetRouterSessionMemoryForTesting()
		record := routerreplay.RoutingRecord{RequestID: "score", SessionID: "score", Timestamp: time.Now()}
		recordIngestedTurnOutcome(record, "model-a", routerLearningOutcomeGoodFit, 0, provided)
		window := sessiontelemetry.RecentTurnOutcomes(config.RoutingNamespaceKey("", "score"), time.Now())
		if len(window) != 1 || window[0].ConfidenceKnown != provided {
			t.Fatalf("score presence=%t, window=%+v", provided, window)
		}
	}
}

func TestRecordIngestedTurnOutcomeSkipsUnmappedVerdict(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	record := routerreplay.RoutingRecord{
		SessionID: "cost-only",
		Recipe:    "balance",
		Timestamp: time.Now(),
	}
	recordIngestedTurnOutcome(record, "model-a", routerLearningOutcomeOverprovisioned, 0.2, true)

	key := config.RoutingNamespaceKey(config.RecipeName("balance"), "cost-only")
	if window := sessiontelemetry.RecentTurnOutcomes(key, time.Now()); len(window) != 0 {
		t.Fatalf("overprovisioned must not enter the window: %+v", window)
	}
}
