package extproc

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func reviewRecord(id string, minute int, model, status, outcome, risk string, latency int64, cost *float64) routerreplay.RoutingRecord {
	review := &routerreplay.Review{SchemaVersion: 1, IdentificationSource: "entrypoint", Status: status, LatencyMS: latency}
	if outcome != "" {
		review.Assessment = &routerreplay.ReviewAssessment{Outcome: outcome, RiskLevel: risk}
	}
	record := routerreplay.RoutingRecord{
		ID: id, Timestamp: time.Date(2026, 11, 15, 9, minute, 0, 0, time.UTC), SessionID: "agent-session",
		SelectedModel: model, LifecycleState: routerreplay.LifecycleCompleted, DurationMS: latency,
		RequestKind: store.RequestKindCodexApprovalReview, Review: review,
	}
	if cost != nil {
		currency := "USD"
		record.ActualCost, record.Currency = cost, &currency
	}
	return record
}

func newReviewQueryRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(20, 0))
	price := func(value float64) *float64 { return &value }
	for _, record := range []routerreplay.RoutingRecord{
		reviewRecord("allow-1", 0, "codex-auto-review", "completed", "allow", "low", 100, price(0.01)),
		reviewRecord("allow-2", 1, "codex-auto-review", "completed", "allow", "", 200, price(0.02)),
		reviewRecord("allow-3", 30, "local-reviewer", "completed", "allow", "medium", 300, nil),
		reviewRecord("deny-1", 31, "codex-auto-review", "completed", "deny", "high", 1000, price(0.04)),
		reviewRecord("failed-1", 32, "codex-auto-review", "failed", "", "", 50, nil),
		reviewRecord("invalid-1", 33, "local-reviewer", "invalid", "", "", 400, nil),
		reviewRecord("check-1", 34, "codex-auto-review", "tool_call", "", "", 80, price(0.005)),
		{ID: "main-turn", Timestamp: time.Date(2026, 11, 15, 9, 0, 0, 0, time.UTC), SessionID: "agent-session", SelectedModel: "gpt-6-luna", LifecycleState: routerreplay.LifecycleCompleted},
	} {
		_, err := recorder.AddRecord(record)
		require.NoError(t, err)
	}
	return &OpenAIRouter{ReplayRecorder: recorder}
}

func replayAPIJSON(t *testing.T, router *OpenAIRouter, path string, target interface{}) int {
	t.Helper()
	response := router.handleRouterReplayAPI("GET", path)
	require.NotNil(t, response.GetImmediateResponse())
	immediate := response.GetImmediateResponse()
	require.NoError(t, json.Unmarshal(immediate.Body, target))
	return int(immediate.GetStatus().GetCode())
}

func TestReplayListFiltersReviews(t *testing.T) {
	router := newReviewQueryRouter(t)
	for query, want := range map[string][]string{
		"request_kind=codex_approval_review":                     {"check-1", "invalid-1", "failed-1", "deny-1", "allow-3", "allow-2", "allow-1"},
		"request_kind=codex_approval_review&review_outcome=deny": {"deny-1"},
		"review_status=failed":                                   {"failed-1"},
		"review_risk=medium":                                     {"allow-3"},
		"review_outcome=allow&model=local-reviewer":              {"allow-3"},
		"session_id=agent-session&review_status=all":             {"check-1", "invalid-1", "failed-1", "deny-1", "allow-3", "allow-2", "allow-1", "main-turn"},
	} {
		var list routerreplay.ListResponse
		replayAPIJSON(t, router, "/api/v1/observability/replays?limit=100&"+query, &list)
		ids := make([]string, 0, len(list.Data))
		for _, record := range list.Data {
			ids = append(ids, record.ID)
		}
		require.ElementsMatch(t, want, ids, query)
	}
	var failure map[string]interface{}
	require.Equal(t, 400, replayAPIJSON(t, router, "/api/v1/observability/replays?review_outcome=maybe", &failure))
}

// Approval rate counts verdicts only; failures, invalid answers and tool
// checks are shown apart, and the whole filtered set is summarized.
func TestReplayAggregateSummarizesReviews(t *testing.T) {
	router := newReviewQueryRouter(t)
	var aggregate routerreplay.AggregateResponse
	replayAPIJSON(t, router, "/api/v1/observability/replays/aggregate?request_kind=codex_approval_review", &aggregate)
	reviews := aggregate.Reviews
	require.NotNil(t, reviews)
	require.Equal(t, 7, aggregate.RecordCount)
	require.Equal(t, []int{6, 3, 1, 1, 1, 0, 0, 1}, []int{reviews.Reviews, reviews.Allowed, reviews.Denied, reviews.Invalid, reviews.Failed, reviews.Cancelled, reviews.Pending, reviews.ToolCalls})
	require.InDelta(t, 0.75, *reviews.ApprovalRate, 1e-9)
	require.Equal(t, int64(200), *reviews.LatencyP50MS)
	require.Equal(t, int64(1000), *reviews.LatencyP95MS)
	require.Len(t, reviews.Cost, 1)
	require.Equal(t, "USD", reviews.Cost[0].Currency)
	require.InDelta(t, 0.075, reviews.Cost[0].Amount, 1e-9)
	require.Equal(t, 4, reviews.Cost[0].Requests)
	require.Equal(t, 3, reviews.UnpricedRequests)
	require.Equal(t, "minute", reviews.Bucket)
	require.Len(t, reviews.OverTime, 34)
	require.Equal(t, routerreplay.ReviewTimeBucket{Start: time.Date(2026, 11, 15, 9, 31, 0, 0, time.UTC), Denied: 1}, reviews.OverTime[31])
	require.Equal(t, "codex-auto-review", reviews.ByModel[0].Model)
	require.Equal(t, 5, reviews.ByModel[0].Requests)
	require.Equal(t, []string{"codex-auto-review", "local-reviewer"}, aggregate.AvailableReviewModels)
	for _, entry := range reviews.ByModel {
		// The Dashboard lists each model's costs, so an unpriced one is [].
		require.NotNil(t, entry.Cost, entry.Model)
	}

	var denied routerreplay.AggregateResponse
	replayAPIJSON(t, router, "/api/v1/observability/replays/aggregate?review_outcome=deny", &denied)
	require.Equal(t, 1, denied.Reviews.Reviews)
	require.InDelta(t, 0, *denied.Reviews.ApprovalRate, 1e-9)

	var none routerreplay.AggregateResponse
	replayAPIJSON(t, router, "/api/v1/observability/replays/aggregate?model=gpt-6-luna", &none)
	require.Nil(t, none.Reviews, "a filtered set without reviews has no review summary")
}

func TestReviewTimeBucketsWidenWithTheSpan(t *testing.T) {
	accumulator := newReplayReviewAccumulator()
	for _, offset := range []time.Duration{0, 30 * time.Hour, 50 * time.Hour} {
		record := reviewRecord("r", 0, "m", "completed", "allow", "", 1, nil)
		record.Timestamp = record.Timestamp.Add(offset)
		accumulator.add(record)
	}
	summary := accumulator.result()
	require.Equal(t, "hour", summary.Bucket)
	require.Len(t, summary.OverTime, 51)
	require.Equal(t, 1, summary.OverTime[30].Allowed)
	accumulator.add(func() routerreplay.RoutingRecord {
		record := reviewRecord("late", 0, "m", "completed", "deny", "", 1, nil)
		record.Timestamp = record.Timestamp.Add(10 * 24 * time.Hour)
		return record
	}())
	require.Equal(t, "day", accumulator.result().Bucket)
	require.Nil(t, newReplayReviewAccumulator().result())
}

// A review copies the session's transcript, so it must not become a turn of
// the session's trajectory.
func TestTrajectoryLeavesOutReviewTranscripts(t *testing.T) {
	step := []routerreplay.ToolTraceStep{{Type: replayToolStepUserInput, Text: "fix the build"}}
	turns := buildTrajectoryTurns([]routerreplay.RoutingRecord{
		{ID: "turn", ToolTrace: &routerreplay.ToolTrace{Steps: step}},
		{ID: "review", RequestKind: store.RequestKindCodexApprovalReview, TurnIndex: 1, ToolTrace: &routerreplay.ToolTrace{Steps: step}},
	})
	require.Len(t, turns, 1)
}
