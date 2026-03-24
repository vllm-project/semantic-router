package extproc

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestHandleMetaRoutingFeedbackAPIListReturnsSummaryRecords(t *testing.T) {
	router := newMetaRoutingFeedbackTestRouter(t)

	response := router.handleMetaRoutingFeedbackAPI("GET", "/v1/meta_routing_feedback?limit=10")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate meta-routing list response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := int(body["count"].(float64)); got != 3 {
		t.Fatalf("expected count=3, got %d", got)
	}
	if got := body["object"]; got != "meta_routing_feedback.list" {
		t.Fatalf("expected list object, got %#v", got)
	}

	data := mustMetaRoutingData(t, body["data"])
	if got := data[0]["id"]; got != "meta-003" {
		t.Fatalf("expected newest record first, got %#v", got)
	}
	if got := data[0]["mode"]; got != config.MetaRoutingModeShadow {
		t.Fatalf("expected shadow mode, got %#v", got)
	}
	actionTypes := mustStringArray(t, data[1]["action_types"])
	if len(actionTypes) != 1 || actionTypes[0] != config.MetaRoutingActionRerunSignalFamilies {
		t.Fatalf("expected planned action type fallback, got %#v", actionTypes)
	}
	signalFamilies := mustStringArray(t, data[1]["refined_signal_families"])
	if len(signalFamilies) != 1 || signalFamilies[0] != config.SignalTypePreference {
		t.Fatalf("expected planned signal family fallback, got %#v", signalFamilies)
	}
}

func TestHandleMetaRoutingFeedbackAPIListAppliesFilters(t *testing.T) {
	router := newMetaRoutingFeedbackTestRouter(t)

	response := router.handleMetaRoutingFeedbackAPI(
		"GET",
		"/v1/meta_routing_feedback?trigger=partition_conflict&mode=shadow&response_status=503",
	)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected filtered meta-routing list response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := int(body["count"].(float64)); got != 1 {
		t.Fatalf("expected one filtered record, got %d", got)
	}
	data := mustMetaRoutingData(t, body["data"])
	if got := data[0]["id"]; got != "meta-003" {
		t.Fatalf("expected meta-003, got %#v", got)
	}
}

func TestHandleMetaRoutingFeedbackAPIRecordLookupReturnsDetail(t *testing.T) {
	router := newMetaRoutingFeedbackTestRouter(t)

	response := router.handleMetaRoutingFeedbackAPI("GET", "/v1/meta_routing_feedback/meta-001")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate detail response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := body["object"]; got != "meta_routing_feedback.record" {
		t.Fatalf("expected detail object, got %#v", got)
	}
	if got := body["id"]; got != "meta-001" {
		t.Fatalf("expected id meta-001, got %#v", got)
	}
	record, ok := body["record"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected embedded record, got %#v", body["record"])
	}
	outcome, ok := record["outcome"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected outcome object, got %#v", record["outcome"])
	}
	if got := outcome["final_decision_name"]; got != "route-b" {
		t.Fatalf("expected final decision route-b, got %#v", got)
	}
}

func TestHandleMetaRoutingFeedbackAPIRejectsInvalidQuery(t *testing.T) {
	router := newMetaRoutingFeedbackTestRouter(t)

	response := router.handleMetaRoutingFeedbackAPI("GET", "/v1/meta_routing_feedback?response_status=nope")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate error response")
	}
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != typev3.StatusCode_BadRequest {
		t.Fatalf("expected bad request status, got %v", got)
	}
}

func TestHandleMetaRoutingFeedbackAggregateAPIReturnsSummary(t *testing.T) {
	router := newMetaRoutingFeedbackTestRouter(t)

	response := router.handleMetaRoutingFeedbackAPI("GET", "/v1/meta_routing_feedback/aggregate")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate aggregate response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := body["object"]; got != "meta_routing_feedback.aggregate" {
		t.Fatalf("expected aggregate object, got %#v", got)
	}
	summary, ok := body["summary"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected aggregate summary, got %#v", body["summary"])
	}
	if got := summary["top_trigger"]; got != "low_decision_margin" {
		t.Fatalf("expected top trigger low_decision_margin, got %#v", got)
	}
	if got := summary["top_root_cause"]; got != "decision_overlap" {
		t.Fatalf("expected top root cause decision_overlap, got %#v", got)
	}
	if got := summary["planned_refinement_rate"].(float64); got <= 0.9 {
		t.Fatalf("expected planned rate close to 1, got %v", got)
	}
}

func TestHandleMetaRoutingFeedbackAggregateAPIAppliesFilters(t *testing.T) {
	router := newMetaRoutingFeedbackTestRouter(t)

	response := router.handleMetaRoutingFeedbackAPI(
		"GET",
		"/v1/meta_routing_feedback/aggregate?mode=observe&action_type=rerun_signal_families",
	)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate aggregate response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := int(body["record_count"].(float64)); got != 1 {
		t.Fatalf("expected one filtered record, got %d", got)
	}
}

type metaRoutingFeedbackFixture struct {
	id        string
	timestamp time.Time
	record    FeedbackRecord
}

func newMetaRoutingFeedbackTestRouter(t *testing.T) *OpenAIRouter {
	t.Helper()

	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	for i, fixture := range metaRoutingFeedbackFixtures() {
		encoded, err := json.Marshal(fixture.record)
		if err != nil {
			t.Fatalf("failed to marshal fixture %d: %v", i, err)
		}
		if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
			ID:             fixture.id,
			Timestamp:      fixture.timestamp,
			RequestID:      fixture.record.Observation.RequestID,
			Decision:       fixture.record.Outcome.FinalDecisionName,
			SelectedModel:  fixture.record.Outcome.FinalModel,
			RequestBody:    string(encoded),
			ResponseStatus: fixture.record.Outcome.ResponseStatus,
			Streaming:      fixture.record.Outcome.Streaming,
			FromCache:      fixture.record.Outcome.CacheHit,
		}); err != nil {
			t.Fatalf("failed to add meta-routing record %s: %v", fixture.id, err)
		}
	}

	return &OpenAIRouter{FeedbackRecorder: recorder}
}

func metaRoutingFeedbackFixtures() []metaRoutingFeedbackFixture {
	return []metaRoutingFeedbackFixture{
		buildActiveMetaRoutingFixture(),
		buildObserveMetaRoutingFixture(),
		buildShadowMetaRoutingFixture(),
	}
}

func buildActiveMetaRoutingFixture() metaRoutingFeedbackFixture {
	return metaRoutingFeedbackFixture{
		id:        "meta-001",
		timestamp: time.Unix(1, 0).UTC(),
		record: FeedbackRecord{
			Mode: config.MetaRoutingModeActive,
			Observation: FeedbackObservation{
				RequestID:    "req-001",
				RequestModel: "model-a",
				RequestQuery: "Need a business answer with citations",
				Trace: &RoutingTrace{
					Mode:                    config.MetaRoutingModeActive,
					MaxPasses:               2,
					PassCount:               2,
					TriggerNames:            []string{metaRoutingTriggerLowDecisionMargin, metaRoutingTriggerProjectionBoundary},
					RefinedSignalFamilies:   []string{config.SignalTypeEmbedding},
					OverturnedDecision:      true,
					LatencyDeltaMs:          42,
					DecisionMarginDelta:     0.31,
					FinalDecisionName:       "route-b",
					FinalDecisionConfidence: 0.88,
					FinalModel:              "model-b",
					FinalAssessment: &MetaAssessment{
						NeedsRefine: true,
						Triggers:    []string{metaRoutingTriggerLowDecisionMargin, metaRoutingTriggerProjectionBoundary},
						RootCauses:  []string{metaRoutingCauseDecisionOverlap, metaRoutingCauseProjectionBoundary},
					},
					FinalPlan: &RefinementPlan{
						Actions: []RefinementActionPlan{
							{Type: config.MetaRoutingActionRerunSignalFamilies, SignalFamilies: []string{config.SignalTypeEmbedding}},
							{Type: config.MetaRoutingActionDisableCompression},
						},
					},
				},
			},
			Action: FeedbackAction{
				Planned:                true,
				Executed:               true,
				ExecutedPassCount:      2,
				ExecutedActionTypes:    []string{config.MetaRoutingActionDisableCompression, config.MetaRoutingActionRerunSignalFamilies},
				ExecutedSignalFamilies: []string{config.SignalTypeEmbedding},
				Plan: &RefinementPlan{
					Actions: []RefinementActionPlan{
						{Type: config.MetaRoutingActionRerunSignalFamilies, SignalFamilies: []string{config.SignalTypeEmbedding}},
						{Type: config.MetaRoutingActionDisableCompression},
					},
				},
			},
			Outcome: FeedbackOutcome{
				FinalDecisionName:       "route-b",
				FinalDecisionConfidence: 0.88,
				FinalModel:              "model-b",
				ResponseStatus:          200,
				RouterReplayID:          "replay-1",
			},
		},
	}
}

func buildObserveMetaRoutingFixture() metaRoutingFeedbackFixture {
	return metaRoutingFeedbackFixture{
		id:        "meta-002",
		timestamp: time.Unix(2, 0).UTC(),
		record: FeedbackRecord{
			Mode: config.MetaRoutingModeObserve,
			Observation: FeedbackObservation{
				RequestID:    "req-002",
				RequestModel: "model-a",
				RequestQuery: "Prefer a short direct answer",
				Trace: &RoutingTrace{
					Mode:                    config.MetaRoutingModeObserve,
					MaxPasses:               1,
					PassCount:               1,
					TriggerNames:            []string{metaRoutingTriggerRequiredFamilyLowConf},
					OverturnedDecision:      false,
					FinalDecisionName:       "route-a",
					FinalDecisionConfidence: 0.66,
					FinalModel:              "model-a",
					FinalAssessment: &MetaAssessment{
						NeedsRefine: true,
						Triggers:    []string{metaRoutingTriggerRequiredFamilyLowConf},
						RootCauses:  []string{metaRoutingCauseLowConfidenceFamily},
					},
					FinalPlan: &RefinementPlan{
						Actions: []RefinementActionPlan{
							{Type: config.MetaRoutingActionRerunSignalFamilies, SignalFamilies: []string{config.SignalTypePreference}},
						},
					},
				},
			},
			Action: FeedbackAction{
				Planned:           true,
				Executed:          false,
				ExecutedPassCount: 1,
				Plan: &RefinementPlan{
					Actions: []RefinementActionPlan{
						{Type: config.MetaRoutingActionRerunSignalFamilies, SignalFamilies: []string{config.SignalTypePreference}},
					},
				},
			},
			Outcome: FeedbackOutcome{
				FinalDecisionName:       "route-a",
				FinalDecisionConfidence: 0.66,
				FinalModel:              "model-a",
				ResponseStatus:          429,
			},
		},
	}
}

func buildShadowMetaRoutingFixture() metaRoutingFeedbackFixture {
	return metaRoutingFeedbackFixture{
		id:        "meta-003",
		timestamp: time.Unix(3, 0).UTC(),
		record: FeedbackRecord{
			Mode: config.MetaRoutingModeShadow,
			Observation: FeedbackObservation{
				RequestID:    "req-003",
				RequestModel: "model-c",
				RequestQuery: "Resolve a domain conflict safely",
				Trace: &RoutingTrace{
					Mode:                    config.MetaRoutingModeShadow,
					MaxPasses:               2,
					PassCount:               2,
					TriggerNames:            []string{metaRoutingTriggerPartitionConflict},
					RefinedSignalFamilies:   []string{},
					OverturnedDecision:      false,
					LatencyDeltaMs:          15,
					FinalDecisionName:       "route-c",
					FinalDecisionConfidence: 0.74,
					FinalModel:              "model-c",
					FinalAssessment: &MetaAssessment{
						NeedsRefine: true,
						Triggers:    []string{metaRoutingTriggerPartitionConflict},
						RootCauses:  []string{metaRoutingCausePartitionConflict},
					},
					FinalPlan: &RefinementPlan{
						Actions: []RefinementActionPlan{
							{Type: config.MetaRoutingActionDisableCompression},
						},
					},
				},
			},
			Action: FeedbackAction{
				Planned:             true,
				Executed:            true,
				ExecutedPassCount:   2,
				ExecutedActionTypes: []string{config.MetaRoutingActionDisableCompression},
				Plan: &RefinementPlan{
					Actions: []RefinementActionPlan{
						{Type: config.MetaRoutingActionDisableCompression},
					},
				},
			},
			Outcome: FeedbackOutcome{
				FinalDecisionName:       "route-c",
				FinalDecisionConfidence: 0.74,
				FinalModel:              "model-c",
				ResponseStatus:          503,
				Streaming:               true,
			},
		},
	}
}

func mustMetaRoutingData(t *testing.T, value interface{}) []map[string]interface{} {
	t.Helper()

	rawData, ok := value.([]interface{})
	if !ok {
		t.Fatalf("expected meta-routing data array, got %#v", value)
	}
	data := make([]map[string]interface{}, 0, len(rawData))
	for _, item := range rawData {
		record, ok := item.(map[string]interface{})
		if !ok {
			t.Fatalf("expected meta-routing record object, got %#v", item)
		}
		data = append(data, record)
	}
	return data
}

func mustStringArray(t *testing.T, value interface{}) []string {
	t.Helper()

	rawValues, ok := value.([]interface{})
	if !ok {
		t.Fatalf("expected string array, got %#v", value)
	}
	values := make([]string, 0, len(rawValues))
	for _, item := range rawValues {
		value, ok := item.(string)
		if !ok {
			t.Fatalf("expected string item, got %#v", item)
		}
		values = append(values, value)
	}
	return values
}

func TestCollectMetaRoutingFeedbackRecordsReturnsParsedPayloads(t *testing.T) {
	router := newMetaRoutingFeedbackTestRouter(t)
	records := router.collectMetaRoutingFeedbackRecords()
	if len(records) != 3 {
		t.Fatalf("expected 3 parsed feedback records, got %d", len(records))
	}
	if records[0].Outcome.FinalDecisionName != "route-c" {
		t.Fatalf("expected newest parsed record first, got %+v", records[0].Outcome)
	}
}

func TestHandleMetaRoutingFeedbackAPIReturnsNotFoundForUnknownRecord(t *testing.T) {
	router := newMetaRoutingFeedbackTestRouter(t)

	response := router.handleMetaRoutingFeedbackAPI("GET", "/v1/meta_routing_feedback/missing")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate error response")
	}
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != typev3.StatusCode_NotFound {
		t.Fatalf("expected not found status, got %v", got)
	}
}

func TestHandleMetaRoutingFeedbackAPIListDefaultsToBoundedPage(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(64, 0))
	for i := 1; i <= 25; i++ {
		record := FeedbackRecord{
			Mode: config.MetaRoutingModeObserve,
			Observation: FeedbackObservation{
				RequestID:    fmt.Sprintf("req-%03d", i),
				RequestModel: "model-a",
				RequestQuery: "query",
				Trace: &RoutingTrace{
					Mode:              config.MetaRoutingModeObserve,
					PassCount:         1,
					FinalDecisionName: "route-a",
					FinalModel:        "model-a",
				},
			},
			Outcome: FeedbackOutcome{
				FinalDecisionName: "route-a",
				FinalModel:        "model-a",
				ResponseStatus:    200,
			},
		}
		encoded, err := json.Marshal(record)
		if err != nil {
			t.Fatalf("failed to marshal record %d: %v", i, err)
		}
		if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
			ID:          fmt.Sprintf("meta-%03d", i),
			RequestID:   fmt.Sprintf("req-%03d", i),
			RequestBody: string(encoded),
			Timestamp:   time.Unix(int64(i), 0).UTC(),
		}); err != nil {
			t.Fatalf("failed to add record %d: %v", i, err)
		}
	}

	router := &OpenAIRouter{FeedbackRecorder: recorder}
	response := router.handleMetaRoutingFeedbackAPI("GET", "/v1/meta_routing_feedback")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate list response")
	}
	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := int(body["count"].(float64)); got != metaRoutingFeedbackDefaultLimit {
		t.Fatalf("expected default count %d, got %d", metaRoutingFeedbackDefaultLimit, got)
	}
}
