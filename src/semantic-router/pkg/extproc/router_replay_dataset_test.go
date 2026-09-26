package extproc

import (
	"fmt"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const datasetExportQuery = "?seed=seed-a&split=train:8&split=eval:2"

func comparedReplayRecord(t *testing.T, id, recipe string) routerreplay.RoutingRecord {
	t.Helper()
	return routerreplay.RoutingRecord{
		ID:             id,
		Timestamp:      time.Date(2026, 9, 21, 9, 0, 0, 0, time.UTC),
		RequestID:      "req-" + id,
		Recipe:         recipe,
		Decision:       "guard",
		SelectedModel:  "primary-model",
		RequestBody:    `{"messages":[{"role":"user","content":"` + id + `"}]}`,
		ResponseBody:   `{"id":"resp-` + id + `","output":[]}`,
		ResponseStatus: 200,
		LifecycleState: store.LifecycleCompleted,
		Outcomes: []store.Outcome{
			{
				Source:    "primary_response",
				Target:    "model",
				TargetRef: "primary-model",
				Verdict:   "completed",
				Metadata:  map[string]string{"response_sha256": "primary-digest-" + id},
			},
			{
				Source:    "shadow_dispatch",
				Target:    "model",
				TargetRef: "candidate-model",
				Verdict:   "completed",
				Metadata: map[string]string{
					"shadow_model":    "candidate-model",
					"shadow_backend":  "candidate-pool",
					"response_sha256": "shadow-digest-" + id,
				},
			},
		},
	}
}

func newDatasetExportRouter(t *testing.T, records ...routerreplay.RoutingRecord) *OpenAIRouter {
	t.Helper()
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(len(records)+1, 0))
	// Without body capture the recorder stores no request text, and every
	// observation is excluded as request_body_missing rather than compared.
	recorder.SetCapturePolicy(true, true, 0)
	for _, record := range records {
		if _, err := recorder.AddRecord(record); err != nil {
			t.Fatalf("failed to add replay record: %v", err)
		}
	}
	return &OpenAIRouter{
		ReplayRecorder:    recorder,
		ReplayStoreShared: true,
		ReplayRecorders:   map[string]*routerreplay.Recorder{"guard": recorder},
	}
}

func exportDataset(router *OpenAIRouter, query string) *ext_proc.ProcessingResponse {
	return router.handleRouterReplayAPI("GET", "/api/v1/observability/replays/dataset"+query)
}

func assertDatasetStatus(t *testing.T, response *ext_proc.ProcessingResponse, want typev3.StatusCode) {
	t.Helper()
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected an immediate dataset response")
	}
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != want {
		t.Fatalf("dataset status %v, want %v: %s", got, want, response.GetImmediateResponse().Body)
	}
}

// The export reads whole records rather than list summaries. A summary carries
// no request body, so every example would be excluded as request_body_missing
// and the manifest would come back empty while still reporting success.
func TestRouterReplayDatasetExportBuildsExamplesFromWholeRecords(t *testing.T) {
	router := newDatasetExportRouter(t, comparedReplayRecord(t, "replay-1", "vault"))

	response := exportDataset(router, datasetExportQuery)
	assertDatasetStatus(t, response, typev3.StatusCode_OK)
	body := decodeJSONBody(t, response.GetImmediateResponse().Body)

	assertStringField(t, body, "version", "shadow-dataset.v1")
	counts, ok := body["counts"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected manifest counts, got %#v", body["counts"])
	}
	assertIntField(t, counts, "examples", 1)
	if excluded, present := counts["excluded"]; present {
		t.Fatalf("expected no exclusions, got %#v", excluded)
	}
}

// The manifest digest is the dataset's identity, so the same selection under
// the same policy has to hash to the same value on a second export.
func TestRouterReplayDatasetExportRepeatsTheSameDigest(t *testing.T) {
	router := newDatasetExportRouter(t,
		comparedReplayRecord(t, "replay-1", "vault"),
		comparedReplayRecord(t, "replay-2", "vault"),
	)

	first := exportDataset(router, datasetExportQuery)
	assertDatasetStatus(t, first, typev3.StatusCode_OK)
	second := exportDataset(router, datasetExportQuery)
	assertDatasetStatus(t, second, typev3.StatusCode_OK)

	firstDigest := decodeJSONBody(t, first.GetImmediateResponse().Body)["digest"]
	secondDigest := decodeJSONBody(t, second.GetImmediateResponse().Body)["digest"]
	if firstDigest != secondDigest {
		t.Fatalf("digest %v on the second export, want %v", secondDigest, firstDigest)
	}
}

// Filters select which observations the manifest covers, so one that does not
// reach the query would publish a digest for a wider dataset than it names.
func TestRouterReplayDatasetExportAppliesReplayFilters(t *testing.T) {
	router := newDatasetExportRouter(t,
		comparedReplayRecord(t, "replay-1", "vault"),
		comparedReplayRecord(t, "replay-2", "atlas"),
	)

	response := exportDataset(router, datasetExportQuery+"&recipe=vault")
	assertDatasetStatus(t, response, typev3.StatusCode_OK)
	body := decodeJSONBody(t, response.GetImmediateResponse().Body)

	counts, ok := body["counts"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected manifest counts, got %#v", body["counts"])
	}
	assertIntField(t, counts, "records", 1)
	assertIntField(t, counts, "examples", 1)
}

// A page of a larger match is not the dataset the digest claims, so the export
// refuses the selection instead of returning a manifest that cannot be rebuilt.
func TestRouterReplayDatasetExportRefusesAnOversizedSelection(t *testing.T) {
	records := make([]routerreplay.RoutingRecord, 0, routerReplayDatasetMaxRecords+1)
	for index := 0; index <= routerReplayDatasetMaxRecords; index++ {
		records = append(records, comparedReplayRecord(t, fmt.Sprintf("replay-%05d", index), "vault"))
	}
	router := newDatasetExportRouter(t, records...)

	assertDatasetStatus(t, exportDataset(router, datasetExportQuery), typev3.StatusCode_BadRequest)
}

func TestRouterReplayDatasetExportRejectsAnIncompletePolicy(t *testing.T) {
	router := newDatasetExportRouter(t, comparedReplayRecord(t, "replay-1", "vault"))

	for name, query := range map[string]string{
		"missing seed":      "?split=train:8",
		"missing split":     "?seed=seed-a",
		"split without ':'": "?seed=seed-a&split=train",
		"split weight":      "?seed=seed-a&split=train:most",
	} {
		t.Run(name, func(t *testing.T) {
			assertDatasetStatus(t, exportDataset(router, query), typev3.StatusCode_BadRequest)
		})
	}
}

func TestRouterReplayDatasetExportRejectsANonReadMethod(t *testing.T) {
	router := newDatasetExportRouter(t, comparedReplayRecord(t, "replay-1", "vault"))

	response := router.handleRouterReplayAPI("POST", "/api/v1/observability/replays/dataset"+datasetExportQuery)
	assertDatasetStatus(t, response, typev3.StatusCode_MethodNotAllowed)
}

// A cap without a group, or a group without a cap, states no balance. Reading
// either one alone would build a dataset under a rule the caller never gave.
func TestRouterReplayDatasetExportRejectsAHalfStatedBalance(t *testing.T) {
	router := newDatasetExportRouter(t, comparedReplayRecord(t, "replay-1", "vault"))

	for name, query := range map[string]string{
		"group without a cap": datasetExportQuery + "&balance_by=recipe",
		"cap without a group": datasetExportQuery + "&balance_max=1",
		"cap is not a number": datasetExportQuery + "&balance_by=recipe&balance_max=some",
		"group is not a key":  datasetExportQuery + "&balance_by=caller&balance_max=1",
	} {
		t.Run(name, func(t *testing.T) {
			assertDatasetStatus(t, exportDataset(router, query), typev3.StatusCode_BadRequest)
		})
	}
}

func TestRouterReplayDatasetExportAppliesTheBalanceCap(t *testing.T) {
	router := newDatasetExportRouter(t,
		comparedReplayRecord(t, "replay-1", "vault"),
		comparedReplayRecord(t, "replay-2", "vault"),
		comparedReplayRecord(t, "replay-3", "atlas"),
	)

	response := exportDataset(router, datasetExportQuery+"&balance_by=recipe&balance_max=1")
	assertDatasetStatus(t, response, typev3.StatusCode_OK)
	body := decodeJSONBody(t, response.GetImmediateResponse().Body)

	counts, ok := body["counts"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected manifest counts, got %#v", body["counts"])
	}
	assertIntField(t, counts, "records", 3)
	assertIntField(t, counts, "examples", 2)
	excluded, ok := counts["excluded"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected an exclusion tally, got %#v", counts["excluded"])
	}
	assertIntField(t, excluded, "balance_cap", 1)
}
