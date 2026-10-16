//go:build !windows

package apiserver

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

const batchClassifierUnavailableMessage = "Batch classification requires unified classifier. Please ensure models are available in ./models/ directory."

type batchClassificationHTTPCase struct {
	name           string
	requestBody    string
	expectedStatus int
	expectedError  string
}

func TestHandleBatchClassificationRejectsInvalidRequests(t *testing.T) {
	apiServer := newBatchClassificationTestServer(nil)
	for _, tt := range invalidBatchClassificationCases() {
		runBatchClassificationHTTPCase(t, apiServer, tt)
	}
}

func TestHandleBatchClassificationReturnsUnavailableWithoutClassifier(t *testing.T) {
	apiServer := newBatchClassificationTestServer(nil)
	for _, tt := range unavailableBatchClassificationCases() {
		runBatchClassificationHTTPCase(t, apiServer, tt)
	}
}

func TestBatchClassificationConfiguration(t *testing.T) {
	for _, tt := range batchClassificationConfigCases() {
		apiServer := newBatchClassificationTestServer(tt.config)
		runBatchClassificationHTTPCase(t, apiServer, tt.batchClassificationHTTPCase)
	}
}

func invalidBatchClassificationCases() []batchClassificationHTTPCase {
	return []batchClassificationHTTPCase{
		{
			name:           "Invalid task_type - jailbreak",
			requestBody:    `{"texts": ["test text"], "task_type": "jailbreak"}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "invalid task_type 'jailbreak'. Supported values: [intent pii security all]",
		},
		{
			name:           "Invalid task_type - random",
			requestBody:    `{"texts": ["test text"], "task_type": "invalid_type"}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "invalid task_type 'invalid_type'. Supported values: [intent pii security all]",
		},
		{
			name:           "Empty texts array",
			requestBody:    `{"texts": [], "task_type": "intent"}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "texts array cannot be empty",
		},
		{
			name:           "Missing texts field",
			requestBody:    `{"task_type": "intent"}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "texts field is required",
		},
		{
			name:           "Invalid JSON",
			requestBody:    `{"texts": [invalid json`,
			expectedStatus: http.StatusBadRequest,
		},
	}
}

func unavailableBatchClassificationCases() []batchClassificationHTTPCase {
	return []batchClassificationHTTPCase{
		{
			name:           "Valid small batch",
			requestBody:    `{"texts": ["What is machine learning?", "How to invest in stocks?"], "task_type": "intent"}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Valid task_type - pii",
			requestBody:    `{"texts": ["test text"], "task_type": "pii"}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Valid task_type - security",
			requestBody:    `{"texts": ["test text"], "task_type": "security"}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Valid task_type - all",
			requestBody:    `{"texts": ["test text"], "task_type": "all"}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Empty task_type defaults to intent",
			requestBody:    `{"texts": ["test text"]}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Valid large batch",
			requestBody:    marshalBatchTexts(50),
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Valid batch with options",
			requestBody:    `{"texts": ["What is quantum physics?"], "task_type": "intent", "options": {"include_probabilities": true}}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Batch too large still checks classifier availability first",
			requestBody:    marshalBatchTexts(101),
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
	}
}

type batchClassificationConfigCase struct {
	batchClassificationHTTPCase
	config *config.RouterConfig
}

func batchClassificationConfigCases() []batchClassificationConfigCase {
	return []batchClassificationConfigCase{
		{
			batchClassificationHTTPCase: batchClassificationHTTPCase{
				name:           "Custom max batch size",
				requestBody:    `{"texts": ["text1", "text2", "text3", "text4"]}`,
				expectedStatus: http.StatusServiceUnavailable,
				expectedError:  batchClassifierUnavailableMessage,
			},
			config: batchClassificationMetricsConfig(),
		},
		{
			batchClassificationHTTPCase: batchClassificationHTTPCase{
				name:           "Default config when config is nil",
				requestBody:    marshalBatchTexts(101),
				expectedStatus: http.StatusServiceUnavailable,
				expectedError:  batchClassifierUnavailableMessage,
			},
			config: nil,
		},
		{
			batchClassificationHTTPCase: batchClassificationHTTPCase{
				name:           "Valid request within custom limits",
				requestBody:    `{"texts": ["text1", "text2"]}`,
				expectedStatus: http.StatusServiceUnavailable,
				expectedError:  batchClassifierUnavailableMessage,
			},
			config: batchClassificationMetricsConfig(),
		},
		{
			batchClassificationHTTPCase: batchClassificationHTTPCase{
				name:           "Batch at max_batch_size passes the bound",
				requestBody:    marshalBatchTexts(3),
				expectedStatus: http.StatusServiceUnavailable,
				expectedError:  batchClassifierUnavailableMessage,
			},
			config: batchClassificationMaxBatchSizeConfig(3),
		},
		{
			batchClassificationHTTPCase: batchClassificationHTTPCase{
				name:           "Batch above max_batch_size is rejected",
				requestBody:    marshalBatchTexts(4),
				expectedStatus: http.StatusBadRequest,
				expectedError:  "texts array exceeds max_batch_size 3",
			},
			config: batchClassificationMaxBatchSizeConfig(3),
		},
	}
}

func TestBatchClassificationMaxBatchSizeFollowsRuntimeConfig(t *testing.T) {
	apiServer := newBatchClassificationTestServer(batchClassificationMaxBatchSizeConfig(0))
	live := batchClassificationMaxBatchSizeConfig(3)
	apiServer.runtimeConfig = newLiveRuntimeConfig(nil, func() *config.RouterConfig { return live }, nil)

	runBatchClassificationHTTPCase(t, apiServer, batchClassificationHTTPCase{
		name:           "Resolver config bounds the batch",
		requestBody:    marshalBatchTexts(4),
		expectedStatus: http.StatusBadRequest,
		expectedError:  "texts array exceeds max_batch_size 3",
	})
}

func batchClassificationMaxBatchSizeConfig(maxBatchSize int) *config.RouterConfig {
	return &config.RouterConfig{
		APIServer: config.APIServer{
			API: config.APIConfig{
				BatchClassification: config.BatchClassificationConfig{MaxBatchSize: maxBatchSize},
			},
		},
	}
}

func batchClassificationMetricsConfig() *config.RouterConfig {
	return &config.RouterConfig{
		APIServer: config.APIServer{
			API: config.APIConfig{
				BatchClassification: config.BatchClassificationConfig{
					Metrics: config.BatchClassificationMetricsConfig{Enabled: true},
				},
			},
		},
	}
}

func marshalBatchTexts(count int) string {
	texts := make([]string, count)
	for i := range texts {
		texts[i] = fmt.Sprintf("Test text %d", i)
	}
	payload, _ := json.Marshal(map[string]interface{}{"texts": texts, "task_type": "intent"})
	return string(payload)
}

func newBatchClassificationTestServer(cfg *config.RouterConfig) *ClassificationAPIServer {
	return &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            cfg,
	}
}

func runBatchClassificationHTTPCase(t *testing.T, apiServer *ClassificationAPIServer, tt batchClassificationHTTPCase) {
	t.Helper()

	t.Run(tt.name, func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/diagnostics/classify/batch", bytes.NewBufferString(tt.requestBody))
		req.Header.Set("Content-Type", "application/json")

		rr := httptest.NewRecorder()
		apiServer.handleBatchClassification(rr, req)

		if rr.Code != tt.expectedStatus {
			t.Fatalf("expected status %d, got %d: %s", tt.expectedStatus, rr.Code, rr.Body.String())
		}
		if tt.expectedError != "" {
			assertJSONErrorMessage(t, rr.Body.Bytes(), tt.expectedError)
		}
	})
}

func assertJSONErrorMessage(t *testing.T, body []byte, expected string) {
	t.Helper()

	var errorResponse map[string]interface{}
	if err := json.Unmarshal(body, &errorResponse); err != nil {
		t.Fatalf("failed to unmarshal error response: %v", err)
	}

	errorData, ok := errorResponse["error"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected error response object, got %+v", errorResponse)
	}

	message, ok := errorData["message"].(string)
	if !ok {
		t.Fatalf("expected error message string, got %+v", errorData)
	}
	if message != expected {
		t.Fatalf("expected error message %q, got %q", expected, message)
	}
}

func batchBoolPtr(v bool) *bool { return &v }

func batchAllUnifiedResponse() *services.UnifiedBatchResponse {
	return &services.UnifiedBatchResponse{
		IntentResults: []classification.IntentResult{
			{Category: "economics", Confidence: 0.5, Probabilities: []float32{0.5, 0.5}},
			{Category: "law", Confidence: 0.25},
			{Category: "math", Confidence: 0.75},
		},
		PIIResults: []classification.PIIResult{
			{HasPII: true, PIITypes: []string{"EMAIL_ADDRESS"}, Confidence: 0.97, ScoresAvailable: batchBoolPtr(true)},
			// Scores unavailable: the numeric zero value must not leak as a probability.
			{HasPII: true, PIITypes: []string{"PERSON"}, ScoresAvailable: batchBoolPtr(false)},
			{HasPII: false, Confidence: 0.125, ScoresAvailable: batchBoolPtr(true)},
		},
		SecurityResults: []classification.SecurityResult{
			{IsJailbreak: true, ThreatType: "jailbreak", Confidence: 0.75, ScoresAvailable: batchBoolPtr(true)},
			// Categorical guard: decision detail is present while the score stays null.
			{IsJailbreak: false, ThreatType: "benign", Decision: &tasks.LabelDecision{Label: "benign", SourceLabel: "safe", Categories: []string{"prompt_injection"}}},
			// Explicitly available zero score must stay 0, not null.
			{IsJailbreak: false, ThreatType: "benign", Confidence: 0, ScoresAvailable: batchBoolPtr(true)},
		},
		ProcessingTimeMs: 30,
		TotalTexts:       3,
	}
}

func buildBatchResponseForTask(unified *services.UnifiedBatchResponse, taskType string, options *ClassificationOptions) BatchClassificationResponse {
	apiServer := newBatchClassificationTestServer(nil)
	return apiServer.buildBatchClassificationResponse(unified, BatchClassificationRequest{
		Texts:    []string{"a", "b", "c"},
		TaskType: taskType,
		Options:  options,
	})
}

func marshalBatchResults(t *testing.T, results []BatchClassificationResult) string {
	t.Helper()
	payload, err := json.Marshal(results)
	if err != nil {
		t.Fatalf("failed to marshal results: %v", err)
	}
	return string(payload)
}

func TestBatchClassificationAllIncludesIntentPIIAndSecurity(t *testing.T) {
	resp := buildBatchResponseForTask(batchAllUnifiedResponse(), "all", nil)
	if len(resp.Results) != 3 || resp.TotalCount != 3 {
		t.Fatalf("expected 3 results, got %d (total_count %d)", len(resp.Results), resp.TotalCount)
	}

	for i, wantIntent := range []struct {
		category   string
		confidence float64
	}{{"economics", 0.5}, {"law", 0.25}, {"math", 0.75}} {
		got := resp.Results[i]
		if got.Category != wantIntent.category || got.Confidence != wantIntent.confidence {
			t.Errorf("text %d: intent = (%s, %v), want (%s, %v)", i, got.Category, got.Confidence, wantIntent.category, wantIntent.confidence)
		}
		if got.ProcessingTimeMs != 10 {
			t.Errorf("text %d: processing_time_ms = %d, want 10", i, got.ProcessingTimeMs)
		}
		if got.PII == nil || got.Security == nil {
			t.Fatalf("text %d: expected pii and security objects, got %+v", i, got)
		}
	}

	// PII: types, confidence, and null confidence when scores are unavailable.
	pii0 := resp.Results[0].PII
	if !pii0.HasPII || !reflect.DeepEqual(pii0.PIITypes, []string{"EMAIL_ADDRESS"}) || pii0.Confidence == nil || *pii0.Confidence != 0.97 {
		t.Errorf("text 0 pii = %+v", pii0)
	}
	pii1 := resp.Results[1].PII
	if !pii1.HasPII || !reflect.DeepEqual(pii1.PIITypes, []string{"PERSON"}) || pii1.Confidence != nil {
		t.Errorf("text 1 pii should have nil confidence, got %+v", pii1)
	}
	pii2 := resp.Results[2].PII
	if pii2.HasPII || len(pii2.PIITypes) != 0 || pii2.Confidence == nil || *pii2.Confidence != 0.125 {
		t.Errorf("text 2 pii = %+v", pii2)
	}

	sec0 := resp.Results[0].Security
	if !sec0.IsJailbreak || sec0.ThreatType != "jailbreak" || sec0.Confidence == nil || *sec0.Confidence != 0.75 {
		t.Errorf("text 0 security = %+v", sec0)
	}
	sec1 := resp.Results[1].Security
	if sec1.IsJailbreak || sec1.ThreatType != "benign" || sec1.Confidence != nil {
		t.Errorf("text 1 security should have nil confidence, got %+v", sec1)
	}
	sec2 := resp.Results[2].Security
	if sec2.IsJailbreak || sec2.Confidence == nil || *sec2.Confidence != 0 {
		t.Errorf("text 2 security = %+v", sec2)
	}
}

func TestBatchClassificationAllWireFormat(t *testing.T) {
	unified := batchAllUnifiedResponse()
	resp := buildBatchResponseForTask(unified, "all", nil)
	got := marshalBatchResults(t, resp.Results)
	want := `[` +
		`{"category":"economics","confidence":0.5,"processing_time_ms":10,` +
		`"pii":{"has_pii":true,"pii_types":["EMAIL_ADDRESS"],"confidence":0.97,"scores_available":true},` +
		`"security":{"is_jailbreak":true,"threat_type":"jailbreak","confidence":0.75,"scores_available":true}},` +
		`{"category":"law","confidence":0.25,"processing_time_ms":10,` +
		`"pii":{"has_pii":true,"pii_types":["PERSON"],"confidence":null,"scores_available":false},` +
		`"security":{"is_jailbreak":false,"threat_type":"benign","confidence":null,"decision":{"label":"benign","source_label":"safe","categories":["prompt_injection"]}}},` +
		`{"category":"math","confidence":0.75,"processing_time_ms":10,` +
		`"pii":{"has_pii":false,"confidence":0.125,"scores_available":true},` +
		`"security":{"is_jailbreak":false,"threat_type":"benign","confidence":0,"scores_available":true}}` +
		`]`
	if got != want {
		t.Fatalf("unexpected wire format\n got: %s\nwant: %s", got, want)
	}

	// The nested objects must encode confidence exactly like the classification types.
	for i, r := range resp.Results {
		assertSameJSON(t, fmt.Sprintf("text %d pii", i), unified.PIIResults[i], r.PII)
		assertSameJSON(t, fmt.Sprintf("text %d security", i), unified.SecurityResults[i], r.Security)
	}
}

func assertSameJSON(t *testing.T, label string, want, got any) {
	t.Helper()
	var wantMap, gotMap map[string]any
	for _, c := range []struct {
		in  any
		out *map[string]any
	}{{want, &wantMap}, {got, &gotMap}} {
		payload, err := json.Marshal(c.in)
		if err != nil {
			t.Fatalf("%s: marshal failed: %v", label, err)
		}
		if err := json.Unmarshal(payload, c.out); err != nil {
			t.Fatalf("%s: unmarshal failed: %v", label, err)
		}
	}
	if !reflect.DeepEqual(wantMap, gotMap) {
		t.Errorf("%s: got %v, want %v", label, gotMap, wantMap)
	}
}

func TestBatchClassificationAllKeepsIntentStatisticsAndProbabilities(t *testing.T) {
	unified := batchAllUnifiedResponse()
	options := &ClassificationOptions{ReturnProbabilities: true}
	all := buildBatchResponseForTask(unified, "all", options)
	intent := buildBatchResponseForTask(unified, "intent", options)

	if !reflect.DeepEqual(all.Statistics, intent.Statistics) {
		t.Errorf("statistics differ: all=%+v intent=%+v", all.Statistics, intent.Statistics)
	}
	if all.Results[0].Probabilities["economics"] != 0.5 {
		t.Errorf("expected probabilities from intent result, got %+v", all.Results[0].Probabilities)
	}
	for i := range all.Results {
		stripped := all.Results[i]
		stripped.PII, stripped.Security = nil, nil
		if !reflect.DeepEqual(stripped, intent.Results[i]) {
			t.Errorf("text %d: all minus pii/security != intent: %+v vs %+v", i, stripped, intent.Results[i])
		}
	}
}

func TestBatchClassificationAllHandlesMismatchedResultLengths(t *testing.T) {
	unified := batchAllUnifiedResponse()
	unified.PIIResults = unified.PIIResults[:1]
	unified.SecurityResults = nil

	resp := buildBatchResponseForTask(unified, "all", nil)
	if len(resp.Results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(resp.Results))
	}
	if resp.Results[0].PII == nil || resp.Results[1].PII != nil || resp.Results[2].PII != nil {
		t.Errorf("pii should only be present at index 0: %+v", resp.Results)
	}
	for i, r := range resp.Results {
		if r.Security != nil {
			t.Errorf("text %d: security should be omitted", i)
		}
	}
	if got := marshalBatchResults(t, resp.Results[1:2]); got != `[{"category":"law","confidence":0.25,"processing_time_ms":10}]` {
		t.Errorf("omitted objects must not appear on the wire, got %s", got)
	}

	// Extra PII/security entries beyond the intent results are ignored.
	unified = batchAllUnifiedResponse()
	unified.IntentResults = unified.IntentResults[:1]
	if resp = buildBatchResponseForTask(unified, "all", nil); len(resp.Results) != 1 {
		t.Errorf("expected 1 result, got %d", len(resp.Results))
	}

	empty := buildBatchResponseForTask(&services.UnifiedBatchResponse{}, "all", nil)
	if len(empty.Results) != 0 {
		t.Errorf("expected no results, got %d", len(empty.Results))
	}
}

// Existing task types must not gain the new objects or change their output.
func TestBatchClassificationOtherTaskTypesUnchanged(t *testing.T) {
	unified := batchAllUnifiedResponse()
	intentWant := `[{"category":"economics","confidence":0.5,"processing_time_ms":10},` +
		`{"category":"law","confidence":0.25,"processing_time_ms":10},` +
		`{"category":"math","confidence":0.75,"processing_time_ms":10}]`
	cases := []struct {
		taskType string
		want     string
	}{
		{"intent", intentWant},
		{"", intentWant},
		{"pii", `[{"category":"EMAIL_ADDRESS","confidence":0.9700000286102295,"processing_time_ms":10},` +
			`{"category":"PERSON","confidence":0,"processing_time_ms":10},` +
			`{"category":"no_pii","confidence":0.125,"processing_time_ms":10}]`},
		{"security", `[{"category":"jailbreak","confidence":0.75,"processing_time_ms":10},` +
			`{"category":"safe","confidence":0,"processing_time_ms":10},` +
			`{"category":"safe","confidence":0,"processing_time_ms":10}]`},
	}
	for _, tc := range cases {
		t.Run("task_type="+tc.taskType, func(t *testing.T) {
			resp := buildBatchResponseForTask(unified, tc.taskType, nil)
			if got := marshalBatchResults(t, resp.Results); got != tc.want {
				t.Fatalf("unexpected results\n got: %s\nwant: %s", got, tc.want)
			}
		})
	}
}

// batchFixtureService serves the fixture unified response through the same
// service seam the handler uses in production.
type batchFixtureService struct {
	classificationService
	response *services.UnifiedBatchResponse
}

func (s *batchFixtureService) HasUnifiedClassifier() bool { return true }

func (s *batchFixtureService) ClassifyBatchUnifiedWithOptions(_ []string, _ interface{}) (*services.UnifiedBatchResponse, error) {
	return s.response, nil
}

func TestHandleBatchClassificationReturnsResultsPerTaskType(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc: &batchFixtureService{response: batchAllUnifiedResponse()},
	}
	intentWant := `[{"category":"economics","confidence":0.5,"processing_time_ms":10},` +
		`{"category":"law","confidence":0.25,"processing_time_ms":10},` +
		`{"category":"math","confidence":0.75,"processing_time_ms":10}]`
	allWant := `[` +
		`{"category":"economics","confidence":0.5,"processing_time_ms":10,` +
		`"pii":{"has_pii":true,"pii_types":["EMAIL_ADDRESS"],"confidence":0.97,"scores_available":true},` +
		`"security":{"is_jailbreak":true,"threat_type":"jailbreak","confidence":0.75,"scores_available":true}},` +
		`{"category":"law","confidence":0.25,"processing_time_ms":10,` +
		`"pii":{"has_pii":true,"pii_types":["PERSON"],"confidence":null,"scores_available":false},` +
		`"security":{"is_jailbreak":false,"threat_type":"benign","confidence":null,"decision":{"label":"benign","source_label":"safe","categories":["prompt_injection"]}}},` +
		`{"category":"math","confidence":0.75,"processing_time_ms":10,` +
		`"pii":{"has_pii":false,"confidence":0.125,"scores_available":true},` +
		`"security":{"is_jailbreak":false,"threat_type":"benign","confidence":0,"scores_available":true}}` +
		`]`
	cases := []struct {
		name string
		body string
		want string
	}{
		{"all", `{"texts":["a","b","c"],"task_type":"all"}`, allWant},
		{"intent", `{"texts":["a","b","c"],"task_type":"intent"}`, intentWant},
		{"default", `{"texts":["a","b","c"]}`, intentWant},
		{
			"pii", `{"texts":["a","b","c"],"task_type":"pii"}`,
			`[{"category":"EMAIL_ADDRESS","confidence":0.9700000286102295,"processing_time_ms":10},` +
				`{"category":"PERSON","confidence":0,"processing_time_ms":10},` +
				`{"category":"no_pii","confidence":0.125,"processing_time_ms":10}]`,
		},
		{
			"security", `{"texts":["a","b","c"],"task_type":"security"}`,
			`[{"category":"jailbreak","confidence":0.75,"processing_time_ms":10},` +
				`{"category":"safe","confidence":0,"processing_time_ms":10},` +
				`{"category":"safe","confidence":0,"processing_time_ms":10}]`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/api/v1/diagnostics/classify/batch", bytes.NewBufferString(tc.body))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()
			apiServer.handleBatchClassification(rr, req)

			if rr.Code != http.StatusOK {
				t.Fatalf("expected status 200, got %d: %s", rr.Code, rr.Body.String())
			}
			var decoded struct {
				Results    json.RawMessage `json:"results"`
				TotalCount int             `json:"total_count"`
			}
			if err := json.Unmarshal(rr.Body.Bytes(), &decoded); err != nil {
				t.Fatalf("failed to decode response: %v", err)
			}
			if decoded.TotalCount != 3 {
				t.Errorf("total_count = %d, want 3", decoded.TotalCount)
			}
			var compact bytes.Buffer
			if err := json.Compact(&compact, decoded.Results); err != nil {
				t.Fatalf("failed to compact results: %v", err)
			}
			if compact.String() != tc.want {
				t.Fatalf("unexpected results\n got: %s\nwant: %s", compact.String(), tc.want)
			}
		})
	}
}
