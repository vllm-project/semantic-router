package handlers

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const validControlledPairRequestJSON = `{
  "client_request_id":"df2c738b-b4d2-4378-b9fb-c0dfce81e448",
  "baseline_source_run_id":"e698c676-1119-4dcb-86fc-3d53c1ecae50",
  "candidate_source_run_id":"d3e802cf-d413-4104-9be8-1e8fc451eb09",
  "baseline_run_id":"05f97c0e-bbc2-4bf6-94bc-2c65c4819d7f",
  "candidate_run_id":"09b53ed3-20de-4480-a861-604156229bbb"
}`

func TestControlledPairWireRejectsClientTargetAndVersionClaims(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := newAuthenticatedEvaluationTestHandler(service, false)

	for _, field := range []string{
		`"endpoint":"https://attacker.invalid"`,
		`"baseline_endpoint":"https://baseline.invalid"`,
		`"candidate_label":"pretend-v2"`,
		`"credential_env":"ATTACKER_KEY"`,
	} {
		body := strings.Replace(validControlledPairRequestJSON, "\n}", ",\n  "+field+"\n}", 1)
		response := httptest.NewRecorder()
		handler.ControlledPairs(
			response,
			httptest.NewRequest(http.MethodPost, evaluationAPIBase+"/controlled-pairs", strings.NewReader(body)),
		)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("client target field %s status=%d body=%s", field, response.Code, response.Body.String())
		}
	}

	readonly := newAuthenticatedEvaluationTestHandler(service, true)
	denied := httptest.NewRecorder()
	readonly.ControlledPairs(
		denied,
		httptest.NewRequest(http.MethodPost, evaluationAPIBase+"/controlled-pairs", strings.NewReader(validControlledPairRequestJSON)),
	)
	if denied.Code != http.StatusForbidden {
		t.Fatalf("readonly controlled pair status=%d body=%s", denied.Code, denied.Body.String())
	}
}

func TestControlledPairLifecycleHTTPContractAndPermissions(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := newAuthenticatedEvaluationTestHandler(service, false)
	pairID := "df2c738b-b4d2-4378-b9fb-c0dfce81e448"

	for _, test := range []struct {
		name, method, path, body string
		want                     int
		allow                    string
	}{
		{"cancel not found", http.MethodPost, evaluationAPIBase + "/controlled-pairs/" + pairID + "/cancel", "", http.StatusNotFound, ""},
		{"delete not found", http.MethodDelete, evaluationAPIBase + "/controlled-pairs/" + pairID, "", http.StatusNotFound, ""},
		{"malformed path", http.MethodPost, evaluationAPIBase + "/controlled-pairs/" + pairID + "/start", "", http.StatusBadRequest, ""},
		{"get not found", http.MethodGet, evaluationAPIBase + "/controlled-pairs/" + pairID, "", http.StatusNotFound, ""},
		{"resource unsupported method", http.MethodPut, evaluationAPIBase + "/controlled-pairs/" + pairID, "", http.StatusMethodNotAllowed, "GET, DELETE"},
		{"cancel unsupported method", http.MethodDelete, evaluationAPIBase + "/controlled-pairs/" + pairID + "/cancel", "", http.StatusMethodNotAllowed, "POST"},
		{"resource query", http.MethodGet, evaluationAPIBase + "/controlled-pairs/" + pairID + "?view=full", "", http.StatusBadRequest, ""},
		{"delete dry run", http.MethodDelete, evaluationAPIBase + "/controlled-pairs/" + pairID + "?dry_run=true", "", http.StatusBadRequest, ""},
		{"cancel body", http.MethodPost, evaluationAPIBase + "/controlled-pairs/" + pairID + "/cancel", `{}`, http.StatusBadRequest, ""},
		{"delete body", http.MethodDelete, evaluationAPIBase + "/controlled-pairs/" + pairID, `{}`, http.StatusBadRequest, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			var body *strings.Reader
			if test.body != "" {
				body = strings.NewReader(test.body)
			} else {
				body = strings.NewReader("")
			}
			handler.ControlledPairLifecycle(response, httptest.NewRequest(test.method, test.path, body))
			if response.Code != test.want {
				t.Fatalf("status=%d body=%s, want %d", response.Code, response.Body.String(), test.want)
			}
			if test.allow != "" && response.Header().Get("Allow") != test.allow {
				t.Fatalf("Allow=%q, want %q", response.Header().Get("Allow"), test.allow)
			}
		})
	}

	readonly := newAuthenticatedEvaluationTestHandler(service, true)
	denied := httptest.NewRecorder()
	readonly.ControlledPairLifecycle(
		denied,
		httptest.NewRequest(http.MethodPost, evaluationAPIBase+"/controlled-pairs/"+pairID+"/cancel", nil),
	)
	if denied.Code != http.StatusForbidden {
		t.Fatalf("readonly lifecycle status=%d body=%s", denied.Code, denied.Body.String())
	}
	read := httptest.NewRecorder()
	readonly.ControlledPairLifecycle(
		read,
		httptest.NewRequest(http.MethodGet, evaluationAPIBase+"/controlled-pairs/"+pairID, nil),
	)
	if read.Code != http.StatusNotFound {
		t.Fatalf("readonly GET passed through mutation gate: status=%d body=%s", read.Code, read.Body.String())
	}
}

func TestControlledPairCreateRejectsUnsupportedShape(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := newAuthenticatedEvaluationTestHandler(service, false)
	for _, test := range []struct {
		name, method, path string
		want               int
		allow              string
	}{
		{"query", http.MethodPost, evaluationAPIBase + "/controlled-pairs?auto_start=false", http.StatusBadRequest, ""},
		{"trailing path", http.MethodPost, evaluationAPIBase + "/controlled-pairs/extra", http.StatusBadRequest, ""},
		{"method", http.MethodGet, evaluationAPIBase + "/controlled-pairs", http.StatusMethodNotAllowed, "POST"},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.ControlledPairs(response, httptest.NewRequest(test.method, test.path, strings.NewReader(validControlledPairRequestJSON)))
			if response.Code != test.want {
				t.Fatalf("status=%d body=%s, want %d", response.Code, response.Body.String(), test.want)
			}
			if test.allow != "" && response.Header().Get("Allow") != test.allow {
				t.Fatalf("Allow=%q, want %q", response.Header().Get("Allow"), test.allow)
			}
		})
	}
}
