package handlers

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSRBenchExperimentRoutesPreserveRequests(t *testing.T) {
	const experiment = "exp-0123456789abcdef0123456789abcdef"
	type forwardedRequest struct{ method, uri, body string }
	var received []forwardedRequest
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		received = append(received, forwardedRequest{r.Method, r.URL.RequestURI(), string(body)})
		_, _ = io.WriteString(w, `{"status":"ready"}`)
	}))
	defer upstream.Close()
	handler, err := NewSRBenchHandler(upstream.URL, "service-secret", false)
	if err != nil {
		t.Fatal(err)
	}
	for _, request := range []forwardedRequest{
		{http.MethodGet, "/datasets/selection?profile=quick", ""},
		{http.MethodGet, "/runs/run-1/replay-options?after=preview%3A123&limit=25", ""},
		{http.MethodPost, "/runs/run-1/candidate-plan", `{"target_ids":["balance"],"mode":"preview"}`},
		{http.MethodGet, "/experiments", ""},
		{http.MethodPost, "/experiments", `{"name":"Routing comparison"}`},
		{http.MethodGet, "/experiments/" + experiment, ""},
		{http.MethodGet, "/experiments/" + experiment + "/runs", ""},
		{http.MethodPost, "/experiments/" + experiment + "/runs", `{"run_id":"run-1","role":"baseline","hypothesis":"Reference"}`},
	} {
		request.uri = SRBenchAPIPath + request.uri
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, srBenchTestRequest(request.method, request.uri, request.body))
		if response.Code != http.StatusOK || received[len(received)-1] != request {
			t.Fatalf("request changed: want=%+v status=%d received=%v", request, response.Code, received)
		}
	}
}

func TestSRBenchExperimentRoutesRejectInvalidPathsAndMethods(t *testing.T) {
	const experiment = "exp-0123456789abcdef0123456789abcdef"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("invalid request reached sr-bench")
		_, _ = io.WriteString(w, `{}`)
	}))
	defer upstream.Close()
	handler, err := NewSRBenchHandler(upstream.URL, "service-secret", false)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		method, path, allow string
		status              int
	}{
		{http.MethodPost, "/datasets/selection", "GET", http.StatusMethodNotAllowed},
		{http.MethodPost, "/runs/run-1/replay-options", "GET", http.StatusMethodNotAllowed},
		{http.MethodGet, "/runs/run-1/candidate-plan", "POST", http.StatusMethodNotAllowed},
		{http.MethodDelete, "/experiments", "GET, POST", http.StatusMethodNotAllowed},
		{http.MethodPost, "/experiments/" + experiment, "GET", http.StatusMethodNotAllowed},
		{http.MethodDelete, "/experiments/" + experiment + "/runs", "GET, POST", http.StatusMethodNotAllowed},
		{http.MethodGet, "/experiments/" + strings.ToUpper(experiment), "", http.StatusNotFound},
		{http.MethodGet, "/experiments/" + experiment[:len(experiment)-1], "", http.StatusNotFound},
		{http.MethodGet, "/experiments/" + experiment + "0", "", http.StatusNotFound},
		{http.MethodGet, "/experiments/" + strings.ReplaceAll(experiment, "f", "g"), "", http.StatusNotFound},
		{http.MethodGet, "/experiments/" + experiment + "/runs/extra", "", http.StatusNotFound},
		{http.MethodPost, "/experiments/" + experiment + "/cancel", "", http.StatusNotFound},
		{http.MethodGet, "/experiments/" + experiment + "%2fruns", "", http.StatusNotFound},
		{http.MethodGet, "/datasets/selection/extra", "", http.StatusNotFound},
		{http.MethodGet, "/runs/run-1/replay-options/extra", "", http.StatusNotFound},
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, srBenchTestRequest(tc.method, SRBenchAPIPath+tc.path, "{}"))
		if response.Code != tc.status || response.Header().Get("Allow") != tc.allow {
			t.Errorf("%s %s: status=%d allow=%q", tc.method, tc.path, response.Code, response.Header().Get("Allow"))
		}
	}
	handler.readonly = true
	for _, path := range []string{"/experiments", "/experiments/" + experiment + "/runs", "/runs/run-1/candidate-plan"} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, srBenchTestRequest(http.MethodPost, SRBenchAPIPath+path, "{}"))
		if response.Code != http.StatusForbidden {
			t.Errorf("read-only dashboard permitted %s: %d", path, response.Code)
		}
	}
}
