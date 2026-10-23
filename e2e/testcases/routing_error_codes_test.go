package testcases

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRoutingErrorProbesRequireRouterOwnedErrorEnvelopes(t *testing.T) {
	for _, probe := range routingErrorProbes {
		t.Run(probe.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					Model string `json:"model"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil || request.Model != probe.model {
					t.Errorf("request model = %q, want %q: %v", request.Model, probe.model, err)
				}
				w.Header().Set("x-vsr-response-path", "error")
				w.WriteHeader(http.StatusBadRequest)
				_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{
					"type": "invalid_request_error", "code": probe.code, "message": probe.message,
				}})
			}))
			defer server.Close()
			if err := checkRoutingErrorProbe(context.Background(), server.Client(), server.URL, probe); err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, testcase := range []struct {
		name, path, code string
		status           int
	}{
		{"default fallback", "upstream", "no_route", http.StatusOK},
		{"backend error", "upstream", "no_route", http.StatusBadRequest},
		{"wrong reason", "error", "model_not_found", http.StatusBadRequest},
	} {
		t.Run(testcase.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("x-vsr-response-path", testcase.path)
				w.WriteHeader(testcase.status)
				_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{
					"type": "invalid_request_error", "code": testcase.code, "message": "no route matched the request",
				}})
			}))
			defer server.Close()
			probe := routingErrorProbe{model: "e2e-no-route", code: "no_route", message: "no route matched the request"}
			if err := checkRoutingErrorProbe(context.Background(), server.Client(), server.URL, probe); err == nil {
				t.Fatal("an incorrect response satisfied the Router error contract")
			}
		})
	}
}
