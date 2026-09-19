package router

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestSRBenchExperimentsUseAuthenticatedActorAndCSRF(t *testing.T) {
	store, err := auth.NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	svc := auth.NewService(store, "experiment-test-secret", 1)
	const email, password = "experiment@example.com", "test-admin-password"
	if bootstrapErr := svc.EnsureBootstrapAdmin(context.Background(), email, password, "Experiment Admin"); bootstrapErr != nil {
		t.Fatal(bootstrapErr)
	}
	token, user, err := svc.Login(context.Background(), email, password)
	if err != nil {
		t.Fatal(err)
	}
	role := auth.RoleAdmin
	requests := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.Header.Get("Authorization") != "Bearer service-secret" || r.Header.Get("X-SR-Bench-Actor-ID") != user.ID ||
			r.Header.Get("X-SR-Bench-Actor-Role") != role || r.Header.Get("Cookie") != "" || r.Header.Get("X-CSRF-Token") != "" {
			t.Error("proxy did not preserve the trusted actor and isolate browser credentials")
		}
		_, _ = io.WriteString(w, `{"status":"ready"}`)
	}))
	defer upstream.Close()
	t.Setenv("BENCH_EXPERIMENT_TEST_TOKEN", "service-secret")
	mux := http.NewServeMux()
	registerSRBenchRoutes(mux, &config.Config{SRBenchURL: upstream.URL, SRBenchTokenEnv: "BENCH_EXPERIMENT_TEST_TOKEN"})
	handler := wrapWithAuth(mux, svc)
	request := func(method, path, origin, csrf string, authenticated bool) *httptest.ResponseRecorder {
		t.Helper()
		r := httptest.NewRequest(method, "http://dashboard.example/api/sr-bench/v1"+path, strings.NewReader(`{}`))
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Origin", origin)
		r.Header.Set("X-CSRF-Token", csrf)
		r.Header.Set("X-SR-Bench-Actor-ID", "untrusted-browser-value")
		r.Header.Set("X-SR-Bench-Actor-Role", "untrusted-browser-role")
		if authenticated {
			r.AddCookie(&http.Cookie{Name: "vsr_session", Value: token})
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, r)
		return response
	}
	initial := request(http.MethodGet, "/experiments", "", "", true)
	var csrf string
	for _, cookie := range initial.Result().Cookies() {
		if cookie.Name == "vsr_csrf" {
			csrf = cookie.Value
		}
	}
	if initial.Code != http.StatusOK || csrf == "" {
		t.Fatalf("authenticated initial read failed: %d", initial.Code)
	}
	const experiment = "/experiments/exp-0123456789abcdef0123456789abcdef"
	writes := []string{"/experiments", experiment + "/runs", "/runs/run-1/candidate-plan"}
	for _, path := range writes {
		before := requests
		for _, tc := range []struct {
			origin, csrf  string
			authenticated bool
			status        int
		}{
			{"http://dashboard.example", csrf, false, http.StatusUnauthorized},
			{"http://dashboard.example", "", true, http.StatusForbidden},
			{"http://dashboard.example", "invalid", true, http.StatusForbidden},
			{"https://other.example", csrf, true, http.StatusForbidden},
		} {
			response := request(http.MethodPost, path, tc.origin, tc.csrf, tc.authenticated)
			if response.Code != tc.status || requests != before {
				t.Fatalf("offline action auth/CSRF failure for %s: status=%d requests=%d", path, response.Code, requests)
			}
		}
		response := request(http.MethodPost, path, "http://dashboard.example", csrf, true)
		if response.Code != http.StatusOK || requests != before+1 {
			t.Fatalf("authenticated offline action failed for %s: status=%d", path, response.Code)
		}
	}
	if _, err := store.UpdateUserRoleOrStatus(context.Background(), user.ID, auth.RoleRead, ""); err != nil {
		t.Fatal(err)
	}
	role = auth.RoleRead
	for _, path := range []string{"/experiments", experiment, experiment + "/runs", "/datasets/selection", "/replay-options", "/comparison-options"} {
		response := request(http.MethodGet, path, "", "", true)
		if response.Code != http.StatusOK {
			t.Fatalf("viewer could not read %s: %d", path, response.Code)
		}
	}
	before := requests
	for _, path := range writes {
		response := request(http.MethodPost, path, "http://dashboard.example", csrf, true)
		if response.Code != http.StatusForbidden || requests != before {
			t.Fatalf("viewer wrote %s: status=%d requests=%d", path, response.Code, requests)
		}
	}
}
