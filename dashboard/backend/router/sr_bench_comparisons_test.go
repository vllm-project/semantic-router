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

func TestSRBenchComparisonAllowsReadonlyViewerWithCSRF(t *testing.T) {
	store, err := auth.NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	svc := auth.NewService(store, "comparison-test-secret", 1)
	const email, password = "comparison@example.com", "test-admin-password"
	if bootstrapErr := svc.EnsureBootstrapAdmin(context.Background(), email, password, "Comparison Admin"); bootstrapErr != nil {
		t.Fatal(bootstrapErr)
	}
	token, user, err := svc.Login(context.Background(), email, password)
	if err != nil {
		t.Fatal(err)
	}
	if _, roleErr := store.UpdateUserRoleOrStatus(context.Background(), user.ID, auth.RoleRead, ""); roleErr != nil {
		t.Fatal(roleErr)
	}
	requests := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.Header.Get("Authorization") != "Bearer service-secret" || r.Header.Get("X-SR-Bench-Actor-ID") != user.ID ||
			r.Header.Get("X-SR-Bench-Actor-Role") != auth.RoleRead || r.Header.Get("Cookie") != "" || r.Header.Get("X-CSRF-Token") != "" {
			t.Error("comparison must preserve owner identity and isolate browser credentials")
		}
		if r.Method == http.MethodPost {
			body, readErr := io.ReadAll(r.Body)
			if readErr != nil || r.URL.Path != "/api/sr-bench/v1/comparisons" || string(body) != `{"baseline_run_id":"run-a","candidate_run_id":"run-b"}` {
				t.Error("only the exact comparison read may reach the service")
			}
		}
		_, _ = io.WriteString(w, `{"comparison":"saved-evidence"}`)
	}))
	defer upstream.Close()
	t.Setenv("BENCH_COMPARISON_TEST_TOKEN", "service-secret")
	for _, readonly := range []bool{false, true} {
		mux := http.NewServeMux()
		registerSRBenchRoutes(mux, &config.Config{SRBenchURL: upstream.URL, SRBenchTokenEnv: "BENCH_COMPARISON_TEST_TOKEN", ReadonlyMode: readonly})
		handler := wrapWithAuth(mux, svc)
		request := func(method, path, origin, csrf string, authenticated bool) *httptest.ResponseRecorder {
			t.Helper()
			r := httptest.NewRequest(method, "http://dashboard.example/api/sr-bench/v1"+path,
				strings.NewReader(`{"baseline_run_id":"run-a","candidate_run_id":"run-b"}`))
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("Origin", origin)
			r.Header.Set("X-CSRF-Token", csrf)
			if authenticated {
				r.AddCookie(&http.Cookie{Name: "vsr_session", Value: token})
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, r)
			return response
		}
		initial := request(http.MethodGet, "/runs", "", "", true)
		var csrf string
		for _, cookie := range initial.Result().Cookies() {
			if cookie.Name == "vsr_csrf" {
				csrf = cookie.Value
			}
		}
		if initial.Code != http.StatusOK || csrf == "" {
			t.Fatalf("viewer initial read failed: %d", initial.Code)
		}
		for _, tc := range []struct {
			origin, csrf  string
			authenticated bool
			status        int
		}{
			{"http://dashboard.example", csrf, false, http.StatusUnauthorized},
			{"http://dashboard.example", "", true, http.StatusForbidden},
			{"http://dashboard.example", "invalid", true, http.StatusForbidden},
			{"https://other.example", csrf, true, http.StatusForbidden},
			{"http://dashboard.example", csrf, true, http.StatusOK},
		} {
			before := requests
			response := request(http.MethodPost, "/comparisons", tc.origin, tc.csrf, tc.authenticated)
			forwarded := response.Code == http.StatusOK
			if response.Code != tc.status || (requests == before+1) != forwarded {
				t.Fatalf("readonly=%v comparison status=%d want=%d requests=%d", readonly, response.Code, tc.status, requests)
			}
		}
		for _, path := range []string{"/replays", "/experiments", "/runs", "/runs/run-a/cancel", "/runs/run-a/recover"} {
			before := requests
			response := request(http.MethodPost, path, "http://dashboard.example", csrf, true)
			if response.Code != http.StatusForbidden || requests != before {
				t.Fatalf("viewer unexpectedly mutated %s: status=%d", path, response.Code)
			}
		}
	}
}
