package auth

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestSRBenchOfflineExperimentActionsRequireWriteWithoutRun(t *testing.T) {
	for _, path := range []string{
		"/api/sr-bench/v1/experiments",
		"/api/sr-bench/v1/experiments/exp-0123456789abcdef0123456789abcdef/runs",
		"/api/sr-bench/v1/runs/run-1/candidate-plan",
	} {
		for _, removed := range []string{PermEvalRun, PermEvalWrite} {
			t.Run(path+"/without-"+removed, func(t *testing.T) {
				svc := newTestAuthService(t)
				writer := newTestUser(t, svc, "writer@example.com", RoleWrite, "active")
				if _, err := svc.store.db.Exec(`DELETE FROM role_permissions WHERE role = ? AND permission_key = ?`, RoleWrite, removed); err != nil {
					t.Fatal(err)
				}
				if actual := RequiredPermissions(http.MethodPost, path); !reflect.DeepEqual(actual, []string{PermEvalWrite}) {
					t.Fatalf("offline action requires unexpected permissions: %v", actual)
				}
				called := false
				handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					called = true
					w.WriteHeader(http.StatusNoContent)
				}))
				response := httptest.NewRecorder()
				handler.ServeHTTP(response, newAuthenticatedRequest(t, svc, writer, http.MethodPost, path, `{}`))
				want := http.StatusNoContent
				if removed == PermEvalWrite {
					want = http.StatusForbidden
				}
				if response.Code != want || called != (want == http.StatusNoContent) {
					t.Fatalf("status=%d called=%v want=%d", response.Code, called, want)
				}
			})
		}
	}
}
