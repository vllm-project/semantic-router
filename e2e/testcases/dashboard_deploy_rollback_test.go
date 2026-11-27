package testcases

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestPostDashboardDeployKeepsTheBackupVersionOfEveryAnswer(t *testing.T) {
	for _, tc := range []struct {
		name          string
		status        int
		body          string
		wantConfirmed bool
		wantErr       bool
	}{
		{name: "confirmed", status: http.StatusOK, body: `{"status":"success","version":"v1"}`, wantConfirmed: true},
		{name: "unconfirmed", status: http.StatusAccepted, body: `{"status":"unconfirmed","version":"v1"}`},
		{name: "deferred to a rollout", status: http.StatusAccepted, body: `{"status":"persisted","version":"v1"}`, wantErr: true},
		{name: "failed after saving", status: http.StatusInternalServerError, body: `{"status":"error","version":"v1"}`, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()

			deploy, err := postDashboardDeploy(context.Background(), server.Client(), server.URL, "token", "recipes: []\n")
			if (err != nil) != tc.wantErr {
				t.Fatalf("err = %v, want error %t", err, tc.wantErr)
			}
			if deploy.Version != "v1" {
				t.Fatalf("version = %q, want v1 so the journey can roll the save back", deploy.Version)
			}
			if deploy.Confirmed != tc.wantConfirmed {
				t.Fatalf("confirmed = %t, want %t", deploy.Confirmed, tc.wantConfirmed)
			}
		})
	}
}
