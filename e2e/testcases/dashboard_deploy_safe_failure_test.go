package testcases

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestDashboardSafeFailureRequiresTheUnknownModelRejection(t *testing.T) {
	tests := []struct {
		name    string
		message string
		wantErr bool
	}{
		{name: "unknown target", message: `routing decision references unknown model "e2e-nonexistent-model"`},
		{name: "unrelated broken active config", message: "legacy config: unexpected top-level field model_config", wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadRequest)
				_ = json.NewEncoder(w).Encode(map[string]string{"error": "config_validation_error", "message": test.message})
			}))
			defer server.Close()
			_, err := postRejectedDashboardDeploy(context.Background(), server.Client(), server.URL, "test-token", []byte(`{}`))
			if (err != nil) != test.wantErr {
				t.Fatalf("postRejectedDashboardDeploy() error = %v, wantErr %t", err, test.wantErr)
			}
		})
	}
}
