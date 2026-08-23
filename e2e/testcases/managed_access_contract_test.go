package testcases

import (
	"encoding/json"
	"errors"
	"net/http"
	"testing"
)

func TestManagedAccessRuntimeDiagnosticsPending(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name   string
		status int
		code   string
		want   bool
	}{
		{name: "namespace publication is not visible", status: http.StatusNotFound, code: "not_found", want: true},
		{name: "authentication failure", status: http.StatusUnauthorized, code: "unauthenticated", want: false},
		{name: "authorization failure", status: http.StatusForbidden, code: "forbidden", want: false},
		{name: "service failure", status: http.StatusServiceUnavailable, code: "unavailable", want: false},
		{name: "unrelated missing route", status: http.StatusNotFound, code: "route_not_found", want: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			err := &managedAccessResponseError{status: test.status, code: test.code}
			if got := managedAccessRuntimeDiagnosticsPending(test.status, err); got != test.want {
				t.Fatalf("managedAccessRuntimeDiagnosticsPending(%d, %q) = %t, want %t", test.status, test.code, got, test.want)
			}
		})
	}

	if managedAccessRuntimeDiagnosticsPending(http.StatusNotFound, errors.New("not found")) {
		t.Fatal("untyped 404 error must not be retried")
	}
	if managedAccessRuntimeDiagnosticsPending(http.StatusNotFound, nil) {
		t.Fatal("successful 404 response must not be retried")
	}
}

func TestManagedAccessResponseCode(t *testing.T) {
	t.Parallel()

	for body, want := range map[string]string{
		`{"code":"not_found"}`:                 "not_found",
		`{"error":{"code":"unauthenticated"}}`: "unauthenticated",
		`{"code":`:                             "",
		`{"message":"missing"}`:                "",
	} {
		if got := managedAccessResponseCode([]byte(body)); got != want {
			t.Errorf("managedAccessResponseCode(%q) = %q, want %q", body, got, want)
		}
	}
}

func TestManagedAccessReplicaConvergenceReasonHonorsBarrierRequirement(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name   string
		mutate func(*managedAccessRuntimeDiagnostics)
		want   string
	}{
		{name: "expansive publication"},
		{
			name: "expansive publication rejects unexpected acknowledgement",
			mutate: func(diagnostics *managedAccessRuntimeDiagnostics) {
				diagnostics.Namespace.Publication.BarrierAcknowledgements = []string{"router-a"}
			},
			want: "non-restrictive publication reported barrier acknowledgement state",
		},
		{
			name: "expansive publication rejects unexpected missing acknowledgement",
			mutate: func(diagnostics *managedAccessRuntimeDiagnostics) {
				diagnostics.Namespace.Publication.MissingBarrierAcks = []string{"router-a"}
			},
			want: "non-restrictive publication reported barrier acknowledgement state",
		},
		{
			name: "restrictive publication",
			mutate: func(diagnostics *managedAccessRuntimeDiagnostics) {
				diagnostics.Namespace.Publication.BarrierAcknowledgementsRequired = true
				diagnostics.Namespace.Publication.BarrierAcknowledgements = []string{"router-a", "router-b"}
			},
		},
		{
			name: "restrictive publication requires every acknowledgement",
			mutate: func(diagnostics *managedAccessRuntimeDiagnostics) {
				diagnostics.Namespace.Publication.BarrierAcknowledgementsRequired = true
				diagnostics.Namespace.Publication.BarrierAcknowledgements = []string{"router-a"}
				diagnostics.Namespace.Publication.MissingBarrierAcks = []string{"router-b"}
			},
			want: "access barrier acknowledgements do not cover both Router replicas",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			diagnostics := managedAccessConvergedDiagnostics(t)
			if test.mutate != nil {
				test.mutate(&diagnostics)
			}
			revision, reason := managedAccessReplicaConvergenceReason(diagnostics, "namespace-1", 1)
			if revision != 2 {
				t.Fatalf("applied revision = %d, want 2", revision)
			}
			if reason != test.want {
				t.Fatalf("convergence reason = %q, want %q", reason, test.want)
			}
		})
	}
}

func managedAccessConvergedDiagnostics(t *testing.T) managedAccessRuntimeDiagnostics {
	t.Helper()
	payload, err := json.Marshal(map[string]any{
		"status": "ready",
		"namespace": map[string]any{
			"namespaceId": "namespace-1",
			"publication": map[string]any{
				"readiness": map[string]any{
					"ready": true, "runtimeEpoch": 1, "desiredRevision": 2, "appliedRevision": 2,
				},
				"activeReplicas":                  []string{"router-a", "router-b"},
				"recordedRequiredReplicas":        []string{"router-a", "router-b"},
				"barrierAcknowledgementsRequired": false,
				"barrierAcknowledgements":         []string{},
				"routingAcknowledgements":         []string{"router-a", "router-b"},
				"missingBarrierAcks":              []string{},
				"missingRoutingAcks":              []string{},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var diagnostics managedAccessRuntimeDiagnostics
	if err := json.Unmarshal(payload, &diagnostics); err != nil {
		t.Fatal(err)
	}
	return diagnostics
}
