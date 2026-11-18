package store

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/retention"
)

func TestRouteDiagnosticsRetentionRoundTripAndClone(t *testing.T) {
	value := true
	record := Record{RouteDiagnostics: &RouteDiagnostics{
		Retention: &retention.Outcome{
			SchemaVersion: retention.SchemaVersion,
			Requested:     &retention.Directive{Drop: &value},
			Status:        retention.StatusUnsupported,
			Reason:        "no_retention_adapter",
		},
	}}
	encoded, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Record
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.RouteDiagnostics == nil || decoded.RouteDiagnostics.Retention == nil {
		t.Fatal("retention outcome was not persisted")
	}
	if decoded.RouteDiagnostics.Retention.Status != retention.StatusUnsupported ||
		decoded.RouteDiagnostics.Retention.Reason != "no_retention_adapter" {
		t.Fatalf("retention outcome changed: %+v", decoded.RouteDiagnostics.Retention)
	}
	clone := cloneRouteDiagnostics(record.RouteDiagnostics)
	*clone.Retention.Requested.Drop = false
	if !*record.RouteDiagnostics.Retention.Requested.Drop {
		t.Fatal("retention clone mutation changed original")
	}
}
