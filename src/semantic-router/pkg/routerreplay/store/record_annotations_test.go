package store

import "testing"

func TestCloneRecordClonesDecisionAnnotations(t *testing.T) {
	record := Record{
		RouteDiagnostics: &RouteDiagnostics{
			Annotations:  map[string]interface{}{"reason": "privacy"},
			SignalErrors: map[string]string{"classifier:risk": "timeout"},
		},
	}
	cloned := cloneRecord(record)
	cloned.RouteDiagnostics.Annotations["reason"] = "changed"
	cloned.RouteDiagnostics.SignalErrors["classifier:risk"] = "changed"

	if record.RouteDiagnostics.Annotations["reason"] != "privacy" {
		t.Fatalf("original annotations mutated: %v", record.RouteDiagnostics.Annotations)
	}
	if record.RouteDiagnostics.SignalErrors["classifier:risk"] != "timeout" {
		t.Fatalf("original signal errors mutated: %v", record.RouteDiagnostics.SignalErrors)
	}
}
