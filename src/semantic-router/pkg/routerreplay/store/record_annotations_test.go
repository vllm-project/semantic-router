package store

import "testing"

func TestCloneRecordClonesDecisionAnnotations(t *testing.T) {
	record := Record{
		RouteDiagnostics: &RouteDiagnostics{
			Annotations:            map[string]interface{}{"reason": "privacy"},
			SignalErrors:           map[string]string{"classifier:risk": "timeout"},
			AppliedUnknownPolicies: map[string]string{"guarded": "no_match"},
		},
	}
	cloned := cloneRecord(record)
	cloned.RouteDiagnostics.Annotations["reason"] = "changed"
	cloned.RouteDiagnostics.SignalErrors["classifier:risk"] = "changed"
	cloned.RouteDiagnostics.AppliedUnknownPolicies["guarded"] = "match"

	if record.RouteDiagnostics.Annotations["reason"] != "privacy" {
		t.Fatalf("original annotations mutated: %v", record.RouteDiagnostics.Annotations)
	}
	if record.RouteDiagnostics.SignalErrors["classifier:risk"] != "timeout" {
		t.Fatalf("original signal errors mutated: %v", record.RouteDiagnostics.SignalErrors)
	}
	if record.RouteDiagnostics.AppliedUnknownPolicies["guarded"] != "no_match" {
		t.Fatalf("original unknown policies mutated: %v", record.RouteDiagnostics.AppliedUnknownPolicies)
	}
}
