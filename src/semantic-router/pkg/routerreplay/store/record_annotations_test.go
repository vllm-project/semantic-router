package store

import "testing"

func TestCloneRecordClonesDecisionAnnotations(t *testing.T) {
	record := Record{
		RouteDiagnostics: &RouteDiagnostics{
			Annotations:            map[string]interface{}{"reason": "privacy"},
			SignalErrors:           map[string]string{"classifier:risk": "timeout"},
			SignalDeployments:      map[string]string{"systemone:needs_tools": "cpu-backup"},
			AppliedUnknownPolicies: map[string]string{"guarded": "no_match"},
		},
	}
	cloned := cloneRecord(record)
	cloned.RouteDiagnostics.Annotations["reason"] = "changed"
	cloned.RouteDiagnostics.SignalErrors["classifier:risk"] = "changed"
	cloned.RouteDiagnostics.SignalDeployments["systemone:needs_tools"] = "changed"
	cloned.RouteDiagnostics.AppliedUnknownPolicies["guarded"] = "match"

	if record.RouteDiagnostics.Annotations["reason"] != "privacy" {
		t.Fatalf("original annotations mutated: %v", record.RouteDiagnostics.Annotations)
	}
	if record.RouteDiagnostics.SignalErrors["classifier:risk"] != "timeout" {
		t.Fatalf("original signal errors mutated: %v", record.RouteDiagnostics.SignalErrors)
	}
	if record.RouteDiagnostics.SignalDeployments["systemone:needs_tools"] != "cpu-backup" {
		t.Fatalf("original signal deployments mutated: %v", record.RouteDiagnostics.SignalDeployments)
	}
	if record.RouteDiagnostics.AppliedUnknownPolicies["guarded"] != "no_match" {
		t.Fatalf("original unknown policies mutated: %v", record.RouteDiagnostics.AppliedUnknownPolicies)
	}
}
