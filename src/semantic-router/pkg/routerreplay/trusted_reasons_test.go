package routerreplay

import (
	"reflect"
	"testing"
)

func TestNewTrustedFactsReasonMinimized(t *testing.T) {
	sources := []string{"operator-policy"}
	r := NewTrustedFactsReason("authoritative", "final", sources, "allow")
	if r.Enforcement != "authoritative" || r.Stage != "final" || r.Outcome != "allow" {
		t.Fatalf("unexpected reason %+v", r)
	}
	if !reflect.DeepEqual(r.TrustSources, []string{"operator-policy"}) {
		t.Fatalf("unexpected sources %+v", r.TrustSources)
	}
	// The reason must be decoupled from the caller slice: later mutations
	// must not leak into the recorded reason.
	sources[0] = "prompt text that must never be recorded"
	if r.TrustSources[0] != "operator-policy" {
		t.Fatalf("reason aliases caller slice: %+v", r.TrustSources)
	}
	// The reason carries only the bounded vocabulary: no prompt, argument,
	// result, credential, or reasoning fields exist to populate.
	v := reflect.ValueOf(r)
	if v.NumField() != 4 {
		t.Fatalf("reason must carry exactly enforcement/stage/sources/outcome, got %d fields", v.NumField())
	}
	for _, field := range []string{"Enforcement", "Stage", "TrustSources", "Outcome"} {
		if !v.FieldByName(field).IsValid() {
			t.Fatalf("reason is missing field %q", field)
		}
	}
}
