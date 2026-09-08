package routerreplay

import (
	"strings"
	"testing"
)

func TestNewTrustedFactsReasonMinimized(t *testing.T) {
	r := NewTrustedFactsReason("authoritative", "final", []string{"operator-policy"}, "allow")
	if r.Enforcement != "authoritative" || r.Stage != "final" || r.Outcome != "allow" {
		t.Fatalf("unexpected reason %+v", r)
	}
	if len(r.TrustSources) != 1 || r.TrustSources[0] != "operator-policy" {
		t.Fatalf("unexpected sources %+v", r.TrustSources)
	}
	// Ensure no raw content field exists (serialized form must not contain prompt/args).
	serialized := r.Enforcement + r.Stage + strings.Join(r.TrustSources, ",") + r.Outcome
	if strings.Contains(serialized, "prompt") || strings.Contains(serialized, "credential") {
		t.Fatalf("reason leaks raw content: %q", serialized)
	}
}
