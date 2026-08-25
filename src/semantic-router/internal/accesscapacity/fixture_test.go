package accesscapacity

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
)

func TestFixtureCompilesIndependentPoliciesAndVisibility(t *testing.T) {
	config := DefaultConfig()
	config.KeyCount = 4
	config.Concurrency = 4
	fixture, err := BuildFixture(config, time.Date(2026, 8, 25, 1, 2, 3, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	publication, err := accesspublisher.Compile(fixture.Desired)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if len(publication.Access) != config.KeyCount || len(publication.Credentials) != config.KeyCount ||
		len(fixture.Credentials) != config.KeyCount || len(fixture.Targets) != config.KeyCount {
		t.Fatalf("fixture cardinality = access %d credentials %d plaintext %d targets %d",
			len(publication.Access), len(publication.Credentials), len(fixture.Credentials), len(fixture.Targets))
	}
	seenBindings := make(map[string]struct{}, config.KeyCount)
	for _, document := range publication.Access {
		if len(document.Projection.Grants) != 1 || len(document.Projection.RateBindings) != 1 {
			t.Fatalf("projection %q = %+v", document.KeyID, document.Projection)
		}
		binding := document.Projection.RateBindings[0].BindingID
		if _, exists := seenBindings[binding]; exists {
			t.Fatalf("quota binding %q is shared", binding)
		}
		seenBindings[binding] = struct{}{}
	}
	if fixture.Targets[0] == fixture.Targets[1] || fixture.Targets[0] != fixture.Targets[2] {
		t.Fatalf("visibility fixtures = %v", fixture.Targets)
	}
}
