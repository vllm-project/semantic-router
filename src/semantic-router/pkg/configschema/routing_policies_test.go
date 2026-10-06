package configschema

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestRecipePoliciesAreDiscoverable(t *testing.T) {
	for _, path := range []string{"routing.candidate_requirements", "recipes.routing.candidate_requirements", "routing.data_policy"} {
		representation, err := Render(ViewOptions{View: ViewSection, Path: path, Expanded: true})
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		var doc map[string]any
		if err := json.Unmarshal(representation.Body, &doc); err != nil {
			t.Fatal(err)
		}
		if strings.Contains(path, "candidate_requirements") && (!strings.Contains(string(representation.Body), "declared") || !strings.Contains(string(representation.Body), "known_limits")) {
			t.Fatalf("policy enum missing: %s", path)
		}
	}
	surface, err := Render(ViewOptions{View: ViewSurface, SurfaceKind: "algorithm", SurfaceName: "multi_factor", Expanded: true})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(surface.Body), "latency_metric") {
		t.Fatal("algorithm schema lost metric")
	}
}
