package configschema

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestRecipePoliciesAreDiscoverable(t *testing.T) {
	for _, path := range []string{"recipes.candidate_requirements"} {
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

func TestReplayCaptureDefaultsAndOverridesAreDiscoverable(t *testing.T) {
	global, err := Render(ViewOptions{View: ViewSection, Path: "global.services.router_replay", Expanded: true})
	if err != nil {
		t.Fatal(err)
	}
	plugin, err := Render(ViewOptions{View: ViewSurface, SurfaceKind: "plugin", SurfaceName: "router_replay", Expanded: true})
	if err != nil {
		t.Fatal(err)
	}
	capture := []string{"enabled", "prompt_capture", "capture_request_body", "capture_response_body", "capture_personal_data", "capture_tool_results", "max_records", "max_body_bytes", "max_tool_trace_bytes", "max_tool_trace_steps"}
	for _, field := range capture {
		if !strings.Contains(string(plugin.Body), `"`+field+`"`) {
			t.Errorf("decision plugin replay schema omits %s", field)
		}
		if strings.Contains(string(global.Body), `"`+field+`"`) {
			t.Errorf("the global replay store still declares %s, which the router_replay plugin sets", field)
		}
	}
	for _, field := range []string{"store_backend", "ttl_seconds", "async_writes"} {
		if !strings.Contains(string(global.Body), `"`+field+`"`) {
			t.Errorf("global replay store schema omits %s", field)
		}
	}
	for _, path := range []string{"routing.data_policy", "recipes.data_policy"} {
		if _, err := Render(ViewOptions{View: ViewSection, Path: path}); err == nil {
			t.Errorf("removed policy remains discoverable: %s", path)
		}
	}
}
