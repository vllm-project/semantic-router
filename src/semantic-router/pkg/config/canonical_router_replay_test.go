package config

import (
	"testing"

	"gopkg.in/yaml.v2"
)

func TestCanonicalExportKeepsRouterReplayDisabled(t *testing.T) {
	cfg := &RouterConfig{
		RouterReplay: RouterReplayConfig{
			Enabled:      false,
			StoreBackend: "memory",
			TTLSeconds:   600,
		},
	}

	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatalf("marshal canonical config: %v", err)
	}

	var document map[interface{}]interface{}
	if err := yaml.Unmarshal(encoded, &document); err != nil {
		t.Fatalf("unmarshal canonical config: %v", err)
	}
	global := requireYAMLMap(t, document["global"], "global")
	services := requireYAMLMap(t, global["services"], "global.services")
	replay := requireYAMLMap(t, services["router_replay"], "global.services.router_replay")
	if got, ok := replay["enabled"]; !ok || got != false {
		t.Fatalf("router_replay.enabled = %#v, want explicit false; YAML:\n%s", got, encoded)
	}
}

func requireYAMLMap(t *testing.T, value interface{}, path string) map[interface{}]interface{} {
	t.Helper()
	result, ok := value.(map[interface{}]interface{})
	if !ok {
		t.Fatalf("%s = %#v, want YAML mapping", path, value)
	}
	return result
}
