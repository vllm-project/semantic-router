package config

import (
	"strings"
	"testing"
)

func TestReplicaPlacementsHaveOneLogicalIdentity(t *testing.T) {
	valid := []ModelDeployment{
		{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-2.0-4B", Replicas: []ModelReplica{{Device: "rocm:0"}, {Device: "rocm:0"}, {Device: "rocm:1"}}},
		{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Replicas: []ModelReplica{{Endpoint: "http://worker-a:8100", ServedName: "first"}, {Endpoint: "http://worker-b:8100", ServedName: "second"}}},
	}
	for _, d := range valid {
		if err := d.WithDefaults().validate(&RouterConfig{}); err != nil {
			t.Fatal(err)
		}
		if d.WithDefaults().Device != "" {
			t.Fatal("replica declaration acquired an ambiguous top-level device")
		}
	}
	invalid := []ModelDeployment{
		{Provider: ModelRuntimeProvider, Artifact: "org/model", Device: "cpu", Replicas: []ModelReplica{{Device: "cpu"}}},
		{Provider: ModelRuntimeProvider, Artifact: "org/model", Replicas: []ModelReplica{{Endpoint: "http://same:8100"}, {Endpoint: "http://same:8100/"}}},
		{Provider: ModelRuntimeProvider, Artifact: "org/model", Replicas: []ModelReplica{{Endpoint: "http://worker:8100", Device: "cpu"}}},
		{Provider: ModelRuntimeProvider, Replicas: []ModelReplica{{Endpoint: "http://worker:8100"}}},
		{Provider: ModelRuntimeProvider, Artifact: "org/model", Replicas: []ModelReplica{{Device: "cpu", ServedName: "hidden"}}},
	}
	for _, d := range invalid {
		if d.WithDefaults().validate(&RouterConfig{}) == nil {
			t.Fatalf("accepted ambiguous placement: %+v", d)
		}
	}
}

func TestReplicaConfigCopiesDoNotAliasAndRetiredProcessIsRejected(t *testing.T) {
	original := map[string]ModelDeployment{"primary": {Replicas: []ModelReplica{{Device: "cpu"}}}}
	cloned := cloneModelMap(original)
	cloned["primary"].Replicas[0].Device = "rocm:0"
	if original["primary"].Replicas[0].Device != "cpu" {
		t.Fatal("canonical export can mutate active worker placements")
	}
	_, err := ParseYAMLBytes([]byte("version: v0.3\nglobal:\n  model_catalog:\n    deployments:\n      primary:\n        provider: model_runtime\n        artifact: org/model\n        process: old-group\n"))
	if err == nil || !strings.Contains(err.Error(), "process") {
		t.Fatalf("retired grouping accepted: %v", err)
	}
}
