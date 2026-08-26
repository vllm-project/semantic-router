package managementserver

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestRoutingSnapshotExportEncodesRequiredEmptyCollectionsAsArrays(t *testing.T) {
	emptyWire, err := json.Marshal(routingSnapshotExportDTO(routingsnapshot.Snapshot{}))
	if err != nil {
		t.Fatal(err)
	}
	var emptyPayload map[string]json.RawMessage
	if err := json.Unmarshal(emptyWire, &emptyPayload); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"models", "recipes", "entrypoints"} {
		if string(emptyPayload[field]) != "[]" {
			t.Errorf("RoutingSnapshotExport.%s = %s, want []: %s", field, emptyPayload[field], emptyWire)
		}
	}

	wire, err := json.Marshal(routingSnapshotExportDTO(routingsnapshot.Snapshot{Bundle: routingsnapshot.Bundle{
		Models:      []routingsnapshot.Model{{}},
		Recipes:     []routingsnapshot.Recipe{{}},
		Entrypoints: []routingsnapshot.Entrypoint{{}},
	}}))
	if err != nil {
		t.Fatal(err)
	}
	var payload struct {
		Models []struct {
			Backends json.RawMessage `json:"backends"`
		} `json:"models"`
		Recipes []struct {
			Decisions json.RawMessage `json:"decisions"`
		} `json:"recipes"`
		Entrypoints []struct {
			Aliases json.RawMessage `json:"aliases"`
			Rules   json.RawMessage `json:"rules"`
		} `json:"entrypoints"`
	}
	if err := json.Unmarshal(wire, &payload); err != nil {
		t.Fatal(err)
	}
	if string(payload.Models[0].Backends) != "[]" || string(payload.Recipes[0].Decisions) != "[]" ||
		string(payload.Entrypoints[0].Aliases) != "[]" || string(payload.Entrypoints[0].Rules) != "[]" {
		t.Fatalf("nested required collections must be arrays: %s", wire)
	}
}
