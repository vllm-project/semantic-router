package config

import (
	"strings"
	"testing"
)

func TestHumanV03ModelValuesCompileIntoGeneratedSnapshotIdentity(t *testing.T) {
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(humanAuthoringFixture))
	if err != nil {
		t.Fatal(err)
	}
	model := cfg.ModelConfig["local/primary"]
	if !strings.HasPrefix(model.ResourceID, "mdl_") || model.ResourceRevision != 1 {
		t.Fatalf("generated Model identity = %+v", model)
	}
	if model.Execution.MaxRetries != 2 || model.Execution.RequestTimeout != "30s" || model.Execution.StreamTimeout != "2m" {
		t.Fatalf("Model control = %+v", model.Execution)
	}
	if model.RuntimePricing.InputCostPerMillionTokens == nil || *model.RuntimePricing.InputCostPerMillionTokens != "0.5" {
		t.Fatalf("Model pricing = %+v", model.RuntimePricing)
	}
}
